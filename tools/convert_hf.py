"""Converts a Hugging Face model (safetensors: BF16, F16, FP8) straight to a quantized GGUF, with no full-size
intermediate file: llama.cpp's own converter (convert_hf_to_gguf.py and its conversion/ classes) reads the model, names
the tensors and writes the metadata and tokenizer; just before each tensor is written, the tensors your rules file
names are quantized (the i-quants on the GPU with --gpu, the rest through ggml on the CPU). Every architecture
llama.cpp converts works here as soon as llama.cpp does: update llama.cpp, run ./setup.sh.

usage: convert_hf.py <hf model folder> <out.gguf> [--rules=<file>] [--imatrix=<imatrix.gguf>] [--gpu[=<device>]]
                     [--outtype=bf16|q8_0|f16] [--split-max-size=50G] [--llama=<llama.cpp folder>] [--dry-run]
  --outtype   what tensors no rule names become (llama.cpp's converter decides which stay F32: norms, routers ...);
              default q8_0
  --rules     "<tensor name regex> <type>" lines as for requant_experts.py, matched against the GGUF names
              (blk.N.ffn_up_exps.weight ...); the first match wins. Any ggml type; IQ1_*/IQ2_* need --imatrix.
  --imatrix   an imatrix of this model (tools/imatrix.py run ... on a first conversion without the IQ1/IQ2 types).
              A tensor it has no data for gets --outtype instead of an IQ1/IQ2 type, with a warning.
  --split-max-size  passed to the converter (shards of at most that size, e.g. 50G); the out name gets -0000N-of-...

The usual way for a new model (no GGUF published yet):
  1. convert_hf.py <hf folder> <first.gguf> --outtype=q8_0 --split-max-size=50G
     (or, when even Q8_0 does not fit your SSD: --rules with iq4_xs for the experts, which needs no imatrix)
  2. tools/imatrix.py run <first shard 1> calib.txt imatrix.gguf
  3. convert_hf.py <hf folder> <final.gguf> --rules=<file> --imatrix=imatrix.gguf --gpu --split-max-size=50G
     (from the original weights again, so no rounding is added twice); delete first.gguf
Needs torch, transformers and sentencepiece in the venv (./setup.sh installs them).
"""
import ctypes
import os
import re
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# llama.cpp's converter and its gguf-py go together (a pip gguf lacks the newest architectures): load that gguf-py
# before anything imports gguf
_LLAMA = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--llama=")), None)
if not _LLAMA:
    from requant_experts import _env as _e0  # noqa: E402  (only reads trench.env)
    _LLAMA = _e0("LLAMA_DIR")
if _LLAMA:
    os.environ["GGUF_PY"] = os.path.join(_LLAMA, "gguf-py")
    for _m in [m for m in sys.modules if m == "gguf" or m.startswith("gguf.") or m == "requant_experts"]:
        del sys.modules[_m]
from requant_experts import (_env, LIBGGML, NEEDS_IMATRIX, Q, TypeTraits, fit_type, gguf, np,  # noqa: E402
                             read_imatrix, read_rules)

CUR = threading.local()   # the GGUF name of the tensor llama.cpp's converter is about to quantize


def main() -> int:
    flags = dict((a.split("=", 1) + [""])[:2] for a in sys.argv[1:] if a.startswith("--"))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        print(__doc__)
        return 2
    hf_dir, out = args
    llama = _LLAMA
    if not llama or not os.path.exists(os.path.join(llama, "convert_hf_to_gguf.py")):
        sys.exit("pass --llama=<llama.cpp folder> (or run ./setup.sh, which writes LLAMA_DIR to trench.env)")
    sys.path.insert(0, llama)
    sys.path.insert(1, os.path.join(llama, "gguf-py"))
    try:
        import torch  # noqa: F401
        from conversion import ModelBase, ModelType, get_model_architecture, get_model_class
    except ImportError as e:
        sys.exit("%s - install the converter's packages: pip install -r %s/requirements/requirements-convert_hf_to_gguf.txt"
                 % (e, llama))
    import gguf as lgguf   # (llama.cpp's gguf-py, the one its converter uses)
    from gguf.lazy import LazyNumpyTensor

    rules = read_rules(flags["--rules"]) if flags.get("--rules") else []
    imat = read_imatrix(flags["--imatrix"]) if flags.get("--imatrix") else {}
    if not imat and any(t in NEEDS_IMATRIX for _, t in rules):
        sys.exit("the rules ask for IQ1/IQ2 types, which need --imatrix (make a first conversion without them, then "
                 "tools/imatrix.py run)")
    outtype = {"bf16": lgguf.LlamaFileType.MOSTLY_BF16, "q8_0": lgguf.LlamaFileType.MOSTLY_Q8_0,
               "f16": lgguf.LlamaFileType.MOSTLY_F16}[flags.get("--outtype") or "q8_0"]
    dense = {lgguf.LlamaFileType.MOSTLY_BF16: Q.BF16, lgguf.LlamaFileType.MOSTLY_Q8_0: Q.Q8_0,
             lgguf.LlamaFileType.MOSTLY_F16: Q.F16}[outtype]

    lib = ctypes.CDLL(LIBGGML)
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                        ctypes.c_int64, ctypes.c_void_p]
    lib.ggml_quantize_init.argtypes = [ctypes.c_int]
    lib.ggml_get_type_traits.restype = ctypes.POINTER(TypeTraits)
    lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
    gq = None
    if "--gpu" in flags:
        here = os.environ.get("GPUQ_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gpuq"))
        gq = ctypes.CDLL(os.path.join(here, _env("GPUQ_LIB") or "libgpuq.so"))
        gq.gpuq_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
        gq.gpuq_quantize.restype = ctypes.c_size_t
        gq.gpuq_quantize.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                     ctypes.c_void_p]
        gq.gpuq_block_bytes.restype = ctypes.c_size_t
        if gq.gpuq_init(os.path.join(here, "gpuq_tables.bin").encode(), int(flags["--gpu"] or 0)) != 0:
            sys.exit("gpuq_init failed")
    stats = {"gpu": 0, "cpu": 0}

    def quantize_array(a, typ, name):
        """a: float rows (..., ne0) -> the GGUF bytes of typ, one expert (one 2-D slice) at a time"""
        a = np.ascontiguousarray(a, dtype=np.float32)
        ne0 = a.shape[-1]
        mats = a.reshape(-1, a.shape[-2] if a.ndim >= 2 else 1, ne0)
        im = imat.get(name)
        bs, ts = gguf.GGML_QUANT_SIZES[typ]
        per = mats.shape[1] * ne0 // bs * ts
        outb = np.empty((mats.shape[0], per), dtype=np.uint8)
        lib.ggml_quantize_init(int(typ))
        for e in range(mats.shape[0]):
            w = None if im is None else np.ascontiguousarray(im[e if len(im) > 1 else 0], dtype=np.float32)
            if gq is not None and w is not None and gq.gpuq_block_bytes(int(typ)) > 0:
                got = gq.gpuq_quantize(int(typ), mats[e].ctypes.data, outb[e].ctypes.data, mats.shape[1], ne0,
                                       w.ctypes.data)
                stats["gpu"] += 1
            else:
                got = lib.ggml_quantize_chunk(int(typ), mats[e].ctypes.data, outb[e].ctypes.data, 0, mats.shape[1], ne0,
                                              None if w is None else w.ctypes.data)
                stats["cpu"] += 1
            if got != per:
                raise RuntimeError("%s expert %d: %d bytes, expected %d" % (name, e, got, per))
        return outb.reshape(*a.shape[:-1], ne0 // bs * ts)

    picked = {}   # GGUF name -> our type
    refit = {}    # GGUF name -> the type written instead (rows the asked type cannot hold: llama-quantize's fallback)

    orig_quantize = lgguf.quants.quantize

    def quantize(data, qtype):
        name = getattr(CUR, "name", None)
        typ = picked.get(name)
        if typ is None or Q(int(qtype)) != typ or isinstance(data, lgguf.LazyChunkedTensor):
            return orig_quantize(data, qtype)
        fit = fit_type(typ, data.shape[-1])
        if fit != typ:
            print("note: %s rows of %d: %s instead of %s" % (name, data.shape[-1], fit.name, typ.name))
            refit[name] = fit
            if fit == Q.F16:
                return orig_quantize(data, lgguf.GGMLQuantizationType.F16)
            typ = fit
        shape_b = lgguf.quants.quant_shape_to_byte_shape(data.shape, lgguf.GGMLQuantizationType(int(typ)))
        if isinstance(data, LazyNumpyTensor):
            return LazyNumpyTensor._wrap_fn(lambda a, t=typ, n=name: quantize_array(a, t, n),
                                            meta_noop=(np.uint8, lambda s, sb=shape_b: sb))(data)
        return quantize_array(data, typ, name)

    lgguf.quants.quantize = quantize

    hparams = ModelBase.load_hparams(__import__("pathlib").Path(hf_dir), False)
    arch = get_model_architecture(hparams, ModelType.TEXT)
    base = get_model_class(arch)

    class Model(base):
        model_arch = base.model_arch   # (the converter checks that each subclass names one)
        def tensor_force_quant(self, name, new_name, bid, n_dims):
            CUR.name = new_name
            want = super().tensor_force_quant(name, new_name, bid, n_dims)
            typ = next((t for pat, t in rules if pat.search(new_name)), None)
            if typ is None or n_dims < 2:
                return want
            if typ in NEEDS_IMATRIX and new_name not in imat:
                print("warning: %s has no imatrix data: %s instead of %s" % (new_name, dense.name, typ.name))
                return dense
            picked[new_name] = typ
            return lgguf.GGMLQuantizationType(int(typ))

    print("architecture %s, converter %s, untouched tensors -> %s, %d rules%s%s" % (
        arch, base.__name__, dense.name, len(rules), ", imatrix %d tensors" % len(imat) if imat else "",
        ", GPU" if gq is not None else ""))
    import pathlib
    split = flags.get("--split-max-size") or "0"
    mult = {"K": 1000, "M": 1000**2, "G": 1000**3}
    split_bytes = int(split[:-1]) * mult[split[-1]] if split[-1] in mult else int(split)
    m = Model(pathlib.Path(hf_dir), outtype, pathlib.Path(out), eager=False, split_max_size=split_bytes,
              dry_run="--dry-run" in flags)
    add = m.gguf_writer.add_tensor

    def add_tensor(name, tensor, *a, **kw):
        if name in refit:
            kw["raw_dtype"] = lgguf.GGMLQuantizationType(int(refit[name]))
        return add(name, tensor, *a, **kw)

    m.gguf_writer.add_tensor = add_tensor
    m.write()
    print("done: %d slices quantized on the GPU, %d on the CPU" % (stats["gpu"], stats["cpu"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
