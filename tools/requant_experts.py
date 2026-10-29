"""Re-quantizes the routed experts of a (split) GGUF to smaller i-quants with an imatrix; copies everything else byte
for byte. Rules may name 2-D tensors too (attention, shared experts, embeddings: e.g. BF16 -> Q8_0 / Q5_K), quantized
the same way as one expert.

usage: requant_experts.py <in shard 1> <out dir> <rules file> --imatrix=<imatrix.gguf> [--check] [--threads=N]
                         [--gpu[=<device>]]
  rules file: one "<tensor name regex> <type>" a line (# comments), the first match wins, e.g.
      blk\\.(2[4-9]|30)\\.ffn_(gate|up)_exps\\.weight  iq3_xxs
      blk\\.\\d+\\.ffn_down_exps\\.weight             iq2_s
Each input shard becomes one output shard of the same name (the split metadata is copied as is); a missing shard is
skipped, and one gguf-py cannot read (DeepSeek V4.1's Engram-only shard) is copied as is.  Each expert is dequantized with ggml's own to_float
(any source type ggml reads: MXFP4, the i-quants, the k-quants) and quantized by ggml_quantize_chunk with that
expert's imatrix row (llama-quantize's normalization: in_sum2 / counts, 1 for an expert the calibration never routed
to).  The source's rounding error and the new one add up: start from the most precise file to hand.

Rows that the asked type cannot hold (ne0 not a multiple of its block, e.g. Qwen3.8's 640-long down rows) get
llama-quantize's fallback type (IQ1_*/IQ2_*/IQ3_*/IQ4_XS -> IQ4_NL, Q2_K/Q3_K -> Q4_0, Q4_K -> Q5_0, Q5_K -> Q5_1,
Q6_K -> Q8_0, F16 when even that does not fit).  A tensor the imatrix has no data for (e.g. Ornith's NextN layer) is
quantized without one when the type allows it; the very low-bit types (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S) need it,
as in llama-quantize, so such a tensor is left as is - give it a rule of its own with another type (iq4_xs, q4_k ...).

--gpu quantizes IQ1_S / IQ1_M / IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS / IQ4_NL on the GPU with
gpuq/libgpuq.so (ggml's quantizers line for line: byte-identical output, checked by gpuq/test_gpuq.py); sources in
F32 / F16 / BF16 / Q8_0 / MXFP4 are dequantized on the GPU as well (others on the CPU threads); other target types, and
tensors without imatrix data, stay on the CPU.  GPUQ_LIB picks the library (libgpuq_cuda.so for CUDA), GPUQ_DIR its
folder.

Environment: GGUF_PY = llama.cpp/gguf-py (default: the installed gguf package), LIBGGML = <llama.cpp build>/bin/
libggml-base.so (the CPU quantizer, and the reference the GPU output matches).
"""
import concurrent.futures
import ctypes
import os
import re
import shutil
import sys
import threading
import time

if os.environ.get("GGUF_PY"):   # llama.cpp/gguf-py; without it the installed gguf package (pip install gguf)
    sys.path.insert(0, os.environ["GGUF_PY"])
import numpy as np  # noqa: E402
import gguf  # noqa: E402
from gguf import GGMLQuantizationType as Q  # noqa: E402

def _env(name):
    """name from the environment, else from trench.env (written by setup.sh) - never a libggml-base.so of another
    version found on the library path (a system one gave other IQ1_M bytes)"""
    if os.environ.get(name):
        return os.environ[name]
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "trench.env")
    if os.path.exists(p):
        for line in open(p):
            if line.startswith("export %s=" % name):
                return line.split("=", 1)[1].strip()
    return None


LIBGGML = _env("LIBGGML")   # <llama.cpp build>/bin/libggml-base.so
if not LIBGGML:
    sys.exit("LIBGGML is not set and there is no trench.env: run ./setup.sh, or export LIBGGML=<llama.cpp build>/bin/libggml-base.so")


class TypeTraits(ctypes.Structure):
    _fields_ = [("type_name", ctypes.c_char_p), ("blck_size", ctypes.c_int64), ("blck_size_interleave", ctypes.c_int64),
                ("type_size", ctypes.c_size_t), ("is_quantized", ctypes.c_bool), ("to_float", ctypes.c_void_p),
                ("from_float_ref", ctypes.c_void_p)]


TO_FLOAT = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)


def read_rules(path):
    rules = []
    for line in open(path, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if line:
            pat, typ = line.rsplit(None, 1)
            rules.append((re.compile(pat), Q[typ.upper()]))
    return rules


FALLBACK = {Q.IQ1_S: Q.IQ4_NL, Q.IQ1_M: Q.IQ4_NL, Q.IQ2_XXS: Q.IQ4_NL, Q.IQ2_XS: Q.IQ4_NL, Q.IQ2_S: Q.IQ4_NL,
            Q.IQ3_XXS: Q.IQ4_NL, Q.IQ3_S: Q.IQ4_NL, Q.IQ4_XS: Q.IQ4_NL, Q.Q2_K: Q.Q4_0, Q.Q3_K: Q.Q4_0,
            Q.Q4_K: Q.Q5_0, Q.Q5_K: Q.Q5_1, Q.Q6_K: Q.Q8_0}   # (llama-quantize's tensor_type_fallback)
NEEDS_IMATRIX = {Q.IQ1_S, Q.IQ1_M, Q.IQ2_XXS, Q.IQ2_XS, Q.IQ2_S}


def fit_type(typ, ne0):
    """The type llama-quantize would write for rows of length ne0 when asked for typ."""
    if ne0 % gguf.GGML_QUANT_SIZES[typ][0] == 0:
        return typ
    fb = FALLBACK.get(typ, typ)
    return fb if ne0 % gguf.GGML_QUANT_SIZES[fb][0] == 0 else Q.F16


def read_imatrix(path):
    r = gguf.GGUFReader(path)
    sums, counts = {}, {}
    for t in r.tensors:
        if t.name.endswith(".in_sum2"):
            sums[t.name[:-len(".in_sum2")]] = np.asarray(t.data, dtype=np.float32)
        elif t.name.endswith(".counts"):
            counts[t.name[:-len(".counts")]] = np.asarray(t.data, dtype=np.float32).reshape(-1)
    out = {}
    for name, s in sums.items():
        c = counts[name]
        s = s.reshape(len(c), -1)
        out[name] = np.where(c[:, None] > 0, s / np.maximum(c, 1)[:, None], 1.0).astype(np.float32)
    return out


def main() -> int:
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in flags
    imat_path = next((a.split("=", 1)[1] for a in flags if a.startswith("--imatrix=")), None)
    threads = int(next((a.split("=", 1)[1] for a in flags if a.startswith("--threads=")), os.cpu_count() or 8))
    gpu = next((a for a in flags if a == "--gpu" or a.startswith("--gpu=")), None)
    if len(args) != 3 or imat_path is None:
        print(__doc__)
        return 2
    shard1, out_dir, rules_path = args
    rules = read_rules(rules_path)
    imat = read_imatrix(imat_path)
    lib = ctypes.CDLL(LIBGGML)
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                        ctypes.c_int64, ctypes.c_void_p]
    lib.ggml_get_type_traits.restype = ctypes.POINTER(TypeTraits)
    lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
    lib.ggml_quantize_init.argtypes = [ctypes.c_int]
    to_float = {}
    gq = None
    gq_lock = threading.Lock()
    if gpu is not None:
        here = os.environ.get("GPUQ_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gpuq"))
        gq = ctypes.CDLL(os.path.join(here, _env("GPUQ_LIB") or "libgpuq.so"))   # (CUDA: libgpuq_cuda.so)
        gq.gpuq_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
        gq.gpuq_quantize.restype = ctypes.c_size_t
        gq.gpuq_quantize.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                     ctypes.c_void_p]
        gq.gpuq_block_bytes.restype = ctypes.c_size_t
        gq.gpuq_requantize.restype = ctypes.c_size_t
        gq.gpuq_requantize.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64,
                                       ctypes.c_int64, ctypes.c_void_p]
        rc = gq.gpuq_init(os.path.join(here, "gpuq_tables.bin").encode(), int(gpu.split("=", 1)[1]) if "=" in gpu else 0)
        if rc != 0:
            print("gpuq_init failed (%d)" % rc)
            return 1

    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", os.path.basename(shard1))
    shards = [shard1] if m is None else [
        os.path.join(os.path.dirname(shard1), "%s-%05d-of-%s.gguf" % (m.group(1), i, m.group(3)))
        for i in range(1, int(m.group(3)) + 1)]
    os.makedirs(out_dir, exist_ok=True)
    total_old = total_new = 0
    arch = None
    for sh in shards:
        if not os.path.isfile(sh):
            print("%s is not there: skipped" % sh)
            continue
        try:
            r = gguf.GGUFReader(sh)
        except ValueError as e:   # a tensor type gguf-py does not know (DeepSeek V4.1's Engram shard): no experts in it
            out = os.path.join(out_dir, os.path.basename(sh))
            print("%s: gguf-py cannot read it (%s): kept as is (hard link or copy)" % (sh, e), flush=True)
            if check:
                continue
            if os.path.exists(out):
                os.remove(out)
            try:
                os.link(os.path.realpath(sh), out)   # (no space used on the same disk)
            except OSError:
                shutil.copyfile(sh, out)
            continue
        if gguf.Keys.General.ARCHITECTURE in r.fields:   # (the later shards of a split carry only the split keys)
            arch = r.fields[gguf.Keys.General.ARCHITECTURE].contents()
        out = os.path.join(out_dir, os.path.basename(sh))
        print("%s -> %s" % (sh, out), flush=True)
        plan = []
        no_im = set()
        for t in r.tensors:
            new_type = next((typ for pat, typ in rules if pat.search(t.name)), None)
            if new_type is not None and len(t.shape) not in (2, 3):
                print("  %s: not a matrix: left as is" % t.name)
                new_type = None
            if new_type is not None:
                ne0, ne1, ne2 = (int(x) for x in t.shape) if len(t.shape) == 3 else (int(t.shape[0]), int(t.shape[1]), 1)
                fit = fit_type(new_type, ne0)
                if fit != new_type:
                    print("  %s: rows of %d do not fit %s: %s instead (llama-quantize's fallback)" % (
                        t.name, ne0, new_type.name, fit.name))
                    new_type = fit
                if t.name not in imat and new_type in NEEDS_IMATRIX:
                    print("  %s: no imatrix data, which %s needs: left as is (give it a rule with another type)" % (
                        t.name, new_type.name))
                    new_type = None
                elif t.name in imat and imat[t.name].shape != (ne2, ne0):   # (a 2-D tensor: one row)
                    if new_type in NEEDS_IMATRIX:
                        print("  %s: imatrix %s does not fit [%d x %d experts]: left as is" % (
                            t.name, imat[t.name].shape, ne0, ne2))
                        new_type = None
                    else:   # (e.g. DeepSeek V4's grouped attn_output_a: the imatrix saw the 8 groups as one row)
                        print("  %s: imatrix %s does not fit [%d x %d]: quantized without it" % (
                            t.name, imat[t.name].shape, ne0, ne2))
                        no_im.add(t.name)
            if new_type is not None and new_type == t.tensor_type:
                new_type = None
            if new_type is None:
                plan.append((t, None, 0))
                continue
            bs, ts = gguf.GGML_QUANT_SIZES[new_type]
            nbytes = ne2 * ne1 * (ne0 // bs) * ts
            plan.append((t, new_type, nbytes))
            total_old += t.n_bytes
            total_new += nbytes
            print("  %-34s %-7s %8.2f GiB -> %-7s %8.2f GiB" % (t.name, t.tensor_type.name, t.n_bytes / 2**30,
                                                               new_type.name, nbytes / 2**30))
        if check:
            continue
        w = gguf.GGUFWriter(out, arch)
        for field in r.fields.values():
            if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
                continue
            if field.name == gguf.Keys.General.ALIGNMENT:   # (keep the source's padding)
                w.add_custom_alignment(int(field.contents()))
                continue
            val_type = field.types[0]
            sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
            w.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
        for t, new_type, nbytes in plan:
            if new_type is not None:
                ne1, ne2 = int(t.shape[1]), (int(t.shape[2]) if len(t.shape) == 3 else 1)
                shape = (ne2, ne1, nbytes // (ne2 * ne1)) if len(t.shape) == 3 else (ne1, nbytes // ne1)
                w.add_tensor_info(t.name, shape, np.uint8, nbytes, raw_dtype=new_type)
            else:
                w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_ti_data_to_file()
        t0 = time.time()
        for t, new_type, nbytes in plan:
            if new_type is None:
                w.write_tensor_data(t.data, tensor_endianess=r.endianess)
                continue
            ne0, ne1, ne2 = (int(x) for x in t.shape) if len(t.shape) == 3 else (int(t.shape[0]), int(t.shape[1]), 1)
            src = t.tensor_type
            if src not in to_float:
                to_float[src] = TO_FLOAT(lib.ggml_get_type_traits(int(src)).contents.to_float)
            lib.ggml_quantize_init(int(new_type))
            raw = np.ascontiguousarray(np.asarray(t.data).reshape(ne2, -1))
            per = nbytes // ne2
            dst = np.empty((ne2, per), dtype=np.uint8)
            im = None if t.name in no_im else imat.get(t.name)   # (None: quantized without one, as llama-quantize does)

            on_gpu = gq is not None and im is not None and gq.gpuq_block_bytes(int(new_type)) > 0
            gpu_dq = on_gpu and gq.gpuq_can_dequantize(int(src)) == 1   # (read the source bytes on the GPU too)

            def one(e):
                if gpu_dq:
                    with gq_lock:
                        return gq.gpuq_requantize(int(src), raw[e].ctypes.data, int(new_type), dst[e].ctypes.data, ne1,
                                                  ne0, im[e].ctypes.data)
                f32 = np.empty(ne1 * ne0, dtype=np.float32)
                to_float[src](raw[e].ctypes.data, f32.ctypes.data, ne1 * ne0)
                if on_gpu:
                    with gq_lock:
                        return gq.gpuq_quantize(int(new_type), f32.ctypes.data, dst[e].ctypes.data, ne1, ne0,
                                                im[e].ctypes.data)
                got = lib.ggml_quantize_chunk(int(new_type), f32.ctypes.data, dst[e].ctypes.data, 0, ne1, ne0,
                                              None if im is None else im[e].ctypes.data)
                return got

            with concurrent.futures.ThreadPoolExecutor(threads) as ex:
                for e, got in enumerate(ex.map(one, range(ne2))):
                    if got != per:
                        print("ggml_quantize_chunk wrote %d bytes, expected %d for %s expert %d" % (got, per, t.name, e))
                        return 1
            w.write_tensor_data(dst.reshape((ne2, ne1, per // ne1) if len(t.shape) == 3 else (ne1, per // ne1)),
                                tensor_endianess=r.endianess)
            print("  %s -> %s in %.0f s%s" % (t.name, new_type.name, time.time() - t0, " (gpu)" if gpu_dq else " (gpu quant, cpu dequant)" if on_gpu else ""),
                  flush=True)
        w.close()
    print("re-quantized %.2f GiB -> %.2f GiB%s" % (total_old / 2**30, total_new / 2**30, " (check only)" if check else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
