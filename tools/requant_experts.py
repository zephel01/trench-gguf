"""Re-quantizes the routed experts of a (split) GGUF to smaller i-quants with an imatrix; copies everything else byte
for byte. Rules may name 2-D tensors too (attention, shared experts, embeddings: e.g. BF16 -> Q8_0 / Q5_K), quantized
the same way as one expert.

usage: requant_experts.py <in shard 1> <out dir> <rules file> --imatrix=<imatrix.gguf> [--check] [--threads=N]
                         [--gpu=<device>] [--cpu] [--hessian=<dir> [--damp=0.01] [--damp-iq1=D] [--order=seq|act] [--weights=imatrix|diag|schur]
                         [--gated] [--no-feedback] [--sub=32|8 [--scale-search=K]]]
                         [--dump=<hessian_dump dir> [--dump-device=cpu|cuda|numpy]] [--ldlq-cpu] [--ldlq-f32] [--gpu-threads=N]
                         [--dense-dump=<hessian_dump --dense dir> [--dense-max-rows=N]] [--only-matched]
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

The GPU quantizes by default (gpuq: byte-identical to ggml_quantize_chunk, checked by gpuq/test_gpuq.py): IQ1_S /
IQ1_M / IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS / IQ4_NL, the k-quants Q2_K / Q3_K / Q4_K / Q5_K / Q6_K
and Q8_0, the 2-D tensors the rules name included (a big one in row batches). Without imatrix data the k-quants, Q8_0,
IQ4_NL and IQ4_XS take ggml's no-imatrix path on the GPU as well; IQ3_XXS / IQ3_S without one stay on the CPU.
Sources in F32 / F16 / BF16 / Q8_0 / MXFP4 are dequantized on the GPU too (others on the CPU threads). --gpu=N picks
the device, --cpu forces the CPU threads; with no built library (./build.sh / ./setup.sh) or no GPU it says so and uses
the CPU. GPUQ_LIB picks the library (default: libgpuq_cuda.so, then libgpuq.so), GPUQ_DIR its folder, GPUQ_BATCH_MB
the floats on the GPU a call (default 2048).

--hessian=<dir> turns on error-feedback rounding (tools/ldlq.py: GPTQ / block-LDLQ around ggml's own quantizer, the
output stays the plain GGUF format) for the expert tensors the folder has a Hessian for: blk.L.in.npy [n_expert, ne0,
ne0] for ffn_gate/up_exps, blk.L.mid.npy for ffn_down_exps, blk.L.counts.npy (tools/hessian_hf.py writes them;
--gated takes the routing-weight-squared versions blk.L.in_g / mid_g). --damp is the ridge added to the Hessian as a
share of its mean diagonal (GPTQ's 1%; on OLMoE 0.3 beat 0.1 for IQ2_XXS and 1.0 beat 0.3 for IQ1_S / IQ1_M), --damp-iq1
overrides it for the IQ1_S / IQ1_M tensors, --order=act quantizes the super-blocks in order of descending input energy,
--weights picks the importance handed to ggml for each super-block (imatrix = the imatrix row as without --hessian;
diag = diag(H)/n; schur = the conditional Hessian's diagonal), --no-feedback runs the same path without the error
update (a check: the bytes must then equal the plain run). --sub=32 / 8 (IQ2_XXS only) runs the feedback every
sub-block / codebook group inside the super-block with ggml's scales kept and the points chosen again against the
inference grid; --scale-search=K also tries the 4-bit sub-block scales K steps around ggml's. Tensors without a Hessian
file are quantized as before.
--dump=<dir> takes the Hessians from a tools/hessian_dump dump instead (blk.L.x.bf16 + ids): built per expert on the
fly (tools/hessian_from_dump.py; the down input recomputed from the expert's own gate/up), the factorization in torch
on --dump-device (cuda = ROCm/CUDA; numpy = no torch at all), gate and up of an expert done together. For a model
with 256 experts of 4096 inputs this keeps one expert's Hessian in memory at a time instead of 20 GB a layer.

--dense-dump=<dir> runs the same error feedback on the 2-D (dense) tensors the rules name, with Hessians from a
tools/hessian_dump --dense dump (dense.txt: weight -> input rows; DeepSeek V4's grouped attn_output_a gets one Hessian
per group, its rows split into the groups). The k-quants Q4_K / Q5_K / Q6_K take the feedback too (their quantizers are
closed per 256-weight super-block, as the i-quants). A tensor without imatrix data (or one that does not fit, as the
grouped attn_output_a) takes diag(H)/n as the importance. --dense-max-rows caps the rows a Hessian is summed over.
The down input recompute (--dump) applies the model's SwiGLU clamp (<arch>.swiglu_clamp_exp, DeepSeek V4 / V4.1) as
llama.cpp's ggml_swiglu_clamp does.

With the GPU (the default) and --dump-device=cuda the whole error-feedback loop runs on the GPU (tools/ldlq.py quantize_matrix_torch:
the weights stay on the device, gpuq's device kernels quantize and dequantize each super-block column, the error update
is a GPU GEMM in float64 as on the CPU; gate and up of an expert go through as one batch). The bytes are not bit for bit
the CPU loop's (another GEMM summation order), the quality is the same within run noise. --ldlq-cpu keeps the CPU loop
(numpy, --threads), --ldlq-f32 does the error update in float32 (faster where float64 is slow, a little more drift),
--gpu-threads=N experts in flight on the GPU at once (default 4; each holds one expert's Hessian and factor).

--only-matched writes only the re-quantized tensors: one GGUF a source shard that has any (same file name, shard 1's
metadata minus the split keys, no copies of the untouched tensors). For donors of per-layer mixes (an Abyss pack's
native_experts.txt can take any layer's experts from such a file) and for overlays (a first shard over another
model's shards, iq_pack.py --first-wins).

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


def load_gpuq(device=0, quiet=False):
    """gpuq (the GPU quantizer) on device, or None with a one-line reason (no library built, no GPU, an old library):
    the caller then quantizes on the CPU. GPUQ_LIB / trench.env pick the library; else libgpuq_cuda.so, then
    libgpuq.so (ROCm) in GPUQ_DIR (default gpuq/ next to tools/)."""
    here = os.environ.get("GPUQ_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gpuq"))
    names = [_env("GPUQ_LIB")] if _env("GPUQ_LIB") else ["libgpuq_cuda.so", "libgpuq.so"]
    why = []
    for nm in names:
        path = os.path.join(here, nm)
        if not os.path.exists(path):
            why.append("%s not built" % nm)
            continue
        try:
            gq = ctypes.CDLL(path)
        except OSError as e:   # (e.g. the CUDA library on a box without the CUDA runtime)
            why.append("%s: %s" % (nm, str(e).splitlines()[0][:120]))
            continue
        gq.gpuq_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
        gq.gpuq_quantize.restype = ctypes.c_size_t
        gq.gpuq_quantize.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                     ctypes.c_void_p]
        gq.gpuq_block_bytes.restype = ctypes.c_size_t
        gq.gpuq_requantize.restype = ctypes.c_size_t
        gq.gpuq_requantize.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64,
                                       ctypes.c_int64, ctypes.c_void_p]
        gq.batch = hasattr(gq, "gpuq_quantize_batch")   # (a library from before v0.4.1: no k-quants, no batches)
        if gq.batch:
            gq.gpuq_quantize_batch.restype = ctypes.c_size_t
            gq.gpuq_quantize_batch.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                               ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
            gq.gpuq_requantize_batch.restype = ctypes.c_size_t
            gq.gpuq_requantize_batch.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                                 ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
            gq.gpuq_can_quantize.argtypes = [ctypes.c_int, ctypes.c_int]
        rc = gq.gpuq_init(os.path.join(here, "gpuq_tables.bin").encode(), int(device))
        if rc != 0:
            why.append("%s: gpuq_init %d (no GPU %d, or gpuq_tables.bin missing)" % (nm, rc, int(device)))
            continue
        gq.name = nm
        return gq
    if not quiet:
        print("GPU quantizer not available (%s): quantizing on the CPU. Build it with ./build.sh (or ./setup.sh); "
              "--cpu silences this." % "; ".join(why), flush=True)
    return None


def gpuq_takes(gq, typ, with_imatrix):
    """whether gpuq quantizes typ (with or without an imatrix) byte-identically to ggml"""
    if gq is None:
        return False
    if gq.batch:
        return gq.gpuq_can_quantize(int(typ), 1 if with_imatrix else 0) == 1
    return with_imatrix and gq.gpuq_block_bytes(int(typ)) > 0   # (an old library: the imatrix i-quants only)


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


LDLQ_TYPES = {Q.IQ1_S, Q.IQ1_M, Q.IQ2_XXS, Q.IQ2_XS, Q.IQ2_S, Q.IQ3_XXS, Q.IQ3_S, Q.IQ4_XS,
              Q.Q4_K, Q.Q5_K, Q.Q6_K}   # super-block = 256 (the k-quants: --dense-dump)
MIN_ROWS = int(os.environ.get("LDLQ_MIN_ROWS", 64))   # an expert with fewer calibration rows is rounded plain


def hess_file(hess_dir, name, gated):
    """(Hessian .npy, counts .npy) for an expert tensor name, or None when the folder has none for it."""
    m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", name)
    if m is None:
        return None
    kind = "mid" if m.group(2) == "down" else "in"
    h = os.path.join(hess_dir, "blk.%s.%s%s.npy" % (m.group(1), kind, "_g" if gated else ""))
    c = os.path.join(hess_dir, "blk.%s.counts.npy" % m.group(1))
    return (h, c) if os.path.exists(h) and os.path.exists(c) else None


_FORK_FN = None
_FORK_DST = None


def _fork_call(e):
    got = _FORK_FN(e)
    return got, _FORK_DST(e)


class Shards:
    """All shards of a split GGUF by tensor name, opened lazily; dequantizes one expert through ggml's to_float."""

    def __init__(self, shard1, lib=None):
        m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", os.path.basename(shard1))
        self.paths = [shard1] if m is None else [
            os.path.join(os.path.dirname(shard1), "%s-%05d-of-%s.gguf" % (m.group(1), i, m.group(3)))
            for i in range(1, int(m.group(3)) + 1)]
        self.readers = {}
        self.where = {}
        for p in self.paths:
            if not os.path.isfile(p):
                continue
            try:
                r = gguf.GGUFReader(p)
            except ValueError:
                continue
            self.readers[p] = r
            for t in r.tensors:
                self.where[t.name] = (p, t)
        # NVFP4 experts carry a per-expert second scale (blk.L.ffn_*_exps.scale, F32 [n_expert]) that the matmul
        # applies after the block-scaled product (llama.cpp build_lora_mm_id's w_s): the expert's real weights are
        # to_float(bytes) x scale[e].  (.input_scale is the activations' and stays out of the weights.)
        self.escale = {}
        for name, (p, t) in self.where.items():
            if name.endswith("_exps.scale") and len(t.shape) == 1:
                w = name[:-len(".scale")] + ".weight"
                if w in self.where:
                    self.escale[w] = np.asarray(t.data, dtype=np.float32).reshape(-1).copy()
        self.lib = lib or ctypes.CDLL(LIBGGML)
        self.lib.ggml_get_type_traits.restype = ctypes.POINTER(TypeTraits)
        self.lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
        self.to_float = {}

    def tensor(self, name):
        return self.where[name][1]

    def dequant_expert(self, name, e):
        """expert e of a 3-D tensor as float32 [ne1, ne0]"""
        t = self.tensor(name)
        ne0, ne1, ne2 = (int(x) for x in t.shape)
        src = t.tensor_type
        if src not in self.to_float:
            self.to_float[src] = TO_FLOAT(self.lib.ggml_get_type_traits(int(src)).contents.to_float)
        raw = np.ascontiguousarray(np.asarray(t.data).reshape(ne2, -1)[e])
        f32 = np.empty(ne1 * ne0, dtype=np.float32)
        self.to_float[src](raw.ctypes.data, f32.ctypes.data, ne1 * ne0)
        if name in self.escale:
            f32 *= self.escale[name][e]
        return f32.reshape(ne1, ne0)


def open_shards(shard1):
    return Shards(shard1)


def dequant_expert(shards, name, e):
    return shards.dequant_expert(name, e)


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


LLAMA_FTYPE = {"F32": 0, "F16": 1, "Q4_0": 2, "Q4_1": 3, "Q8_0": 7, "Q5_0": 8, "Q5_1": 9, "Q2_K": 10, "Q3_K_S": 11,
               "Q3_K_M": 12, "Q3_K_L": 13, "Q4_K_S": 14, "Q4_K_M": 15, "Q5_K_S": 16, "Q5_K_M": 17, "Q6_K": 18,
               "IQ2_XXS": 19, "IQ2_XS": 20, "Q2_K_S": 21, "IQ3_XS": 22, "IQ3_XXS": 23, "IQ1_S": 24, "IQ4_NL": 25,
               "IQ3_S": 26, "IQ3_M": 27, "IQ2_S": 28, "IQ2_M": 29, "IQ4_XS": 30, "IQ1_M": 31, "BF16": 32,
               "MXFP4_MOE": 38}   # (llama.h's llama_ftype)
GGML_QNT_VERSION = 2


def llama_order(name):
    """llama_model_loader's weight_name_comparer: block number first (-1 outside the blocks), then the name"""
    m = re.match(r"blk\.(\d+)\.", name)
    return (int(m.group(1)) if m else -1, name.encode())


def imatrix_meta(path):
    """the quantize.imatrix.* keys llama-quantize writes for a GGUF imatrix: (file, dataset, entries, chunks)"""
    r = gguf.GGUFReader(path)
    ds = r.fields["imatrix.datasets"].contents() if "imatrix.datasets" in r.fields else []
    chunks = int(r.fields["imatrix.chunk_count"].contents()) if "imatrix.chunk_count" in r.fields else 0
    entries = sum(1 for t in r.tensors if t.name.endswith(".in_sum2"))
    return os.path.basename(path), (os.path.basename(ds[0]) if ds else None), entries, chunks


def main() -> int:
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    check = "--check" in flags
    imat_path = next((a.split("=", 1)[1] for a in flags if a.startswith("--imatrix=")), None)
    threads = int(next((a.split("=", 1)[1] for a in flags if a.startswith("--threads=")), os.cpu_count() or 8))
    gpu = next((a for a in flags if a == "--gpu" or a.startswith("--gpu=")), None)
    force_cpu = "--cpu" in flags
    gpu_batch_mb = int(os.environ.get("GPUQ_BATCH_MB", "2048"))
    hess_dir = next((a.split("=", 1)[1] for a in flags if a.startswith("--hessian=")), None)
    hess_opt = {"damp": float(next((a.split("=", 1)[1] for a in flags if a.startswith("--damp=")), 0.01)),
                "damp_iq1": next((float(a.split("=", 1)[1]) for a in flags if a.startswith("--damp-iq1=")), None),
                "order": next((a.split("=", 1)[1] for a in flags if a.startswith("--order=")), "seq"),
                "weights": next((a.split("=", 1)[1] for a in flags if a.startswith("--weights=")), "imatrix"),
                "gated": "--gated" in flags, "feedback": "--no-feedback" not in flags,
                "sub": int(next((a.split("=", 1)[1] for a in flags if a.startswith("--sub=")), 0)),
                "scale_search": int(next((a.split("=", 1)[1] for a in flags if a.startswith("--scale-search=")), 0))}
    hess_cache = {}
    stash = {}
    dump_dir = next((a.split("=", 1)[1] for a in flags if a.startswith("--dump=")), None)
    dump_device = next((a.split("=", 1)[1] for a in flags if a.startswith("--dump-device=")), "cpu")
    dump = src_shards = None
    dense_dir = next((a.split("=", 1)[1] for a in flags if a.startswith("--dense-dump=")), None)
    dense_max = int(next((a.split("=", 1)[1] for a in flags if a.startswith("--dense-max-rows=")), 0))
    only_matched = "--only-matched" in flags
    # --ftype=<llama-quantize type, e.g. Q8_0> writes the file exactly as llama-quantize does (its header
    # keys, its tensor order); --header-only stops after the header (to check it against a llama-quantize file)
    ftype_arg = next((a.split("=", 1)[1] for a in flags if a.startswith("--ftype=")), None)
    llama_ftype = None if ftype_arg is None else (int(ftype_arg) if ftype_arg.isdigit() else LLAMA_FTYPE[ftype_arg.upper()])
    header_only = "--header-only" in flags
    ldlq_cpu = "--ldlq-cpu" in flags
    ldlq_f64 = "--ldlq-f32" not in flags
    gpu_threads = int(next((a.split("=", 1)[1] for a in flags if a.startswith("--gpu-threads=")), min(threads, 4)))
    dense = None
    if hess_dir or dump_dir or dense_dir:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import ldlq
    for d_, what in ((dump_dir, "--dump"), (dense_dir, "--dense-dump")):
        # a dump without info.txt is missing or unfinished: without this check every tensor would be rounded plain
        # with no word (the bytes of a run without --dump)
        if d_ and not os.path.isfile(os.path.join(d_, "info.txt")):
            print("%s=%s: no info.txt there - not a finished tools/hessian_dump output" % (what, d_), file=sys.stderr)
            return 2
    if dump_dir:
        from hessian_from_dump import DumpHessians
        dump = DumpHessians(dump_dir, dump_device if dump_device != "numpy" else "cpu")
    if dense_dir:
        from hessian_from_dump import DenseHessians
        dense = DenseHessians(dense_dir, dump_device if dump_device != "numpy" else "cpu", max_rows=dense_max)
    if len(args) != 3 or imat_path is None:
        print(__doc__)
        return 2
    shard1, out_dir, rules_path = args
    rules = read_rules(rules_path)
    imat = read_imatrix(imat_path)
    lib = ctypes.CDLL(LIBGGML)
    if dump_dir:
        src_shards = Shards(shard1, lib)
    escale = (src_shards or Shards(shard1, lib)).escale   # (NVFP4's per-expert scales, folded into every dequant)
    if escale:
        print("source: %d expert tensors with a per-expert scale (NVFP4): folded into the weights; their .scale / "
              ".input_scale tensors are left out of re-quantized output" % len(escale), flush=True)
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                        ctypes.c_int64, ctypes.c_void_p]
    lib.ggml_get_type_traits.restype = ctypes.POINTER(TypeTraits)
    lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
    lib.ggml_quantize_init.argtypes = [ctypes.c_int]
    to_float = {}
    gq = None
    gq_lock = threading.Lock()
    if not force_cpu and not check:
        gq = load_gpuq(int(gpu.split("=", 1)[1]) if gpu and "=" in gpu else 0)
        if gq is not None:
            print("quantizing on the GPU (gpuq/%s; --cpu for the CPU threads)" % gq.name, flush=True)
    # the error-feedback loop on the GPU: gpuq's device kernels + torch on the same device
    gpu_ldlq = gq is not None and dump_device == "cuda" and not ldlq_cpu and (hess_dir or dump_dir or dense_dir)
    torch = None
    if gpu_ldlq:
        import torch
        if not torch.cuda.is_available():
            print("--dump-device=cuda: torch has no GPU here")
            return 1
        if not hasattr(gq, "gpuq_quantize_dev"):
            print("this libgpuq.so has no device-pointer API (rebuild it: ./build.sh): the LDLQ loop runs on the CPU")
            gpu_ldlq = False

    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", os.path.basename(shard1))
    shards = [shard1] if m is None else [
        os.path.join(os.path.dirname(shard1), "%s-%05d-of-%s.gguf" % (m.group(1), i, m.group(3)))
        for i in range(1, int(m.group(3)) + 1)]
    os.makedirs(out_dir, exist_ok=True)
    total_old = total_new = 0
    arch = None
    r_first = None   # shard 1's reader: its metadata for --only-matched files, and the SwiGLU clamp
    clamp_exp = []
    if os.path.isfile(shards[0]):
        r_first = gguf.GGUFReader(shards[0])
        if gguf.Keys.General.ARCHITECTURE in r_first.fields:
            a0 = r_first.fields[gguf.Keys.General.ARCHITECTURE].contents()
            f = r_first.fields.get("%s.swiglu_clamp_exp" % a0)
            if f is not None:
                v = f.contents()
                clamp_exp = [float(x) for x in (v if isinstance(v, list) else [v])]
                print("swiglu clamp (%s.swiglu_clamp_exp): %s" % (a0, sorted(set(clamp_exp))), flush=True)
    for sh in shards:
        if not os.path.isfile(sh):
            print("%s is not there: skipped" % sh)
            continue
        try:
            r = gguf.GGUFReader(sh)
        except ValueError as e:   # a tensor type gguf-py does not know (DeepSeek V4.1's Engram shard): no experts in it
            out = os.path.join(out_dir, os.path.basename(sh))
            print("%s: gguf-py cannot read it (%s): kept as is (hard link or copy)" % (sh, e), flush=True)
            if check or only_matched:
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
                if dense is not None and len(t.shape) == 2 and dense.has(t.name) and dense.groups(t.name) > 1:
                    if t.name in imat and imat[t.name].shape[-1] != ne0:
                        no_im.add(t.name)   # (the imatrix saw the groups as one long row: diag(H)/n instead)
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
        if escale:
            redone = {t.name for t, nt, _ in plan if nt is not None}
            plan = [p_ for p_ in plan if not (p_[1] is None and (
                (p_[0].name.endswith("_exps.scale") and p_[0].name[:-6] + ".weight" in redone) or
                (p_[0].name.endswith("_exps.input_scale") and p_[0].name[:-12] + ".weight" in redone)))]
        if only_matched:
            plan_w = [p for p in plan if p[1] is not None]
            if not plan_w:
                print("  nothing re-quantized in this shard: no file (--only-matched)", flush=True)
                continue
            kv_src = r_first if r_first is not None else r
        else:
            plan_w = plan
            kv_src = r
        if llama_ftype is not None:   # (llama-quantize writes the tensors in its loader's order)
            plan_w = sorted(plan_w, key=lambda p_: llama_order(p_[0].name))
        w = gguf.GGUFWriter(out, arch)
        for field in kv_src.fields.values():
            if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
                continue
            if llama_ftype is not None and field.name in ("general.quantization_version", "general.file_type",
                                                          "split.no", "split.count", "split.tensors.count"):
                continue   # (llama-quantize: re-set at the end / removed)
            if only_matched and field.name.startswith("split."):
                continue
            if field.name == gguf.Keys.General.ALIGNMENT:   # (keep the source's padding)
                w.add_custom_alignment(int(field.contents()))
                continue
            val_type = field.types[0]
            sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
            w.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
        if llama_ftype is not None:   # (gguf_set_val: remove, then append; then the imatrix keys as kv overrides)
            w.add_uint32("general.quantization_version", GGML_QNT_VERSION)
            w.add_uint32("general.file_type", llama_ftype)
            if imat_path:
                f_, ds_, ne_, nc_ = imatrix_meta(imat_path)
                w.add_string("quantize.imatrix.file", f_[:127])
                if ds_:
                    w.add_string("quantize.imatrix.dataset", ds_[:127])
                w.add_uint32("quantize.imatrix.entries_count", ne_)
                if nc_ > 0:
                    w.add_uint32("quantize.imatrix.chunks_count", nc_)
        for t, new_type, nbytes in plan_w:
            if new_type is not None:
                ne1, ne2 = int(t.shape[1]), (int(t.shape[2]) if len(t.shape) == 3 else 1)
                shape = (ne2, ne1, nbytes // (ne2 * ne1)) if len(t.shape) == 3 else (ne1, nbytes // ne1)
                w.add_tensor_info(t.name, shape, np.uint8, nbytes, raw_dtype=new_type)
            else:
                shp = tuple(t.data.shape)
                if llama_ftype is not None:   # (ggml_n_dims: trailing 1s of the ggml shape = leading 1s here dropped)
                    while len(shp) > 1 and shp[0] == 1:
                        shp = shp[1:]
                w.add_tensor_info(t.name, shp, t.data.dtype, t.data.nbytes, t.tensor_type)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_ti_data_to_file()
        if header_only:
            w.close()
            print("  header only: %s" % out, flush=True)
            continue
        t0 = time.time()
        for t, new_type, nbytes in plan_w:
            if new_type is None:
                w.write_tensor_data(t.data, tensor_endianess=r.endianess)
                continue
            ne0, ne1, ne2 = (int(x) for x in t.shape) if len(t.shape) == 3 else (int(t.shape[0]), int(t.shape[1]), 1)
            src = t.tensor_type
            if src not in to_float:
                to_float[src] = TO_FLOAT(lib.ggml_get_type_traits(int(src)).contents.to_float)
            lib.ggml_quantize_init(int(new_type))
            if dense is not None and len(t.shape) == 2 and dense.has(t.name) and dense.groups(t.name) > 1:
                G = dense.groups(t.name)   # (DeepSeek V4's attn_output_a: the rows of group g = a virtual expert g)
                if ne1 % G == 0:
                    ne1, ne2 = ne1 // G, G
            raw = np.ascontiguousarray(np.asarray(t.data).reshape(ne2, -1))
            per = nbytes // ne2
            dst = np.empty((ne2, per), dtype=np.uint8)
            im = None if t.name in no_im else imat.get(t.name)   # (None: quantized without one, as llama-quantize does)

            on_gpu = gpuq_takes(gq, new_type, im is not None)
            esc = escale.get(t.name)   # (NVFP4: expert e's weights x esc[e])
            gpu_dq = on_gpu and esc is None and gq.gpuq_can_dequantize(int(src)) == 1   # (read the source bytes on the GPU too)

            # error-feedback rounding: a Hessian provider per tensor, from files (--hessian) or from a dump (--dump)
            hess = None
            get_H = None
            if hess_dir and len(t.shape) == 3:
                hf = hess_file(hess_dir, t.name, hess_opt["gated"])
                if hf is not None:
                    H_all = np.load(hf[0], mmap_mode="r")
                    H_counts = np.load(hf[1])
                    if H_all.shape != (ne2, ne0, ne0):
                        print("  %s: Hessian %s does not fit [%d x %d x %d]: plain rounding" % (t.name, H_all.shape, ne2, ne0, ne0))
                    else:
                        hess = os.path.basename(hf[0])
                        get_H = lambda e, H_all=H_all, H_counts=H_counts: (np.asarray(H_all[e], dtype=np.float64), H_counts[e])
                        share_key = hf[0]   # (gate and up share the Hessian: factor it once)
                        hkind = "file"
            elif dump is not None and len(t.shape) == 3:
                mm = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", t.name)
                if mm and dump.has(int(mm.group(1))):
                    Ld = int(mm.group(1))
                    hess = "dump blk.%d" % Ld
                    if mm.group(2) == "down":
                        def get_H(e, Ld=Ld, torch_out=False, W_dev=None):
                            if W_dev is not None:   # (the GPU loop: gate / up decoded on the device)
                                Wg = W_dev("blk.%d.ffn_gate_exps.weight" % Ld, e)
                                Wu = W_dev("blk.%d.ffn_up_exps.weight" % Ld, e)
                            else:
                                Wg = src_shards.dequant_expert("blk.%d.ffn_gate_exps.weight" % Ld, e)
                                Wu = src_shards.dequant_expert("blk.%d.ffn_up_exps.weight" % Ld, e)
                            cl = clamp_exp[Ld] if Ld < len(clamp_exp) else 0.0
                            return dump.hessian_mid(Ld, e, Wg, Wu, gated=hess_opt["gated"], clamp=cl, torch_out=torch_out)
                        share_key = None
                        hkind = "dump-down"
                    else:
                        get_H = lambda e, Ld=Ld, torch_out=False, W_dev=None: dump.hessian_in(Ld, e, gated=hess_opt["gated"], torch_out=torch_out)
                        share_key = "dump-in-%d" % Ld
                        hkind = "dump-in"
            elif dense is not None and len(t.shape) == 2 and dense.has(t.name) and (ne2 == 1 or ne2 == dense.groups(t.name)):
                hess = "dense %s" % dense.file(t.name)
                get_H = lambda e, nm=t.name: dense.hessian(nm, e)
                share_key = "dense:" + dense.file(t.name)
                mm = None
                hkind = "dense"
            is_dense = hess is not None and hess.startswith("dense")
            if hess is not None and ((im is None and not is_dense) or ne0 % ldlq.QK_K or new_type not in LDLQ_TYPES):
                hess = None
                is_dense = False
            # the importance a super-block's quantizer gets: the imatrix row, or diag(H)/n where there is none
            w_mode = hess_opt["weights"] if im is not None else "diag"
            im_row = (lambda e: im[min(e, im.shape[0] - 1)]) if im is not None else (lambda e: np.ones(ne0, dtype=np.float32))
            if hess is not None:
                if new_type not in to_float:
                    to_float[new_type] = TO_FLOAT(lib.ggml_get_type_traits(int(new_type)).contents.to_float)
                bs_q, tsize = gguf.GGML_QUANT_SIZES[new_type]
                iq2xxs_tab = ldlq.IQ2XXS(to_float[new_type]) if hess_opt["sub"] and new_type == Q.IQ2_XXS else None

                def quant_blk(blk, qw):
                    rawb = np.empty((blk.shape[0], tsize), dtype=np.uint8)
                    if on_gpu:
                        with gq_lock:
                            got = gq.gpuq_quantize(int(new_type), blk.ctypes.data, rawb.ctypes.data, blk.shape[0],
                                                   blk.shape[1], qw.ctypes.data)
                    else:
                        got = lib.ggml_quantize_chunk(int(new_type), blk.ctypes.data, rawb.ctypes.data, 0, blk.shape[0],
                                                      blk.shape[1], qw.ctypes.data)
                    assert got == rawb.nbytes
                    return rawb

                def dequant_blk(rawb):
                    f = np.empty((rawb.shape[0], bs_q), dtype=np.float32)
                    to_float[new_type](rawb.ctypes.data, f.ctypes.data, f.size)
                    return f

                # from a dump, gate and up of one expert are done together (one factorization, no cache of 256 U's):
                # the partner's bytes wait in `stash` until its own turn
                partner = None
                if dump is not None and hess.startswith("dump") and mm.group(2) in ("gate", "up"):
                    pname = t.name.replace("ffn_gate_exps", "ffn_up_exps") if mm.group(2) == "gate" else t.name.replace("ffn_up_exps", "ffn_gate_exps")
                    pplan = next((p for p in plan if p[0].name == pname and p[1] is not None), None)
                    if pname in stash:
                        pass   # (computed with the partner already)
                    elif pplan is not None and pplan[1] == new_type:
                        partner = (pname, pplan[2] // ne2)
                        stash[pname] = np.empty((ne2, partner[1]), dtype=np.uint8)
                if hess_cache.get("file") != share_key or share_key is None:
                    hess_cache.clear()
                    hess_cache["file"] = share_key

                damp_t = hess_opt["damp_iq1"] if hess_opt["damp_iq1"] is not None and new_type in (Q.IQ1_S, Q.IQ1_M) else hess_opt["damp"]

                def factor(H, n):   # (torch on --dump-device for a dump, and for --hessian files when it is a GPU)
                    if (dump is not None and dump_device != "numpy") or dump_device == "cuda":
                        return ldlq.prepare_hessian_torch(H, n, damp=damp_t, order=hess_opt["order"], device=dump_device)
                    return ldlq.prepare_hessian(H, n, damp=damp_t, order=hess_opt["order"])

                def ldlq_one_matrix(W, U, perm, diag_n, imrow):
                    if hess_opt["sub"] and new_type == Q.IQ2_XXS:
                        out_b, deq, _ = ldlq.quantize_matrix_iq2xxs_sub(W, quant_blk, iq2xxs_tab, U, perm, imrow,
                                                                        sub=hess_opt["sub"], scale_search=hess_opt["scale_search"])
                    else:
                        out_b, deq, _ = ldlq.quantize_matrix(W, quant_blk, dequant_blk, tsize, U=U, perm=perm, diag_n=diag_n,
                                                             imatrix=imrow, weights=w_mode,
                                                             feedback=hess_opt["feedback"])
                    return out_b.reshape(-1)

                fell_back = []   # experts quantized plain: no (or too few) calibration rows for a Hessian

                # the GPU loop: this type's gpuq device kernels (the IQ2_XXS sub-block variant stays on the CPU)
                on_gpu_loop = gpu_ldlq and on_gpu and not (hess_opt["sub"] and new_type == Q.IQ2_XXS) and \
                    ldlq.GpuQuant.supported(gq, new_type)
                if on_gpu_loop:
                    gqt = ldlq.GpuQuant(gq, new_type, tsize)
                    dev = torch.device("cuda")

                    def W_dev(name, e):
                        """expert e of a source tensor as f32 [ne1, ne0] on the device (gpuq decodes F16/BF16/Q8_0/MXFP4
                        bytes there; other source types through ggml's to_float on the CPU)"""
                        if name == t.name:
                            rb, st, n1, n0 = raw[e], src, ne1, ne0
                        else:
                            tt = src_shards.tensor(name)
                            n0, n1 = int(tt.shape[0]), int(tt.shape[1])
                            rb, st = np.asarray(tt.data).reshape(int(tt.shape[2]), -1)[e], tt.tensor_type
                        f = gqt.dequant_src(st, rb, n1 * n0, dev) if name not in escale else None
                        if f is None:
                            if name == t.name:
                                h = np.empty(n1 * n0, dtype=np.float32)
                                to_float[src](rb.ctypes.data, h.ctypes.data, h.size)
                                if esc is not None:
                                    h *= esc[e]
                            else:
                                h = src_shards.dequant_expert(name, e)
                            f = torch.from_numpy(h).to(dev)
                        return f.reshape(n1, n0)

                    def get_H_dev(e):
                        if hkind == "dump-down":
                            H, n = get_H(e, torch_out=True, W_dev=W_dev)
                        elif hkind == "dump-in":
                            H, n = get_H(e, torch_out=True)
                        else:
                            H, n = get_H(e)
                            H = torch.as_tensor(np.asarray(H)).to(dev)
                        return H, n

                    def one_ldlq_gpu(e):
                        if t.name in stash:
                            dst[e] = stash[t.name][e]
                            return per
                        if share_key is not None and e in hess_cache:   # (U kept on the host between gate and up)
                            U = perm = diag_n = None
                            if hess_cache[e][0] is not None:
                                U, perm, diag_n = (torch.from_numpy(x).to(dev) for x in hess_cache[e][:3])
                        else:
                            H, n = get_H_dev(e)
                            if n < MIN_ROWS:   # (plain rounding, as without a Hessian: the imatrix row still applies)
                                fell_back.append((e, int(n)))
                                U = perm = diag_n = dead = None
                            else:
                                U, perm, diag_n, dead = ldlq.prepare_hessian_torch(H, n, damp=damp_t, order=hess_opt["order"],
                                                                                   device="cuda", out="torch")
                            del H
                            if share_key is not None and partner is None:
                                hess_cache[e] = (None, None, None, None) if U is None else (
                                    U.cpu().numpy(), perm.cpu().numpy(), diag_n.cpu().numpy(), None)
                        names = [t.name] + ([partner[0]] if partner is not None else [])
                        Ws = torch.stack([W_dev(nm, e) for nm in names])
                        ims = torch.from_numpy(np.stack([im_row(e) if nm == t.name else imat[nm][e] for nm in names])).to(dev)
                        if U is None:
                            out_b = ldlq.quantize_matrix_torch(Ws, gqt, imatrix=ims, feedback=False)
                        else:
                            out_b = ldlq.quantize_matrix_torch(Ws, gqt, U=U, perm=perm, diag_n=diag_n, imatrix=ims, weights=w_mode,
                                                               feedback=hess_opt["feedback"], f64=ldlq_f64)
                        out_h = out_b.cpu().numpy()
                        dst[e] = out_h[0].reshape(-1)
                        if partner is not None:
                            stash[partner[0]][e] = out_h[1].reshape(-1)
                        return per

                def one_ldlq(e):
                    if t.name in stash:
                        dst[e] = stash[t.name][e]
                        return per
                    f32 = np.empty(ne1 * ne0, dtype=np.float32)
                    to_float[src](raw[e].ctypes.data, f32.ctypes.data, ne1 * ne0)
                    if esc is not None:
                        f32 *= esc[e]
                    W = f32.reshape(ne1, ne0)
                    if share_key is not None and e in hess_cache:
                        U, perm, diag_n, dead = hess_cache[e]
                    else:
                        H, n = get_H(e)
                        if n < MIN_ROWS:   # (plain rounding, as without a Hessian: the imatrix row still applies)
                            fell_back.append((e, int(n)))
                            U = perm = diag_n = dead = None
                        else:
                            U, perm, diag_n, dead = factor(H, n)
                        if share_key is not None and partner is None:
                            hess_cache[e] = (U, perm, diag_n, dead)
                    if U is None:
                        dst[e] = ldlq.quantize_matrix(W, quant_blk, dequant_blk, tsize, imatrix=im_row(e), feedback=False)[0].reshape(-1)
                        if partner is not None:
                            Wp_ = src_shards.dequant_expert(partner[0], e)
                            stash[partner[0]][e] = ldlq.quantize_matrix(Wp_, quant_blk, dequant_blk, tsize, imatrix=imat[partner[0]][e],
                                                                        feedback=False)[0].reshape(-1)
                        return per
                    dst[e] = ldlq_one_matrix(W, U, perm, diag_n, im_row(e))
                    if partner is not None:
                        Wp_ = src_shards.dequant_expert(partner[0], e)
                        stash[partner[0]][e] = ldlq_one_matrix(Wp_, U, perm, diag_n, imat[partner[0]][e])
                    return per

            def one(e):
                if gpu_dq:
                    with gq_lock:
                        return gq.gpuq_requantize(int(src), raw[e].ctypes.data, int(new_type), dst[e].ctypes.data, ne1,
                                                  ne0, None if im is None else im[e].ctypes.data)
                f32 = np.empty(ne1 * ne0, dtype=np.float32)
                to_float[src](raw[e].ctypes.data, f32.ctypes.data, ne1 * ne0)
                if esc is not None:
                    f32 *= esc[e]
                if on_gpu:
                    with gq_lock:
                        return gq.gpuq_quantize(int(new_type), f32.ctypes.data, dst[e].ctypes.data, ne1, ne0,
                                                None if im is None else im[e].ctypes.data)
                got = lib.ggml_quantize_chunk(int(new_type), f32.ctypes.data, dst[e].ctypes.data, 0, ne1, ne0,
                                              None if im is None else im[e].ctypes.data)
                return got

            if hess is not None and hess_opt["sub"]:
                # the sub-block search is many small numpy ops (GIL-bound): processes instead of threads (fork: the
                # closures, the mapped source and the ggml library are inherited; a child returns its expert's bytes)
                import multiprocessing
                global _FORK_FN
                _FORK_FN = one_ldlq
                global _FORK_DST
                _FORK_DST = lambda e: dst[e].tobytes()
                with multiprocessing.get_context("fork").Pool(threads) as pool:
                    for e, (got, b) in enumerate(pool.imap(_fork_call, range(ne2))):
                        if got != per:
                            print("quantizer wrote %d bytes, expected %d for %s expert %d" % (got, per, t.name, e))
                            return 1
                        dst[e] = np.frombuffer(b, dtype=np.uint8)
                ex_map = []
            elif on_gpu and gq.batch and hess is None:
                # the plain path on the GPU, GPUQ_BATCH_MB of floats a call: k experts of a 3-D tensor (each its own
                # imatrix row), or k rows of a 2-D one (a 118 GiB per-layer embedding fits no GPU, nor RAM as floats);
                # the next batch's source bytes are read from disk (or decoded on the CPU) while the GPU works
                rows2d = ne2 == 1
                unit_f = ne0 if rows2d else ne1 * ne0          # floats a unit (a row / an expert)
                nunit = ne1 if rows2d else ne2
                k = max(1, min(nunit, (gpu_batch_mb << 20) // max(1, unit_f * 4)))
                imb = None if im is None else np.ascontiguousarray(im, dtype=np.float32)
                if imb is not None and imb.shape[0] != ne2:
                    imb = np.ascontiguousarray(np.broadcast_to(imb[:1], (ne2, ne0)))
                src_u = raw.reshape(nunit, -1)                  # (source bytes a unit)
                dst_u = dst.reshape(nunit, -1)
                per_u = dst_u.shape[1]

                def load(u0):
                    u1 = min(nunit, u0 + k)
                    if gpu_dq:
                        return np.ascontiguousarray(src_u[u0:u1])   # (the copy pages the source in)
                    f32 = np.empty((u1 - u0, unit_f), dtype=np.float32)
                    step = max(1, (u1 - u0) // threads) if rows2d else 1

                    def dq(i):
                        j = min(u1 - u0, i + step)
                        to_float[src](src_u[u0 + i:u0 + j].ctypes.data if rows2d else src_u[u0 + i].ctypes.data,
                                      f32[i].ctypes.data, (j - i) * unit_f)
                        if esc is not None:
                            f32[i] *= esc[u0 + i]
                    with concurrent.futures.ThreadPoolExecutor(threads) as dex:
                        list(dex.map(dq, range(0, u1 - u0, step)))
                    return f32

                with concurrent.futures.ThreadPoolExecutor(1) as pre:
                    nxt = pre.submit(load, 0)
                    for u0 in range(0, nunit, k):
                        u1 = min(nunit, u0 + k)
                        buf = nxt.result()
                        if u1 < nunit:
                            nxt = pre.submit(load, u1)
                        nm, nr = (1, u1 - u0) if rows2d else (u1 - u0, ne1)
                        w_ = None if imb is None else imb[0 if rows2d else u0].ctypes.data
                        with gq_lock:
                            if gpu_dq:
                                got = gq.gpuq_requantize_batch(int(src), buf.ctypes.data, int(new_type),
                                                               dst_u[u0].ctypes.data, nm, nr, ne0, w_)
                            else:
                                got = gq.gpuq_quantize_batch(int(new_type), buf.ctypes.data, dst_u[u0].ctypes.data, nm,
                                                             nr, ne0, w_)
                        if got != per_u * (u1 - u0):
                            print("gpuq wrote %d bytes, expected %d for %s, %s %d-%d" % (
                                got, per_u * (u1 - u0), t.name, "rows" if rows2d else "experts", u0, u1 - 1))
                            return 1
                ex_map = []
            else:
                ex_map = None
            gpu_loop = hess is not None and on_gpu_loop
            with concurrent.futures.ThreadPoolExecutor(gpu_threads if gpu_loop else threads) as ex:
                for e, got in enumerate(ex_map if ex_map is not None else ex.map(
                        (one_ldlq_gpu if gpu_loop else one_ldlq) if hess is not None else one, range(ne2))):
                    if got != per:
                        print("ggml_quantize_chunk wrote %d bytes, expected %d for %s expert %d" % (got, per, t.name, e))
                        return 1
            w.write_tensor_data(dst.reshape((ne2, ne1, per // ne1) if len(t.shape) == 3 else (ne2 * ne1, per // ne1)),
                                tensor_endianess=r.endianess)
            stash.pop(t.name, None)
            print("  %s -> %s in %.0f s%s%s%s" % (t.name, new_type.name, time.time() - t0, " (gpu)" if gpu_dq else " (gpu quant, cpu dequant)" if on_gpu else "",
                                                   " (ldlq%s %s)" % (" gpu" if gpu_loop else "", hess) if hess is not None else "",
                                                   "" if hess is None or not fell_back else " [%d experts plain: fewer than %d rows, e.g. %s]" % (
                                                       len(fell_back), MIN_ROWS, ", ".join("%d:%d" % fb for fb in fell_back[:4]))),
                  flush=True)
        w.close()
    print("re-quantized %.2f GiB -> %.2f GiB%s" % (total_old / 2**30, total_new / 2**30, " (check only)" if check else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
