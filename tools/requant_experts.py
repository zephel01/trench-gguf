"""Re-quantizes the routed experts of a (split) GGUF to smaller i-quants with an imatrix; copies everything else byte
for byte.

usage: requant_experts.py <in shard 1> <out dir> <rules file> --imatrix=<imatrix.gguf> [--check] [--threads=N]
                         [--gpu[=<device>]]
  rules file: one "<tensor name regex> <type>" a line (# comments), the first match wins, e.g.
      blk\\.(2[4-9]|30)\\.ffn_(gate|up)_exps\\.weight  iq3_xxs
      blk\\.\\d+\\.ffn_down_exps\\.weight             iq2_s
Each input shard becomes one output shard of the same name (the split metadata is copied as is); a missing shard is
skipped (DeepSeek V4.1: a shard holding only the Engram table).  Each expert is dequantized with ggml's own to_float
(any source type ggml reads: MXFP4, the i-quants, the k-quants) and quantized by ggml_quantize_chunk with that
expert's imatrix row (llama-quantize's normalization: in_sum2 / counts, 1 for an expert the calibration never routed
to).  The source's rounding error and the new one add up: start from the most precise file to hand.

--gpu quantizes IQ2_XXS / IQ2_S / IQ1_S / IQ1_M on the GPU with gpuq/libgpuq.so (ggml's quantizers line for line:
byte-identical output, checked by gpuq/test_gpuq.py); the dequantizing stays on the CPU threads, other types stay on
the CPU.  GPUQ_LIB picks the library (libgpuq_cuda.so for CUDA), GPUQ_DIR its folder.

Environment: GGUF_PY = llama.cpp/gguf-py (default: the installed gguf package), LIBGGML = <llama.cpp build>/bin/
libggml-base.so (the CPU quantizer, and the reference the GPU output matches).
"""
import concurrent.futures
import ctypes
import os
import re
import sys
import threading
import time

if os.environ.get("GGUF_PY"):   # llama.cpp/gguf-py; without it the installed gguf package (pip install gguf)
    sys.path.insert(0, os.environ["GGUF_PY"])
import numpy as np  # noqa: E402
import gguf  # noqa: E402
from gguf import GGMLQuantizationType as Q  # noqa: E402

LIBGGML = os.environ.get("LIBGGML", "libggml-base.so")   # <llama.cpp build>/bin/libggml-base.so


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
        gq = ctypes.CDLL(os.path.join(here, os.environ.get("GPUQ_LIB", "libgpuq.so")))   # (CUDA: libgpuq_cuda.so)
        gq.gpuq_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
        gq.gpuq_quantize.restype = ctypes.c_size_t
        gq.gpuq_quantize.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                     ctypes.c_void_p]
        gq.gpuq_block_bytes.restype = ctypes.c_size_t
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
        r = gguf.GGUFReader(sh)
        if gguf.Keys.General.ARCHITECTURE in r.fields:   # (the later shards of a split carry only the split keys)
            arch = r.fields[gguf.Keys.General.ARCHITECTURE].contents()
        out = os.path.join(out_dir, os.path.basename(sh))
        print("%s -> %s" % (sh, out), flush=True)
        plan = []
        for t in r.tensors:
            new_type = next((typ for pat, typ in rules if pat.search(t.name)), None)
            if new_type is not None and (len(t.shape) != 3 or t.name not in imat):
                print("  %s: not a 3-D expert tensor with imatrix data: left as is" % t.name)
                new_type = None
            if new_type is not None and new_type == t.tensor_type:
                new_type = None
            if new_type is None:
                plan.append((t, None, 0))
                continue
            ne0, ne1, ne2 = (int(x) for x in t.shape)
            bs, ts = gguf.GGML_QUANT_SIZES[new_type]
            if ne0 % bs != 0 or imat[t.name].shape != (ne2, ne0):
                print("  %s: ne0 %d / imatrix %s do not fit %s: left as is" % (t.name, ne0, imat[t.name].shape,
                                                                              new_type.name))
                plan.append((t, None, 0))
                continue
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
                ne0, ne1, ne2 = (int(x) for x in t.shape)
                w.add_tensor_info(t.name, (ne2, ne1, nbytes // (ne2 * ne1)), np.uint8, nbytes, raw_dtype=new_type)
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
            ne0, ne1, ne2 = (int(x) for x in t.shape)
            src = t.tensor_type
            if src not in to_float:
                to_float[src] = TO_FLOAT(lib.ggml_get_type_traits(int(src)).contents.to_float)
            lib.ggml_quantize_init(int(new_type))
            raw = np.ascontiguousarray(np.asarray(t.data).reshape(ne2, -1))
            per = nbytes // ne2
            dst = np.empty((ne2, per), dtype=np.uint8)
            im = imat[t.name]

            on_gpu = gq is not None and gq.gpuq_block_bytes(int(new_type)) > 0

            def one(e):
                f32 = np.empty(ne1 * ne0, dtype=np.float32)
                to_float[src](raw[e].ctypes.data, f32.ctypes.data, ne1 * ne0)
                if on_gpu:
                    with gq_lock:
                        return gq.gpuq_quantize(int(new_type), f32.ctypes.data, dst[e].ctypes.data, ne1, ne0,
                                                im[e].ctypes.data)
                got = lib.ggml_quantize_chunk(int(new_type), f32.ctypes.data, dst[e].ctypes.data, 0, ne1, ne0,
                                              im[e].ctypes.data)
                return got

            with concurrent.futures.ThreadPoolExecutor(threads) as ex:
                for e, got in enumerate(ex.map(one, range(ne2))):
                    if got != per:
                        print("ggml_quantize_chunk wrote %d bytes, expected %d for %s expert %d" % (got, per, t.name, e))
                        return 1
            w.write_tensor_data(dst.reshape(ne2, ne1, per // ne1), tensor_endianess=r.endianess)
            print("  %s -> %s in %.0f s%s" % (t.name, new_type.name, time.time() - t0, " (gpu)" if on_gpu else ""),
                  flush=True)
        w.close()
    print("re-quantized %.2f GiB -> %.2f GiB%s" % (total_old / 2**30, total_new / 2**30, " (check only)" if check else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
