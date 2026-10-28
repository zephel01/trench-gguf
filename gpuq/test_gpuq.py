"""gpuq/test_gpuq.py (trench-gguf) - the GPU quantizer (libgpuq.so) against ggml_quantize_chunk, byte for byte.

    test_gpuq.py <gguf shard> <imatrix.gguf> [--tensor=<regex>] [--experts=N] [--types=iq2_xxs,iq2_s,iq1_s,iq1_m]
    test_gpuq.py --selftest [--types=...]      (no model files: random weights, a coarse 4-bit-like copy with many
                                                equal values, and a random imatrix)

Takes the first tensor matching --tensor (an expert tensor, ne2 experts), dequantizes N experts with ggml's to_float,
quantizes each to every type on the CPU (ggml) and on the GPU, and prints the share of identical blocks and the time.
"""
import ctypes
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from requant_experts import LIBGGML, TO_FLOAT, TypeTraits, read_imatrix, gguf, np, Q  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    flags = {a.split("=", 1)[0]: (a.split("=", 1) + [""])[1] for a in sys.argv[1:] if a.startswith("--")}
    selftest = "--selftest" in flags
    if not selftest:
        shard, imat_path = [a for a in sys.argv[1:] if not a.startswith("--")]
    pat = re.compile(flags.get("--tensor", r"ffn_gate_exps"))
    n_exp = int(flags.get("--experts", "4"))
    types = flags.get("--types", "iq2_xxs,iq2_s,iq1_s,iq1_m").split(",")

    lib = ctypes.CDLL(LIBGGML)
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                        ctypes.c_int64, ctypes.c_void_p]
    lib.ggml_get_type_traits.restype = ctypes.POINTER(TypeTraits)
    lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
    lib.ggml_quantize_init.argtypes = [ctypes.c_int]
    gq = ctypes.CDLL(os.path.join(HERE, os.environ.get("GPUQ_LIB", "libgpuq.so")))
    gq.gpuq_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    gq.gpuq_quantize.restype = ctypes.c_size_t
    gq.gpuq_quantize.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                 ctypes.c_void_p]
    gq.gpuq_block_bytes.restype = ctypes.c_size_t
    rc = gq.gpuq_init(os.path.join(HERE, "gpuq_tables.bin").encode(), int(os.environ.get("GPUQ_DEVICE", "0")))
    assert rc == 0, "gpuq_init %d" % rc

    if selftest:
        rng = np.random.default_rng(1234)
        ne0, ne1 = 2304, 256
        g = rng.standard_normal((ne1, ne0)).astype(np.float32) * 0.02
        coarse = (np.round(g / 0.005) * 0.005).astype(np.float32)          # ties, as MXFP4 / k-quant sources have
        heavy = (rng.standard_t(3, (ne1, ne0)) * 0.01).astype(np.float32)  # outliers
        xs = [g.reshape(-1), coarse.reshape(-1), heavy.reshape(-1)]
        imv = [rng.gamma(1.0, 1.0, ne0).astype(np.float32) for _ in xs]
        n_exp = len(xs)
        print("selftest: %d random matrices [%d x %d]" % (n_exp, ne0, ne1))
    else:
        imat = read_imatrix(imat_path)
        r = gguf.GGUFReader(shard)
        t = next((t for t in r.tensors if pat.search(t.name) and len(t.shape) == 3), None)
        if t is None:
            sys.exit("no 3-D expert tensor matching --tensor in %s (a split model keeps its experts in some shards only:\n"
                     "pass a shard that holds them)" % shard)
        ne0, ne1, ne2 = (int(x) for x in t.shape)
        print("%s %s [%d x %d x %d], %d experts" % (t.name, t.tensor_type.name, ne0, ne1, ne2, n_exp))
        to_float = TO_FLOAT(lib.ggml_get_type_traits(int(t.tensor_type)).contents.to_float)
        raw = np.ascontiguousarray(np.asarray(t.data).reshape(ne2, -1))
        im = imat[t.name]
        xs = []
        for e in range(n_exp):
            f32 = np.empty(ne1 * ne0, dtype=np.float32)
            to_float(raw[e].ctypes.data, f32.ctypes.data, ne1 * ne0)
            xs.append(f32)
        imv = [im[e] for e in range(n_exp)]
    bad = 0
    for name in types:
        typ = Q[name.upper()]
        bb = gq.gpuq_block_bytes(int(typ))
        lib.ggml_quantize_init(int(typ))
        same = blocks = 0
        tc = tg = 0.0
        for e in range(n_exp):
            nbytes = ne1 * (ne0 // 256) * bb
            a = np.empty(nbytes, dtype=np.uint8)
            b = np.empty(nbytes, dtype=np.uint8)
            t0 = time.time()
            got = lib.ggml_quantize_chunk(int(typ), xs[e].ctypes.data, a.ctypes.data, 0, ne1, ne0, imv[e].ctypes.data)
            t1 = time.time()
            got2 = gq.gpuq_quantize(int(typ), xs[e].ctypes.data, b.ctypes.data, ne1, ne0, imv[e].ctypes.data)
            t2 = time.time()
            assert got == nbytes and got2 == nbytes, (got, got2, nbytes)
            tc += t1 - t0
            tg += t2 - t1
            ab, bb_ = a.reshape(-1, bb), b.reshape(-1, bb)
            eq = np.all(ab == bb_, axis=1)
            same += int(eq.sum())
            blocks += len(eq)
            if not eq.all() and e == 0:
                i = int(np.argmin(eq))
                print("   first differing block %d:\n   cpu %s\n   gpu %s" % (i, ab[i].tobytes().hex(), bb_[i].tobytes().hex()))
        print("%-8s identical blocks %d / %d (%.4f%%)   cpu 1 thread %.2f s/expert, gpu %.3f s/expert (x%.0f)" %
              (name, same, blocks, 100.0 * same / blocks, tc / n_exp, tg / n_exp, tc / max(tg, 1e-9)))
        bad += blocks - same
    print("OK: byte-identical to ggml" if bad == 0 else "MISMATCH: %d blocks differ" % bad)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
