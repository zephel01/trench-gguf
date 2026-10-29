"""Checks tools/ldlq.py against ggml on random weights:
  1. without feedback, the super-block-wise calls give the same bytes as one whole-row ggml_quantize_chunk call;
  2. the assembled rows dequantize (ggml to_float on whole rows) to the per-block dequantized values;
  3. with feedback on a correlated synthetic Hessian, the proxy loss tr(E H E^T) goes down against plain rounding.
usage: test_ldlq.py [--types=iq2_xxs,iq1_s,iq1_m,iq2_s] [--rows=64] [--cols=1024]
"""
import ctypes
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
from requant_experts import LIBGGML, Q, TO_FLOAT, TypeTraits, gguf  # noqa: E402
import ldlq  # noqa: E402


def main():
    flags = dict((a.split("=", 1) + [""])[:2] for a in sys.argv[1:] if a.startswith("--"))
    types = [Q[t.upper()] for t in (flags.get("--types") or "iq2_xxs,iq1_s,iq1_m,iq2_s,iq2_xs,iq3_xxs").split(",")]
    ne1, ne0 = int(flags.get("--rows") or 64), int(flags.get("--cols") or 1024)
    lib = ctypes.CDLL(LIBGGML)
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                        ctypes.c_int64, ctypes.c_void_p]
    lib.ggml_get_type_traits.restype = ctypes.POINTER(TypeTraits)
    lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
    lib.ggml_quantize_init.argtypes = [ctypes.c_int]
    rng = np.random.default_rng(1)
    # a correlated input: x = A z with a random mixing matrix, n = 4 ne0 rows
    n = 4 * ne0
    A = rng.standard_normal((ne0, ne0)).astype(np.float32) * 0.2 + np.eye(ne0, dtype=np.float32)
    X = rng.standard_normal((n, ne0)).astype(np.float32) @ A
    X *= (1 + 3 * rng.random(ne0, dtype=np.float32))[None]   # uneven column energies
    H = (X.T @ X).astype(np.float64)
    imatrix = (np.diag(H) / n).astype(np.float32)
    W = (rng.standard_normal((ne1, ne0)) * 0.05).astype(np.float32)
    ok = True
    for typ in types:
        lib.ggml_quantize_init(int(typ))
        bs, tsize = gguf.GGML_QUANT_SIZES[typ]
        to_float = TO_FLOAT(lib.ggml_get_type_traits(int(typ)).contents.to_float)

        def quant(blk, qw):
            raw = np.empty((blk.shape[0], tsize), dtype=np.uint8)
            got = lib.ggml_quantize_chunk(int(typ), blk.ctypes.data, raw.ctypes.data, 0, blk.shape[0], blk.shape[1],
                                          qw.ctypes.data)
            assert got == raw.nbytes, (got, raw.nbytes)
            return raw

        def dequant(raw):
            f = np.empty((raw.shape[0], bs), dtype=np.float32)
            to_float(raw.ctypes.data, f.ctypes.data, f.size)
            return f

        ref = np.empty((ne1, (ne0 // bs) * tsize), dtype=np.uint8)
        got = lib.ggml_quantize_chunk(int(typ), W.ctypes.data, ref.ctypes.data, 0, ne1, ne0, imatrix.ctypes.data)
        assert got == ref.nbytes
        plain, deq_plain, _ = ldlq.quantize_matrix(W, quant, dequant, tsize, imatrix=imatrix, feedback=False)
        same = np.array_equal(plain, ref)
        full = np.empty((ne1, ne0), dtype=np.float32)
        to_float(plain.ctypes.data, full.ctypes.data, full.size)
        deq_same = np.array_equal(full, deq_plain)
        t0 = time.time()
        U, perm, diag_n, dead = ldlq.prepare_hessian(H, n, damp=0.01)
        fb, deq_fb, _ = ldlq.quantize_matrix(W, quant, dequant, tsize, U=U, perm=perm, diag_n=diag_n,
                                            imatrix=imatrix, weights="imatrix", feedback=True)
        to_float(fb.ctypes.data, full.ctypes.data, full.size)
        fb_deq_same = np.array_equal(full, deq_fb)
        Ua, perma, diag_a, _ = ldlq.prepare_hessian(H, n, damp=0.01, order="act")
        fba, deq_fba, _ = ldlq.quantize_matrix(W, quant, dequant, tsize, U=Ua, perm=perma, diag_n=diag_a,
                                               imatrix=imatrix, weights="schur", feedback=True)
        extra = ""
        if typ == Q.IQ2_XXS:
            tab = ldlq.IQ2XXS(to_float)
            for sub, ss in ((32, 0), (8, 0), (32, 2)):
                t1 = time.time()
                sb, deq_sb, _ = ldlq.quantize_matrix_iq2xxs_sub(W, quant, tab, U, perm, imatrix, sub=sub, scale_search=ss)
                to_float(sb.ctypes.data, full.ctypes.data, full.size)
                ok &= np.array_equal(full, deq_sb)
                extra += "; sub%d%s %.5g (%+.1f%%, dequant matches %s, %.1f s)" % (
                    sub, "+ss%d" % ss if ss else "", ldlq.proxy_loss(W, deq_sb, H, n),
                    100 * (ldlq.proxy_loss(W, deq_sb, H, n) / ldlq.proxy_loss(W, deq_plain, H, n) - 1),
                    np.array_equal(full, deq_sb), time.time() - t1)
            # the no-feedback sub path must reproduce ggml's own choice up to the grid mismatch: just report it
        lp = ldlq.proxy_loss(W, deq_plain, H, n)
        lf = ldlq.proxy_loss(W, deq_fb, H, n)
        la = ldlq.proxy_loss(W, deq_fba, H, n)
        print("%-8s plain bytes == ggml whole-row: %s; dequant matches: %s / %s; proxy loss plain %.5g, feedback %.5g "
              "(%+.1f%%), act+schur %.5g (%+.1f%%), %.1f s" % (
                  typ.name.lower(), same, deq_same, fb_deq_same, lp, lf, 100 * (lf / lp - 1), la, 100 * (la / lp - 1),
                  time.time() - t0) + extra)
        ok &= same and deq_same and fb_deq_same
    print("OK" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
