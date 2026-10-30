"""Error-feedback rounding (GPTQ / block-LDLQ) around ggml's own i-quant quantizers; the output is the plain GGUF block
format of the target type.

Idea (GPTQ, QuIP#'s BlockLDLQ): quantize the input columns of a weight matrix a block at a time; the rounding error of
a block is pushed into the not-yet-quantized columns so that W x stays close for the inputs x seen in calibration.
The Hessian H = X^T X (per expert, over the rows routed to it) says how: with U the upper Cholesky factor of H^-1
(H^-1 = U^T U), after a block B of columns is quantized with error E = W_B - Q_B (rows x |B|),
    W_rest -= (E U_BB^-1) U_B,rest
which for |B| = 1 is GPTQ's column update and for a joint block quantizer is the exact least-squares compensation of
the remaining columns (the Schur complement of the already fixed columns is U_rest,rest^T U_rest,rest).

The block is one super-block of the type (256 columns: QK_K), because ggml's IQ1/IQ2/IQ3 quantizers choose the
super-block scale d jointly over its 256 weights, and the quantizer for one super-block depends on nothing outside it
(sigma2 and the scales are per super-block) - so calling ggml_quantize_chunk on a rows x 256 slice gives exactly the
bytes a whole-row call would give for those weights. The slice's bytes are placed at their super-block's position in
the row. With feedback the later super-blocks see modified weights, so the bytes differ from plain rounding, but every
byte is still ggml's output for some weights: the format is unchanged, and no inference code knows the difference.

The inner call is pluggable (quant(block, weights) -> bytes, dequant(bytes) -> floats): ggml_quantize_chunk here, the
GPU kernel of gpuq later (it is per super-block as well).

Importance weights handed to the inner quantizer for a block (its "imatrix"): 'imatrix' = the imatrix row (what plain
rounding uses), 'diag' = diag(H)/n (the same thing from the collected Hessian), 'schur' = diag((U_BB^T U_BB)^-1), the
conditional Hessian of the block given that the rest will compensate (what the block's error really costs).

Order: 'seq' = super-blocks in row order; 'act' = super-blocks by descending sum of diag(H) (GPTQ's act-order at
super-block granularity; the bytes still go to their own positions).
"""
import numpy as np

QK_K = 256


def prepare_hessian(H, n, damp=0.01, order="seq", bs=QK_K):
    """H: [ne0, ne0] sum X^T X (float32/64), n: rows it was summed over. Returns (U, perm, diag_n, dead) where U is
    the upper Cholesky factor of the inverse of the damped, normalized, block-permuted Hessian, perm the column
    permutation (blocks of bs), diag_n = diag(H)/n in the permuted order, dead = columns with no energy."""
    ne0 = H.shape[0]
    Hn = np.asarray(H, dtype=np.float64) / max(float(n), 1.0)
    d = np.diag(Hn).copy()
    dead = d <= 0
    if dead.any():   # (GPTQ: a column no input ever used costs nothing; keep H invertible)
        Hn = Hn.copy()
        Hn[dead, :] = 0
        Hn[:, dead] = 0
        Hn[dead, dead] = 1.0
        d = np.diag(Hn).copy()
    nblk = ne0 // bs
    if order == "act":
        blk_energy = d.reshape(nblk, bs).sum(1)
        blk_order = np.argsort(-blk_energy, kind="stable")
    else:
        blk_order = np.arange(nblk)
    perm = (blk_order[:, None] * bs + np.arange(bs)[None]).reshape(-1)
    Hp = Hn[np.ix_(perm, perm)]
    mean_d = float(np.mean(np.diag(Hp)))
    Hp[np.diag_indices(ne0)] += damp * mean_d
    try:
        Lc = np.linalg.cholesky(Hp)
    except np.linalg.LinAlgError:   # (more damping until it factors)
        Hp[np.diag_indices(ne0)] += 10 * damp * mean_d
        Lc = np.linalg.cholesky(Hp)
    # H^-1 = Lc^-T Lc^-1 ; its upper Cholesky factor U: H^-1 = U^T U
    Linv = np.linalg.solve(Lc, np.eye(ne0))          # (lower)
    Hinv = Linv.T @ Linv
    Hinv = 0.5 * (Hinv + Hinv.T)
    U = np.linalg.cholesky(Hinv).T                    # upper, Hinv = U^T U
    return U, perm, d[perm], dead


def quantize_matrix(W, quant, dequant, tsize, U=None, perm=None, diag_n=None, imatrix=None, weights="imatrix",
                    feedback=True, bs=QK_K):
    """W: [ne1, ne0] float32. quant(block [ne1, bs] contiguous f32, qw [bs] f32) -> uint8 [ne1, tsize];
    dequant(uint8 [ne1, tsize]) -> float32 [ne1, bs]. imatrix: [ne0] the row's importance (needed for weights=
    'imatrix'). Returns (bytes [ne1, nblk*tsize] in GGUF row layout, dequantized W [ne1, ne0], stats)."""
    ne1, ne0 = W.shape
    nblk = ne0 // bs
    if perm is None:
        perm = np.arange(ne0)
    Wp = np.ascontiguousarray(W[:, perm], dtype=np.float32)
    out = np.empty((ne1, nblk, tsize), dtype=np.uint8)
    deq = np.empty((ne1, ne0), dtype=np.float32)
    blk_order = perm.reshape(nblk, bs)[:, 0] // bs
    err_plain = 0.0   # sum of squared error of each block against the weights it saw (diagnostics)
    for k in range(nblk):
        cols = slice(k * bs, (k + 1) * bs)
        if weights == "imatrix":
            qw = np.ascontiguousarray(imatrix[perm[cols]], dtype=np.float32)
        elif weights == "diag":
            qw = np.ascontiguousarray(diag_n[cols], dtype=np.float32)
        elif weights == "schur":
            Ubb = U[cols, cols]
            qw = np.ascontiguousarray(np.diag(np.linalg.inv(Ubb.T @ Ubb)), dtype=np.float32)
        else:
            raise ValueError(weights)
        blk = np.ascontiguousarray(Wp[:, cols])
        raw = quant(blk, qw)
        q = dequant(raw)
        out[:, blk_order[k], :] = raw
        deq[:, perm[cols]] = q
        if feedback and k + 1 < nblk:
            E = blk - q
            err_plain += float((E * E).sum())
            rest = slice((k + 1) * bs, ne0)
            Es = np.linalg.solve(U[cols, cols].T, E.T).T        # E U_BB^-1
            Wp[:, rest] -= (Es @ U[cols, rest]).astype(np.float32)
    return out.reshape(ne1, nblk * tsize), deq, {"err": err_plain}


def proxy_loss(W, deq, H, n):
    """tr((W-Q) H (W-Q)^T) / n: the calibration-input squared output error, what the feedback minimizes."""
    D = (W - deq).astype(np.float64)
    return float(np.einsum("ij,jk,ik->", D, np.asarray(H, dtype=np.float64), D) / max(float(n), 1.0))


# ---------------------------------------------------------------------------------------------------------------
# Finer feedback for IQ2_XXS: ggml's quantizer fixes the super-block scale d (fp16) and the 4-bit sub-block scales;
# with those kept, the 8-weight codebook points and signs of each 32-weight sub-block (or each group of 8) are chosen
# again on the error-updated weights, against the values inference really produces (iq2xxs_grid: 8/25/43/59 x d x
# (2l+1)/8, not the 1/3/5/7 the quantizer searches), and the error is pushed on before the next sub-block / group.
# Optionally the 4-bit sub-block scale is searched too (per row; d stays).

class IQ2XXS:
    """Tables for the IQ2_XXS block format, recovered from ggml's own dequantizer (to_float)."""
    TSIZE = 66

    def __init__(self, to_float):
        grid = np.zeros((256, 8), dtype=np.float32)
        raw = np.zeros((256, self.TSIZE), dtype=np.uint8)
        raw[:, 0:2] = np.frombuffer(np.float16(1.0).tobytes(), dtype=np.uint8)     # d = 1
        q = np.zeros((256, 2), dtype=np.uint32)
        q[:, 0] = np.arange(256)                      # grid index in group 0, signs 0
        q[:, 1] = 3 << 28                             # l = 3: db = (0.5 + 3) * 0.25 = 0.875
        raw[:, 2:10] = q.view(np.uint8)
        f = np.empty((256, 256), dtype=np.float32)
        to_float(raw.ctypes.data, f.ctypes.data, f.size)
        grid = np.rint(f[:, :8] / 0.875).astype(np.float32)   # 8, 25, 43 or 59
        assert set(np.unique(grid).tolist()) <= {8.0, 25.0, 43.0, 59.0}, np.unique(grid)
        self.grid = grid
        # sign index s (7 bits) -> 8-bit mask with even parity (ksigns_iq2xs)
        s = np.arange(128, dtype=np.uint8)
        par = np.array([bin(int(v)).count("1") & 1 for v in s], dtype=np.uint8)
        self.ksigns = s | (par << 7)
        self.signmask = np.array([(-1.0 if (self.ksigns[k] >> j) & 1 else 1.0) for k in range(128) for j in range(8)],
                                 dtype=np.float32).reshape(128, 8)

    def parse(self, raw):
        """raw [ne1, 66] -> d [ne1] f32, ls [ne1, 8] int (the 4-bit sub-block scales)"""
        d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(-1)
        q = raw[:, 2:66].copy().view(np.uint32).reshape(-1, 8, 2)
        ls = (q[:, :, 1] >> 28).astype(np.int64)
        return d, ls

    def best_points(self, x, w, db):
        """x [ne1, 8g] weights of g groups of 8, w the same shape (importance), db [ne1] the sub-block's dequant
        scale -> (grid idx [ne1, g] int, sign idx [ne1, g] int (7 bits), dequantized [ne1, 8g], cost [ne1])
        Exhaustive over the 256 points; signs follow x, parity fixed by flipping the cheapest weight (as ggml)."""
        ne1, n = x.shape
        g = n // 8
        xg = x.reshape(ne1, g, 1, 8)
        wg = w.reshape(ne1, g, 1, 8)
        v = (self.grid[None, None] * db[:, None, None, None])          # [ne1, 1, 256, 8] candidate magnitudes
        ax = np.abs(xg)
        err = wg * (ax - v) ** 2                                        # [ne1, g, 256, 8]
        cost = err.sum(-1)
        # parity: the product of the signs must be +1 (even number of minus signs); x < 0 gives the sign
        neg = (xg < 0)
        odd = (neg.sum(-1) & 1).astype(bool)                            # [ne1, g, 1]
        flip_cost = 4 * wg * ax * v                                     # [ne1, g, 256, 8]
        imin = flip_cost.argmin(-1)                                     # [ne1, g, 256]
        add = np.take_along_axis(flip_cost, imin[..., None], -1)[..., 0]
        cost = cost + np.where(odd, add, 0.0)
        best = cost.argmin(-1)                                          # [ne1, g]
        bc = np.take_along_axis(cost, best[..., None], -1)[..., 0]
        # signs for the chosen point
        sgn = np.where(neg[:, :, 0, :], -1.0, 1.0).astype(np.float32)   # [ne1, g, 8]
        fl = np.take_along_axis(imin, best[..., None], -1)[..., 0]      # [ne1, g] the flipped position
        oddg = odd[:, :, 0]
        rows, cols = np.nonzero(oddg)
        sgn[rows, cols, fl[rows, cols]] *= -1
        vals = self.grid[best] * db[:, None, None]                     # [ne1, g, 8]
        deq = (vals * sgn).reshape(ne1, n)
        bits = (sgn < 0).astype(np.uint8)
        s8 = (bits << np.arange(8, dtype=np.uint8)).sum(-1).astype(np.uint8)   # even parity -> low 7 bits = index
        assert ((s8 >> 7) == np.array([bin(int(v)).count("1") & 1 for v in (s8 & 127).reshape(-1)]).reshape(s8.shape)).all()
        return best, (s8 & 127).astype(np.int64), deq, bc.sum(-1)

    def encode(self, d, ls, gidx, sidx):
        """d [ne1], ls [ne1, 8], gidx [ne1, 8, 4], sidx [ne1, 8, 4] -> raw [ne1, 66]"""
        ne1 = d.shape[0]
        raw = np.zeros((ne1, self.TSIZE), dtype=np.uint8)
        raw[:, 0:2] = d.astype(np.float16).view(np.uint8).reshape(ne1, 2)
        q = np.zeros((ne1, 8, 2), dtype=np.uint32)
        for k in range(4):
            q[:, :, 0] |= (gidx[:, :, k].astype(np.uint32) << (8 * k))
            q[:, :, 1] |= (sidx[:, :, k].astype(np.uint32) << (7 * k))
        q[:, :, 1] |= (ls.astype(np.uint32) << 28)
        raw[:, 2:66] = q.reshape(ne1, 16).view(np.uint8)
        return raw


def quantize_matrix_iq2xxs_sub(W, quant, tab, U, perm, imatrix, sub=32, scale_search=0, bs=QK_K):
    """Like quantize_matrix for IQ2_XXS, but the feedback runs every `sub` columns (32 = sub-block, 8 = codebook
    group) inside each super-block: ggml's pass fixes d and the sub-block scales, the points are chosen again on the
    updated weights. scale_search=k also tries the 4-bit scales l-k..l+k per row (the best total cost wins).
    Returns (bytes, dequantized W, stats)."""
    ne1, ne0 = W.shape
    nblk = ne0 // bs
    tsize = tab.TSIZE
    Wp = np.ascontiguousarray(W[:, perm], dtype=np.float32)
    out = np.empty((ne1, nblk, tsize), dtype=np.uint8)
    deq = np.empty((ne1, ne0), dtype=np.float32)
    blk_order = perm.reshape(nblk, bs)[:, 0] // bs
    nsub = bs // sub
    for k in range(nblk):
        cols = slice(k * bs, (k + 1) * bs)
        qw = np.ascontiguousarray(imatrix[perm[cols]], dtype=np.float32)
        blk = np.ascontiguousarray(Wp[:, cols])
        raw = quant(blk, qw)                                   # ggml's pass: d and the sub-block scales
        d, ls = tab.parse(raw)
        sigma2 = (blk * blk).sum(1) / bs                       # (what ggml weights the search with)
        weff = qw[None] * np.sqrt(sigma2[:, None] + blk * blk)
        gidx = np.zeros((ne1, 8, 4), dtype=np.int64)
        sidx = np.zeros((ne1, 8, 4), dtype=np.int64)
        new_ls = ls.copy()
        for j in range(nsub):
            c0 = k * bs + j * sub
            sc = slice(c0, c0 + sub)
            x = np.ascontiguousarray(Wp[:, sc])
            w = np.ascontiguousarray(weff[:, j * sub:(j + 1) * sub])
            ib = (j * sub) // 32                                # the sub-block(s) these columns belong to
            g0 = (j * sub) % 32 // 8
            ng = sub // 8
            if sub == 32:
                cand = range(-scale_search, scale_search + 1)
                bestc = None
                for dl in cand:
                    l_try = np.clip(ls[:, ib] + dl, 0, 15)
                    db = d * (0.5 + l_try) * 0.25
                    gi, si, dq, c = tab.best_points(x, w, db)
                    if bestc is None:
                        bestc, bl, bg, bsn, bd = c, l_try, gi, si, dq
                    else:
                        better = c < bestc
                        bestc = np.where(better, c, bestc)
                        bl = np.where(better, l_try, bl)
                        bg = np.where(better[:, None], gi, bg)
                        bsn = np.where(better[:, None], si, bsn)
                        bd = np.where(better[:, None], dq, bd)
                new_ls[:, ib] = bl
                gidx[:, ib, :] = bg
                sidx[:, ib, :] = bsn
                q = bd
            else:   # groups of 8 with the sub-block's scale as it is
                db = d * (0.5 + new_ls[:, ib]) * 0.25
                gi, si, q, _ = tab.best_points(x, w, db)
                gidx[:, ib, g0:g0 + ng] = gi
                sidx[:, ib, g0:g0 + ng] = si
            zero = d == 0
            if zero.any():                                      # (an all-zero super-block stays all zero)
                q[zero] = 0
                gidx[zero, ib] = 0
                sidx[zero, ib] = 0
            deq[:, perm[sc]] = q
            if c0 + sub < ne0:
                E = x - q
                rest = slice(c0 + sub, ne0)
                Es = np.linalg.solve(U[sc, sc].T, E.T).T
                Wp[:, rest] -= (Es @ U[sc, rest]).astype(np.float32)
        enc = tab.encode(d, new_ls, gidx, sidx)
        out[:, blk_order[k], :] = enc
    return out.reshape(ne1, nblk * tsize), deq, {}


def prepare_hessian_torch(H, n, damp=0.01, order="seq", bs=QK_K, device="cuda", out="numpy"):
    """prepare_hessian with torch on `device` (float32 on a GPU is enough for a damped H; float64 on the CPU).
    H: numpy [ne0, ne0], or a torch tensor already on `device`. Returns numpy (U float64, perm, diag_n, dead) like
    prepare_hessian, or with out="torch" the same four as tensors on `device` (U float64; for quantize_matrix_torch)."""
    import torch
    dt = torch.float64 if device == "cpu" else torch.float32
    ne0 = H.shape[0]
    if isinstance(H, torch.Tensor):
        Hn = H.to(device=device, dtype=torch.float64) / max(float(n), 1.0)
    else:
        Hn = torch.as_tensor(np.asarray(H), dtype=torch.float64).to(device) / max(float(n), 1.0)
    d = torch.diagonal(Hn).clone()
    dead = d <= 0
    if bool(dead.any()):
        Hn[dead, :] = 0
        Hn[:, dead] = 0
        Hn[dead, dead] = 1.0
        d = torch.diagonal(Hn).clone()
    nblk = ne0 // bs
    if order == "act":
        blk_order = torch.argsort(-d.reshape(nblk, bs).sum(1), stable=True)
    else:
        blk_order = torch.arange(nblk, device=device)
    perm = (blk_order[:, None] * bs + torch.arange(bs, device=device)[None]).reshape(-1)
    Hp64 = Hn[perm][:, perm]
    mean_d = float(torch.diagonal(Hp64).mean())

    def factor(dtype, extra):
        Hp = Hp64.to(dtype).clone()
        Hp.diagonal().add_((damp + extra) * mean_d)
        Lc = torch.linalg.cholesky(Hp)
        Hinv = torch.cholesky_inverse(Lc)
        Hinv = 0.5 * (Hinv + Hinv.T)
        return torch.linalg.cholesky(Hinv, upper=True)

    # float32 on the GPU first; a Hessian whose ridge is swamped by a few huge diagonal entries (massive activations)
    # is not positive-definite in float32 - then float64, then more ridge (a non-PD case is rare: it costs seconds)
    U = None
    for dtype, extra in ((dt, 0.0), (torch.float64, 0.0), (torch.float64, 10 * damp), (torch.float64, 100 * damp)):
        try:
            U = factor(dtype, extra)
            break
        except Exception:
            continue
    if U is None:
        raise RuntimeError("Hessian could not be factored even with %.0f%% ridge" % (100 * (damp + 100 * damp)))
    if out == "torch":
        return U.double(), perm, d[perm], dead
    return (U.double().cpu().numpy(), perm.cpu().numpy(), d[perm].cpu().numpy(), dead.cpu().numpy())


# ---------------------------------------------------------------------------------------------------------------
# The same loop with the weights on the GPU (torch): the block quantizer and its dequantizer are gpuq's device-pointer
# kernels (gpuq_quantize_dev / gpuq_dequantize_dev, ggml's bytes and ggml's to_float), the error update is a GPU GEMM.
# The math is quantize_matrix's: U float64, E U_BB^-1 by a float64 triangular solve, the update cast to float32 and
# subtracted from the float32 weights (f64=False does the update in float32: faster on a GPU with slow float64, the
# bytes drift a little more from the CPU path). Several matrices that share U, perm and the feedback (gate and up of
# one expert) go through together as a batch: one kernel launch a block quantizes all their rows.

class GpuQuant:
    """gpuq's device-pointer quantizer / dequantizer of one type for torch tensors (ctypes handle gq, see gpuq.hip)."""

    def __init__(self, gq, typ, tsize, bs=QK_K):
        import ctypes
        self.gq, self.typ, self.tsize, self.bs = gq, int(typ), tsize, bs
        if not hasattr(gq, "_dev_api"):
            gq.gpuq_quantize_dev.restype = ctypes.c_size_t
            gq.gpuq_quantize_dev.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
                                             ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
            gq.gpuq_dequantize_dev.restype = ctypes.c_size_t
            gq.gpuq_dequantize_dev.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
            gq.gpuq_dequantize_src_dev.restype = ctypes.c_size_t
            gq.gpuq_dequantize_src_dev.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
            gq.gpuq_can_dequantize_type.restype = ctypes.c_int
            gq._dev_api = True

    @staticmethod
    def supported(gq, typ):
        GpuQuant(gq, typ, 0)
        return gq.gpuq_block_bytes(int(typ)) > 0 and int(typ) != 20 and gq.gpuq_can_dequantize_type(int(typ)) == 1

    @staticmethod
    def _stream():
        import torch
        return torch.cuda.current_stream().cuda_stream

    def quant(self, blk, qw, raw):
        """blk [B, rows, bs] f32 contiguous on the device, qw [B, bs] f32 contiguous -> raw [B, rows, tsize] uint8 (filled)"""
        B, rows = blk.shape[0], blk.shape[1]
        got = self.gq.gpuq_quantize_dev(self.typ, blk.data_ptr(), raw.data_ptr(), B * rows, qw.data_ptr(), B * rows, rows,
                                        self._stream())
        if got != B * rows * self.tsize:
            raise RuntimeError("gpuq_quantize_dev failed (type %d, %d blocks)" % (self.typ, B * rows))

    def dequant(self, raw):
        """raw [B, rows, tsize] uint8 contiguous -> f32 [B, rows, bs]"""
        import torch
        B, rows = raw.shape[0], raw.shape[1]
        f = torch.empty((B, rows, self.bs), dtype=torch.float32, device=raw.device)
        if self.gq.gpuq_dequantize_dev(self.typ, raw.data_ptr(), f.data_ptr(), B * rows, self._stream()) == 0:
            raise RuntimeError("gpuq_dequantize_dev failed (type %d)" % self.typ)
        return f

    def dequant_src(self, src, raw_host, n, device):
        """the source tensor's bytes (ggml type src: F32/F16/BF16/Q8_0/MXFP4, numpy uint8) -> f32 [n] on the device,
        or None when gpuq cannot read that type"""
        import torch
        if self.gq.gpuq_can_dequantize(int(src)) != 1:
            return None
        rb = torch.from_numpy(np.ascontiguousarray(raw_host).reshape(-1)).to(device)
        f = torch.empty(n, dtype=torch.float32, device=device)
        if self.gq.gpuq_dequantize_src_dev(int(src), rb.data_ptr(), f.data_ptr(), n, self._stream()) == 0:
            raise RuntimeError("gpuq_dequantize_src_dev failed (src %d)" % int(src))
        return f


def quantize_matrix_torch(W, gqt, U=None, perm=None, diag_n=None, imatrix=None, weights="imatrix", feedback=True,
                          bs=QK_K, f64=True):
    """quantize_matrix on the GPU. W: torch f32 [B, ne1, ne0] on the device (B matrices sharing U and perm, e.g. gate
    and up of one expert); gqt: GpuQuant of the target type; U: [ne0, ne0] (float32/64) on the device; perm: [ne0] long
    on the device; diag_n: [ne0] (permuted order); imatrix: [B, ne0] f32 on the device (the rows' importance).
    Returns uint8 [B, ne1, nblk*tsize] on the device in GGUF row layout (the caller moves it to the host)."""
    import torch
    B, ne1, ne0 = W.shape
    nblk = ne0 // bs
    tsize = gqt.tsize
    dev = W.device
    if perm is None:
        perm = torch.arange(ne0, device=dev)
    Wp = W.index_select(2, perm).contiguous()
    Ud = None
    if feedback:
        Ud = U.to(torch.float64 if f64 else torch.float32)
    if weights == "imatrix":
        imp = imatrix.index_select(1, perm).contiguous()
    elif weights == "diag":
        imp = diag_n.to(torch.float32)[None].expand(B, ne0).contiguous()
    elif weights != "schur":
        raise ValueError(weights)
    blk_order = (perm.view(nblk, bs)[:, 0] // bs).tolist()
    out = torch.empty((nblk, B, ne1, tsize), dtype=torch.uint8, device=dev)   # slot-major: one block's bytes contiguous
    for k in range(nblk):
        c0, c1 = k * bs, (k + 1) * bs
        blk = Wp[:, :, c0:c1].contiguous()
        if weights == "schur":
            Ubb = U[c0:c1, c0:c1].double()
            qw = torch.diagonal(torch.linalg.inv(Ubb.T @ Ubb)).to(torch.float32)[None].expand(B, bs).contiguous()
        else:
            qw = imp[:, c0:c1].contiguous()
        raw = out[blk_order[k]]
        gqt.quant(blk, qw, raw)
        if feedback and k + 1 < nblk:
            q = gqt.dequant(raw)
            E = (blk - q).to(Ud.dtype)
            Es = torch.linalg.solve_triangular(Ud[c0:c1, c0:c1], E, upper=True, left=False)   # E U_BB^-1
            Wp[:, :, c1:] -= (Es @ Ud[c0:c1, c1:]).to(torch.float32)
    return out.permute(1, 2, 0, 3).reshape(B, ne1, nblk * tsize)
