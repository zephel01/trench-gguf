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
