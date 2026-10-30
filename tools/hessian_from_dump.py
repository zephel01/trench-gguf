"""Per-expert Hessians from a tools/hessian_dump dump (blk.L.x.bf16 + blk.L.ids.i32 [+ blk.L.a.bf16]).

As a module (used by requant_experts.py --dump): DumpHessians(dir, device) gives, per layer and expert,
H_in = sum x x^T over the rows routed to the expert (gate/up input) and H_mid = sum a a^T (down input), a taken from
blk.L.a.bf16 when the dump has it, else recomputed as silu(Wg x) * (Wu x) from the expert's own gate/up weights (the
float32 rows handed in). The sums run in torch (float32 GEMMs, float64 accumulation) on `device` (cpu, or cuda = ROCm
/ CUDA); for DeepSeek V4 (23K rows x 4096 an expert at 1M tokens, 1.7 TFLOP an expert) the GPU is the practical choice.

As a command, it writes the same files as tools/hessian_hf.py (blk.L.in.npy, blk.L.mid.npy, blk.L.counts.npy,
blk.L.gsum2.npy, and *_g.npy when the dump has the routing weights), for small models and for checks:
  hessian_from_dump.py <dump dir> <out dir> [--layers=a-b] [--model=<gguf shard 1>] [--device=cpu|cuda] [--threads=N]
                       [--recompute]
--model is needed when the dump has no blk.L.a.bf16 (the gate/up weights come from the GGUF, dequantized by ggml);
--recompute ignores blk.L.a.bf16 and recomputes anyway (a check of the two against each other).
"""
import os
import re
import sys

import numpy as np


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


class LayerDump:
    def __init__(self, ddir, L):
        pre = os.path.join(ddir, "blk.%d." % L)
        info = dict(line.split(None, 1) for line in open(os.path.join(ddir, "info.txt")) if " " in line.strip())
        self.n_embd, self.n_ids = int(info["n_embd"]), int(info["n_expert_used"])
        self.n_ff = int(info["n_ff"])
        self.ids = np.memmap(pre + "ids.i32", dtype=np.int32, mode="r").reshape(-1, self.n_ids)
        self.n = self.ids.shape[0]
        self.x = np.memmap(pre + "x.bf16", dtype=np.uint16, mode="r").reshape(self.n, self.n_embd)
        self.a = np.memmap(pre + "a.bf16", dtype=np.uint16, mode="r").reshape(self.n, self.n_ids, self.n_ff) \
            if os.path.exists(pre + "a.bf16") else None
        self.w = None   # the routing weights at the last stage the graph names (scaled > norm > raw)
        for v in ("scaled", "norm", "raw"):
            if os.path.exists(pre + "w_%s.f32" % v):
                self.w = np.memmap(pre + "w_%s.f32" % v, dtype=np.float32, mode="r").reshape(self.n, self.n_ids)
                break
        # rows per expert, once: sort the (token, slot) pairs by expert
        flat = np.asarray(self.ids).reshape(-1)
        order = np.argsort(flat, kind="stable")
        self.bounds = np.searchsorted(flat[order], np.arange(flat.max() + 2))
        self.tok = (order // self.n_ids).astype(np.int64)
        self.slot = (order % self.n_ids).astype(np.int64)
        self.n_expert = len(self.bounds) - 1

    def rows(self, e):
        if e + 1 >= len(self.bounds):   # (an expert no token was routed to, above the largest id seen)
            return self.tok[:0], self.slot[:0]
        s = slice(self.bounds[e], self.bounds[e + 1])
        return self.tok[s], self.slot[s]


class DumpHessians:
    def __init__(self, ddir, device="cpu", chunk=8192, recompute=False):
        import torch
        self.torch = torch
        self.ddir, self.device, self.chunk, self.recompute = ddir, device, chunk, recompute
        self.layers = {}

    def layer(self, L):
        if L not in self.layers:
            self.layers = {L: LayerDump(self.ddir, L)}   # (one layer at a time)
        return self.layers[L]

    def has(self, L):
        return os.path.exists(os.path.join(self.ddir, "blk.%d.x.bf16" % L))

    def _xtx(self, gen, dim):
        """sum over row chunks of X^T X, float32 GEMM, float64 accumulation on the device"""
        torch = self.torch
        H = torch.zeros(dim, dim, dtype=torch.float64, device=self.device)
        for xc in gen:
            H += (xc.T @ xc).double()
        return H

    def _bf16(self, u16):
        """bf16 rows (uint16 numpy) -> float32 tensor on the device (the shift runs on the device)"""
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(u16).view(np.int16)).to(self.device)
        return ((x.to(torch.int32) & 0xFFFF) << 16).view(torch.float32)

    def _x_chunks(self, d, tok, gated):
        torch = self.torch
        for c0 in range(0, len(tok), self.chunk):
            t = tok[c0:c0 + self.chunk]
            xc = self._bf16(np.asarray(d.x[t]))
            if gated is not None:
                xc = xc * torch.from_numpy(gated[c0:c0 + self.chunk]).to(self.device)[:, None]
            yield xc

    def hessian_in(self, L, e, gated=False):
        """(H_in [n_embd, n_embd] float64 numpy, n rows)"""
        d = self.layer(L)
        tok, slot = d.rows(e)
        g = None
        if gated and d.w is not None:
            g = np.asarray(d.w)[tok, slot].astype(np.float32)
        H = self._xtx(self._x_chunks(d, tok, g), d.n_embd)
        return H.cpu().numpy(), len(tok)

    def hessian_mid(self, L, e, Wg=None, Wu=None, gated=False, clamp=0.0):
        """(H_mid [n_ff, n_ff] float64 numpy, n rows). Wg, Wu: float32 [n_ff, n_embd] (needed without blk.L.a).
        clamp > 0: the model's SwiGLU limit (DeepSeek V4 / V4.1 swiglu_clamp_exp, llama.cpp's ggml_swiglu_clamp):
        a = silu(min(g, clamp)) * clamp(u, -clamp, clamp) in the recompute."""
        torch = self.torch
        d = self.layer(L)
        tok, slot = d.rows(e)
        g = np.asarray(d.w)[tok, slot].astype(np.float32) if gated and d.w is not None else None

        def gen():
            if d.a is not None and not self.recompute:
                for c0 in range(0, len(tok), self.chunk):
                    t, s = tok[c0:c0 + self.chunk], slot[c0:c0 + self.chunk]
                    ac = self._bf16(np.asarray(d.a[t, s]))
                    if g is not None:
                        ac = ac * torch.from_numpy(g[c0:c0 + self.chunk]).to(self.device)[:, None]
                    yield ac
            else:
                wg = torch.from_numpy(np.ascontiguousarray(Wg)).to(self.device)
                wu = torch.from_numpy(np.ascontiguousarray(Wu)).to(self.device)
                for xc in self._x_chunks(d, tok, None):
                    gx, ux = xc @ wg.T, xc @ wu.T
                    if clamp and clamp > 0:
                        gx, ux = torch.clamp(gx, max=clamp), torch.clamp(ux, -clamp, clamp)
                    ac = torch.nn.functional.silu(gx) * ux
                    if g is not None:
                        pass   # (handled below: scale rows)
                    yield ac
        # the down input's width: from the dump when it has blk.L.a (info.txt n_ff), else from the gate weights
        n_ff = d.n_ff if (d.a is not None and not self.recompute) else int(Wg.shape[0])
        if (d.a is None or self.recompute) and g is not None:   # gated recompute: scale after the nonlinearity
            def gen2():
                for i, ac in enumerate(gen()):
                    yield ac * torch.from_numpy(g[i * self.chunk:(i + 1) * self.chunk]).to(self.device)[:, None]
            H = self._xtx(gen2(), n_ff)
        else:
            H = self._xtx(gen(), n_ff)
        return H.cpu().numpy(), len(tok)


class DenseHessians:
    """Hessians of the dense (2-D, and grouped 3-D) tensors from a tools/hessian_dump --dense dump: dense.txt maps a
    weight to its input file blk.L.din.<name>.bf16 ([rows][groups][ne0] bf16); H = sum x x^T over the rows (per group
    for a grouped weight such as DeepSeek V4's attn_output_a). The shared experts' gate/up and the router read the MoE
    input, so without a din file of their own they take blk.L.x.bf16 (all dumped tokens, or the first max_rows).
    The H of one input file is kept until another file is asked for (weights sharing an input share the work)."""

    IMPLICIT = re.compile(r"^blk\.(\d+)\.(ffn_gate_shexp|ffn_up_shexp|ffn_gate_inp)\.weight$")

    def __init__(self, ddir, device="cpu", chunk=8192, max_rows=0):
        import torch
        self.torch, self.ddir, self.device, self.chunk, self.max_rows = torch, ddir, device, chunk, max_rows
        self.map = {}   # weight -> (file, ne0, groups)
        p = os.path.join(ddir, "dense.txt")
        if os.path.exists(p):
            for line in open(p):
                if line.startswith("#") or not line.strip():
                    continue
                w, f, ne0, g, _rows = line.split()
                self.map[w] = (f, int(ne0), int(g))
        info = os.path.join(ddir, "info.txt")
        self.n_embd = 0
        if os.path.exists(info):
            kv = dict(line.split(None, 1) for line in open(info) if " " in line.strip())
            self.n_embd = int(kv.get("n_embd", "0"))
        self.cache_file, self.cache = None, None

    def _entry(self, name):
        if name in self.map:
            return self.map[name]
        m = self.IMPLICIT.match(name)
        if m and self.n_embd and os.path.exists(os.path.join(self.ddir, "blk.%s.x.bf16" % m.group(1))):
            return ("blk.%s.x.bf16" % m.group(1), self.n_embd, 1)
        return None

    def has(self, name):
        return self._entry(name) is not None

    def groups(self, name):
        return self._entry(name)[2]

    def file(self, name):
        return self._entry(name)[0]

    def hessian(self, name, g=0):
        """(H [ne0, ne0] float64 numpy, n rows) of group g"""
        f, ne0, G = self._entry(name)
        if self.cache_file != f:
            torch = self.torch
            x = np.memmap(os.path.join(self.ddir, f), dtype=np.uint16, mode="r").reshape(-1, G, ne0)
            n = x.shape[0] if not self.max_rows else min(x.shape[0], self.max_rows)
            H = torch.zeros(G, ne0, ne0, dtype=torch.float64, device=self.device)
            for c0 in range(0, n, self.chunk):
                u = torch.from_numpy(np.ascontiguousarray(x[c0:min(n, c0 + self.chunk)]).view(np.int16)).to(self.device)
                xc = ((u.to(torch.int32) & 0xFFFF) << 16).view(torch.float32)   # [c, G, ne0]
                H += torch.einsum("cgi,cgj->gij", xc, xc).double()
            self.cache_file, self.cache = f, (H.cpu().numpy(), n)
        H, n = self.cache
        return H[g], n


def main():
    flags = dict((a.split("=", 1) + [""])[:2] for a in sys.argv[1:] if a.startswith("--"))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 2:
        sys.exit(__doc__)
    ddir, out = args
    import torch
    torch.set_num_threads(int(flags.get("--threads") or 12))
    dev = flags.get("--device") or "cpu"
    dh = DumpHessians(ddir, dev, recompute="--recompute" in flags)
    info = dict(line.split(None, 1) for line in open(os.path.join(ddir, "info.txt")) if " " in line.strip())
    layers = [int(x) for x in info["layers"].split()]
    if flags.get("--layers"):
        a, b = (flags["--layers"].split("-") + [flags["--layers"]])[:2]
        layers = [L for L in layers if int(a) <= L <= int(b)]
    model = None
    if flags.get("--model"):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from requant_experts import gguf, open_shards, dequant_expert
        model = open_shards(flags["--model"])
    os.makedirs(out, exist_ok=True)
    for L in layers:
        d = dh.layer(L)
        n_exp = d.n_expert
        Hin = np.zeros((n_exp, d.n_embd, d.n_embd), dtype=np.float32)
        Hmid = np.zeros((n_exp, d.n_ff, d.n_ff), dtype=np.float32)
        gated = d.w is not None
        Hin_g = np.zeros_like(Hin) if gated else None
        Hmid_g = np.zeros_like(Hmid) if gated else None
        counts = np.zeros(n_exp)
        gsum2 = np.zeros(n_exp)
        for e in range(n_exp):
            H, n = dh.hessian_in(L, e)
            Hin[e] = H
            counts[e] = n
            Wg = Wu = None
            if d.a is None or dh.recompute:
                if model is None:
                    sys.exit("the dump has no blk.%d.a.bf16: pass --model=<gguf> to recompute the down input" % L)
                Wg = dequant_expert(model, "blk.%d.ffn_gate_exps.weight" % L, e)
                Wu = dequant_expert(model, "blk.%d.ffn_up_exps.weight" % L, e)
            Hmid[e] = dh.hessian_mid(L, e, Wg, Wu)[0]
            if gated:
                Hin_g[e] = dh.hessian_in(L, e, gated=True)[0]
                Hmid_g[e] = dh.hessian_mid(L, e, Wg, Wu, gated=True)[0]
                tok, slot = d.rows(e)
                gsum2[e] = float((np.asarray(d.w)[tok, slot].astype(np.float64) ** 2).sum())
        np.save(os.path.join(out, "blk.%d.in.npy" % L), Hin)
        np.save(os.path.join(out, "blk.%d.mid.npy" % L), Hmid)
        np.save(os.path.join(out, "blk.%d.counts.npy" % L), counts)
        if gated:
            np.save(os.path.join(out, "blk.%d.in_g.npy" % L), Hin_g)
            np.save(os.path.join(out, "blk.%d.mid_g.npy" % L), Hmid_g)
            np.save(os.path.join(out, "blk.%d.gsum2.npy" % L), gsum2)
        print("layer %d: %d experts, rows min %d median %d max %d" % (L, n_exp, counts.min(), np.median(counts), counts.max()), flush=True)


if __name__ == "__main__":
    main()
