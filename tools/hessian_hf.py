"""Collects per-expert Hessians (X^T X of the inputs of each expert's ffn_gate/ffn_up and of its ffn_down) from a
Hugging Face MoE model over a calibration text, one decoder layer at a time, on the CPU with torch.

usage: hessian_hf.py <hf model dir> <calib.txt> <out dir> [--ctx=2048] [--batch=4] [--threads=12] [--layers=a-b]
                     [--max-chunks=N]

Writes, per layer L (the sums, not normalized, float32; plus the counts):
  <out dir>/blk.L.in.npy      [n_expert, d_model, d_model]   sum x x^T of the rows routed to the expert (gate/up input)
  <out dir>/blk.L.in_g.npy    the same with each row scaled by its routing weight g (sum g^2 x x^T)
  <out dir>/blk.L.mid.npy     [n_expert, d_ff, d_ff]         sum a a^T, a = silu(gate x) * (up x) (down input)
  <out dir>/blk.L.mid_g.npy   the same weighted by g^2
  <out dir>/blk.L.counts.npy  [n_expert] rows per expert; blk.L.gsum2.npy [n_expert] sum g^2
The diagonal of blk.L.in / counts is what llama-imatrix stores for ffn_gate_exps / ffn_up_exps (in_sum2 / counts), so
the two can be compared.

The text is tokenized with the model's tokenizer (no BOS unless the tokenizer adds one) and cut into chunks of --ctx
tokens, as llama-imatrix does with -c. The whole hidden state of every chunk is kept in RAM (N tokens x d_model fp32:
1.2 GB for 150K tokens of OLMoE) and each layer is run over all chunks before the next layer's weights are loaded, so
only one layer's weights and Hessians are in memory at a time (OLMoE: 1.6 GB + 2 x 1.3 GB).

Supported: OLMoE (OlmoeForCausalLM; transformers' OlmoeDecoderLayer does attention and the norms, the MoE block is
re-implemented here so the rows routed to each expert and their routing weights are at hand). Other architectures
need the three functions marked ARCH below. For a model too big for this (DeepSeek V4), the same per-layer loop would
run inside an engine (llama.cpp eval callback on MUL_MAT_ID: gather the rows of the input by the ids tensor).
"""
import json
import os
import re
import sys
import time

import numpy as np


def parse_layers(s, n):
    if not s:
        return list(range(n))
    a, b = (s.split("-") + [s])[:2]
    return list(range(int(a), int(b) + 1))


def main():
    flags = dict((a.split("=", 1) + [""])[:2] for a in sys.argv[1:] if a.startswith("--"))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 3:
        sys.exit(__doc__)
    model_dir, text_path, out_dir = args
    ctx = int(flags.get("--ctx") or 2048)
    batch = int(flags.get("--batch") or 4)
    threads = int(flags.get("--threads") or 12)
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer
    torch.set_num_threads(threads)
    torch.set_grad_enabled(False)

    config = AutoConfig.from_pretrained(model_dir)
    arch = config.architectures[0]
    if arch != "OlmoeForCausalLM":
        sys.exit("%s: only OlmoeForCausalLM is wired up (see ARCH in the source)" % arch)
    from transformers.models.olmoe.modeling_olmoe import OlmoeDecoderLayer, OlmoeRotaryEmbedding
    config._attn_implementation = "sdpa"

    # ARCH: tensor names
    def layer_prefix(L):
        return "model.layers.%d." % L
    EMB = "model.embed_tokens.weight"
    n_layers = config.num_hidden_layers
    n_exp, top_k, d_model, d_ff = config.num_experts, config.num_experts_per_tok, config.hidden_size, config.intermediate_size
    norm_topk = config.norm_topk_prob

    tok = AutoTokenizer.from_pretrained(model_dir)
    text = open(text_path, encoding="utf-8").read()
    ids = tok(text)["input_ids"]
    n_chunks = len(ids) // ctx
    if flags.get("--max-chunks"):
        n_chunks = min(n_chunks, int(flags["--max-chunks"]))
    ids = torch.tensor(ids[:n_chunks * ctx], dtype=torch.long).view(n_chunks, ctx)
    print("%d tokens -> %d chunks of %d (bos added by tokenizer: %s)" % (len(ids.reshape(-1)), n_chunks, ctx,
                                                                         bool(getattr(tok, "add_bos_token", False))), flush=True)

    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
    files = {}

    def get(name):
        f = index[name]
        if f not in files:
            files[f] = safe_open(os.path.join(model_dir, f), "pt")
        return files[f].get_tensor(name)

    emb = get(EMB).float()
    h = emb[ids]   # [n_chunks, ctx, d_model] fp32
    del emb
    rotary = OlmoeRotaryEmbedding(config)
    pos = torch.arange(ctx)[None]
    cos, sin = rotary(h[:1], pos)
    mask = torch.full((ctx, ctx), torch.finfo(torch.float32).min).triu(1)[None, None]
    os.makedirs(out_dir, exist_ok=True)
    layers = parse_layers(flags.get("--layers"), n_layers)

    for L in range(n_layers):
        t0 = time.time()
        layer = OlmoeDecoderLayer(config, L)
        pre = layer_prefix(L)
        sd = {k[len(pre):]: get(k).float() for k in index if k.startswith(pre)}
        missing, unexpected = layer.load_state_dict(sd, strict=False)
        if missing or unexpected:
            sys.exit("layer %d: missing %s unexpected %s" % (L, missing, unexpected))
        layer.eval()
        collect = L in layers
        # ARCH: the expert weights as [n_exp, out, in]
        Wg = torch.stack([sd["mlp.experts.%d.gate_proj.weight" % e] for e in range(n_exp)])
        Wu = torch.stack([sd["mlp.experts.%d.up_proj.weight" % e] for e in range(n_exp)])
        Wd = torch.stack([sd["mlp.experts.%d.down_proj.weight" % e] for e in range(n_exp)])
        Wr = sd["mlp.gate.weight"]
        del sd
        if collect:
            H_in = torch.zeros(n_exp, d_model, d_model, dtype=torch.float64)
            H_in_g = torch.zeros_like(H_in)
            H_mid = torch.zeros(n_exp, d_ff, d_ff, dtype=torch.float64)
            H_mid_g = torch.zeros_like(H_mid)
            counts = torch.zeros(n_exp, dtype=torch.float64)
            gsum2 = torch.zeros(n_exp, dtype=torch.float64)

        def moe_forward(hidden_states):   # ARCH: OLMoE's router (softmax then top-k, no renormalization)
            B, T, D = hidden_states.shape
            x = hidden_states.reshape(-1, D)
            probs = F.softmax(x @ Wr.T, dim=-1, dtype=torch.float32)
            topw, topi = torch.topk(probs, top_k, dim=-1)
            if norm_topk:
                topw = topw / topw.sum(-1, keepdim=True)
            out = torch.zeros_like(x)
            for e in range(n_exp):
                tokens, slots = (topi == e).nonzero(as_tuple=True)
                if len(tokens) == 0:
                    continue
                xe = x[tokens]
                g = topw[tokens, slots]
                a = F.silu(xe @ Wg[e].T) * (xe @ Wu[e].T)
                out.index_add_(0, tokens, (a @ Wd[e].T) * g[:, None])
                if collect:
                    H_in[e] += (xe.T @ xe).double()
                    xg = xe * g[:, None]
                    H_in_g[e] += (xg.T @ xg).double()
                    H_mid[e] += (a.T @ a).double()
                    ag = a * g[:, None]
                    H_mid_g[e] += (ag.T @ ag).double()
                    counts[e] += len(tokens)
                    gsum2[e] += (g * g).sum().double()
            return out.view(B, T, D), None

        layer.mlp.forward = moe_forward
        for c0 in range(0, n_chunks, batch):
            hb = h[c0:c0 + batch]
            out = layer(hb, attention_mask=mask, position_ids=pos.expand(hb.shape[0], -1), position_embeddings=(cos, sin))
            h[c0:c0 + batch] = out[0] if isinstance(out, tuple) else out
        if collect:
            np.save(os.path.join(out_dir, "blk.%d.in.npy" % L), H_in.float().numpy())
            np.save(os.path.join(out_dir, "blk.%d.in_g.npy" % L), H_in_g.float().numpy())
            np.save(os.path.join(out_dir, "blk.%d.mid.npy" % L), H_mid.float().numpy())
            np.save(os.path.join(out_dir, "blk.%d.mid_g.npy" % L), H_mid_g.float().numpy())
            np.save(os.path.join(out_dir, "blk.%d.counts.npy" % L), counts.numpy())
            np.save(os.path.join(out_dir, "blk.%d.gsum2.npy" % L), gsum2.numpy())
            c = counts.numpy()
            print("layer %d: %.0f s, rows per expert min %d median %d max %d%s" % (
                L, time.time() - t0, c.min(), np.median(c), c.max(), "" if L == n_layers - 1 else ""), flush=True)
            del H_in, H_in_g, H_mid, H_mid_g
        else:
            print("layer %d: %.0f s (forward only)" % (L, time.time() - t0), flush=True)
        if L >= max(layers):
            break
    json.dump({"model": model_dir, "text": text_path, "ctx": ctx, "chunks": n_chunks, "tokens": n_chunks * ctx,
               "arch": arch}, open(os.path.join(out_dir, "info.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
