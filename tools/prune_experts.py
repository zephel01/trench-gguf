"""Writes a GGUF set that keeps, in every MoE layer, only the listed routed experts: the experts' weights, the router's
rows and the router biases are sliced byte for byte (nothing re-quantized), and <arch>.expert_count becomes the kept
count. Which experts to keep is the caller's choice - e.g. the most-routed ones on calibration text of every language
and task the model should keep (routing counts per domain, then a kept-route check on held-out text).

usage: prune_experts.py <shard 1> <keep.json> <out dir> <out stem> [--check] [--no-copy]
  keep.json: {"<layer>": [expert ids], ...}, the same number of ids in every MoE layer (the order does not matter;
  the kept experts stay in id order). Each input shard becomes <out stem>-NNNNN-of-MMMMM.gguf; a shard with nothing
  to slice (DeepSeek V4.1's Engram table) is hard-linked (a copy on another disk), or left out with --no-copy (when
  you replace it anyway). GGUF_PY as in requant_experts.py.

Sliced per layer L: blk.L.ffn_{gate,up,down}_exps.weight (the expert dimension), blk.L.ffn_gate_inp.weight (the
router's rows), blk.L.exp_probs_b.bias and any other blk.L tensor whose outer dimension is the expert count
(DeepSeek V4.1: exp_probs_b_vl.bias). Models with hash-routed layers (<arch>.hash_layer_count > 0) are refused:
their token -> expert tables would point at removed experts.
"""
import json
import os
import re
import shutil
import sys

if os.environ.get("GGUF_PY"):   # llama.cpp/gguf-py; without it the installed gguf package (pip install gguf)
    sys.path.insert(0, os.environ["GGUF_PY"])
import numpy as np  # noqa: E402
import gguf  # noqa: E402


def shards_of(first):
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", os.path.basename(first))
    if m is None:
        return [first]
    return [os.path.join(os.path.dirname(first), "%s-%05d-of-%s.gguf" % (m.group(1), i, m.group(3)))
            for i in range(1, int(m.group(3)) + 1)]


def link_or_copy(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(os.path.realpath(src), dst)
    except OSError:
        shutil.copyfile(src, dst)


def main():
    check = "--check" in sys.argv
    no_copy = "--no-copy" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 4:
        print(__doc__)
        return 2
    first, keep_path, out_dir, stem = args
    keep = {int(k): sorted(set(int(e) for e in v)) for k, v in json.load(open(keep_path)).items()}
    counts = {len(v) for v in keep.values()}
    if len(counts) != 1:
        print("keep.json: every layer must keep the same number of experts (got %s)" % sorted(counts))
        return 1
    n_keep = counts.pop()
    shards = shards_of(first)
    r1 = gguf.GGUFReader(shards[0])
    arch = r1.fields[gguf.Keys.General.ARCHITECTURE].contents()
    n_expert = int(r1.fields["%s.expert_count" % arch].contents())
    hash_key = "%s.hash_layer_count" % arch
    if hash_key in r1.fields and int(r1.fields[hash_key].contents()) > 0:
        print("%s > 0: hash-routed layers map tokens to fixed experts - not pruned" % hash_key)
        return 1
    if any(e < 0 or e >= n_expert for v in keep.values() for e in v):
        print("keep.json: an expert id outside 0..%d" % (n_expert - 1))
        return 1
    print("%s: %d experts -> %d in %d layers" % (arch, n_expert, n_keep, len(keep)))
    lay = re.compile(r"^blk\.(\d+)\.")
    os.makedirs(out_dir, exist_ok=True)
    seen_layers = set()
    old_b = new_b = 0
    for sh in shards:
        m = re.match(r".*-(\d{5}-of-\d{5})\.gguf$", os.path.basename(sh))
        out = os.path.join(out_dir, "%s-%s.gguf" % (stem, m.group(1)) if m else "%s.gguf" % stem)
        try:
            r = gguf.GGUFReader(sh)
        except ValueError:   # a tensor type gguf-py does not know: nothing of ours in it
            print("%s: gguf-py cannot read it: kept as is" % os.path.basename(sh))
            if not check and not no_copy:
                link_or_copy(sh, out)
            continue
        cut = {}
        for t in r.tensors:
            mm = lay.match(t.name)
            if mm is None or int(mm.group(1)) not in keep:
                continue
            # the outer (slowest) dimension is the expert one: shape is ggml order, the last entry the outer
            if int(t.shape[-1]) == n_expert and (t.name.endswith("_exps.weight") or "gate_inp" in t.name or
                                                 t.name.endswith(".bias")):
                cut[t.name] = keep[int(mm.group(1))]
                seen_layers.add(int(mm.group(1)))
        has_count = "%s.expert_count" % arch in r.fields
        print("%s -> %s: %d tensors sliced" % (os.path.basename(sh), os.path.basename(out), len(cut)), flush=True)
        if check:
            for t in r.tensors:
                if t.name in cut:
                    old_b += t.n_bytes
                    new_b += t.n_bytes * n_keep // n_expert
            continue
        if not cut and not has_count:
            if no_copy:
                print("  (unchanged: left out, --no-copy)")
            else:
                link_or_copy(sh, out)
            continue
        w = gguf.GGUFWriter(out, arch)
        for field in r.fields.values():
            if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
                continue
            if field.name == gguf.Keys.General.ALIGNMENT:
                w.add_custom_alignment(int(field.contents()))
                continue
            val_type = field.types[0]
            sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
            val = n_keep if field.name == "%s.expert_count" % arch else field.contents()
            w.add_key_value(field.name, val, val_type, sub_type=sub_type)
        datas = []
        for t in r.tensors:
            d = t.data
            if t.name in cut:
                if d.shape[0] != n_expert:
                    print("%s: outer dimension %d of the data is not the expert count" % (t.name, d.shape[0]))
                    return 1
                d = np.ascontiguousarray(d[cut[t.name]])
                old_b += t.n_bytes
                new_b += d.nbytes
            datas.append((t, d))
            w.add_tensor_info(t.name, d.shape, d.dtype, d.nbytes, t.tensor_type)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_ti_data_to_file()
        for t, d in datas:
            w.write_tensor_data(d, tensor_endianess=r.endianess)
        w.close()
    missing = sorted(set(keep) - seen_layers)
    if missing:
        print("warning: no expert tensors found for layers %s" % missing)
    print("sliced: %.2f GB -> %.2f GB%s" % (old_b / 1e9, new_b / 1e9, " (check only)" if check else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
