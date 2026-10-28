"""Builds a GGUF set whose expert tensors come, per layer, from one of two GGUF sets of the same model: the base set's
files with the listed layers' ffn_{gate,up,down}_exps taken byte for byte from the donor set (nothing re-quantized), so
the result is exactly the mix that a native_experts.txt row swap measured.

usage: merge_experts.py <base shard 1> <donor shard 1> <out dir> <out stem> <layers, e.g. 0,1,2,5-9> [--check]
  Each base shard becomes <out stem>-NNNNN-of-MMMMM.gguf (the base's split); a shard with no swapped tensor is copied
  as is. Donor shards that are not there are skipped (V4.1: the Engram-only shard). GGUF_PY as in requant_experts.py.
"""
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


def parse_layers(s):
    out = set()
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def main():
    check = "--check" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 5:
        print(__doc__)
        return 2
    base1, donor1, out_dir, stem, layers = args
    take = parse_layers(layers)
    pat = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
    donor = {}
    for sh in shards_of(donor1):
        if not os.path.isfile(sh):
            print("donor %s is not there: skipped" % os.path.basename(sh))
            continue
        r = gguf.GGUFReader(sh)
        for t in r.tensors:
            m = pat.match(t.name)
            if m and int(m.group(1)) in take:
                donor[t.name] = t
    want = {"blk.%d.ffn_%s_exps.weight" % (l, p) for l in take for p in ("gate", "up", "down")}
    if want - set(donor):
        print("the donor lacks: %s" % ", ".join(sorted(want - set(donor))[:6]))
        return 1
    os.makedirs(out_dir, exist_ok=True)
    total_old = total_new = 0
    for sh in shards_of(base1):
        m = re.match(r".*-(\d{5}-of-\d{5})\.gguf$", os.path.basename(sh))
        out = os.path.join(out_dir, "%s-%s.gguf" % (stem, m.group(1)) if m else "%s.gguf" % stem)
        r = gguf.GGUFReader(sh)
        swap = [t.name for t in r.tensors if t.name in donor]
        print("%s -> %s: %d tensors swapped" % (os.path.basename(sh), os.path.basename(out), len(swap)), flush=True)
        for t in r.tensors:
            if t.name in donor:
                total_old += t.n_bytes
                total_new += donor[t.name].n_bytes
        if check:
            continue
        if not swap:
            shutil.copyfile(sh, out)
            continue
        arch = None
        if gguf.Keys.General.ARCHITECTURE in r.fields:
            arch = r.fields[gguf.Keys.General.ARCHITECTURE].contents()
        w = gguf.GGUFWriter(out, arch)
        for field in r.fields.values():
            if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
                continue
            if field.name == gguf.Keys.General.ALIGNMENT:
                w.add_custom_alignment(int(field.contents()))
                continue
            val_type = field.types[0]
            sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
            w.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
        src = [donor.get(t.name, t) for t in r.tensors]
        for t in src:
            w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_ti_data_to_file()
        for t in src:
            w.write_tensor_data(t.data, tensor_endianess=r.endianess)
        w.close()
    print("experts swapped: %.2f GiB -> %.2f GiB (%s layers %s)%s" % (total_old / 2**30, total_new / 2**30, len(take),
                                                                 sorted(take), " (check only)" if check else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
