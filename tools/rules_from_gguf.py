"""Reads the expert type layout of an existing (split) GGUF and writes it as a rules file for requant_experts.py, so a
published quant's layout (unsloth's, ISTA's ...) can be re-made from the original with your own imatrix, or shifted a
step up or down to hit another size.

usage: rules_from_gguf.py <shard 1 of a GGUF> [--shift=N] [--all] [--sizes=<shard 1 of the source you will quantize>]
  Prints one "<exact tensor name regex> <type>" line per routed-expert tensor (ffn_*_exps; --all: every quantized
  tensor), in file order, then a size summary (# comments). Only the header of each shard is read.
  --shift=N moves every expert tensor N steps along the ladder IQ1_S < IQ1_M < IQ2_XXS < IQ2_XS < IQ2_S < IQ3_XXS <
  IQ3_S < IQ4_XS (clamped at both ends; other types stay): -1 for a smaller file, +1 for a better one.
  --sizes=<source>: also prints what the experts would weigh when the rules are applied to that source (types whose
  block does not fit a row fall back as in requant_experts.py, which this count ignores).

example: python tools/rules_from_gguf.py ~/models/V4.1/UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf > ud-iq2xxs.rules
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from requant_experts import gguf, Q  # noqa: E402  (the same gguf-py as the other tools)

LADDER = [Q.IQ1_S, Q.IQ1_M, Q.IQ2_XXS, Q.IQ2_XS, Q.IQ2_S, Q.IQ3_XXS, Q.IQ3_S, Q.IQ4_XS]
EXPERT = re.compile(r"ffn_(gate|up|down|gate_up)_exps")


def shards(first):
    m = re.match(r"(.*)-00001-of-(\d{5})\.gguf$", first)
    return sorted(glob.glob("%s-*-of-%s.gguf" % (m.group(1), m.group(2)))) if m else [first]


def tensors(first):
    for path in shards(first):
        try:
            r = gguf.GGUFReader(path)
        except Exception as e:   # (DeepSeek V4.1's Engram-only shard: nothing to read here)
            print("# skipped %s (%s)" % (os.path.basename(path), type(e).__name__))
            continue
        for t in r.tensors:
            yield t


def nbytes(t, typ):
    bs, ts = gguf.GGML_QUANT_SIZES[typ]
    n = 1
    for d in t.shape:
        n *= int(d)
    return n // bs * ts


def main():
    flags = dict((a.split("=", 1) + [""])[:2] for a in sys.argv[1:] if a.startswith("--"))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit(__doc__)
    shift = int(flags.get("--shift") or 0)
    rules, before, after = {}, 0, 0
    print("# layout of %s%s" % (os.path.basename(args[0]), " shifted %+d" % shift if shift else ""))
    for t in tensors(args[0]):
        typ = Q(t.tensor_type)
        expert = bool(EXPERT.search(t.name))
        if not expert and ("--all" not in flags or typ in (Q.F32, Q.F16, Q.BF16)):
            continue
        new = typ
        if expert and shift and typ in LADDER:
            new = LADDER[max(0, min(len(LADDER) - 1, LADDER.index(typ) + shift))]
        rules[t.name] = new
        before += nbytes(t, typ)
        after += nbytes(t, new)
        print("%-44s %s" % ("^" + re.escape(t.name).replace("\\_", "_") + "$", new.name.lower()))
    counts = {}
    for typ in rules.values():
        counts[typ.name] = counts.get(typ.name, 0) + 1
    print("# %d tensors: %s" % (len(rules), ", ".join("%s %d" % kv for kv in sorted(counts.items()))))
    print("# in this file: %.2f GiB%s" % (before / 2**30, ", with the shift: %.2f GiB" % (after / 2**30) if shift else ""))
    if flags.get("--sizes"):
        total = sum(nbytes(t, rules[t.name]) for t in tensors(flags["--sizes"]) if t.name in rules)
        print("# applied to %s: %.2f GiB" % (os.path.basename(flags["--sizes"]), total / 2**30))


if __name__ == "__main__":
    main()
