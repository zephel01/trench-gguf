#!/usr/bin/env python3
"""gguf_set_split_no.py <file> <n> - sets the u16 split.no of a GGUF shard in place (header only, 2 bytes).

For the llama.cpp view of DeepSeek V4.1 UD-Q4_K_XL that tools/hessian_dump reads (README, "LDLQ walkthrough"): llama.cpp
ac8e1694d does not know the type-43 tensor of unsloth's shard 8 (the Engram table) and checks every shard's split.no
against its position, so the view puts unsloth UD-IQ2_XXS's Engram shard (Q5_K, split.no 5) at position 6 and a COPY
of the Q4_K_XL shard 6 at position 8 with split.no set to 7. Only ever run it on a copy: it edits the file in place."""
import struct
import sys

path, n = sys.argv[1], int(sys.argv[2])
with open(path, "r+b") as f:
    rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))
    assert f.read(4) == b"GGUF"
    rd("I")
    rd("Q")
    nk, = rd("Q")
    sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    def skip(t):
        if t in sz:
            f.seek(sz[t], 1)
        elif t == 8:
            k, = rd("Q"); f.seek(k, 1)
        elif t == 9:
            at, = rd("I"); m, = rd("Q")
            for _ in range(m):
                skip(at)
    for _ in range(nk):
        k, = rd("Q")
        key = f.read(k).decode()
        t, = rd("I")
        if key == "split.no":
            assert t == 2, "split.no is type %d, not u16" % t
            at = f.tell()
            old, = rd("H")
            f.seek(at)
            f.write(struct.pack("<H", n))
            print("%s: split.no %d -> %d" % (path, old, n))
            sys.exit(0)
        skip(t)
sys.exit("%s: no split.no key" % path)
