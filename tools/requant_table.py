"""Re-quantizes one big 2-D table of a GGUF shard (DeepSeek V4.1's Engram table, per_layer_token_embd.weight:
256 x 768,022,850 values) to another type on the GPU, streaming it in row chunks, and writes it as a shard that drops
into a split set in place of the old one (the split.* keys are taken from a template shard of that set).

usage: requant_table.py <src shard> <tensor> <type> <template shard> <out.gguf> [--rows-per-chunk N] [--limit-rows N]

  src shard   the shard holding the table. Its type may be one ggml knows (read through LIBGGML's to_float), or
              unsloth's type 43 of the UD-Q4_K_XL Engram shard: rows of 32-value blocks, each [an E8M0 scale byte |
              32 E4M3 bytes] (an MXFP8 layout; current ggml's type 43 is plain F8_E4M3 - a different layout, so the
              reader here decodes the rows itself and checks the size adds up)
  type        the new type (e.g. IQ4_NL; ISTA's and our V4.1 GGUFs keep the Engram table at IQ4_NL, 4.5 bits)
  template    a shard of the target set whose split.no / split.count / split.tensors.count the new shard keeps
              (e.g. the old Engram shard 00006-of-00006)
  --limit-rows  only the first N rows (a quick check: the output is then not a usable shard)

Environment: GGUF_PY, LIBGGML, GPUQ_DIR + GPUQ_LIB as for requant_experts.py (the GPU is device 0, or TRENCH_GPU).
No importance matrix: a lookup table's rows are read whole, every value counts the same (gpuq is given weights of 1).
"""
import ctypes
import os
import sys
import time

if os.environ.get("GGUF_PY"):
    sys.path.insert(0, os.environ["GGUF_PY"])
import numpy as np  # noqa: E402
import gguf  # noqa: E402

Q = gguf.GGMLQuantizationType


def e4m3_lut():
    lut = np.zeros(256, np.float32)
    for i in range(256):
        s = -1.0 if i & 0x80 else 1.0
        e, m = (i >> 3) & 0xF, i & 7
        if e == 15 and m == 7:
            lut[i] = 0.0          # NaN in E4M3: none expected; read as 0 (counted below)
        elif e == 0:
            lut[i] = s * (m / 8.0) * 2.0 ** -6
        else:
            lut[i] = s * (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return lut


def raw_header(path):
    """(data offset, tensors) of a shard whose tensor types gguf-py may not know: the GGUF header by hand."""
    import struct
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    ver, n_t, n_kv = struct.unpack("<IQQ", f.read(20))

    def rstr():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode()
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    align = 32

    def skip_val(t):
        if t == 8:
            rstr()
        elif t == 9:
            st, n = struct.unpack("<IQ", f.read(12))
            for _ in range(n):
                skip_val(st)
        else:
            f.read(sizes[t])
    for _ in range(n_kv):
        k = rstr()
        t = struct.unpack("<I", f.read(4))[0]
        if k == "general.alignment":
            align = struct.unpack("<I", f.read(4))[0]
        else:
            skip_val(t)
    ts = []
    for _ in range(n_t):
        name = rstr()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = struct.unpack("<%dQ" % nd, f.read(8 * nd))
        typ, off = struct.unpack("<IQ", f.read(12))
        ts.append((name, dims, typ, off))
    pos = f.tell()
    return (pos + align - 1) // align * align, ts


def main():
    a = [x for x in sys.argv[1:]]
    opt = {}
    for k in ("--rows-per-chunk", "--limit-rows"):
        if k in a:
            i = a.index(k)
            opt[k] = int(a[i + 1])
            del a[i:i + 2]
    if len(a) != 5:
        print(__doc__)
        return 2
    src, tname, tnew, tmpl, out = a
    new_type = Q[tnew.upper()]
    data_off, ts = raw_header(src)
    hit = [t for t in ts if t[0] == tname]
    if not hit:
        print("%s: no tensor %s" % (src, tname))
        return 1
    _, dims, typ, toff = hit[0]
    ne0, ne1 = int(dims[0]), int(dims[1])
    base = data_off + toff
    mm = np.memmap(src, np.uint8, "r")
    lut = e4m3_lut()
    if typ == 43:
        # unsloth's layout: ne0/32 blocks of 33 bytes a row - check it is what the file holds
        rb = ne0 // 32 * 33
        if ne0 % 32 or base + rb * ne1 > mm.size:
            print("type 43 does not fit the MXFP8 row layout (%d values, %d bytes a row)" % (ne0, rb))
            return 1
        exp2 = np.array([2.0 ** (i - 127) if i < 255 else 0.0 for i in range(256)], np.float64).astype(np.float32)

        def rows_f32(r0, r1):
            blk = np.asarray(mm[base + r0 * rb: base + r1 * rb]).reshape(r1 - r0, ne0 // 32, 33)
            return (lut[blk[:, :, 1:]] * exp2[blk[:, :, 0]][:, :, None]).reshape(r1 - r0, ne0)
        src_desc = "unsloth type 43 (MXFP8 rows: E8M0 scale + 32 x E4M3)"
    else:
        lib = ctypes.CDLL(os.environ["LIBGGML"])

        class TT(ctypes.Structure):
            _fields_ = [("type_name", ctypes.c_char_p), ("blck_size", ctypes.c_int64), ("blck_size_interleave", ctypes.c_int64),
                        ("type_size", ctypes.c_size_t), ("is_quantized", ctypes.c_bool), ("to_float", ctypes.c_void_p),
                        ("from_float_ref", ctypes.c_void_p)]
        lib.ggml_get_type_traits.restype = ctypes.POINTER(TT)
        lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
        tt = lib.ggml_get_type_traits(int(typ)).contents
        to_float = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)(tt.to_float)
        rb = ne0 // tt.blck_size * tt.type_size

        def rows_f32(r0, r1):
            raw = np.ascontiguousarray(mm[base + r0 * rb: base + r1 * rb])
            f = np.empty((r1 - r0, ne0), np.float32)
            to_float(raw.ctypes.data, f.ctypes.data, f.size)
            return f
        src_desc = "ggml type %d" % typ
    n_rows = min(ne1, opt.get("--limit-rows", ne1))
    bs, tsz = gguf.GGML_QUANT_SIZES[new_type]
    if ne0 % bs:
        print("%d values a row is not a multiple of %s's block of %d" % (ne0, new_type.name, bs))
        return 1
    orb = ne0 // bs * tsz
    print("%s: %s [%d x %d], %s -> %s (%d -> %d bytes a row), %d rows" % (os.path.basename(src), tname, ne0, ne1,
          src_desc, new_type.name, rb, orb, n_rows), flush=True)

    here = os.environ.get("GPUQ_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gpuq"))
    gq = ctypes.CDLL(os.path.join(here, os.environ.get("GPUQ_LIB", "libgpuq.so")))
    gq.gpuq_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    gq.gpuq_quantize.restype = ctypes.c_size_t
    gq.gpuq_quantize.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                 ctypes.c_void_p]
    if gq.gpuq_init(os.path.join(here, "gpuq_tables.bin").encode(), int(os.environ.get("TRENCH_GPU", "0"))) != 0:
        print("gpuq_init failed")
        return 1

    # the header: the template's keys (split.*, ...) and the one tensor
    tr = gguf.GGUFReader(tmpl)
    arch = tr.fields[gguf.Keys.General.ARCHITECTURE].contents() if gguf.Keys.General.ARCHITECTURE in tr.fields else None
    w = gguf.GGUFWriter(out, arch)
    align = 32
    for field in tr.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        if field.name == gguf.Keys.General.ALIGNMENT:
            align = int(field.contents())
            w.add_custom_alignment(align)
            continue
        vt = field.types[0]
        w.add_key_value(field.name, field.contents(), vt, sub_type=field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None)
    if len(tr.tensors) != 1:
        print("warning: the template shard holds %d tensors; the new one holds 1" % len(tr.tensors))
    w.add_tensor_info(tname, (n_rows, orb), np.dtype(np.uint8), n_rows * orb, new_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    w.fout[0].flush()
    f = w.fout[0]
    pad = (-f.tell()) % align
    f.write(b"\0" * pad)
    t0 = time.time()
    step = opt.get("--rows-per-chunk", 1 << 20)
    buf = np.empty((step, orb), np.uint8)
    ones = np.ones(ne0, np.float32)   # (gpuq takes importance weights; a table has none)
    err_n = err_d = 0.0
    for r0 in range(0, n_rows, step):
        r1 = min(n_rows, r0 + step)
        x = np.ascontiguousarray(rows_f32(r0, r1))
        ob = buf[: r1 - r0]
        got = gq.gpuq_quantize(int(new_type), x.ctypes.data, ob.ctypes.data, r1 - r0, ne0, ones.ctypes.data)
        if got != ob.nbytes:
            print("gpuq_quantize wrote %d of %d bytes" % (got, ob.nbytes))
            return 1
        f.write(ob.tobytes())
        if r0 % (step * 32) == 0:   # a sample of the error: the first 4096 rows of every 32nd chunk
            n = min(4096, r1 - r0)
            back = gguf.quants.dequantize(ob[:n], new_type).reshape(n, ne0)
            err_n += float(((back - x[:n]) ** 2).sum())
            err_d += float((x[:n] ** 2).sum())
            el = time.time() - t0
            print("  rows %d / %d  %.0f s, ~%.0f s left, rel. RMS error so far %.4f" % (r1, n_rows, el,
                  el / r1 * (n_rows - r1), (err_n / max(err_d, 1e-30)) ** 0.5), flush=True)
    f.flush()
    w.close()
    print("wrote %s: %.2f GB, %.0f s, rel. RMS error (sampled) %.4f" % (out, os.path.getsize(out) / 1e9,
          time.time() - t0, (err_n / max(err_d, 1e-30)) ** 0.5))
    return 0


if __name__ == "__main__":
    sys.exit(main())
