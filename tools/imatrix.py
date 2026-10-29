"""Makes an imatrix the same way every time: a calibration text from a public dataset at a fixed revision, then
llama.cpp's llama-imatrix with the flags that work for MoE models bigger than RAM, then the sha256 of the result.

usage:
  imatrix.py calib <out.txt> [--dataset-dir=<dir>]
      Downloads eaddario/imatrix-calibration at revision REVISION (MIT; only the *_micro files of PLAN) into
      --dataset-dir (default: ./imatrix-calibration) and writes the text: from each file, whole lines from the start
      until the section reaches its byte budget, the sections in PLAN's order. The text does not depend on the model
      (byte budgets, not token budgets): the same revision gives the same bytes on every machine.
  imatrix.py run <model shard 1> <calib.txt> <out imatrix.gguf> [--llama-bin=<dir with llama-imatrix>] [--gpu[=<device>]]
                 [--threads=N] [--chunks=N] [--ub=N] [--dry-run]
      --ub: tokens per GPU batch (default 8192; 16384 with the patched llama.cpp is the fastest measured, below).
      --llama-bin defaults to $LLAMA_BIN (trench.env: setup.sh's GPU build of llama.cpp).
      --gpu: a ROCm or CUDA build of llama.cpp; the dense tensors go to the GPU (-ngl 99), the routed experts stay
      memory-mapped on the host (-ot exps=CPU) and llama.cpp pushes them to the GPU for each 8,192-token batch (its
      op offload). Without --gpu: a CPU build. Both: -c 2048 -b 8192 -ub 8192, the whole text, --fit off, saved once at
      the end.
      GGML_HIP_ENABLE_UNIFIED_MEMORY is removed from the environment: with it, a model bigger than RAM sat on one
      thread for 20 minutes (DeepSeek V4 Flash 0731, 150 GB on 94 GB). The run goes into its own systemd scope when
      systemd-run exists, with no memory cap: if the desktop's systemd-oomd steps in, it stops this scope, not the
      session (a capped scope was killed at once).
      Measured (DeepSeek V4 Flash 0731 Q8_K_XL, 150 GB on a 94 GB machine + R9700, seconds per 1,000 tokens): CPU build
      46; ROCm build as is: stalled (no pass in 28 min: llama.cpp copies all experts from the unpinned mmap to the GPU
      every batch); ROCm --no-op-offload 22; ROCm with patches/llama-expert-upload.diff (setup.sh applies it) 19 at
      --ub=8192, 16 at --ub=16384.
      The imatrix of a GPU build and of a CPU build differ in the last digits (another summation order): note which
      one you used.
"""
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "eaddario/imatrix-calibration"
REVISION = "e87ed55dcba9d9c3a3e41539f3e728e981b1daa4"
# (file, byte budget): ~146K tokens on DeepSeek's tokenizer, 71 chunks of 2,048
PLAN = [("text_en_micro", 180_000), ("text_jp_micro", 80_000), ("text_cn_micro", 50_000),
        ("code_micro", 180_000), ("math_micro", 55_000), ("tools_micro", 60_000)]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def calib(out, ddir):
    import pyarrow.parquet as pq
    ddir.mkdir(parents=True, exist_ok=True)
    need = [f"{n}.parquet" for n, _ in PLAN if not (ddir / f"{n}.parquet").exists()]
    if need:
        from huggingface_hub import hf_hub_download
        for n in need:
            hf_hub_download(REPO, n, repo_type="dataset", revision=REVISION, local_dir=str(ddir))
    parts = []
    for name, budget in PLAN:
        text = pq.read_table(ddir / f"{name}.parquet").column("content")[0].as_py()
        lines, size = [], 0
        for line in text.split("\n"):
            if size >= budget:
                break
            lines.append(line)
            size += len(line.encode("utf-8")) + 1
        parts.append("\n".join(lines))
        print(f"{name}: {len(lines)} lines, {size} bytes")
    out.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(f"{out}: {out.stat().st_size} bytes, sha256 {sha256(out)}")
    print(f"source: {REPO} revision {REVISION}")


def run(model, text, out, flags):
    bindir = Path(flags.get("--llama-bin") or os.environ.get("LLAMA_BIN", ""))
    exe = bindir / "llama-imatrix"
    if not exe.exists():
        sys.exit(f"no {exe}")
    threads = flags.get("--threads", str(min(16, os.cpu_count() or 8)))
    ub = flags.get("--ub") or "8192"
    cmd = [str(exe), "-m", model, "-f", text, "-o", out, "-c", "2048", "-b", ub, "-ub", ub,
           "-t", threads, "-fit", "off", "-ofreq", "100000"]   # (one save at the end: each save also computes
    # per-tensor statistics on one thread, ~2.5 min for DeepSeek V4 Flash - 20 of 65 minutes went to saves every 10)
    env = {k: v for k, v in os.environ.items() if k != "GGML_HIP_ENABLE_UNIFIED_MEMORY"}
    if "--gpu" in flags:
        cmd += ["-ngl", "99", "-ot", "exps=CPU"]
        dev = flags["--gpu"]
        if dev:
            env["HIP_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"] = dev
    if flags.get("--chunks"):   # a short timing run; the real imatrix uses the whole text
        cmd += ["--chunks", flags["--chunks"]]
    if shutil.which("systemd-run"):
        cmd = ["systemd-run", "--user", "--scope", "-q"] + cmd
    print("text:", text, "sha256", sha256(text))
    print("command:", " ".join(cmd), flush=True)
    if "--dry-run" in flags:
        return
    p = subprocess.run(cmd, env=env)
    if p.returncode or not Path(out).exists():
        sys.exit(f"llama-imatrix failed (exit {p.returncode})")
    print(f"{out}: sha256 {sha256(out)}")


def main():
    flags = dict((a.split("=", 1) + [""])[:2] for a in sys.argv[1:] if a.startswith("--"))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) == 2 and args[0] == "calib":
        calib(Path(args[1]), Path(flags.get("--dataset-dir") or "imatrix-calibration"))
    elif len(args) == 4 and args[0] == "run" and (flags.get("--llama-bin") or os.environ.get("LLAMA_BIN")):
        run(args[1], args[2], args[3], flags)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
