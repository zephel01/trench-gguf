<div align="center">

# trench-gguf

**Quantize GGUF i-quants (IQ1_S, IQ1_M, IQ2_XXS, IQ2_S) on the GPU — byte-identical to llama-quantize.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)
![ROCm](https://img.shields.io/badge/ROCm-checked-ED1C24?logo=amd&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-builds%20(unchecked)-76B900?logo=nvidia&logoColor=white)
![GPU](https://img.shields.io/badge/tested-AI%20PRO%20R9700-ED1C24)
![llama.cpp](https://img.shields.io/badge/llama.cpp-ac8e1694d-4D6BFE)

**English** · [日本語](README.ja.md)

[Quick start](#-quick-start) · [Walkthrough](#-walkthrough-deepseek-v41-flash-step-by-step) · [Usage](#️-usage) · [Performance](#-performance) · [How it works](#-how-it-works) ·
[Limits](#️-limits) · [Credits](#-credits)

</div>

---

llama.cpp quantizes its i-quants with an importance matrix on the CPU only, and slowly: for every 8 weights it tries
every point of a codebook (256 to 2,048 of them) at several scales. Re-quantizing the experts of a large MoE model
(DeepSeek V4.1 Flash: 544 billion expert weights) takes about an hour on 32 threads — every time you try another
per-layer mix.

trench-gguf runs that same quantizer on the GPU. The kernels are ggml's `quantize_row_*_impl` copied line for line,
searching the tables ggml itself writes, so the output is **the same bytes `llama-quantize` writes** — no shortcut in
the search, no change in quality. A whole DeepSeek V4.1 expert set took 15.5 minutes on an AI PRO R9700.

> [!NOTE]
> trench-gguf grew out of [Abyss](https://github.com/zephel01/Abyss), an engine that runs DeepSeek V4 / V4.1 Flash on
> one consumer GPU, where it is used to search per-layer quant mixes. It does not need Abyss.

## ✨ Highlights

- **Byte-identical to llama.cpp** — every block of IQ2_XXS / IQ2_S / IQ1_S / IQ1_M matches `ggml_quantize_chunk`
  (checked on DeepSeek V4.1 experts and on random weights; a re-quantized GGUF layer is `cmp`-identical).
- **5x a 32-thread CPU, 50-100x one thread** — one 384-expert tensor in about 5 s instead of 27 s.
- **Re-quantize the experts only** — `tools/requant_experts.py` takes any GGUF ggml can read (MXFP4, i-quants,
  k-quants), re-quantizes the MoE experts per a regex rules file with an imatrix, and copies everything else as is.
- **Mix two GGUFs per layer** — `tools/merge_experts.py` builds a GGUF set whose expert layers come from one of two
  sets, byte for byte, so the file you publish is exactly the mix you measured.
- **One setup script** — `./setup.sh` fetches llama.cpp, builds only `libggml-base`, finds your GPU, builds the
  library and runs a self-test against ggml.
- **ROCm and CUDA from one source.**

## 🚀 Quick start

```bash
cd ~
git clone https://github.com/zephel01/trench-gguf
cd trench-gguf
./setup.sh                 # llama.cpp, libggml-base, the GPU library, a Python venv, the self-test
source ~/trench-gguf/trench.env    # in every new terminal
```

`setup.sh` ends with the self-test. `OK: byte-identical to ggml` means the GPU output matches your llama.cpp.

| Option | What it does |
| --- | --- |
| `--llama <folder>`, `--llama-build <folder>` | use a llama.cpp you already have |
| `--backend hip\|cuda`, `--arch gfx1201\|sm_120` | override the GPU detection |
| `--ref <git ref>` | pin llama.cpp |

### Requirements

| What | Version checked |
| --- | --- |
| Linux x86-64 | Ubuntu, glibc 2.39 |
| git, cmake, a C compiler, Python 3 with venv | Ubuntu's |
| ROCm (`hipcc`, `rocminfo`) **or** CUDA (`nvcc`, `nvidia-smi`) | HIP 7.15 (ROCm 10 / TheRock); CUDA 13.3 builds |
| a GPU | AI PRO R9700 (gfx1201); RTX 50 (sm_120) needs CUDA 12.8 or later |
| GPU memory | a few hundred MB |

### What the machine needs

The tool itself is small; the models are not. For the [walkthrough](#-walkthrough-deepseek-v41-flash-step-by-step):

| What | How much | Why |
| --- | --- | --- |
| free disk space | **about 600 GB** (+ 470 GB for step 6) | the source GGUF 467 GB + the output 132 GB |
| disk type | an NVMe SSD (a HDD works, but slowly) | 270 GB is read and 132 GB written |
| RAM | 32 GB or more (estimate) | the GGUF is memory-mapped; each CPU thread holds one expert as floats (47 MB) |
| CPU | any x86-64; more cores = faster | turning the source weights back into floats runs on all threads |
| GPU | one ROCm or CUDA GPU, 1 GB of free memory | the quantizer; checked on an AI PRO R9700 |
| network | a fast line | 467 GB: about 1 h at 1 Gbps, about 9 h at 100 Mbps |
| the tool | about 750 MB | trench-gguf + llama.cpp + the Python venv |

Tested on: Ryzen with 16 cores / 32 threads, 94 GB RAM, AI PRO R9700, a Gen4 NVMe SSD.

## 📘 Walkthrough: DeepSeek V4.1 Flash, step by step

Every command below runs as is. What you get: DeepSeek V4.1 Flash (1017) with **every expert one step below unsloth's
UD-IQ2_XXS**, made from unsloth's UD-Q4_K_XL (whose experts are the released MXFP4 weights) and unsloth's imatrix.

**Step 0 — the tool** (once; skip it if you did the [Quick start](#-quick-start))

```bash
cd ~
git clone https://github.com/zephel01/trench-gguf
cd trench-gguf
./setup.sh
```

**Step 1 — a work folder.** Put it on a disk with about 600 GB free. `df -h .` prints the free space (`Avail`).

```bash
source ~/trench-gguf/trench.env
mkdir -p ~/gguf-work/DeepSeek-V4.1
cd ~/gguf-work/DeepSeek-V4.1
df -h .
```

If your big disk is mounted somewhere else (for example `/mnt/data`), use that instead:
`mkdir -p /mnt/data/gguf-work/DeepSeek-V4.1 && cd /mnt/data/gguf-work/DeepSeek-V4.1`.

**Step 2 — download the source (467 GB) and the imatrix.** `hf` was installed by `setup.sh`. If the download stops,
run the same command again: it continues.

```bash
hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF --local-dir . --include "UD-Q4_K_XL/*" "imatrix_unsloth.gguf_file"
```

**Step 3 — the rules file.** Which type each expert tensor gets: layer 0-9 gate/up IQ1_S, and so on.

```bash
cat > rules-down1.txt <<'RULES'
blk\.[0-9]\.ffn_(gate|up)_exps\.weight              iq1_s
blk\.[0-9]\.ffn_down_exps\.weight                   iq1_m
blk\.(1[0-9]|2[0-4])\.ffn_(gate|up)_exps\.weight    iq1_m
blk\.1[0-9]\.ffn_down_exps\.weight                  iq2_xxs
blk\.2[0-4]\.ffn_down_exps\.weight                  iq2_s
blk\.(2[5-9]|3[0-9])\.ffn_(gate|up)_exps\.weight    iq2_xxs
blk\.(2[5-9]|3[0-4])\.ffn_down_exps\.weight         iq2_s
blk\.3[5-9]\.ffn_down_exps\.weight                  iq2_xxs
RULES
```

**Step 4 — a dry run** (writes nothing). The last line should read `re-quantized 268.95 GiB -> 122.61 GiB (check only)`.

```bash
python ~/trench-gguf/tools/requant_experts.py \
  UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf down1 rules-down1.txt \
  --imatrix=imatrix_unsloth.gguf_file --check
```

**Step 5 — the real run** (15.5 min on the test machine). Each expert tensor prints one line ending in `(gpu)`.

```bash
python ~/trench-gguf/tools/requant_experts.py \
  UD-Q4_K_XL/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf down1 rules-down1.txt \
  --imatrix=imatrix_unsloth.gguf_file --gpu
ls -lh down1
```

`down1/` now holds 9 shards with the source's names. Shard 8 (the 189 GB Engram table, no experts) is a hard link to
the source's, so it takes no space on the same disk.

**Step 6 (optional) — mix per layer.** Keep unsloth's UD-IQ2_XXS (280 GB) and take only layers 0-3 from `down1`:

```bash
hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF --local-dir . --include "UD-IQ2_XXS/*"
python ~/trench-gguf/tools/merge_experts.py \
  UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf \
  down1/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf \
  mix DeepSeek-V4.1-Flash-1017-mix 0-3
ls -lh mix
```

Only shard 2 (layers 0-3) is written again (about 45 GB); the other shards are hard links.

**If something goes wrong**

| Message | What to do |
| --- | --- |
| `No space left on device` | free space or move the work folder to a bigger disk (Step 1) |
| `hf: command not found` / `python: ... numpy` | run `source ~/trench-gguf/trench.env` in this terminal |
| `gpuq_init failed` | the GPU is not visible: `rocminfo` or `nvidia-smi` should list it; run `./setup.sh` again |
| `MISMATCH` in `setup.sh`'s self-test | do not use `--gpu`; open an issue with `setup-test.log` |

## 🛠️ Usage

### Re-quantize a model's experts

The general form (the [walkthrough](#-walkthrough-deepseek-v41-flash-step-by-step) fills it in):

```bash
python tools/requant_experts.py <source shard 1> <out folder> <rules file> --imatrix=<imatrix.gguf> --gpu
```

The rules file has one tensor-name regex and one type per line; the first match wins:

```text
blk\.[0-9]\.ffn_(gate|up)_exps\.weight              iq1_s
blk\.(2[5-9]|3[0-9])\.ffn_(gate|up)_exps\.weight    iq2_xxs
blk\.\d+\.ffn_down_exps\.weight                     iq2_s
```

- Start from the most precise file you have: the source's rounding error adds to the new one. unsloth's UD-Q4_K_XL
  of DeepSeek V4.1 keeps the experts as the released MXFP4, so nothing is lost before the re-quantization.
- `--gpu` runs IQ2_XXS / IQ2_S / IQ1_S / IQ1_M on the GPU (`--gpu=1` for GPU 1); other types stay on the CPU.
  Lines ending in `(gpu)` ran on the GPU.
- `--check` prints the plan and the sizes without writing.

### Mix two GGUF sets per layer

```bash
python tools/merge_experts.py <base shard 1> <donor shard 1> <out folder> <out name> 0-9,12,35-39
```

The listed layers' `ffn_{gate,up,down}_exps` come from the donor, everything else from the base, nothing is
re-quantized. Shards without a swapped tensor are copied as they are.

### Check the match yourself

```bash
python gpuq/test_gpuq.py --selftest
python gpuq/test_gpuq.py <GGUF shard> <imatrix.gguf> --experts=3 --tensor='blk\.30\.ffn_down_exps'
```

Run it again after updating llama.cpp (and `./setup.sh`, which writes the tables again), and on a new GPU.

## 📊 Performance

AI PRO R9700 (OCuLink Gen4 x4), 16-core Ryzen (32 threads), DeepSeek V4.1 Flash experts, 2026-10-29.

| | CPU (ggml) | GPU (trench-gguf) |
| --- | --- | --- |
| one expert (11.8 M weights) | 0.5-2.5 s, one thread | 0.010-0.028 s |
| one tensor (384 experts) | about 27 s, 32 threads | about 5 s |
| one layer (gate / up / down) into a GGUF | 1 min 54 s | 29 s, `cmp`-identical |
| all 120 expert tensors (269 GiB MXFP4 → 123 GiB) | about 1 h (estimate) | 15.5 min |

About half of the GPU's time is the float weights crossing PCIe.

## 🧠 How it works

- **The same code**: each GPU thread quantizes one 256-weight super-block with ggml's `quantize_row_*_impl`, ported
  line for line.
- **The same tables**: `gpuq/gpuq_tables.c` includes `ggml-quants.c` itself and writes the grid, map and neighbour
  lists that `iq2xs_init_impl` builds.
- **The same arithmetic**: llama.cpp's `libggml-base` is built without FMA, so the GPU build turns FMA contraction off
  (`-ffp-contract=off`, nvcc `-fmad=false`) and uses correctly rounded division and square root.
- **The same order**: IQ1_S / IQ1_M sort weights before they search; glibc 2.39's `qsort` is a stable merge sort at
  these sizes, so the GPU uses a stable sort and ties come out in the same order.
- **The same fp16 rounding**: ggml's `ggml_compute_fp32_to_fp16`, copied as is.

## ⚠️ Limits

- **Types**: IQ2_XXS, IQ2_S, IQ1_S, IQ1_M. IQ2_XS and IQ3_XXS are next; other types run on the CPU.
- **CUDA**: builds, but the match has not been checked on an NVIDIA GPU yet.
- **Platforms**: checked on x86-64 Linux only. On ARM, gcc fuses FMA by default, so ggml's own CPU output changes;
  on Windows, MSVC's `qsort` may order ties differently. The self-test tells.
- **llama.cpp version**: ported from bed0a8566 and re-checked on ac8e1694d. Older llama.cpp searches IQ1_M another
  way. After an update, run the self-test.
- **An imatrix is required** (as in llama.cpp for these types).

## 🗺️ Roadmap

- IQ2_XS and IQ3_XXS on the GPU
- the CUDA match on an RTX 5090
- overlap the PCIe copies with the kernels
- a faster CPU path that stays byte-identical (the codebook search in SIMD lanes), for machines without a GPU

## 🙏 Credits

- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) — the i-quants and their quantizers, which this ports.
- [unsloth](https://huggingface.co/unsloth) — the GGUFs and imatrix files used for the checks.

## 📄 License

MIT — see [LICENSE](LICENSE). The ported ggml code keeps the ggml authors' copyright notice.
