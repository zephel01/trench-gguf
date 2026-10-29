<div align="center">

# trench-gguf

**Quantize GGUF i-quants (IQ1_S, IQ1_M, IQ2_XXS, IQ2_S) on the GPU — byte-identical to llama-quantize.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Linux-lightgrey)
![ROCm](https://img.shields.io/badge/ROCm-checked-ED1C24?logo=amd&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-builds%20(unchecked)-76B900?logo=nvidia&logoColor=white)
![GPU](https://img.shields.io/badge/tested-AI%20PRO%20R9700-ED1C24)
![Version](https://img.shields.io/badge/version-0.2.1-555)
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

- **Byte-identical to llama.cpp** — every block of IQ1_S / IQ1_M / IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S /
  IQ4_XS / IQ4_NL matches `ggml_quantize_chunk` (checked on DeepSeek V4.1 experts and on random weights; a
  re-quantized GGUF layer is `cmp`-identical).
- **The source is read on the GPU too** — MXFP4 / Q8_0 / F16 / BF16 / F32 experts go to the GPU as they are stored
  (1/8 to 1/2 of the bytes of floats) and are dequantized there, bit for bit as ggml does.
- **5x a 32-thread CPU, 50-100x one thread** — one 384-expert tensor in about 5 s instead of 27 s.
- **Re-quantize the experts only** — `tools/requant_experts.py` takes any GGUF ggml can read (MXFP4, i-quants,
  k-quants), re-quantizes the MoE experts per a regex rules file with an imatrix, and copies everything else as is.
- **From the original weights in one go** — `tools/convert_hf.py` turns a Hugging Face model (safetensors, BF16 /
  F16 / FP8) straight into a quantized GGUF with your rules, with no full-size intermediate file. It drives llama.cpp's
  own converter, so a new architecture works as soon as llama.cpp converts it.
- **Check the quality** — `tools/teacher` writes a reference (the top-32 log-probs of a big model) and prints KL, its
  99th percentile and the same-top-token share of a quant against it, on the CPU or the GPU.
- **Copy a published layout** — `tools/rules_from_gguf.py` reads the expert types of any GGUF (unsloth's, ISTA's ...)
  into a rules file, optionally a step smaller or larger, so you can re-make that layout from the original.
- **Mix two GGUFs per layer** — `tools/merge_experts.py` builds a GGUF set whose expert layers come from one of two
  sets, byte for byte, so the file you publish is exactly the mix you measured.
- **One setup script** — `./setup.sh` fetches llama.cpp, builds `libggml-base` and a patched GPU build (llama-imatrix,
  the quality check), finds your GPU, builds the
  library and runs a self-test against ggml.
- **ROCm and CUDA from one source.**

## 🚀 Quick start

```bash
cd ~
git clone https://github.com/zephel01/trench-gguf
cd trench-gguf
./setup.sh                 # llama.cpp (2 builds), the GPU library, a Python venv, the self-test: 20-40 min the first time
source ~/trench-gguf/trench.env    # in every new terminal
```

`setup.sh` ends with the self-test. `OK: byte-identical to ggml` means the GPU output matches your llama.cpp.

| Option | What it does |
| --- | --- |
| `--llama <folder>`, `--llama-build <folder>` | use a llama.cpp you already have |
| `--llama-gpu-build <folder>` | use a ROCm/CUDA build of llama.cpp you already have (without the patch, the GPU imatrix of a model bigger than RAM stalls) |
| `--backend hip\|cuda`, `--arch gfx1201\|sm_120` | override the GPU detection |
| `--ref <git ref>` | pin llama.cpp |

### Requirements

| What | Version checked |
| --- | --- |
| Linux x86-64 | Ubuntu, glibc 2.39 |
| git, cmake, a C compiler, Python 3 with venv | Ubuntu's |
| ROCm (`hipcc`, `rocminfo`) **or** CUDA (`nvcc`, `nvidia-smi`) | HIP 7.15 (ROCm 10 / TheRock); CUDA 13.3 builds |
| a GPU | AI PRO R9700 (gfx1201); RTX 50 (sm_120) needs CUDA 12.8 or later |
| GPU memory | a few hundred MB (quantizing); the dense part of the model plus a few GB (imatrix, quality check on the GPU) |
| disk for the tool | about 3 GB (llama.cpp twice 1.2 GB, the Python venv with torch for the CPU 1.3 GB) |

### What the machine needs

The tool itself is small; the models are not. For the [walkthrough](#-walkthrough-deepseek-v41-flash-step-by-step):

| What | How much | Why |
| --- | --- | --- |
| free disk space | **about 650 GB** (+ 350 GB for step 6) | the source GGUF 501 GB + the output 141 GB |
| disk type | an NVMe SSD (a HDD works, but slowly) | 290 GB is read and 141 GB written |
| RAM | 32 GB or more (estimate) | the GGUF is memory-mapped; each CPU thread holds one expert as floats (47 MB) |
| CPU | any x86-64; more cores = faster | turning the source weights back into floats runs on all threads |
| GPU | one ROCm or CUDA GPU, 1 GB of free memory | the quantizer; checked on an AI PRO R9700 |
| network | a fast line | 501 GB: about 1 h 10 min at 1 Gbps, about 11 h at 100 Mbps |
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

**Step 1 — a work folder.** Put it on a disk with about 650 GB free. `df -h .` prints the free space (`Avail`).

```bash
source ~/trench-gguf/trench.env
mkdir -p ~/gguf-work/DeepSeek-V4.1
cd ~/gguf-work/DeepSeek-V4.1
df -h .
```

If your big disk is mounted somewhere else (for example `/mnt/data`), use that instead:
`mkdir -p /mnt/data/gguf-work/DeepSeek-V4.1 && cd /mnt/data/gguf-work/DeepSeek-V4.1`.

**Step 2 — download the source (501 GB) and the imatrix.** `hf` was installed by `setup.sh`. If the download stops,
run the same command again: it continues.

```bash
hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF --revision 85f81fb970add013033f25e94006713ca2c0aaf2 \
  --local-dir . --include "UD-Q4_K_XL/*" --include "imatrix_unsloth.gguf_file"
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

`down1/` now holds 9 shards (141 GB without shard 8) with the source's names. Shard 8 (the 203 GB Engram table, no experts) is a hard link to
the source's, so it takes no space on the same disk.

**Step 6 (optional) — mix per layer.** Keep unsloth's UD-IQ2_XXS (301 GB) and take only layers 0-3 from `down1`:

```bash
hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF --revision 85f81fb970add013033f25e94006713ca2c0aaf2 \
  --local-dir . --include "UD-IQ2_XXS/*"
python ~/trench-gguf/tools/merge_experts.py \
  UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf \
  down1/DeepSeek-V4.1-Flash-1017-UD-Q4_K_XL-00001-of-00009.gguf \
  mix DeepSeek-V4.1-Flash-1017-mix 0-3
ls -lh mix
```

Only shard 2 (layers 0-3) is written again (about 49 GB); the other shards are hard links.

**If something goes wrong**

| Message | What to do |
| --- | --- |
| `No space left on device` | free space or move the work folder to a bigger disk (Step 1) |
| `hf: command not found` / `python: ... numpy` | run `source ~/trench-gguf/trench.env` in this terminal |
| `gpuq_init failed` | the GPU is not visible: `rocminfo` or `nvidia-smi` should list it; run `./setup.sh` again |
| `MISMATCH` in `setup.sh`'s self-test | do not use `--gpu`; open an issue with `setup-test.log` |

## 🧩 Supported models

trench-gguf re-quantizes the **routed experts of MoE models**: the 3-D `ffn_gate_exps` / `ffn_up_exps` /
`ffn_down_exps` tensors (or a fused `ffn_gate_up_exps`) that llama.cpp's MoE GGUFs use. The source can be any type
ggml reads (BF16, Q8_0, MXFP4, k-quants, i-quants). Checked 2026-10-29 on the GGUFs at hand (headers of ~300 files,
plus the runs below):

| Model (llama.cpp arch) | Experts (row length ne0) | unsloth imatrix | Checked | Status |
| --- | --- | --- | --- | --- |
| DeepSeek V4.1 Flash 1017 (`deepseek41`) | gate/up 5120, down 2304, 384 experts | yes | full run, 15.5 min | ◎ |
| Qwen3.8-Flash-Next (`qwen4exp`) | gate/up 2560, **down 640**, 512 experts | yes | full run (119.5 → 40.4 GiB), loads and answers in llama.cpp; GPU match 100% | ◎ (down → IQ4_NL) |
| GLM-5.3-Flash (`glm5-next`) | gate/up 4096, down 2048, 288 experts | yes | `--check`, GPU match 100% | ○ |
| Ornith-1.5-122B, Qwen3.5-122B (`qwen35moe`) | gate/up 3072, down 1024, 256 experts | own | `--check`, GPU match 100% | ○ |
| DeepSeek V4 Flash 0731 (`deepseek4`) | gate/up 4096, down 2048, 256 experts | **no** | headers | ○ (needs an imatrix) |
| MiniMax M3.1 Flash (`minimax-m3`) | gate/up 6144, down 3072, 128 experts | — | headers | ○ |
| 35B-A3B family (`qwen35moe`: Ornith-1.0/1.5-35B, Qwen-AgentWorld, Holo4, Tiel-Coder, ...), Laguna-S-2.1 | all multiples of 256 | — | headers | ○ |
| Qwen3 MoE (`qwen3moe`: llm-jp-4.1-32b-a3b, ...) | gate/up 2048, **down 768 / 960** | — | headers | △ (down → IQ4_NL) |
| Gemma 4 26B-A4B (`gemma4`) | fused gate_up 2816, **down 704** | — | headers | △ (down → IQ4_NL; write a rule for `ffn_gate_up_exps`) |

◎ run end to end, ○ shapes fit (and `--check` / the GPU match where listed), △ partly: some tensors fall back.

**Why some tensors cannot take the asked type**

| Case | Example | Cause | What trench-gguf does |
| --- | --- | --- | --- |
| Row length not a multiple of 256 | Qwen3.8 down (640), Qwen3 MoE down (768 / 960), Gemma 4 down (704) | IQ1_\*/IQ2_\*/IQ3_\* and the k-quants pack 256 weights a block; a row that 256 does not divide cannot be stored. A limit of the formats, not of the tool | llama-quantize's fallback: IQ1_\*/IQ2_\*/IQ3_\*/IQ4_XS → IQ4_NL, Q2_K/Q3_K → Q4_0, Q4_K → Q5_0, Q5_K → Q5_1, Q6_K → Q8_0 (F16 if even that does not fit). unsloth's and ISTA's Qwen3.8 GGUFs use IQ4_NL there too |
| No imatrix data for the tensor | the NextN/MTP layer: Ornith-1.5 `blk.48`, GLM-5.3-Flash `blk.45` | the calibration run never reached that layer. IQ1_S/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S need an imatrix (llama-quantize stops: "the result will be garbage") | left as is; give it a rule of its own with a type that does not need one, e.g. `blk\.48\.ffn_\w+_exps\.weight  iq4_xs` above the others |
| Fused gate and up | Gemma 4 `ffn_gate_up_exps` | one tensor instead of two | works, but the rule must name `ffn_gate_up_exps` |

**Not supported**

| Case | Example | Cause |
| --- | --- | --- |
| Dense models | Qwen3.8-27B, Gemma 4 12B / 31B, MiniCPM5 | no expert tensors; trench-gguf only rewrites MoE experts (dense support is on the roadmap) |
| Fork-only tensor types | ROCmFP4 builds (type 101), Ternary Bonsai PQ2_0 / PTQ1_0 (types 142 / 143) | not in upstream llama.cpp; gguf-py cannot read the header and ggml cannot turn them back into floats |
| No imatrix for the model | DeepSeek V4 Flash 0731 (unsloth publishes none) | the very low-bit types need one; make it with `llama-imatrix` (or take one from the model's publisher) |

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
- `--gpu` runs IQ1_S / IQ1_M / IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS / IQ4_NL on the GPU (`--gpu=1`
  for GPU 1); other types (k-quants) stay on the CPU. Lines ending in `(gpu)` ran entirely on the GPU; `(gpu quant,
  cpu dequant)` means the source type (a k-quant or an i-quant) was dequantized on the CPU first.
- `--check` prints the plan and the sizes without writing.

### Convert a Hugging Face model (no GGUF published yet)

```bash
python tools/convert_hf.py ~/models/NewModel ~/gguf/NewModel-Q8_0.gguf --outtype=q8_0 --split-max-size=50G
python tools/imatrix.py calib calib.txt
python tools/imatrix.py run ~/gguf/NewModel-Q8_0-00001-of-00004.gguf calib.txt ~/gguf/NewModel.imatrix.gguf
python tools/convert_hf.py ~/models/NewModel ~/gguf/NewModel-IQ2.gguf --rules=newmodel.rules \
       --imatrix=~/gguf/NewModel.imatrix.gguf --gpu --split-max-size=50G
```

1. A first GGUF to make the imatrix from (Q8_0 is about half the size of BF16; when even that does not fit, give the
   experts `iq4_xs` in a rules file: it needs no imatrix).
2. The imatrix, from that file.
3. The final GGUF, from the original weights again (no rounding added twice). Then delete the first one.

A bigger calibration text and a check of how many tokens each expert got:

```bash
python tools/imatrix.py calib calib-1m.txt --size=small --scale=7     # ~1M tokens instead of ~146K
python tools/imatrix.py stats ~/gguf/NewModel.imatrix.gguf
```

`stats` prints, e.g. for the ~146K-token imatrix of DeepSeek V4 Flash 0731 (256 experts, 6 a token):

```text
imatrix-v4-0731.gguf: 129 expert tensors, 33024 experts; tokens per expert min 16, 1st pct 297, median 2692; under 1,000: 3114 (9.4%)
```

Tensors no rule names get `--outtype` (llama.cpp's converter keeps norms, routers and the like in F32). The i-quants
are quantized on the GPU with `--gpu`, the rest through ggml; the result is byte-identical to converting to BF16 and
running `requant_experts.py` (checked on a small Qwen3-MoE model). It needs torch, transformers and sentencepiece,
which `./setup.sh` installs.

### Check the quality against a reference

```bash
tools/teacher ref ~/gguf/NewModel-Q8_0-00001-of-00004.gguf ref/ seq0.ids seq1.ids --gpu
tools/teacher kl  ~/gguf/NewModel-IQ2-00001-of-00002.gguf  ref/ seq0.ids seq1.ids --gpu
```

`seqN.ids` are comma-separated token ids (`llama-tokenize -m <model> -f text.txt --ids`). `ref` writes `ref/seqN.topk`
(the top-32 log-probs of every position, the format Abyss reads as `STRATA_GLM_SCORE_TOPK`); `kl` prints, e.g. for
DeepSeek V4 Flash 0731 with unsloth UD-IQ1_S's types (8 sequences of 1,024 tokens against UD-Q8_K_XL):

```text
all: 8183 positions, nll 2.47236 | KL 0.284858 (99th pct 3.0088, max 10.8833), same top token 76.61%
```

The KL is over the reference's top 32 plus one bucket for the rest of the vocabulary, as in Abyss. `--gpu` keeps the
experts in RAM (mapped from the file) and runs the rest on the GPU; without it everything runs on the CPU. Qwen3.8
UD-IQ4_XS, 628 tokens: 30.4 s on the CPU, 12.5 s with `--gpu`.

### Copy a published layout

```bash
python tools/rules_from_gguf.py ~/models/DeepSeek-V4.1-Flash-1017-GGUF/UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf > ud-iq2xxs.rules
python tools/rules_from_gguf.py ~/models/DeepSeek-V4.1-Flash-1017-GGUF/UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf --shift=-1 > ud-iq2xxs-minus1.rules
```

One exact-name rule per expert tensor, then a summary, e.g. for unsloth's V4.1 UD-IQ2_XXS (only the headers are read,
a few seconds):

```text
# 120 tensors: IQ1_M 33, IQ1_S 20, IQ2_S 15, IQ2_XS 45, IQ2_XXS 6, Q2_K 1
# in this file: 146.11 GiB, with the shift: 129.99 GiB
```

`--shift=N` moves every expert tensor N steps on IQ1_S < IQ1_M < IQ2_XXS < IQ2_XS < IQ2_S < IQ3_XXS < IQ3_S < IQ4_XS
(other types stay). `--sizes=<source shard 1>` adds what the experts would weigh when applied to that source. Feed
the file to `requant_experts.py` as the rules file.

`--all` lists every tensor, not just the experts. Rules may name 2-D tensors (attention, shared experts, embeddings) as
well, so a source with BF16 attention (unsloth's UD-Q8_K_XL) comes out at the published file's size, type for type:

```bash
python tools/rules_from_gguf.py ~/models/DeepSeek-V4-Flash-0731-GGUF/UD-IQ1_S/DeepSeek-V4-Flash-0731-UD-IQ1_S-00001-of-00003.gguf --all | grep -v " i32$" > v4-ud-iq1s-all.rules
```

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
| all 120 expert tensors (289 GB MXFP4 → 132 GB) | about 1 h (estimate) | 15.5 min |

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

- **Types**: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL on the GPU; the k-quants (Q2_K ...
  Q6_K) run on the CPU through ggml. Sources dequantized on the GPU: F32, F16, BF16, Q8_0, MXFP4.
- **Models**: MoE experts only; see [Supported models](#-supported-models) for rows that fall back and why.
- **CUDA**: builds, but the match has not been checked on an NVIDIA GPU yet.
- **Platforms**: checked on x86-64 Linux only. On ARM, gcc fuses FMA by default, so ggml's own CPU output changes;
  on Windows, MSVC's `qsort` may order ties differently. The self-test tells.
- **llama.cpp version**: ported from bed0a8566 and re-checked on ac8e1694d. Older llama.cpp searches IQ1_M another
  way (a January 2026 libggml-base in /usr/local/lib gave 2 other IQ1_M blocks in 6,912). The tools take
  `libggml-base.so` from `LIBGGML` or `trench.env`, never from the library path. After an update, run the self-test.
- **An imatrix is required** (as in llama.cpp for the IQ1/IQ2 types). A tensor the imatrix has no data for is quantized
  without one when its type allows it, and left as is for the IQ1/IQ2 types.

## 🗺️ Roadmap

- the CUDA match on an RTX 5090
- overlap the PCIe copies with the kernels
- a faster CPU path that stays byte-identical (the codebook search in SIMD lanes), for machines without a GPU

## 🙏 Credits

- [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) — the i-quants and their quantizers, which this ports.
- [unsloth](https://huggingface.co/unsloth) — the GGUFs and imatrix files used for the checks.

## 📄 License

MIT — see [LICENSE](LICENSE). The ported ggml code keeps the ggml authors' copyright notice.
