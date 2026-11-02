# Changelog

Every release is on GitHub (Releases) with these notes. Update: `git pull`, then `./setup.sh` (it rebuilds only what
changed).

## v0.3.0 - 2026-11-03

**LDLQ: better rounding at the same types and bytes, on the GPU.** The tools behind
[zephel01/DeepSeek-V4.1-Flash-1017-LDLQ-32G96G-GGUF](https://huggingface.co/zephel01/DeepSeek-V4.1-Flash-1017-LDLQ-32G96G-GGUF),
with the steps in the README (LDLQ walkthrough).

- `tools/hessian_dump`: one llama.cpp pass over a calibration text writes every layer's expert input, the routing ids
  and weights (`--dense` also the inputs of the dense tensors). Built by `setup.sh`.
- `tools/requant_experts.py --dump=<dir>`: error-feedback rounding (GPTQ / block-LDLQ around ggml's own quantizers,
  `tools/ldlq.py`) with per-expert Hessians from the dump; `--order=act`, `--damp`, `--damp-iq1`, `--dense-dump` for
  2-D tensors (k-quants too), `--only-matched` (only the re-made tensors, for `merge_experts.py`). With `--gpu
  --dump-device=cuda` the whole loop runs on the GPU (DeepSeek V4.1, 120 expert tensors: 77 min on an RTX 5090).
  DeepSeek V4.1 at unsloth UD-IQ1_M's exact layout: KL 0.194 (plain rounding 0.404, UD-IQ1_M 0.198; 8 sequences).
- gpuq: a device-pointer API (quantize / dequantize super-blocks on the caller's stream, ggml's `to_float` of the
  i-quants on the GPU).
- CUDA checked on an RTX 5090: the self-test is byte-identical for all 9 types and the 4 GPU-read sources.
- `setup.sh --torch-gpu [--torch-index cu130|rocm7.2]`: the converter's torch version built for the GPU.
- `tools/gguf_set_split_no.py`, `recipes/deepseek-v4.1-1017-ldlq/` (the 132G / 139G rules files).

## v0.2.1 - 2026-10-30

- `tools/imatrix.py calib --size=micro|tiny|small|medium --scale=N`: a bigger calibration text from the same pinned
  dataset (`--size=small --scale=7`: 4,239,316 bytes, ~1M tokens on DeepSeek's tokenizer).
- `tools/imatrix.py stats <imatrix.gguf>`: tokens per expert (min, 1st percentile, median, how many under 1,000). The
  ~146K-token imatrix of DeepSeek V4 Flash 0731 leaves 9.4% of its experts under 1,000 tokens.
- README: the `teacher kl` example is a real run now (DeepSeek V4 Flash 0731).

## v0.2.0 - 2026-10-29

**More types on the GPU, the source read on the GPU, Hugging Face conversion, imatrix on the GPU, a quality check.**

- GPU quantizer: IQ2_XS, IQ3_XXS, IQ3_S, IQ4_XS and IQ4_NL join IQ1_S, IQ1_M, IQ2_XXS and IQ2_S - 9 types, all
  byte-identical to ggml in the self-test, 2-21x one CPU thread a type (AI PRO R9700).
- Sources in F32, F16, BF16, Q8_0 and MXFP4 are dequantized on the GPU too (`gpuq_requantize`); the result is the
  same bytes as the CPU path.
- `tools/requant_experts.py`: rules may name 2-D tensors (attention, shared experts, embeddings), so a BF16-attention
  source comes out at a published file's size type for type. A 2-D tensor whose imatrix does not fit (DeepSeek V4's
  grouped `attn_output_a`) is quantized without it when the type allows.
- `tools/rules_from_gguf.py`: a published GGUF's layout as a rules file (`--all` for every tensor, `--shift=N` one step
  smaller or bigger, `--sizes=` the size on your source).
- `tools/convert_hf.py`: safetensors (BF16 / F16 / FP8) straight to a quantized GGUF through llama.cpp's own
  converter, no full-size intermediate; new architectures work the day llama.cpp supports them.
- `tools/imatrix.py`: calibration text from eaddario/imatrix-calibration at a pinned revision (byte budgets per
  language, the same text for every model), llama-imatrix with flags that work for models bigger than RAM, `--gpu`,
  `--ub`, saved once at the end.
- `patches/llama-expert-upload.diff` (setup.sh applies it to its own copy of llama.cpp): experts reach the GPU through
  pinned buffers. Without it the ROCm imatrix of a model bigger than RAM stalled; DeepSeek V4 Flash 0731 (150 GB on a
  94 GB machine): CPU 46 s per 1,000 tokens, patched GPU 16 s.
- `tools/teacher.cpp`: reference top-32 log-probs, then KL / 99th percentile / same top token, on the CPU or the GPU
  (Qwen3.8 UD-IQ4_XS, 628 tokens: 30.4 s on the CPU, 12.5 s on the GPU).
- `setup.sh`: 8 steps (the patched GPU llama.cpp, teacher, the converter's packages); the tools read `LIBGGML` /
  `GPUQ_LIB` from `trench.env` (a system libggml of another version gave other IQ1_M bytes).

## v0.1.0 - 2026-10-29

**First release: GPU i-quants byte-identical to ggml.**

- IQ1_S, IQ1_M, IQ2_XXS, IQ2_S on the GPU (ROCm; CUDA builds), ggml's quantizers line for line.
- `tools/requant_experts.py` (re-quantize the routed experts of any GGUF ggml reads, by regex rules, with an imatrix),
  `tools/merge_experts.py` (mix two GGUF sets per layer), `setup.sh`.
- llama-quantize's fallback types for rows a type cannot hold; tensors without imatrix data quantized without it where
  the type allows.
- A copy-paste DeepSeek V4.1 Flash walkthrough and a supported-models table in the README.
