#!/usr/bin/env bash
# setup.sh - everything trench-gguf needs, in one go:
#   1. llama.cpp (cloned into third_party/llama.cpp, or --llama <your folder>)
#   2. its libggml-base.so (the CPU quantizer the GPU output must match; nothing else of llama.cpp is built)
#   3. the GPU: ROCm (hipcc + rocminfo) or CUDA (nvcc + nvidia-smi), and its architecture
#   4. gpuq/gpuq_tables.bin + gpuq/libgpuq.so (ROCm) or gpuq/libgpuq_cuda.so (CUDA)   (build.sh)
#   5. llama.cpp on the GPU (third_party/llama.cpp-gpu, the same commit + patches/llama-expert-upload.diff: libllama +
#      llama-imatrix) and tools/teacher (quality check) built on it
#   6. a Python venv (.venv) with numpy, pyarrow (tools/imatrix.py), huggingface_hub (the hf command), llama.cpp's gguf-py
#      and its converter's packages (torch for the CPU, transformers, sentencepiece: tools/convert_hf.py)
#   7. the self-test: random weights quantized on the CPU (ggml) and on the GPU, byte for byte
#   8. trench.env: `source trench.env` sets LIBGGML / GPUQ_LIB / LLAMA_DIR / LLAMA_BIN / TEACHER for tools/*
#
# usage: ./setup.sh [--llama <llama.cpp folder>] [--llama-build <its build folder with bin/libggml-base.so>]
#                   [--llama-gpu-build <a ROCm/CUDA build folder with bin/llama-imatrix, to reuse>]
#                   [--backend auto|hip|cuda] [--arch gfx1201,gfx1100 | sm_120] [--ref <git ref>] [--no-test]
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)
LLAMA="" LBUILD="" GBUILD="" BACKEND=auto ARCH="" REF="" TEST=1
while [ $# -gt 0 ]; do
  case $1 in
    --llama) LLAMA=$2; shift 2 ;;
    --llama-build) LBUILD=$2; shift 2 ;;
    --llama-gpu-build) GBUILD=$2; shift 2 ;;
    --backend) BACKEND=$2; shift 2 ;;
    --arch) ARCH=$2; shift 2 ;;
    --ref) REF=$2; shift 2 ;;
    --no-test) TEST=0; shift ;;
    -h|--help) sed -n 2,15p "$0"; exit 0 ;;
    *) echo "unknown option $1 (--help)"; exit 2 ;;
  esac
done
step() { printf '\n\033[1m[%s] %s\033[0m\n' "$1" "$2"; }
ok() { printf '  \033[32mok\033[0m %s\n' "$*"; }
fail() { printf '  \033[31mfailed:\033[0m %s\n' "$1"; [ -n "${2:-}" ] && printf '  -> %s\n' "$2"; exit 1; }

step 1/8 "llama.cpp"
if [ -z "$LLAMA" ]; then
  LLAMA=$ROOT/third_party/llama.cpp
  if [ ! -d "$LLAMA/.git" ]; then
    command -v git >/dev/null || fail "git is not installed" "sudo apt install git"
    git clone --quiet https://github.com/ggml-org/llama.cpp "$LLAMA"
  fi
  [ -n "$REF" ] && git -C "$LLAMA" checkout --quiet "$REF"
fi
[ -f "$LLAMA/ggml/src/ggml-quants.c" ] || fail "$LLAMA has no ggml/src/ggml-quants.c" "pass --llama <a llama.cpp checkout>"
ok "$LLAMA ($(git -C "$LLAMA" log -1 --format='%h %cd' --date=short 2>/dev/null || echo 'not a git checkout'))"

step 2/8 "libggml-base.so"
if [ -z "$LBUILD" ]; then
  LBUILD=$LLAMA/build-trench
  if [ ! -f "$LBUILD/bin/libggml-base.so" ]; then
    command -v cmake >/dev/null || fail "cmake is not installed" "sudo apt install cmake build-essential"
    cmake -S "$LLAMA" -B "$LBUILD" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF \
          -DGGML_CUDA=OFF -DGGML_HIP=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
          -DLLAMA_BUILD_SERVER=OFF > "$ROOT/setup-cmake.log" 2>&1 || fail "cmake configure" "see setup-cmake.log"
    cmake --build "$LBUILD" --target ggml-base -j"$(nproc)" >> "$ROOT/setup-cmake.log" 2>&1 \
      || fail "building ggml-base" "see setup-cmake.log"
  fi
fi
LIBGGML=$LBUILD/bin/libggml-base.so
[ -f "$LIBGGML" ] || fail "$LIBGGML not found" "pass --llama-build <the folder holding bin/libggml-base.so>"
if command -v objdump >/dev/null && objdump -d "$LIBGGML" --no-show-raw-insn 2>/dev/null | grep -q vfmadd; then
  echo "  warning: $LIBGGML uses FMA instructions - the GPU output may differ from it (the self-test will tell)"
fi
ok "$LIBGGML"

step 3/8 "the GPU"
HIPCC=$(command -v hipcc || ls /opt/rocm/bin/hipcc 2>/dev/null || true)
NVCC=$(command -v nvcc || ls /usr/local/cuda/bin/nvcc 2>/dev/null || true)
hip_arch() { (command -v rocminfo >/dev/null && rocminfo || /opt/rocm/bin/rocminfo) 2>/dev/null |
             grep -o -E 'gfx[0-9a-z-]+' | grep -v generic | sort -u | paste -sd, -; }
cuda_arch() { nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sed 's/\.//; s/^/sm_/' | sort -u | head -1; }
if [ "$BACKEND" = auto ]; then
  if [ -n "$HIPCC" ] && [ -n "$(hip_arch)" ]; then BACKEND=hip
  elif [ -n "$NVCC" ] && [ -n "$(cuda_arch)" ]; then BACKEND=cuda
  else fail "no ROCm or CUDA GPU found" "install ROCm (hipcc, rocminfo) or the CUDA toolkit (nvcc) and the driver, or pass --backend and --arch"
  fi
fi
case $BACKEND in
  hip)  [ -n "$HIPCC" ] || fail "hipcc not found" "install ROCm"; [ -n "$ARCH" ] || ARCH=$(hip_arch)
        export PATH=$(dirname "$HIPCC"):$PATH; LIB=libgpuq.so ;;
  cuda) [ -n "$NVCC" ] || fail "nvcc not found" "install the CUDA toolkit (12.8 or later for RTX 50)"
        [ -n "$ARCH" ] || ARCH=$(cuda_arch); export PATH=$(dirname "$NVCC"):$PATH; LIB=libgpuq_cuda.so ;;
  *) fail "--backend must be auto, hip or cuda" ;;
esac
[ -n "$ARCH" ] || fail "could not tell the GPU's architecture" "pass --arch (e.g. gfx1201 or sm_120)"
ok "$BACKEND, $ARCH"

step 4/8 "gpuq (tables + GPU library)"
./build.sh "$LLAMA" "$LBUILD" "$BACKEND" "$ARCH" > "$ROOT/setup-build.log" 2>&1 || fail "build.sh" "see setup-build.log"
ok "gpuq/gpuq_tables.bin, gpuq/$LIB"

step 5/8 "llama.cpp on the GPU (llama-imatrix, tools/teacher)"
if [ -z "$GBUILD" ]; then
  # a copy of the same llama.cpp commit with patches/llama-expert-upload.diff (experts reach the GPU through pinned
  # memory: without it a model bigger than RAM stalls on the GPU); your checkout is not touched
  GSRC=$ROOT/third_party/llama.cpp-gpu
  if [ ! -d "$GSRC" ]; then
    if git -C "$LLAMA" rev-parse HEAD >/dev/null 2>&1; then
      git -C "$LLAMA" worktree add --quiet --detach "$GSRC" HEAD || fail "git worktree add $GSRC"
    else
      cp -a "$LLAMA" "$GSRC"
    fi
    if (cd "$GSRC" && git apply --check "$ROOT/patches/llama-expert-upload.diff" 2>/dev/null); then
      (cd "$GSRC" && git apply "$ROOT/patches/llama-expert-upload.diff") && echo "  patched: patches/llama-expert-upload.diff"
    else
      echo "  warning: patches/llama-expert-upload.diff does not apply to this llama.cpp - the GPU imatrix may stall"
    fi
  fi
  GBUILD=$GSRC/build-gpu
  if [ ! -x "$GBUILD/bin/llama-imatrix" ]; then
    echo "  building (10-30 minutes the first time; log: setup-cmake-gpu.log)"
    if [ "$BACKEND" = hip ]; then
      GPU_FLAGS=(-DGGML_HIP=ON -DGPU_TARGETS="${ARCH//,/;}" -DCMAKE_HIP_COMPILER="$(hipconfig -l 2>/dev/null)/clang")
    else
      GPU_FLAGS=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${ARCH#sm_}")
    fi
    cmake -S "$GSRC" -B "$GBUILD" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=ON "${GPU_FLAGS[@]}" \
          -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
          > "$ROOT/setup-cmake-gpu.log" 2>&1 || fail "cmake configure (GPU)" "see setup-cmake-gpu.log"
    cmake --build "$GBUILD" --target llama-imatrix -j"$(nproc)" >> "$ROOT/setup-cmake-gpu.log" 2>&1 \
      || fail "building llama-imatrix" "see setup-cmake-gpu.log"
  fi
fi
[ -x "$GBUILD/bin/llama-imatrix" ] || fail "$GBUILD/bin/llama-imatrix not found" "pass --llama-gpu-build <a ROCm/CUDA build folder>"
g++ -O2 -std=c++17 tools/teacher.cpp -I"$LLAMA/include" -I"$LLAMA/ggml/include" -L"$GBUILD/bin" -lllama -lggml -lggml-base \
    -Wl,-rpath,"$GBUILD/bin" -o tools/teacher 2>> "$ROOT/setup-cmake-gpu.log" || fail "building tools/teacher" "see setup-cmake-gpu.log"
ok "$GBUILD/bin/llama-imatrix, tools/teacher"

step 6/8 "Python (.venv)"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv || fail "python3 -m venv" "sudo apt install python3-venv"
fi
.venv/bin/pip install --quiet --upgrade pip > "$ROOT/setup-pip.log" 2>&1
.venv/bin/pip install --quiet -r "$LLAMA/requirements/requirements-convert_hf_to_gguf.txt" >> "$ROOT/setup-pip.log" 2>&1 \
  || fail "pip install (llama.cpp's converter packages: torch, transformers ...)" "see setup-pip.log"
# (huggingface_hub comes with transformers, in the version it supports - upgrading it breaks the converter)
.venv/bin/pip install --quiet pyarrow "$LLAMA/gguf-py" >> "$ROOT/setup-pip.log" 2>&1 \
  || fail "pip install pyarrow gguf-py" "see setup-pip.log"
ok "$(.venv/bin/python -c 'import numpy, torch, transformers, huggingface_hub; print("numpy", numpy.__version__, "/ torch", torch.__version__, "/ transformers", transformers.__version__, "/ huggingface_hub", huggingface_hub.__version__)')" \
  || fail "the converter's packages do not import" "see setup-pip.log"

step 7/8 "self-test (byte-identical to ggml)"
if [ $TEST = 1 ]; then
  LIBGGML=$LIBGGML GPUQ_LIB=$LIB .venv/bin/python gpuq/test_gpuq.py --selftest | tee "$ROOT/setup-test.log" | sed 's/^/  /'
  grep -q "^OK" "$ROOT/setup-test.log" || fail "the GPU output differs from ggml" \
    "do not use --gpu with this build; report setup-test.log at https://github.com/zephel01/trench-gguf/issues"
else
  echo "  skipped (--no-test)"
fi

step 8/8 "trench.env"
cat > trench.env <<EOF
# source trench.env - written by setup.sh $(date +%F)
export LIBGGML=$LIBGGML
export GPUQ_LIB=$LIB
export LLAMA_DIR=$LLAMA
export LLAMA_BIN=$GBUILD/bin
export TEACHER=$ROOT/tools/teacher
export PATH=$ROOT/.venv/bin:\$PATH
EOF
ok "source trench.env, then: python tools/requant_experts.py ... --gpu"
