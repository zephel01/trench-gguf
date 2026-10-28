#!/usr/bin/env bash
# setup.sh - everything trench-gguf needs, in one go:
#   1. llama.cpp (cloned into third_party/llama.cpp, or --llama <your folder>)
#   2. its libggml-base.so (the CPU quantizer the GPU output must match; nothing else of llama.cpp is built)
#   3. the GPU: ROCm (hipcc + rocminfo) or CUDA (nvcc + nvidia-smi), and its architecture
#   4. gpuq/gpuq_tables.bin + gpuq/libgpuq.so (ROCm) or gpuq/libgpuq_cuda.so (CUDA)   (build.sh)
#   5. a Python venv (.venv) with numpy, huggingface_hub (the hf command) and llama.cpp's gguf-py
#   6. the self-test: random weights quantized on the CPU (ggml) and on the GPU, byte for byte
#   7. trench.env: `source trench.env` sets LIBGGML / GPUQ_LIB for tools/*.py
#
# usage: ./setup.sh [--llama <llama.cpp folder>] [--llama-build <its build folder with bin/libggml-base.so>]
#                   [--backend auto|hip|cuda] [--arch gfx1201,gfx1100 | sm_120] [--ref <git ref>] [--no-test]
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)
LLAMA="" LBUILD="" BACKEND=auto ARCH="" REF="" TEST=1
while [ $# -gt 0 ]; do
  case $1 in
    --llama) LLAMA=$2; shift 2 ;;
    --llama-build) LBUILD=$2; shift 2 ;;
    --backend) BACKEND=$2; shift 2 ;;
    --arch) ARCH=$2; shift 2 ;;
    --ref) REF=$2; shift 2 ;;
    --no-test) TEST=0; shift ;;
    -h|--help) sed -n 2,13p "$0"; exit 0 ;;
    *) echo "unknown option $1 (--help)"; exit 2 ;;
  esac
done
step() { printf '\n\033[1m[%s] %s\033[0m\n' "$1" "$2"; }
ok() { printf '  \033[32mok\033[0m %s\n' "$*"; }
fail() { printf '  \033[31mfailed:\033[0m %s\n' "$1"; [ -n "${2:-}" ] && printf '  -> %s\n' "$2"; exit 1; }

step 1/7 "llama.cpp"
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

step 2/7 "libggml-base.so"
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

step 3/7 "the GPU"
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

step 4/7 "gpuq (tables + GPU library)"
./build.sh "$LLAMA" "$LBUILD" "$BACKEND" "$ARCH" > "$ROOT/setup-build.log" 2>&1 || fail "build.sh" "see setup-build.log"
ok "gpuq/gpuq_tables.bin, gpuq/$LIB"

step 5/7 "Python (.venv)"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv || fail "python3 -m venv" "sudo apt install python3-venv"
fi
.venv/bin/pip install --quiet --upgrade pip numpy huggingface_hub "$LLAMA/gguf-py" > "$ROOT/setup-pip.log" 2>&1 \
  || fail "pip install numpy huggingface_hub gguf-py" "see setup-pip.log"
ok "$(.venv/bin/python -c 'import numpy, gguf; print("numpy", numpy.__version__, "/ gguf", getattr(gguf, "__version__", "?"))')"

step 6/7 "self-test (byte-identical to ggml)"
if [ $TEST = 1 ]; then
  LIBGGML=$LIBGGML GPUQ_LIB=$LIB .venv/bin/python gpuq/test_gpuq.py --selftest | tee "$ROOT/setup-test.log" | sed 's/^/  /'
  grep -q "^OK" "$ROOT/setup-test.log" || fail "the GPU output differs from ggml" \
    "do not use --gpu with this build; report setup-test.log at https://github.com/zephel01/trench-gguf/issues"
else
  echo "  skipped (--no-test)"
fi

step 7/7 "trench.env"
cat > trench.env <<EOF
# source trench.env - written by setup.sh $(date +%F)
export LIBGGML=$LIBGGML
export GPUQ_LIB=$LIB
export PATH=$ROOT/.venv/bin:\$PATH
EOF
ok "source trench.env, then: python tools/requant_experts.py ... --gpu"
