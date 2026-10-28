#!/bin/sh
# build.sh <llama.cpp folder> <llama.cpp build folder> [hip|cuda] [arch]
#   writes gpuq/gpuq_tables.bin (ggml's own i-quant search tables) and gpuq/libgpuq.so (ROCm) or gpuq/libgpuq_cuda.so
#   e.g.  ./build.sh ~/llama.cpp ~/llama.cpp/build hip gfx1201      ./build.sh ~/llama.cpp ~/llama.cpp/build cuda sm_120
set -e
L=$1; B=$2; BACKEND=${3:-hip}; ARCH=$4
[ -n "$L" ] && [ -n "$B" ] || { sed -n 2,4p "$0"; exit 2; }
cd "$(dirname "$0")/gpuq"
cc -O2 -D_GNU_SOURCE -I"$L/ggml/include" -I"$L/ggml/src" gpuq_tables.c -L"$B/bin" -Wl,-rpath,"$B/bin" -lggml-base -lm \
   -o gpuq_tables
./gpuq_tables gpuq_tables.bin
if [ "$BACKEND" = cuda ]; then
  nvcc -O3 -fmad=false -prec-div=true -prec-sqrt=true -arch="${ARCH:-sm_120}" -Xcompiler -fPIC -shared -x cu gpuq.hip \
       -o libgpuq_cuda.so
else
  OA=""; for a in $(echo "${ARCH:-gfx1201}" | tr , " "); do OA="$OA --offload-arch=$a"; done   # (a comma list: several GPUs)
  hipcc -O3 -ffp-contract=off -fno-fast-math $OA -fPIC -shared gpuq.hip -o libgpuq.so
fi
echo "built: $(ls gpuq_tables.bin libgpuq*.so)"
