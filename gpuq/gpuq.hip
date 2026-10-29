// gpuq/gpuq.hip (trench-gguf) - ggml's imatrix i-quant quantizers (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
// IQ3_S, IQ4_XS, IQ4_NL) on the GPU, one thread a
// 256-weight super-block. The code is ggml-quants.c's quantize_row_*_impl (llama.cpp bed0a8566) line for line, with
// the same tables (gpuq_tables.bin, written by gpuq_tables from ggml's own iq2xs_init_impl), so the output is meant
// to be byte-identical to ggml_quantize_chunk: ggml-base is built without FMA (no -march=native), so this is built
// with -ffp-contract=off and IEEE division / sqrt; qsort (glibc's merge sort, stable) is a stable insertion sort.
//   (./build.sh does both steps.)
//   ROCm: hipcc -O3 -ffp-contract=off --offload-arch=gfx1201 -fPIC -shared gpuq.hip -o libgpuq.so
//   CUDA: nvcc -O3 -fmad=false -prec-div=true -prec-sqrt=true -arch=sm_120 -Xcompiler -fPIC -shared -x cu gpuq.hip
//         -o libgpuq_cuda.so      (-fmad=false: nvcc contracts a*b+c into FMA by default, ggml-base does not)
// C API (ctypes): gpuq_init(tables, device) -> 0; gpuq_quantize(type, src f32, dst, nrow, n_per_row, imatrix) ->
// bytes written (0 on error); host pointers; the imatrix holds n_per_row weights (as ggml_quantize_chunk's).
#if defined(__CUDACC__) && !defined(__HIP__)
#include <cuda_runtime.h>
#define hipSuccess cudaSuccess
#define hipSetDevice cudaSetDevice
#define hipMalloc cudaMalloc
#define hipFree cudaFree
#define hipMemcpy cudaMemcpy
#define hipMemcpyHostToDevice cudaMemcpyHostToDevice
#define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define hipGetLastError cudaGetLastError
#define hipDeviceSynchronize cudaDeviceSynchronize
#else
#include <hip/hip_runtime.h>
#endif
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cfloat>
#include <vector>

#define QK_K 256
#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define QK4_NL 32
#define IQ1S_DELTA 0.125f
#define IQ1M_DELTA 0.125f
#define IQ1S_BLOCK_SIZE 32
#define IQ1M_BLOCK_SIZE 16
#define NGRID_IQ1S 2048
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

enum { T_IQ2_XXS = 16, T_IQ2_XS = 17, T_IQ3_XXS = 18, T_IQ1_S = 19, T_IQ4_NL = 20, T_IQ3_S = 21, T_IQ2_S = 22,
       T_IQ4_XS = 23, T_IQ1_M = 29 };

typedef uint16_t ggml_half;
struct block_iq2_xxs { ggml_half d; uint16_t qs[QK_K/8]; };
struct block_iq2_s   { ggml_half d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32]; uint8_t scales[QK_K/32]; };
struct block_iq1_s   { ggml_half d; uint8_t qs[QK_K/8]; uint16_t qh[QK_K/32]; };
struct block_iq1_m   { uint8_t qs[QK_K/8]; uint8_t qh[QK_K/16]; uint8_t scales[QK_K/32]; };
struct block_iq2_xs  { ggml_half d; uint16_t qs[QK_K/8]; uint8_t scales[QK_K/32]; };
struct block_iq3_xxs { ggml_half d; uint8_t qs[3*QK_K/8]; };
struct block_iq3_s   { ggml_half d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32]; uint8_t signs[QK_K/8]; uint8_t scales[QK_K/64]; };
struct block_iq4_nl  { ggml_half d; uint8_t qs[QK4_NL/2]; };
struct block_iq4_xs  { ggml_half d; uint16_t scales_h; uint8_t scales_l[QK_K/64]; uint8_t qs[QK_K/2]; };
static_assert(sizeof(block_iq2_xxs) == 66, "iq2_xxs");
static_assert(sizeof(block_iq2_s) == 82, "iq2_s");
static_assert(sizeof(block_iq1_s) == 50, "iq1_s");
static_assert(sizeof(block_iq1_m) == 56, "iq1_m");
static_assert(sizeof(block_iq2_xs) == 74, "iq2_xs");
static_assert(sizeof(block_iq3_xxs) == 98, "iq3_xxs");
static_assert(sizeof(block_iq3_s) == 110, "iq3_s");
static_assert(sizeof(block_iq4_nl) == 18, "iq4_nl");
static_assert(sizeof(block_iq4_xs) == 136, "iq4_xs");

__constant__ int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

struct Tab { const uint64_t * grid; const int * map; const uint16_t * neigh; };

#define QD __device__ static inline
#if defined(__CUDACC__) && !defined(__HIP__)
#define GPUQ_TRAP() __trap()
#else
#define GPUQ_TRAP() __builtin_trap()
#endif

QD uint32_t fp32_to_bits(float f) { return __float_as_uint(f); }
QD float fp32_from_bits(uint32_t w) { return __uint_as_float(w); }

QD ggml_half fp32_to_fp16(float f) {           // ggml_compute_fp32_to_fp16 (ggml-impl.h), verbatim
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;
    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) bias = UINT32_C(0x71000000);
    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

QD int nearest_int(float fval) {
    float val = fval + 12582912.f;
    int i = __float_as_int(val);
    return (i & 0x007fffff) - 0x00400000;
}

QD float make_qp_quants(int n, int nmax, const float * x, uint8_t * L, const float * quant_weights) {
    float max = 0;
    for (int i = 0; i < n; ++i) max = MAX(max, x[i]);
    if (max < GROUP_MAX_EPS) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.f;
    }
    float iscale = nmax / max;
    for (int i = 0; i < n; ++i) L[i] = nearest_int(iscale * x[i]);
    float scale = 1/iscale;
    float best_mse = 0;
    for (int i = 0; i < n; ++i) {
        float diff = x[i] - scale*L[i];
        float w = quant_weights[i];
        best_mse += w*diff*diff;
    }
    for (int is = -4; is <= 4; ++is) {
        if (is == 0) continue;
        float iscale_is = (0.1f*is + nmax)/max;
        float scale_is = 1/iscale_is;
        float mse = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale_is*x[i]);
            l = MIN(nmax, l);
            float diff = x[i] - scale_is*l;
            float w = quant_weights[i];
            mse += w*diff*diff;
        }
        if (mse < best_mse) { best_mse = mse; iscale = iscale_is; }
    }
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = MIN(nmax, l);
        L[i] = l;
        float w = quant_weights[i];
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            float w = quant_weights[i];
            float slx = sumlx - w*x[i]*L[i];
            float sl2 = suml2 - w*L[i]*L[i];
            if (slx > 0 && sl2 > 0) {
                int new_l = nearest_int(x[i] * sl2 / slx);
                new_l = MIN(nmax, new_l);
                if (new_l != L[i]) {
                    slx += w*x[i]*new_l;
                    sl2 += w*new_l*new_l;
                    if (slx*slx*suml2 > sumlx*sumlx*sl2) {
                        L[i] = new_l; sumlx = slx; suml2 = sl2;
                        ++n_changed;
                    }
                }
            }
        }
        if (!n_changed) break;
    }
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

QD int iq2_find_best_neighbour(const uint16_t * neighbours, const uint64_t * grid, const float * xval,
                               const float * weight, float scale, int8_t * L) {
    int num_neighbors = neighbours[0];
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 8; ++i) {
            float q = pg[i];
            float diff = scale*q - xval[i];
            d2 += weight[i]*diff*diff;
        }
        if (d2 < best_d2) { best_d2 = d2; grid_index = neighbours[j]; }
    }
    if (grid_index < 0) GPUQ_TRAP();
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 8; ++i) L[i] = (pg[i] - 1)/2;
    return grid_index;
}

QD int iq1_find_best_neighbour2(const uint16_t * neighbours, const uint64_t * grid, const float * xval,
                                const float * weight, float scale, const float * xg, int8_t * L, int ngrid) {
    int num_neighbors = neighbours[0];
    float best_score = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 8; ++i) {
            float q = xg[(pg[i] - 1)/2];
            float w = weight[i];
            float diff = scale*q - xval[i];
            d2 += w*diff*diff;
        }
        if (d2 < best_score) { best_score = d2; grid_index = neighbours[j]; }
    }
    if (grid_index < 0) {
        for (int i = 0; i < ngrid; ++i) {
            const int8_t * grid_i = (const int8_t *)(grid + i);
            float d2 = 0;
            for (int j = 0; j < 8; ++j) {
                float w = weight[j];
                float q = xg[(grid_i[j] - 1)/2];
                float diff = scale*q - xval[i];      // (ggml's xval[i], kept as is)
                d2 += w*diff*diff;
            }
            if (d2 < best_score) { best_score = d2; grid_index = i; }
        }
    }
    if (grid_index < 0) GPUQ_TRAP();
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 8; ++i) L[i] = (pg[i] - 1)/2;
    return grid_index;
}

// qsort(pairs, n, 2*sizeof(float), iq1_sort_helper): glibc 2.39's qsort is a stable merge sort for these sizes, so
// any stable sort by the key gives the same order (ties keep their index order)
QD void stable_sort(float * key, int * idx, int n) {
    for (int i = 1; i < n; ++i) {
        const float k = key[i]; const int v = idx[i];
        int j = i - 1;
        while (j >= 0 && key[j] > k) { key[j+1] = key[j]; idx[j+1] = idx[j]; --j; }
        key[j+1] = k; idx[j+1] = v;
    }
}

// ---------------------------------------------------------------- IQ2_XXS (quantize_row_iq2_xxs_impl, one block)
QD void q_iq2_xxs(const float * xbl, const float * qwbl, block_iq2_xxs * y, Tab t) {
    const uint64_t * kgrid_q2xs = t.grid;
    const int * kmap_q2xs = t.map;
    const uint16_t * kneighbors_q2xs = t.neigh;
    const int kMaxQ = 3;
    float scales[QK_K/32];
    float weight[32];
    float xval[32];
    int8_t L[32];
    int8_t Laux[32];
    float waux[32];
    uint8_t block_signs[4];
    uint32_t q2[2*(QK_K/32)];

    y->d = fp32_to_fp16(0.f);
    for (int i = 0; i < 2*(QK_K/32); ++i) q2[i] = 0;
    float max_scale = 0;
    float sumx2 = 0;
    for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
    float sigma2 = sumx2/QK_K;

    for (int ib = 0; ib < QK_K/32; ++ib) {
        const float * xb = xbl + 32*ib;
        const float * qw = qwbl + 32*ib;
        for (int i = 0; i < 32; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
        for (int i = 0; i < 32; ++i) waux[i] = sqrtf(weight[i]);
        for (int k = 0; k < 4; ++k) {
            int nflip = 0;
            uint8_t s = 0;
            for (int i = 0; i < 8; ++i) {
                if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                else { xval[8*k + i] = -xb[8*k + i]; ++nflip; s |= (1 << i); }
            }
            if (nflip%2) {
                int imin = 0; float min = weight[8*k+imin]*xb[8*k+imin]*xb[8*k+imin];
                for (int i = 1; i < 8; ++i) {
                    float ax = weight[8*k+i]*xb[8*k+i]*xb[8*k+i];
                    if (ax < min) { min = ax; imin = i; }
                }
                xval[8*k+imin] = -xval[8*k+imin];
                s ^= (1 << imin);
            }
            block_signs[k] = s & 127;
        }
        float max = xval[0];
        for (int i = 1; i < 32; ++i) max = MAX(max, xval[i]);
        if (max < GROUP_MAX_EPS) {
            scales[ib] = 0;
            for (int i = 0; i < 32; ++i) L[i] = 0;
            continue;
        }
        float scale = make_qp_quants(32, kMaxQ+1, xval, (uint8_t*)L, weight);
        float eff_max = scale*kMaxQ;
        if (eff_max <= 0) {
            scales[ib] = 0;
            for (int i = 0; i < 32; ++i) L[i] = 0;
            continue;
        }
        float best = 0;
        for (int is = -6; is <= 6; ++is) {
            float id = (2*kMaxQ-1+is*0.1f)/eff_max;
            float this_scale = 1/id;
            for (int k = 0; k < 4; ++k) {
                for (int i = 0; i < 8; ++i) {
                    int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                    Laux[8*k+i] = MAX(0, MIN(kMaxQ-1, l));
                }
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= (Laux[8*k+i] << 2*i);
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                    grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, this_scale, Laux + 8*k);
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 32; ++i) {
                float w = weight[i];
                float q = 2*Laux[i] + 1;
                sumqx += w*xval[i]*q;
                sumq2 += w*q*q;
            }
            if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                scale = sumqx/sumq2; best = scale*sumqx;
                for (int i = 0; i < 32; ++i) L[i] = Laux[i];
            }
        }
        if (scale > 0) {
            float id = 1/scale;
            for (int k = 0; k < 4; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) {
                    int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                    l = MAX(0, MIN(kMaxQ-1, l));
                    u |= (l << 2*i);
                }
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                    grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, scale, L + 8*k);
                }
                const int8_t * pg = (const int8_t *)(kgrid_q2xs + grid_index);
                for (int i = 0; i < 8; ++i) L[8*k+i] = (pg[i] - 1)/2;
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 32; ++i) {
                float w = weight[i];
                float q = 2*L[i] + 1;
                sumqx += w*xval[i]*q;
                sumq2 += w*q*q;
            }
            if (sumq2 > 0) scale = sumqx/sumq2;
        }
        if (scale < 0) {
            scale = -scale;
            for (int k = 0; k < 4; ++k) block_signs[k] = (~block_signs[k]) & 127;
        }
        for (int k = 0; k < 4; ++k) {
            uint16_t u = 0;
            for (int i = 0; i < 8; ++i) u |= (L[8*k+i] << 2*i);
            int grid_index = kmap_q2xs[u];
            if (grid_index < 0) GPUQ_TRAP();
            q2[2*ib+0] |= ((uint32_t) grid_index << 8*k);
            q2[2*ib+1] |= (block_signs[k] << 7*k);
        }
        scales[ib] = scale;
        max_scale = MAX(max_scale, scale);
    }
    if (!max_scale) {
        for (int i = 0; i < QK_K/8; ++i) y->qs[i] = 0;
        return;
    }
    float d = max_scale/31;
    y->d = fp32_to_fp16(d);
    float id = 1/d;
    for (int ib = 0; ib < QK_K/32; ++ib) {
        int l = nearest_int(0.5f*(id*scales[ib]-1));
        l = MAX(0, MIN(15, l));
        q2[2*ib+1] |= ((uint32_t)l << 28);
    }
    for (int i = 0; i < 2*(QK_K/32); ++i) { y->qs[2*i] = q2[i] & 0xffff; y->qs[2*i+1] = q2[i] >> 16; }
}

// ---------------------------------------------------------------- IQ1_S (quantize_row_iq1_s_impl, one block)
QD void q_iq1_s(const float * xbl, const float * qwbl, block_iq1_s * y, Tab t) {
    const uint64_t * kgrid_q2xs = t.grid;
    const int * kmap_q2xs = t.map;
    const uint16_t * kneighbors_q2xs = t.neigh;
    const int block_size = IQ1S_BLOCK_SIZE;
    float scales[QK_K/IQ1S_BLOCK_SIZE];
    float weight[IQ1S_BLOCK_SIZE];
    int8_t L[IQ1S_BLOCK_SIZE];
    float sumx[IQ1S_BLOCK_SIZE+1];
    float sumw[IQ1S_BLOCK_SIZE+1];
    float key[IQ1S_BLOCK_SIZE];
    int idx[IQ1S_BLOCK_SIZE];
    uint16_t index[IQ1S_BLOCK_SIZE/8];
    int8_t shifts[QK_K/IQ1S_BLOCK_SIZE];
    const float x_p[3] = {-1 + IQ1S_DELTA,  IQ1S_DELTA, 1 + IQ1S_DELTA};
    const float x_m[3] = {-1 - IQ1S_DELTA, -IQ1S_DELTA, 1 - IQ1S_DELTA};

    y->d = fp32_to_fp16(0.f);
    for (int i = 0; i < QK_K/8; ++i) y->qs[i] = 0;
    for (int i = 0; i < QK_K/32; ++i) y->qh[i] = 0;
    float max_scale = 0;
    float sumx2 = 0;
    for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
    float sigma2 = 2*sumx2/QK_K;

    for (int ib = 0; ib < QK_K/block_size; ++ib) {
        const float * xb = xbl + block_size*ib;
        const float * qw = qwbl + block_size*ib;
        for (int i = 0; i < block_size; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
        float max = fabsf(xb[0]);
        for (int i = 1; i < block_size; ++i) max = MAX(max, fabsf(xb[i]));
        if (max < GROUP_MAX_EPS_IQ1_S) {
            scales[ib] = 0;
            shifts[ib] = 1;
            for (int i = 0; i < block_size; ++i) L[i] = 1;
            continue;
        }
        for (int j = 0; j < block_size; ++j) { key[j] = xb[j]; idx[j] = j; }
        stable_sort(key, idx, block_size);
        {
            sumx[0] = sumw[0] = 0;
            for (int j = 0; j < block_size; ++j) {
                int i = idx[j];
                sumx[j+1] = sumx[j] + weight[i]*xb[i];
                sumw[j+1] = sumw[j] + weight[i];
            }
        }
        float best_score = -FLT_MAX, scale = max;
        int besti1 = -1, besti2 = -1, best_shift = 0;
        for (int i1 = 0; i1 <= block_size; ++i1) {
            for (int i2 = i1; i2 <= block_size; ++i2) {
                float sumqx = (sumx[i1] - sumx[0])*x_p[0] + (sumx[i2] - sumx[i1])*x_p[1] + (sumx[block_size] - sumx[i2])*x_p[2];
                float sumq2 = (sumw[i1] - sumw[0])*x_p[0]*x_p[0] + (sumw[i2] - sumw[i1])*x_p[1]*x_p[1] + (sumw[block_size] - sumw[i2])*x_p[2]*x_p[2];
                if (sumq2 > 0 && sumqx*sumqx > best_score*sumq2) {
                    scale = sumqx/sumq2; best_score = scale*sumqx;
                    besti1 = i1; besti2 = i2; best_shift = 1;
                }
                sumqx = (sumx[i1] - sumx[0])*x_m[0] + (sumx[i2] - sumx[i1])*x_m[1] + (sumx[block_size] - sumx[i2])*x_m[2];
                sumq2 = (sumw[i1] - sumw[0])*x_m[0]*x_m[0] + (sumw[i2] - sumw[i1])*x_m[1]*x_m[1] + (sumw[block_size] - sumw[i2])*x_m[2]*x_m[2];
                if (sumq2 > 0 && sumqx*sumqx > best_score*sumq2) {
                    scale = sumqx/sumq2; best_score = scale*sumqx;
                    besti1 = i1; besti2 = i2; best_shift = -1;
                }
            }
        }
        if (besti1 < 0 || besti2 < 0 || best_shift == 0) {
            scales[ib] = 0;
            shifts[ib] = 1;
            for (int i = 0; i < block_size; ++i) L[i] = 1;
            continue;
        }
        for (int j =      0; j < besti1; ++j) L[idx[j]] = 0;
        for (int j = besti1; j < besti2; ++j) L[idx[j]] = 1;
        for (int j = besti2; j < block_size; ++j) L[idx[j]] = 2;
        if (scale < 0) {
            for (int j = 0; j < block_size; ++j) L[j] = 2 - L[j];
            scale = -scale; best_shift = -best_shift;
        }
        bool all_on_grid = true;
        const float * xx = best_shift == 1 ? x_p : x_m;
        for (int k = 0; k < block_size/8; ++k) {
            uint16_t u = 0;
            for (int j = 0; j < 8; ++j) u |= (L[8*k+j] << 2*j);
            int grid_index = kmap_q2xs[u];
            if (grid_index < 0) {
                all_on_grid = false;
                const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                grid_index = iq1_find_best_neighbour2(neighbours, kgrid_q2xs, xb + 8*k, weight + 8*k, scale, xx, L + 8*k, NGRID_IQ1S);
            }
            index[k] = grid_index;
        }
        if (!all_on_grid) {
            float sumqx = 0, sumq2 = 0;
            for (int k = 0; k < block_size/8; ++k) {
                const int8_t * pg = (const int8_t *)(kgrid_q2xs + index[k]);
                for (int j = 0; j < 8; ++j) {
                    float w = weight[8*k + j];
                    float q = xx[(pg[j] - 1)/2];
                    sumqx += w*q*xb[8*k+j];
                    sumq2 += w*q*q;
                }
            }
            if (sumqx > 0 && sumq2 > 0) scale = sumqx/sumq2;
        }
        uint16_t h = 0;
        for (int k = 0; k < block_size/8; ++k) {
            y->qs[(block_size/8)*ib + k] = index[k] & 255;
            h |= (index[k] >> 8) << 3*k;
        }
        y->qh[ib] = h;
        scales[ib] = scale;
        shifts[ib] = best_shift;
        max_scale = MAX(max_scale, scale);
    }
    if (!max_scale) return;
    float d = max_scale/15;
    y->d = fp32_to_fp16(d*1.125f);
    float id = 1/d;
    for (int ib = 0; ib < QK_K/block_size; ++ib) {
        int l = nearest_int(0.5f*(id*scales[ib]-1));
        l = MAX(0, MIN(7, l));
        if (shifts[ib] == -1) l |= 8;
        y->qh[ib] |= (l << 12);
    }
}

// ---------------------------------------------------------------- IQ1_M (quantize_row_iq1_m_impl, one block)
QD void q_iq1_m(const float * xbl, const float * qwbl, block_iq1_m * y, Tab t) {
    const uint64_t * kgrid_q2xs = t.grid;
    const int * kmap_q2xs = t.map;
    const uint16_t * kneighbors_q2xs = t.neigh;
    const int block_size = IQ1M_BLOCK_SIZE;
    float scales[QK_K/IQ1M_BLOCK_SIZE];
    float weight[IQ1M_BLOCK_SIZE];
    int8_t L[IQ1M_BLOCK_SIZE];
    float key[IQ1M_BLOCK_SIZE];
    int idx[IQ1M_BLOCK_SIZE];
    uint16_t index[IQ1M_BLOCK_SIZE/8];
    int8_t shifts[QK_K/IQ1M_BLOCK_SIZE];
    const float x_p[3] = {-1 + IQ1M_DELTA,  IQ1M_DELTA, 1 + IQ1M_DELTA};
    const float x_m[3] = {-1 - IQ1M_DELTA, -IQ1M_DELTA, 1 - IQ1M_DELTA};
    const uint8_t masks[4] = {0x00, 0x80, 0x08, 0x88};
    float sumqx[4], sumq2[4];
    const float * xx;

    for (int i = 0; i < QK_K/8; ++i) y->qs[i] = 0;
    for (int i = 0; i < QK_K/16; ++i) y->qh[i] = 0;
    for (int i = 0; i < QK_K/32; ++i) y->scales[i] = 0;
    float max_scale = 0;
    float sumx2 = 0;
    for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
    float sigma2 = 2*sumx2/QK_K;

    for (int ib = 0; ib < QK_K/block_size; ++ib) {
        const float * xb = xbl + block_size*ib;
        const float * qw = qwbl + block_size*ib;
        for (int i = 0; i < block_size; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
        float max = fabsf(xb[0]);
        for (int i = 1; i < block_size; ++i) max = MAX(max, fabsf(xb[i]));
        if (max < GROUP_MAX_EPS_IQ1_M) {
            scales[ib] = 0;
            shifts[ib] = 0;
            for (int i = 0; i < block_size; ++i) L[i] = 1;
            continue;
        }
        for (int j = 0; j < block_size; ++j) { key[j] = xb[j]; idx[j] = j; }
        stable_sort(key, idx, block_size);
        float best_score = -FLT_MAX, scale = max;
        int besti1 = -1, besti2 = -1, best_k = -1;
        {
            float px[2][IQ1M_BLOCK_SIZE+1];
            float pw[2][IQ1M_BLOCK_SIZE+1];
            px[0][0] = px[1][0] = 0;
            pw[0][0] = pw[1][0] = 0;
            for (int j = 0; j < block_size; ++j) {
                const int i = idx[j];
                const int h = i < block_size/2 ? 0 : 1;
                px[h][j+1]   = px[h][j] + weight[i]*xb[i];
                px[1-h][j+1] = px[1-h][j];
                pw[h][j+1]   = pw[h][j] + weight[i];
                pw[1-h][j+1] = pw[1-h][j];
            }
            const float txs[2] = {px[0][block_size], px[1][block_size]};
            const float tws[2] = {pw[0][block_size], pw[1][block_size]};
            const float p2   = x_p[2], m2 = x_m[2];
            const float cp1  = x_p[0]*x_p[0] - x_p[1]*x_p[1];
            const float cp2  = x_p[1]*x_p[1] - x_p[2]*x_p[2];
            const float cm1  = x_m[0]*x_m[0] - x_m[1]*x_m[1];
            const float cm2  = x_m[1]*x_m[1] - x_m[2]*x_m[2];
            for (int i1 = 0; i1 <= block_size; ++i1) {
                for (int i2 = i1; i2 <= block_size; ++i2) {
                    float qx_p[2], qx_m[2], q2_p[2], q2_m[2];
                    for (int h = 0; h < 2; ++h) {
                        const float sx = px[h][i1] + px[h][i2];
                        qx_p[h] = txs[h]*p2 - sx;
                        qx_m[h] = txs[h]*m2 - sx;
                        q2_p[h] = tws[h]*p2*p2 + pw[h][i1]*cp1 + pw[h][i2]*cp2;
                        q2_m[h] = tws[h]*m2*m2 + pw[h][i1]*cm1 + pw[h][i2]*cm2;
                    }
                    sumqx[0] = qx_p[0] + qx_p[1];
                    sumqx[1] = qx_p[0] + qx_m[1];
                    sumqx[2] = qx_m[0] + qx_p[1];
                    sumqx[3] = qx_m[0] + qx_m[1];
                    sumq2[0] = q2_p[0] + q2_p[1];
                    sumq2[1] = q2_p[0] + q2_m[1];
                    sumq2[2] = q2_m[0] + q2_p[1];
                    sumq2[3] = q2_m[0] + q2_m[1];
                    for (int k = 0; k < 4; ++k) {
                        if (sumq2[k] > 0 && sumqx[k]*sumqx[k] > best_score*sumq2[k]) {
                            scale = sumqx[k]/sumq2[k]; best_score = scale*sumqx[k];
                            besti1 = i1; besti2 = i2; best_k = k;
                        }
                    }
                }
            }
        }
        if (besti1 < 0 || besti2 < 0 || best_k < 0) {
            scales[ib] = 0;
            shifts[ib] = 0;
            for (int i = 0; i < block_size; ++i) L[i] = 1;
            continue;
        }
        for (int j =      0; j < besti1; ++j) L[idx[j]] = 0;
        for (int j = besti1; j < besti2; ++j) L[idx[j]] = 1;
        for (int j = besti2; j < block_size; ++j) L[idx[j]] = 2;
        if (scale < 0) {
            for (int j = 0; j < block_size; ++j) L[j] = 2 - L[j];
            scale = -scale;
            best_k = best_k == 0 ? 3 : best_k == 1 ? 2 : best_k == 2 ? 1 : 0;
        }
        bool all_on_grid = true;
        for (int k = 0; k < block_size/8; ++k) {
            if (k == 0) xx = best_k < 2 ? x_p : x_m;
            else xx = best_k%2 == 0 ? x_p : x_m;
            uint16_t u = 0;
            for (int j = 0; j < 8; ++j) u |= (L[8*k+j] << 2*j);
            int grid_index = kmap_q2xs[u];
            if (grid_index < 0) {
                all_on_grid = false;
                const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                grid_index = iq1_find_best_neighbour2(neighbours, kgrid_q2xs, xb + 8*k, weight + 8*k, scale, xx, L + 8*k, NGRID_IQ1S);
            }
            index[k] = grid_index;
        }
        if (!all_on_grid) {
            float sumqx_f = 0, sumq2_f = 0;
            for (int k = 0; k < block_size/8; ++k) {
                if (k == 0) xx = best_k < 2 ? x_p : x_m;
                else xx = best_k%2 == 0 ? x_p : x_m;
                const int8_t * pg = (const int8_t *)(kgrid_q2xs + index[k]);
                for (int j = 0; j < 8; ++j) {
                    float w = weight[8*k + j];
                    float q = xx[(pg[j] - 1)/2];
                    sumqx_f += w*q*xb[8*k+j];
                    sumq2_f += w*q*q;
                }
            }
            if (sumqx_f > 0 && sumq2_f > 0) scale = sumqx_f/sumq2_f;
        }
        y->qs[2*ib + 0] = index[0] & 255;
        y->qs[2*ib + 1] = index[1] & 255;
        y->qh[ib] = (index[0] >> 8) | ((index[1] >> 8) << 4);
        scales[ib] = scale;
        shifts[ib] = best_k;
        max_scale = MAX(max_scale, scale);
    }
    if (!max_scale) return;
    uint16_t sc[4] = {0, 0, 0, 0};                  // (ggml: uint16_t * sc = (uint16_t *)y->scales, zeroed above)
    float d = max_scale/15;
    float id = 1/d;
    float sumqx_f = 0, sumq2_f = 0;
    for (int ib = 0; ib < QK_K/block_size; ++ib) {
        int l = nearest_int(0.5f*(id*scales[ib+0]-1));
        l = MAX(0, MIN(7, l));
        sc[ib/4] |= (l << 3*(ib%4));
        y->qh[ib] |= masks[shifts[ib]];
        const float * xb = xbl + block_size*ib;
        const float * qw = qwbl + block_size*ib;
        for (int i = 0; i < block_size; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
        for (int k = 0; k < block_size/8; ++k) {
            if (k == 0) xx = shifts[ib] < 2 ? x_p : x_m;
            else xx = shifts[ib]%2 == 0 ? x_p : x_m;
            const int8_t * pg = (const int8_t *)(kgrid_q2xs + y->qs[2*ib+k] + ((y->qh[ib] << (8 - 4*k)) & 0x700));
            for (int j = 0; j < 8; ++j) {
                float w = weight[8*k + j];
                float q = xx[(pg[j] - 1)/2]*(2*l+1);
                sumqx_f += w*q*xb[8*k+j];
                sumq2_f += w*q*q;
            }
        }
    }
    if (sumq2_f > 0) d = sumqx_f/sumq2_f;
    const uint16_t s = fp32_to_fp16(d*1.1125f);
    sc[0] |= ((s & 0x000f) << 12);
    sc[1] |= ((s & 0x00f0) <<  8);
    sc[2] |= ((s & 0x0f00) <<  4);
    sc[3] |= ((s & 0xf000) <<  0);
    for (int i = 0; i < 4; ++i) { y->scales[2*i] = sc[i] & 0xff; y->scales[2*i+1] = sc[i] >> 8; }
}

// ---------------------------------------------------------------- IQ2_S (quantize_row_iq2_s_impl, one block)
QD void q_iq2_s(const float * xbl, const float * qwbl, block_iq2_s * y, Tab t) {
    const uint64_t * kgrid_q2xs = t.grid;
    const int * kmap_q2xs = t.map;
    const uint16_t * kneighbors_q2xs = t.neigh;
    const int kMaxQ = 3;
    float scales[QK_K/16];
    float weight[16];
    float xval[16];
    int8_t L[16];
    int8_t Laux[16];
    float waux[16];
    bool is_on_grid[2];
    bool is_on_grid_aux[2];
    uint8_t block_signs[2];

    { uint8_t * p = (uint8_t *) y; for (int i = 0; i < (int) sizeof(block_iq2_s); ++i) p[i] = 0; }
    y->d = fp32_to_fp16(0.f);
    float max_scale = 0;
    float sumx2 = 0;
    for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
    float sigma2 = 2*sumx2/QK_K;

    for (int ib = 0; ib < QK_K/16; ++ib) {
        const float * xb = xbl + 16*ib;
        const float * qw = qwbl + 16*ib;
        for (int i = 0; i < 16; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
        for (int i = 0; i < 16; ++i) waux[i] = sqrtf(weight[i]);
        for (int k = 0; k < 2; ++k) {
            uint8_t s = 0;
            for (int i = 0; i < 8; ++i) {
                if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                else { xval[8*k + i] = -xb[8*k + i]; s |= (1 << i); }
            }
            block_signs[k] = s;
        }
        float max = xval[0];
        for (int i = 1; i < 16; ++i) max = MAX(max, xval[i]);
        for (int i = 0; i < 16; ++i) L[i] = 0;
        if (max < GROUP_MAX_EPS_IQ2_S) {
            scales[ib] = 0;
            continue;
        }
        float best = 0;
        float scale = max/(2*kMaxQ-1);
        is_on_grid[0] = is_on_grid[1] = true;
        for (int is = -9; is <= 9; ++is) {
            float id = (2*kMaxQ-1+is*0.1f)/max;
            float this_scale = 1/id;
            for (int k = 0; k < 2; ++k) {
                for (int i = 0; i < 8; ++i) {
                    int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                    Laux[8*k+i] = MAX(0, MIN(kMaxQ-1, l));
                }
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= (Laux[8*k+i] << 2*i);
                int grid_index = kmap_q2xs[u];
                is_on_grid_aux[k] = true;
                if (grid_index < 0) {
                    is_on_grid_aux[k] = false;
                    const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                    grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, this_scale, Laux + 8*k);
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 16; ++i) {
                float w = weight[i];
                float q = 2*Laux[i] + 1;
                sumqx += w*xval[i]*q;
                sumq2 += w*q*q;
            }
            if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                scale = sumqx/sumq2; best = scale*sumqx;
                for (int i = 0; i < 16; ++i) L[i] = Laux[i];
                for (int k = 0; k <  2; ++k) is_on_grid[k] = is_on_grid_aux[k];
            }
        }
        int n_not_ongrid = 0;
        for (int k = 0; k < 2; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
        if (n_not_ongrid > 0 && scale > 0) {
            float id = 1/scale;
            for (int k = 0; k < 2; ++k) {
                if (is_on_grid[k]) continue;
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) {
                    int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                    l = MAX(0, MIN(kMaxQ-1, l));
                    u |= (l << 2*i);
                    L[8*k + i] = l;
                }
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                    grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, scale, L + 8*k);
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 16; ++i) {
                float w = weight[i];
                float q = 2*L[i] + 1;
                sumqx += w*xval[i]*q;
                sumq2 += w*q*q;
            }
            if (sumq2 > 0) scale = sumqx/sumq2;
        }
        if (scale < 0) {
            scale = -scale;
            for (int k = 0; k < 2; ++k) block_signs[k] = ~block_signs[k];
        }
        for (int k = 0; k < 2; ++k) {
            uint16_t u = 0;
            for (int i = 0; i < 8; ++i) u |= (L[8*k+i] << 2*i);
            int grid_index = kmap_q2xs[u];
            if (grid_index < 0) GPUQ_TRAP();
            const int i8 = 2*ib + k;
            y->qs[i8] = grid_index & 255;
            y->qh[i8/4] |= ((grid_index >> 8) << 2*(i8%4));
            y->qs[QK_K/8 + i8] = block_signs[k];
        }
        scales[ib] = scale;
        max_scale = MAX(max_scale, scale);
    }
    if (!max_scale) return;
    float d = max_scale/31;
    y->d = fp32_to_fp16(d * 0.9875f);
    float id = 1/d;
    for (int ib = 0; ib < QK_K/16; ++ib) {
        int l = nearest_int(0.5f*(id*scales[ib]-1));
        l = MAX(0, MIN(15, l));
        if (ib%2 == 0) y->scales[ib/2] = l;
        else y->scales[ib/2] |= (l << 4);
    }
}

// ---------------------------------------------------------------- IQ2_XS (quantize_row_iq2_xs_impl, one block)
QD void q_iq2_xs(const float * xbl, const float * qwbl, block_iq2_xs * y, Tab t) {
    const uint64_t * kgrid_q2xs = t.grid;
    const int * kmap_q2xs = t.map;
    const uint16_t * kneighbors_q2xs = t.neigh;
    const int kMaxQ = 3;
    float scales[QK_K/16];
    float weight[16];
    float xval[16];
    int8_t L[16];
    int8_t Laux[16];
    float waux[16];
    bool is_on_grid[2];
    bool is_on_grid_aux[2];
    uint8_t block_signs[2];
    uint16_t q2[2*(QK_K/16)];

    y->d = fp32_to_fp16(0.f);
    for (int i = 0; i < 2*(QK_K/16); ++i) q2[i] = 0;
    for (int i = 0; i < QK_K/32; ++i) y->scales[i] = 0;
    float max_scale = 0;
    float sumx2 = 0;
    for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
    float sigma2 = sumx2/QK_K;

    for (int ib = 0; ib < QK_K/16; ++ib) {
        const float * xb = xbl + 16*ib;
        const float * qw = qwbl + 16*ib;
        for (int i = 0; i < 16; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
        for (int i = 0; i < 16; ++i) waux[i] = sqrtf(weight[i]);
        for (int k = 0; k < 2; ++k) {
            int nflip = 0;
            uint8_t s = 0;
            for (int i = 0; i < 8; ++i) {
                if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                else { xval[8*k + i] = -xb[8*k + i]; ++nflip; s |= (1 << i); }
            }
            if (nflip%2) {
                int imin = 0; float min = weight[8*k+imin]*xb[8*k+imin]*xb[8*k+imin];
                for (int i = 1; i < 8; ++i) {
                    float ax = weight[8*k+i]*xb[8*k+i]*xb[8*k+i];
                    if (ax < min) { min = ax; imin = i; }
                }
                xval[8*k+imin] = -xval[8*k+imin];
                s ^= (1 << imin);
            }
            block_signs[k] = s & 127;
        }
        float max = xval[0];
        for (int i = 1; i < 16; ++i) max = MAX(max, xval[i]);
        for (int i = 0; i < 16; ++i) L[i] = 0;
        if (max < GROUP_MAX_EPS) {
            scales[ib] = 0;
            continue;
        }
        float best = 0;
        float scale = max/(2*kMaxQ-1);
        is_on_grid[0] = is_on_grid[1] = true;
        for (int is = -9; is <= 9; ++is) {
            float id = (2*kMaxQ-1+is*0.1f)/max;
            float this_scale = 1/id;
            for (int k = 0; k < 2; ++k) {
                for (int i = 0; i < 8; ++i) {
                    int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                    Laux[8*k+i] = MAX(0, MIN(kMaxQ-1, l));
                }
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= (Laux[8*k+i] << 2*i);
                int grid_index = kmap_q2xs[u];
                is_on_grid_aux[k] = true;
                if (grid_index < 0) {
                    is_on_grid_aux[k] = false;
                    const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                    grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, this_scale, Laux + 8*k);
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 16; ++i) {
                float w = weight[i];
                float q = 2*Laux[i] + 1;
                sumqx += w*xval[i]*q;
                sumq2 += w*q*q;
            }
            if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                scale = sumqx/sumq2; best = scale*sumqx;
                for (int i = 0; i < 16; ++i) L[i] = Laux[i];
                for (int k = 0; k <  2; ++k) is_on_grid[k] = is_on_grid_aux[k];
            }
        }
        int n_not_ongrid = 0;
        for (int k = 0; k < 2; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
        if (n_not_ongrid > 0 && scale > 0) {
            float id = 1/scale;
            for (int k = 0; k < 2; ++k) {
                if (is_on_grid[k]) continue;
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) {
                    int l = nearest_int(0.5f*(id*xval[8*k+i]-1));
                    l = MAX(0, MIN(kMaxQ-1, l));
                    u |= (l << 2*i);
                    L[8*k + i] = l;
                }
                int grid_index = kmap_q2xs[u];
                if (grid_index < 0) {
                    const uint16_t * neighbours = kneighbors_q2xs - kmap_q2xs[u] - 1;
                    grid_index = iq2_find_best_neighbour(neighbours, kgrid_q2xs, xval + 8*k, waux + 8*k, scale, L + 8*k);
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 16; ++i) {
                float w = weight[i];
                float q = 2*L[i] + 1;
                sumqx += w*xval[i]*q;
                sumq2 += w*q*q;
            }
            if (sumq2 > 0) scale = sumqx/sumq2;
        }
        if (scale < 0) {
            scale = -scale;
            for (int k = 0; k < 2; ++k) block_signs[k] = (~block_signs[k]) & 127;
        }
        for (int k = 0; k < 2; ++k) {
            uint16_t u = 0;
            for (int i = 0; i < 8; ++i) u |= (L[8*k+i] << 2*i);
            int grid_index = kmap_q2xs[u];
            if (grid_index < 0) GPUQ_TRAP();
            q2[2*ib+k] = grid_index | (block_signs[k] << 9);
        }
        scales[ib] = scale;
        max_scale = MAX(max_scale, scale);
    }
    if (!max_scale) {
        for (int i = 0; i < QK_K/8; ++i) y->qs[i] = 0;
        return;
    }
    float d = max_scale/31;
    y->d = fp32_to_fp16(d);
    float id = 1/d;
    for (int ib = 0; ib < QK_K/16; ++ib) {
        int l = nearest_int(0.5f*(id*scales[ib]-1));
        l = MAX(0, MIN(15, l));
        if (ib%2 == 0) y->scales[ib/2] = l;
        else y->scales[ib/2] |= (l << 4);
    }
    for (int i = 0; i < QK_K/8; ++i) y->qs[i] = q2[i];
}

// the IQ3 grids are stored as uint64 (gpuq_tables.c zero-extends ggml's uint32 points); a point is the first 4 bytes
QD int iq3_find_best_neighbour(const uint16_t * neighbours, const uint64_t * grid, const float * xval,
                               const float * weight, float scale, int8_t * L) {
    int num_neighbors = neighbours[0];
    if (num_neighbors <= 0) GPUQ_TRAP();
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 4; ++i) {
            float q = pg[i];
            float diff = scale*q - xval[i];
            d2 += weight[i]*diff*diff;
        }
        if (d2 < best_d2) { best_d2 = d2; grid_index = neighbours[j]; }
    }
    if (grid_index < 0) GPUQ_TRAP();
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 4; ++i) L[i] = (pg[i] - 1)/2;
    return grid_index;
}

// ---------------------------------------------------------------- IQ3_XXS (quantize_row_iq3_xxs_impl(256), one block)
QD void q_iq3_xxs(const float * xbl, const float * qwbl, block_iq3_xxs * y, Tab t) {
    const uint64_t * kgrid_q3xs = t.grid;
    const int * kmap_q3xs = t.map;
    const uint16_t * kneighbors_q3xs = t.neigh;
    const int kMaxQ = 8;
    float scales[QK_K/32];
    float weight[32];
    float xval[32];
    int8_t L[32];
    int8_t Laux[32];
    float waux[32];
    bool is_on_grid[8];
    bool is_on_grid_aux[8];
    uint8_t block_signs[8];
    uint8_t q3[3*(QK_K/8)+QK_K/32];
    uint32_t scales_and_signs[QK_K/32];             // (ggml: (uint32_t *)(q3 + QK_K/4), copied back below)

    y->d = fp32_to_fp16(0.f);
    for (int i = 0; i < 3*QK_K/8+QK_K/32; ++i) q3[i] = 0;
    for (int i = 0; i < QK_K/32; ++i) scales_and_signs[i] = 0;
    float max_scale = 0;
    float sumx2 = 0;
    for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
    float sigma2 = 2*sumx2/QK_K;

    for (int ib = 0; ib < QK_K/32; ++ib) {
        const float * xb = xbl + 32*ib;
        const float * qw = qwbl + 32*ib;
        for (int i = 0; i < 32; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
        for (int i = 0; i < 32; ++i) waux[i] = sqrtf(weight[i]);
        for (int k = 0; k < 4; ++k) {
            int nflip = 0;
            uint8_t s = 0;
            for (int i = 0; i < 8; ++i) {
                if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                else { xval[8*k + i] = -xb[8*k + i]; ++nflip; s |= (1 << i); }
            }
            if (nflip%2) {
                int imin = 0; float min = weight[8*k+imin]*xb[8*k+imin]*xb[8*k+imin];
                for (int i = 1; i < 8; ++i) {
                    float ax = weight[8*k+i]*xb[8*k+i]*xb[8*k+i];
                    if (ax < min) { min = ax; imin = i; }
                }
                xval[8*k+imin] = -xval[8*k+imin];
                s ^= (1 << imin);
            }
            block_signs[k] = s & 127;
        }
        float max = xval[0];
        for (int i = 1; i < 32; ++i) max = MAX(max, xval[i]);
        for (int i = 0; i < 32; ++i) L[i] = 0;
        if (max < GROUP_MAX_EPS_IQ3_XXS) {
            scales[ib] = 0;
            continue;
        }
        float best = 0;
        float scale = max/(2*kMaxQ-1);
        for (int k = 0; k < 8; ++k) is_on_grid[k] = true;
        for (int is = -15; is <= 15; ++is) {
            float id = (2*kMaxQ-1+is*0.2f)/max;
            float this_scale = 1/id;
            for (int k = 0; k < 8; ++k) {
                for (int i = 0; i < 4; ++i) {
                    int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                    Laux[4*k+i] = MAX(0, MIN(kMaxQ-1, l));
                }
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) u |= (Laux[4*k+i] << 3*i);
                int grid_index = kmap_q3xs[u];
                is_on_grid_aux[k] = true;
                if (grid_index < 0) {
                    is_on_grid_aux[k] = false;
                    const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                    grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, this_scale, Laux + 4*k);
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 32; ++i) {
                float w = weight[i];
                float q = 2*Laux[i] + 1;
                sumqx += w*xval[i]*q;
                sumq2 += w*q*q;
            }
            if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                scale = sumqx/sumq2; best = scale*sumqx;
                for (int i = 0; i < 32; ++i) L[i] = Laux[i];
                for (int k = 0; k <  8; ++k) is_on_grid[k] = is_on_grid_aux[k];
            }
        }
        int n_not_ongrid = 0;
        for (int k = 0; k < 8; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
        if (n_not_ongrid > 0 && scale > 0) {
            float id = 1/scale;
            for (int k = 0; k < 8; ++k) {
                if (is_on_grid[k]) continue;
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) {
                    int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                    l = MAX(0, MIN(kMaxQ-1, l));
                    u |= (l << 3*i);
                }
                int grid_index = kmap_q3xs[u];
                if (grid_index < 0) {
                    const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                    grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, scale, L + 4*k);
                }
                const int8_t * pg = (const int8_t *)(kgrid_q3xs + grid_index);
                for (int i = 0; i < 4; ++i) L[4*k+i] = (pg[i] - 1)/2;
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 32; ++i) {
                float w = weight[i];
                float q = 2*L[i] + 1;
                sumqx += w*xval[i]*q;
                sumq2 += w*q*q;
            }
            if (sumq2 > 0) scale = sumqx/sumq2;
        }
        if (scale < 0) {
            scale = -scale;
            for (int k = 0; k < 4; ++k) block_signs[k] = (~block_signs[k]) & 127;
        }
        for (int k = 0; k < 8; ++k) {
            uint16_t u = 0;
            for (int i = 0; i < 4; ++i) u |= (L[4*k+i] << 3*i);
            int grid_index = kmap_q3xs[u];
            if (grid_index < 0) GPUQ_TRAP();
            q3[8*ib+k] = grid_index;
        }
        scales_and_signs[ib] = block_signs[0] | (block_signs[1] << 7) | (block_signs[2] << 14) | (block_signs[3] << 21);
        scales[ib] = scale;
        max_scale = MAX(max_scale, scale);
    }
    if (!max_scale) {
        for (int i = 0; i < 3*QK_K/8; ++i) y->qs[i] = 0;
        return;
    }
    float d = max_scale/31;
    y->d = fp32_to_fp16(d * 1.0125f);
    float id = 1/d;
    for (int ib = 0; ib < QK_K/32; ++ib) {
        int l = nearest_int(0.5f*(id*scales[ib]-1));
        l = MAX(0, MIN(15, l));
        scales_and_signs[ib] |= ((uint32_t)l << 28);
    }
    for (int i = 0; i < QK_K/4; ++i) y->qs[i] = q3[i];
    for (int ib = 0; ib < QK_K/32; ++ib)
        for (int j = 0; j < 4; ++j) y->qs[QK_K/4 + 4*ib + j] = (scales_and_signs[ib] >> 8*j) & 0xff;
}

// ---------------------------------------------------------------- IQ3_S (quantize_row_iq3_s_impl(32), one block)
QD void q_iq3_s(const float * xbl, const float * qwbl, block_iq3_s * y, Tab t) {
    const uint64_t * kgrid_q3xs = t.grid;
    const int * kmap_q3xs = t.map;
    const uint16_t * kneighbors_q3xs = t.neigh;
    const int kMaxQ = 8;
    const int block_size = 32, bs4 = 8, bs8 = 4;
    float scales[QK_K/32];
    float weight[32];
    float xval[32];
    int8_t L[32];
    int8_t Laux[32];
    float waux[32];
    bool is_on_grid[8];
    bool is_on_grid_aux[8];
    uint8_t block_signs[4];

    { uint8_t * p = (uint8_t *) y; for (int i = 0; i < (int) sizeof(block_iq3_s); ++i) p[i] = 0; }
    y->d = fp32_to_fp16(0.f);
    uint8_t * qs = y->qs;
    uint8_t * qh = y->qh;
    uint8_t * signs = y->signs;
    float max_scale = 0;
    float sumx2 = 0;
    for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
    float sigma2 = 2*sumx2/QK_K;

    for (int ib = 0; ib < QK_K/block_size; ++ib) {
        const float * xb = xbl + block_size*ib;
        const float * qw = qwbl + block_size*ib;
        for (int i = 0; i < block_size; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
        for (int i = 0; i < block_size; ++i) waux[i] = sqrtf(weight[i]);
        for (int k = 0; k < bs8; ++k) {
            uint8_t s = 0;
            for (int i = 0; i < 8; ++i) {
                if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                else { xval[8*k + i] = -xb[8*k + i]; s |= (1 << i); }
            }
            block_signs[k] = s;
        }
        float max = xval[0];
        for (int i = 1; i < block_size; ++i) max = MAX(max, xval[i]);
        for (int i = 0; i < block_size; ++i) L[i] = 0;
        if (!max) {
            scales[ib] = 0;
            continue;
        }
        float best = 0;
        float scale = max/(2*kMaxQ-1);
        for (int k = 0; k < bs4; ++k) is_on_grid[k] = false;
        for (int is = -9; is <= 9; ++is) {
            float id = (2*kMaxQ-1+is*0.2f)/max;
            float this_scale = 1/id;
            for (int k = 0; k < bs4; ++k) {
                for (int i = 0; i < 4; ++i) {
                    int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                    Laux[4*k+i] = MAX(0, MIN(kMaxQ-1, l));
                }
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) u |= (Laux[4*k+i] << 3*i);
                int grid_index = kmap_q3xs[u];
                is_on_grid_aux[k] = true;
                if (grid_index < 0) {
                    is_on_grid_aux[k] = false;
                    const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                    grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, this_scale, Laux + 4*k);
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < block_size; ++i) {
                float w = weight[i];
                float q = 2*Laux[i] + 1;
                sumqx += w*xval[i]*q;
                sumq2 += w*q*q;
            }
            if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                scale = sumqx/sumq2; best = scale*sumqx;
                for (int i = 0; i < block_size; ++i) L[i] = Laux[i];
                for (int k = 0; k < bs4; ++k) is_on_grid[k] = is_on_grid_aux[k];
            }
        }
        int n_not_ongrid = 0;
        for (int k = 0; k < bs4; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
        if (n_not_ongrid > 0 && scale > 0) {
            float id = 1/scale;
            for (int k = 0; k < bs4; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) {
                    int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                    l = MAX(0, MIN(kMaxQ-1, l));
                    u |= (l << 3*i);
                }
                int grid_index = kmap_q3xs[u];
                if (grid_index < 0) {
                    const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                    grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, scale, L + 4*k);
                }
                const int8_t * pg = (const int8_t *)(kgrid_q3xs + grid_index);
                for (int i = 0; i < 4; ++i) L[4*k+i] = (pg[i] - 1)/2;
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < block_size; ++i) {
                float w = weight[i];
                float q = 2*L[i] + 1;
                sumqx += w*xval[i]*q;
                sumq2 += w*q*q;
            }
            if (sumq2 > 0) scale = sumqx/sumq2;
        }
        if (scale < 0) {
            scale = -scale;
            for (int k = 0; k < bs8; ++k) block_signs[k] = ~block_signs[k];
        }
        for (int k = 0; k < bs4; ++k) {
            uint16_t u = 0;
            for (int i = 0; i < 4; ++i) u |= (L[4*k+i] << 3*i);
            int grid_index = kmap_q3xs[u];
            if (grid_index < 0) GPUQ_TRAP();
            qs[k] = grid_index & 255;
            qh[(ib*bs4+k)/8] |= ((grid_index >> 8) << ((ib*bs4+k)%8));
        }
        qs += bs4;
        for (int k = 0; k < bs8; ++k) signs[k] = block_signs[k];
        signs += bs8;
        scales[ib] = scale;
        max_scale = MAX(max_scale, scale);
    }
    if (!max_scale) return;
    float d = max_scale/31;
    y->d = fp32_to_fp16(d * 1.033f);
    float id = 1/d;
    for (int ib = 0; ib < QK_K/block_size; ib += 2) {
        int l1 = nearest_int(0.5f*(id*scales[ib+0]-1));
        l1 = MAX(0, MIN(15, l1));
        int l2 = nearest_int(0.5f*(id*scales[ib+1]-1));
        l2 = MAX(0, MIN(15, l2));
        y->scales[ib/2] = l1 | (l2 << 4);
    }
}

// ---------------------------------------------------------------- IQ4_NL / IQ4_XS (quantize_row_iq4_nl_impl, ntry 7)
QD int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

template <int SB>   // super_block_size: QK4_NL (IQ4_NL, no sub-scales) or QK_K (IQ4_XS); block_size 32
QD void iq4_nl_impl(const float * x, ggml_half * dh, uint8_t * q4, uint16_t * scales_h, uint8_t * scales_l,
                    const float * quant_weights) {
    const int block_size = 32, ntry = 7;
    const int8_t * values = kvalues_iq4nl;
    float scales[SB/32];
    float weight[32];
    uint8_t L[SB];
    float sigma2 = 0;
    for (int j = 0; j < SB; ++j) sigma2 += x[j]*x[j];
    sigma2 *= 2.f/SB;
    for (int j = 0; j < SB/2; ++j) q4[j] = 0;
    dh[0] = fp32_to_fp16(0.f);
    float max_scale = 0, amax_scale = 0;
    for (int ib = 0; ib < SB/block_size; ++ib) {
        const float * xb = x + ib*block_size;
        uint8_t * Lb = L + ib*block_size;
        const float * qw = quant_weights + ib*block_size;
        for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        float amax = 0, max = 0;
        for (int j = 0; j < block_size; ++j) {
            float ax = fabsf(xb[j]);
            if (ax > amax) { amax = ax; max = xb[j]; }
        }
        if (amax < GROUP_MAX_EPS) {
            scales[ib] = 0;
            continue;
        }
        float d = ntry > 0 ? -max/values[0] : max/values[0];
        float id = 1/d;
        float sumqx = 0, sumq2 = 0;
        for (int j = 0; j < block_size; ++j) {
            float al = id*xb[j];
            int l = best_index_int8(16, values, al);
            Lb[j] = l;
            float q = values[l];
            float w = weight[j];
            sumqx += w*q*xb[j];
            sumq2 += w*q*q;
        }
        d = sumq2 > 0 ? sumqx/sumq2 : 0.f;
        float best = d*sumqx;
        for (int itry = -ntry; itry <= ntry; ++itry) {
            id = (itry + values[0])/max;
            sumqx = sumq2 = 0;
            for (int j = 0; j < block_size; ++j) {
                float al = id*xb[j];
                int l = best_index_int8(16, values, al);
                float q = values[l];
                float w = weight[j];
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
            if (sumq2 > 0 && sumqx*sumqx > best*sumq2) { d = sumqx/sumq2; best = d * sumqx; }
        }
        scales[ib] = d;
        float abs_d = fabsf(d);
        if (abs_d > amax_scale) { amax_scale = abs_d; max_scale = d; }
    }
    if (SB/block_size > 1) {
        const int nb = SB/block_size;
        for (int i = 0; i < (nb+7)/8; ++i) scales_h[i] = 0;
        float d = -max_scale/32;
        dh[0] = fp32_to_fp16(d);
        float id = d ? 1/d : 0.f;
        for (int ib = 0; ib < SB/block_size; ++ib) {
            int l = nearest_int(id*scales[ib]);
            l = MAX(-32, MIN(31, l));
            float dl = d * l;
            float idl = dl ? 1/dl : 0.f;
            uint8_t * Lb = L + ib*block_size;
            const float * xb = x + ib*block_size;
            for (int j = 0; j < block_size; ++j) Lb[j] = best_index_int8(16, values, idl*xb[j]);
            l += 32;
            uint8_t l_l = l & 0xf;
            uint8_t l_h = l >>  4;
            if (ib%2 == 0) scales_l[ib/2] = l_l;
            else scales_l[ib/2] |= (l_l << 4);
            scales_h[ib/8] |= (l_h << 2*(ib%8));
        }
    } else {
        dh[0] = fp32_to_fp16(scales[0]);
        float id = scales[0] ? 1/scales[0] : 0;
        for (int j = 0; j < SB; ++j) L[j] = best_index_int8(16, values, id*x[j]);
    }
    for (int i = 0; i < SB/32; ++i)
        for (int j = 0; j < 16; ++j) q4[16*i + j] = L[32*i + j] | (L[32*i + 16 + j] << 4);
}

QD void q_iq4_xs(const float * xbl, const float * qwbl, block_iq4_xs * y, Tab) {
    uint16_t sh = 0;
    iq4_nl_impl<QK_K>(xbl, &y->d, y->qs, &sh, y->scales_l, qwbl);
    y->scales_h = sh;
}
QD void q_iq4_nl(const float * xbl, const float * qwbl, block_iq4_nl * y, Tab) {
    uint16_t unused_h;
    iq4_nl_impl<QK4_NL>(xbl, &y->d, y->qs, &unused_h, nullptr, qwbl);
}

// ---------------------------------------------------------------- kernels and the C API
template <typename B, void (*Q)(const float *, const float *, B *, Tab), int BS = QK_K>
__global__ void kq(const float * x, const float * qw, B * y, int64_t nblocks, int64_t nb_row, Tab t) {
    const int64_t b = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nblocks) return;
    Q(x + b*BS, qw + (b % nb_row)*BS, y + b, t);
}

namespace {
struct DevTab { int type; uint64_t * grid; int * map; uint16_t * neigh; };
std::vector<DevTab> g_tabs;
float * g_x = nullptr; void * g_y = nullptr; float * g_qw = nullptr;
size_t g_x_cap = 0, g_y_cap = 0, g_qw_cap = 0;

bool grow(void ** p, size_t * cap, size_t need) {
    if (need <= *cap) return true;
    if (*p) (void) hipFree(*p);
    *p = nullptr; *cap = 0;
    if (hipMalloc(p, need) != hipSuccess) return false;
    *cap = need;
    return true;
}

Tab tab_for(int type) {
    if (type == T_IQ4_NL || type == T_IQ4_XS) return Tab{nullptr, nullptr, nullptr};   // (no search tables)
    const int want = type == T_IQ1_M ? T_IQ1_S : type;     // (IQ1_M uses IQ1_S's tables, as in ggml)
    for (auto & d : g_tabs) if (d.type == want) return Tab{d.grid, d.map, d.neigh};
    return Tab{nullptr, nullptr, nullptr};
}
}

extern "C" int gpuq_init(const char * tables, int device) {
    if (hipSetDevice(device) != hipSuccess) return 1;
    FILE * f = fopen(tables, "rb");
    if (!f) return 2;
    int32_t hdr[4];
    while (fread(hdr, sizeof(hdr), 1, f) == 1) {
        std::vector<uint64_t> grid(hdr[1]); std::vector<int32_t> map(hdr[2]); std::vector<uint16_t> neigh(hdr[3]);
        if (fread(grid.data(), 8, hdr[1], f) != (size_t) hdr[1] || fread(map.data(), 4, hdr[2], f) != (size_t) hdr[2] ||
            fread(neigh.data(), 2, hdr[3], f) != (size_t) hdr[3]) { fclose(f); return 3; }
        DevTab d{hdr[0], nullptr, nullptr, nullptr};
        if (hipMalloc(&d.grid, grid.size()*8) != hipSuccess || hipMalloc(&d.map, map.size()*4) != hipSuccess ||
            hipMalloc(&d.neigh, neigh.size()*2) != hipSuccess) { fclose(f); return 4; }
        (void) hipMemcpy(d.grid, grid.data(), grid.size()*8, hipMemcpyHostToDevice);
        (void) hipMemcpy(d.map, map.data(), map.size()*4, hipMemcpyHostToDevice);
        (void) hipMemcpy(d.neigh, neigh.data(), neigh.size()*2, hipMemcpyHostToDevice);
        g_tabs.push_back(d);
    }
    fclose(f);
    return g_tabs.empty() ? 5 : 0;
}

extern "C" size_t gpuq_block_bytes(int type) {
    switch (type) {
        case T_IQ2_XXS: return sizeof(block_iq2_xxs);
        case T_IQ2_S:   return sizeof(block_iq2_s);
        case T_IQ1_S:   return sizeof(block_iq1_s);
        case T_IQ1_M:   return sizeof(block_iq1_m);
        case T_IQ2_XS:  return sizeof(block_iq2_xs);
        case T_IQ3_XXS: return sizeof(block_iq3_xxs);
        case T_IQ3_S:   return sizeof(block_iq3_s);
        case T_IQ4_NL:  return sizeof(block_iq4_nl);
        case T_IQ4_XS:  return sizeof(block_iq4_xs);
        default:        return 0;
    }
}

// ---------------------------------------------------------------- dequantizing on the GPU (exact: every value below is
// a small integer or a half/bfloat times a power of two or a half, so fp32 holds it exactly and the result equals
// ggml's to_float bit for bit)
enum { S_F32 = 0, S_F16 = 1, S_Q8_0 = 8, S_BF16 = 30, S_MXFP4 = 39 };

QD float fp16_to_fp32(uint16_t h) {                        // exact, as ggml_compute_fp16_to_fp32
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0) {
        if (m == 0) return fp32_from_bits(sign);
        e = 127 - 15 + 1;
        while (!(m & 0x400)) { m <<= 1; --e; }             // subnormal half -> normal float
        m &= 0x3ff;
        return fp32_from_bits(sign | (e << 23) | (m << 13));
    }
    if (e == 31) return fp32_from_bits(sign | 0x7f800000u | (m << 13));
    return fp32_from_bits(sign | ((e + 127 - 15) << 23) | (m << 13));
}
QD float e8m0_to_fp32_half(uint8_t x) { return fp32_from_bits(x < 2 ? 0x00200000u << x : (uint32_t) (x - 1) << 23); }

__global__ void kdq(int src, const uint8_t * in, float * out, int64_t n) {   // one thread a 32-value block
    const int64_t b = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b * 32 >= n) return;
    float * y = out + b*32;
    switch (src) {
        case S_F32:  { const float * x = (const float *) in + b*32; for (int j = 0; j < 32; ++j) y[j] = x[j]; break; }
        case S_F16:  { const uint16_t * x = (const uint16_t *) in + b*32; for (int j = 0; j < 32; ++j) y[j] = fp16_to_fp32(x[j]); break; }
        case S_BF16: { const uint16_t * x = (const uint16_t *) in + b*32;
                       for (int j = 0; j < 32; ++j) y[j] = fp32_from_bits((uint32_t) x[j] << 16); break; }
        case S_Q8_0: { const uint8_t * x = in + b*34;          // ggml_half d; int8_t qs[32]
                       const float d = fp16_to_fp32((uint16_t) (x[0] | (x[1] << 8)));
                       for (int j = 0; j < 32; ++j) y[j] = (int8_t) x[2 + j] * d; break; }
        case S_MXFP4: { const uint8_t * x = in + b*17;         // uint8_t e; uint8_t qs[16]
                        const int8_t kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
                        const float d = e8m0_to_fp32_half(x[0]);
                        for (int j = 0; j < 16; ++j) { y[j] = kv[x[1 + j] & 0x0f]*d; y[j + 16] = kv[x[1 + j] >> 4]*d; }
                        break; }
    }
}

namespace {
uint8_t * g_raw = nullptr; size_t g_raw_cap = 0;

size_t raw_bytes(int src, int64_t n) {
    switch (src) {
        case S_F32: return n*4;
        case S_F16: case S_BF16: return n*2;
        case S_Q8_0: return n/32*34;
        case S_MXFP4: return n/32*17;
        default: return 0;
    }
}

// the quantize kernels over g_x (nrow x n_per_row floats already on the GPU) into dst
size_t quantize_gx(int type, void * dst, int64_t nrow, int64_t n_per_row, const float * imatrix) {
    const size_t bb = gpuq_block_bytes(type);
    const Tab t = tab_for(type);
    const int bs = type == T_IQ4_NL ? QK4_NL : QK_K;
    const int64_t nb_row = n_per_row / bs, nblocks = nrow * nb_row;
    if (!grow(&g_y, &g_y_cap, nblocks * bb) || !grow((void **) &g_qw, &g_qw_cap, n_per_row * 4)) return 0;
    if (hipMemcpy(g_qw, imatrix, n_per_row * 4, hipMemcpyHostToDevice) != hipSuccess) return 0;
    const int th = 64;
    const dim3 grid((unsigned) ((nblocks + th - 1) / th));
    switch (type) {
        case T_IQ2_XXS: kq<block_iq2_xxs, q_iq2_xxs><<<grid, th>>>(g_x, g_qw, (block_iq2_xxs *) g_y, nblocks, nb_row, t); break;
        case T_IQ2_S:   kq<block_iq2_s,   q_iq2_s  ><<<grid, th>>>(g_x, g_qw, (block_iq2_s *)   g_y, nblocks, nb_row, t); break;
        case T_IQ1_S:   kq<block_iq1_s,   q_iq1_s  ><<<grid, th>>>(g_x, g_qw, (block_iq1_s *)   g_y, nblocks, nb_row, t); break;
        case T_IQ1_M:   kq<block_iq1_m,   q_iq1_m  ><<<grid, th>>>(g_x, g_qw, (block_iq1_m *)   g_y, nblocks, nb_row, t); break;
        case T_IQ2_XS:  kq<block_iq2_xs,  q_iq2_xs ><<<grid, th>>>(g_x, g_qw, (block_iq2_xs *)  g_y, nblocks, nb_row, t); break;
        case T_IQ3_XXS: kq<block_iq3_xxs, q_iq3_xxs><<<grid, th>>>(g_x, g_qw, (block_iq3_xxs *) g_y, nblocks, nb_row, t); break;
        case T_IQ3_S:   kq<block_iq3_s,   q_iq3_s  ><<<grid, th>>>(g_x, g_qw, (block_iq3_s *)   g_y, nblocks, nb_row, t); break;
        case T_IQ4_XS:  kq<block_iq4_xs,  q_iq4_xs ><<<grid, th>>>(g_x, g_qw, (block_iq4_xs *)  g_y, nblocks, nb_row, t); break;
        case T_IQ4_NL:  kq<block_iq4_nl,  q_iq4_nl, QK4_NL><<<grid, th>>>(g_x, g_qw, (block_iq4_nl *) g_y, nblocks, nb_row, t); break;
    }
    if (hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess) return 0;
    if (hipMemcpy(dst, g_y, nblocks * bb, hipMemcpyDeviceToHost) != hipSuccess) return 0;
    return nblocks * bb;
}

bool type_ok(int type, int64_t n_per_row, const float * imatrix) {
    const size_t bb = gpuq_block_bytes(type);
    const bool tabless = type == T_IQ4_NL || type == T_IQ4_XS;
    const int bs = type == T_IQ4_NL ? QK4_NL : QK_K;
    return bb && (tabless || tab_for(type).grid) && imatrix && n_per_row % bs == 0;
}
}

extern "C" size_t gpuq_quantize(int type, const float * src, void * dst, int64_t nrow, int64_t n_per_row,
                                const float * imatrix) {
    if (!type_ok(type, n_per_row, imatrix)) return 0;
    if (!grow((void **) &g_x, &g_x_cap, (size_t) nrow * n_per_row * 4)) return 0;
    if (hipMemcpy(g_x, src, (size_t) nrow * n_per_row * 4, hipMemcpyHostToDevice) != hipSuccess) return 0;
    return quantize_gx(type, dst, nrow, n_per_row, imatrix);
}

// 1 when gpuq_requantize can read this ggml type (F32, F16, BF16, Q8_0, MXFP4)
extern "C" int gpuq_can_dequantize(int src) { return raw_bytes(src, 32) > 0; }

// the source tensor's raw bytes (ggml type src) -> floats on the GPU -> type; the same bytes as ggml's to_float then
// ggml_quantize_chunk, with 1/8 (MXFP4) to 1/2 (F16) of the host-to-GPU traffic of gpuq_quantize and no CPU work
extern "C" size_t gpuq_requantize(int src, const void * raw, int type, void * dst, int64_t nrow, int64_t n_per_row,
                                  const float * imatrix) {
    const int64_t n = nrow * n_per_row;
    const size_t rb = raw_bytes(src, n);
    if (!rb || n % 32 || !type_ok(type, n_per_row, imatrix)) return 0;
    if (!grow((void **) &g_raw, &g_raw_cap, rb) || !grow((void **) &g_x, &g_x_cap, (size_t) n * 4)) return 0;
    if (hipMemcpy(g_raw, raw, rb, hipMemcpyHostToDevice) != hipSuccess) return 0;
    const int th = 128;
    kdq<<<(unsigned) ((n/32 + th - 1) / th), th>>>(src, g_raw, g_x, n);
    if (hipGetLastError() != hipSuccess) return 0;
    return quantize_gx(type, dst, nrow, n_per_row, imatrix);
}
