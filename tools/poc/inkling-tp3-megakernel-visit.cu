// mk_visit.cu - go/no-go PoW: one persistent kernel per P100 runs the whole M=1 TP3 decode of 48 SWA visits
// (Inkling 2 Mini shapes, per-rank shards exactly as the pascal-llm TP3 patch) in ONE cooperative launch per token.
// Work per visit and rank (same bytes as the engine path):
//   depth mix + RMSNorm (replicated) -> fused q|r|gate|k|v|sinks BF16 gemv, rows 1/3 -> LL all-gather
//   -> q/k/v/r clip, K/V ShortConv4, V RMS, FP16 KV store, rel table -> SWA-1024 split-K flash (replicated)
//   -> merge * sigmoid(gate) -> wo BF16 gemv rows 1/3 -> LL all-gather -> attn ShortConv4 + residual
//   -> MLP depth mix + RMSNorm -> FP32 router + top-8 -> SF4 gate/up (384-feature shard) + SITU
//   -> SF4 down (fixed-order expert sum) -> LL all-reduce -> MLP ShortConv4 + residual.
// 5 grid barriers + 3 in-kernel P2P exchanges per visit. Weights are random (timing gate, not numerics);
// every rank must end with a bit-identical residual, which checks that all exchanges delivered.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <chrono>
#include <cmath>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

constexpr int H = 2304, QD = 3072, KVD = 1024, NH = 24, NKV = 8, HD = 128, DREL = 16, REL = 1024, WIN = 1024;
constexpr int RING = WIN + 64, NE = 128, TOPK = 8, FS = 384, W = 3, NV = 48, NL = 6, NSP = 14;
constexpr int OQ = 0, OR_ = 3072, OG = 3456, OK_ = 6528, OV = 7552, OS = 8576, NQKV = 8600;
constexpr int WO_ROWS = H / W;                               // 768
constexpr size_t GU_RP = H / 2, GU_RS = H / 16;              // 1152 B packs, 144 B scales per gate/up row
constexpr size_t DN_RP = FS / 2, DN_RS = FS / 16;            // 192 B, 24 B per down row
constexpr size_t DN_OFF = 16 + 2 * FS * (GU_RP + GU_RS);     // 995344
constexpr size_t EXPB = DN_OFF + (size_t)H * (DN_RP + DN_RS);// 1493008
constexpr int NT = 256, NWARP = NT / 32, BUF = 3584;
constexpr float RMS_EPS = 1e-6f, QK_CLIP = 22.627416997969522f, REL_CLIP = 8.0f;

__host__ __device__ inline int qkv_row0(int r) { return (NQKV * r) / W; }

struct Layer {                       // per physical-layer weights on one rank
    const uint16_t* wd;              // [rows_r][H] BF16 (this rank's rows of q|r|gate|k|v|sinks)
    const uint16_t* wo;              // [768][QD] BF16
    const float* rel_proj;           // [16][1024]
    const float4 *sw_k, *sw_v, *sw_a, *sw_m;
    const float *q_res, *q_mlp, *nw_a, *nw_m;
    const float *rw, *rb;            // router [128][H], bias [128]
    const uint8_t* ex;               // 128 x EXPB
};
struct Visit { __half *k, *v; float *hk, *hv, *ha, *hm; };   // KV ring + double-buffered ShortConv histories
struct Params {
    Layer L[NL]; Visit V[NV];
    const float* bstate;             // [3][H] earlier block states (depth mix)
    float* cur0;                     // [H] residual in/out
    uint2* rB[W]; uint2* rF[W]; uint2* rM[W];  // LL receive buffers of every rank (peer-mapped)
    float *qc, *rel, *pm, *pl, *po, *ao, *lg, *hb, *out;
    unsigned* bar; long long* ts;
    int rank, pos, tok; unsigned ep0;
};

__device__ __forceinline__ void ll_put(uint2* p, float x, unsigned ep) {
    asm volatile("st.volatile.global.v2.u32 [%0], {%1,%2};" :: "l"(p), "r"(__float_as_uint(x)), "r"(ep) : "memory");
}
__device__ __forceinline__ float ll_get(const uint2* p, unsigned ep) {
    unsigned x, y;
    do { asm volatile("ld.volatile.global.v2.u32 {%0,%1}, [%2];" : "=r"(x), "=r"(y) : "l"(p)); } while (y != ep);
    return __uint_as_float(x);
}
__device__ __forceinline__ long long gtime() { long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ void gsync(unsigned* bar, unsigned& gen) {
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        if (atomicAdd(&bar[0], 1u) == gridDim.x - 1) { atomicExch(&bar[0], 0u); __threadfence(); atomicExch(&bar[1], gen + 1); }
        else { while (*(volatile unsigned*)&bar[1] == gen) {} }
        __threadfence();
    }
    ++gen;
    __syncthreads();
}
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    return v;
}
__device__ __forceinline__ float bf(uint32_t u, int hi) { return __uint_as_float(hi ? (u & 0xFFFF0000u) : (u << 16)); }
__device__ __forceinline__ float dot8_bf16(uint4 w, const float* x) {
    const float4 a = *reinterpret_cast<const float4*>(x), b = *reinterpret_cast<const float4*>(x + 4);
    float s = bf(w.x, 0) * a.x;
    s = fmaf(bf(w.x, 1), a.y, s); s = fmaf(bf(w.y, 0), a.z, s); s = fmaf(bf(w.y, 1), a.w, s);
    s = fmaf(bf(w.z, 0), b.x, s); s = fmaf(bf(w.z, 1), b.y, s); s = fmaf(bf(w.w, 0), b.z, s); s = fmaf(bf(w.w, 1), b.w, s);
    return s;
}
// --- engine helpers (copied: ink_cuda.cuh / ink_moe_cuda.cuh, SM<100 paths) ---
__device__ __forceinline__ float e4m3_to_f32(uint8_t b) {
    const uint32_t e = (b >> 3) & 15u, m = b & 7u;
    const float v = (e == 0) ? (float)m * 0.001953125f : __uint_as_float(((e + 120u) << 23) | (m << 20));
    const float r = (b & 0x80u) ? -v : v;
    return ((b & 0x7Fu) == 0x7Fu) ? 0.0f : r;
}
__device__ __forceinline__ float2 e4m3x2(uint16_t p) { return make_float2(e4m3_to_f32((uint8_t)(p & 0xFFu)), e4m3_to_f32((uint8_t)(p >> 8))); }
__device__ __forceinline__ void sf4_decode8_h2_prmt(uint32_t w, __half2* o) {
    const uint32_t m = w & 0x77777777u;
    const uint32_t ev = m & 0x07070707u, od = (m >> 4) & 0x07070707u;
    const uint32_t se = ev | (ev >> 4), so = od | (od >> 4);
    const uint32_t E = __byte_perm(0x3E3C3800u, 0x46444240u, (se & 0xFFu) | ((se >> 8) & 0xFF00u));
    const uint32_t O = __byte_perm(0x3E3C3800u, 0x46444240u, (so & 0xFFu) | ((so >> 8) & 0xFF00u));
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t h = __byte_perm(E, O, ((4u + k) << 12) | (k << 4)) & 0xFF00FF00u;
        const uint32_t b = w >> (8 * k);
        const uint32_t v = h | ((b & 0x08u) << 12) | ((b & 0x80u) << 24);
        o[k] = *reinterpret_cast<const __half2*>(&v);
    }
}
__device__ __forceinline__ float dot16_h2(const __half2 (&w)[8], const __half2* xh2) {
    const uint4 v0 = *reinterpret_cast<const uint4*>(xh2), v1 = *reinterpret_cast<const uint4*>(xh2 + 4);
    const __half2* x0 = reinterpret_cast<const __half2*>(&v0);
    const __half2* x1 = reinterpret_cast<const __half2*>(&v1);
    __half2 a0 = __hmul2(w[0], x0[0]), a1 = __hmul2(w[1], x0[1]);
    a0 = __hfma2(w[2], x0[2], a0); a1 = __hfma2(w[3], x0[3], a1);
    a0 = __hfma2(w[4], x1[0], a0); a1 = __hfma2(w[5], x1[1], a1);
    a0 = __hfma2(w[6], x1[2], a0); a1 = __hfma2(w[7], x1[3], a1);
    const float2 f = __half22float2(__hadd2(a0, a1));
    return f.x + f.y;
}
__device__ __forceinline__ float situ(float g, float u) {
    if (!isfinite(g) || !isfinite(u)) return 0.0f;
    const float gc = fminf(fmaxf(g, -80.0f), 80.0f);
    const float sig = 1.0f / (1.0f + __expf(-gc));
    return 4.0f * tanhf(g * 0.25f) * sig * (25.0f * tanhf(u * 0.04f));
}

// Block reduction of NV_ values per thread (all NT threads).
template <int NV_>
__device__ __forceinline__ void block_reduce(float (&v)[NV_], float* sh /*[NWARP*NV_]*/) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
    for (int i = 0; i < NV_; ++i) { const float x = wsum(v[i]); if (lane == 0) sh[warp * NV_ + i] = x; }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < NV_; ++i) { float a = 0.f; for (int w = 0; w < NWARP; ++w) a += sh[w * NV_ + i]; v[i] = a; }
    __syncthreads();
}

// Depth mix over 3 stored block states + the current residual, then RMSNorm * nw (replicated in every CTA).
__device__ void depth_mix(const float* bstate, const float* cur, const float* q, const float* nw, float* out, float* red) {
    constexpr int S = 4, PER = H / NT;   // 9 elements per thread
    float acc[2 * S] = {};
    float xv[S][PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int c = threadIdx.x + NT * j;
        const float qq = __ldg(q + c);
#pragma unroll
        for (int s = 0; s < S; ++s) {
            const float x = (s < 3) ? __ldg(bstate + s * H + c) : cur[c];
            xv[s][j] = x; acc[2 * s] = fmaf(x, x, acc[2 * s]); acc[2 * s + 1] = fmaf(x, qq, acc[2 * s + 1]);
        }
    }
    block_reduce<2 * S>(acc, red);
    float p[S], inv[S], mx = -1e30f, sum = 0.f;
#pragma unroll
    for (int s = 0; s < S; ++s) { inv[s] = rsqrtf(acc[2 * s] / H + RMS_EPS); p[s] = acc[2 * s + 1] * inv[s] * rsqrtf((float)H); mx = fmaxf(mx, p[s]); }
#pragma unroll
    for (int s = 0; s < S; ++s) { p[s] = __expf(p[s] - mx); sum += p[s]; }
    float ss[1] = {0.f}, mv[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        float a = 0.f;
#pragma unroll
        for (int s = 0; s < S; ++s) a = fmaf(p[s] / sum * inv[s], xv[s][j], a);
        mv[j] = a; ss[0] = fmaf(a, a, ss[0]);
    }
    block_reduce<1>(ss, red);
    const float r = rsqrtf(ss[0] / H + RMS_EPS);
#pragma unroll
    for (int j = 0; j < PER; ++j) { const int c = threadIdx.x + NT * j; out[c] = mv[j] * r * __ldg(nw + c); }
    __syncthreads();
}

__global__ void __launch_bounds__(NT, 2) k_mega(Params P) {
    __shared__ alignas(16) float s_cur[H];
    __shared__ alignas(16) float buf[BUF];
    __shared__ float red[NWARP * 8];
    __shared__ float s_w[TOPK];
    __shared__ int s_id[TOPK];
    __shared__ float sm_m[NWARP], sm_l[NWARP];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gw = blockIdx.x * NWARP + warp, nw = gridDim.x * NWARP;
    unsigned gen = *(volatile unsigned*)&P.bar[1];
    const int R = P.rank, par = P.tok & 1, pos = P.pos;
    const bool ts_on = (blockIdx.x == 0 && tid == 0);
    for (int c = tid; c < H; c += NT) s_cur[c] = P.cur0[c];
    __syncthreads();

    for (int v = 0; v < NV; ++v) {
        const Layer& L = P.L[v % NL];
        const Visit& Vi = P.V[v];
        const unsigned ep = P.ep0 + (unsigned)v + 1u;
        const int slot = v & 3;
        uint2* myB = P.rB[R] + (size_t)slot * NQKV;
        uint2* myF = P.rF[R] + (size_t)slot * H;
        uint2* myM = P.rM[R] + (size_t)slot * W * H;
        const bool tsv = ts_on && v == 20;
        if (tsv) P.ts[0] = gtime();

        // A: attention depth mix + RMSNorm -> buf
        depth_mix(P.bstate, s_cur, L.q_res, L.nw_a, buf, red);
        if (tsv) P.ts[1] = gtime();

        // B: fused q|r|gate|k|v|sinks gemv over this rank's rows -> LL all-gather
        {
            const int r0 = qkv_row0(R), nrows = qkv_row0(R + 1) - r0;
            for (int row = gw; row < nrows; row += nw) {
                const uint4* wr = reinterpret_cast<const uint4*>(L.wd + (size_t)row * H);
                uint4 wv[9];
#pragma unroll
                for (int j = 0; j < 9; ++j) wv[j] = __ldg(wr + lane + 32 * j);
                float s = 0.f;
#pragma unroll
                for (int j = 0; j < 9; ++j) s += dot8_bf16(wv[j], buf + (lane + 32 * j) * 8);
                s = wsum(s);
                if (lane < W) ll_put(P.rB[lane] + (size_t)slot * NQKV + r0 + row, s, ep);
            }
        }
        if (tsv) P.ts[2] = gtime();

        // C: post-QKV (64 items as k_ink_post_qkv_fused) + rel table
        for (int it = blockIdx.x; it < 88; it += gridDim.x) {
            if (it < 24) {                     // q clip (tau = 1 on SWA)
                float x[1] = {0.f}; float val = 0.f;
                if (tid < HD) { val = ll_get(myB + OQ + it * HD + tid, ep); x[0] = val * val; }
                block_reduce<1>(x, red);
                if (tid < HD) P.qc[it * HD + tid] = val * fminf(1.f, QK_CLIP / (sqrtf(x[0]) + 1e-6f));
            } else if (it < 40) {              // K/V ShortConv4 + clip / RMS + FP16 KV store
                const bool is_k = it < 32;
                const int kh = is_k ? it - 24 : it - 32;
                float x[1] = {0.f}; float val = 0.f;
                if (tid < HD) {
                    const int c = kh * HD + tid;
                    const float cur = ll_get(myB + (is_k ? OK_ : OV) + c, ep);
                    const float* hr = (is_k ? Vi.hk : Vi.hv) + (size_t)par * 3 * KVD;
                    float* hw = (is_k ? Vi.hk : Vi.hv) + (size_t)(par ^ 1) * 3 * KVD;
                    const float h0 = hr[c], h1 = hr[KVD + c], h2 = hr[2 * KVD + c];
                    hw[c] = h1; hw[KVD + c] = h2; hw[2 * KVD + c] = cur;
                    const float4 sw = __ldg((is_k ? L.sw_k : L.sw_v) + c);
                    val = (cur + (h0 * sw.x + h1 * sw.y + h2 * sw.z + cur * sw.w)) * 0.44721359549995793f;
                    x[0] = val * val;
                }
                block_reduce<1>(x, red);
                if (tid < HD) {
                    val *= is_k ? fminf(1.f, QK_CLIP / (sqrtf(x[0]) + 1e-6f)) : rsqrtf(x[0] / HD + RMS_EPS);
                    __half* dst = (is_k ? Vi.k : Vi.v) + ((size_t)(pos % RING) * NKV + kh) * HD + tid;
                    *dst = __float2half_rn(fminf(fmaxf(val, -65504.f), 65504.f));
                }
            } else {                           // r clip + relative-position table row
                const int h = (it - 40) >> 1, d0 = ((it - 40) & 1) * (REL / 2);
                float* sr = red;               // 16 floats
                __syncthreads();
                if (tid < 32) {
                    const float rv = (lane < DREL) ? ll_get(myB + OR_ + h * DREL + lane, ep) : 0.f;
                    float ss = rv * rv;
#pragma unroll
                    for (int o = 8; o >= 1; o >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, o);
                    if (lane < DREL) sr[lane] = rv * fminf(1.f, REL_CLIP / (sqrtf(ss) + 1e-6f));
                }
                __syncthreads();
                float* dst = P.rel + (size_t)h * (REL + 1);
                for (int d = d0 + tid; d < d0 + REL / 2; d += NT) {
                    float a = 0.f;
#pragma unroll
                    for (int c = 0; c < DREL; ++c) a = fmaf(sr[c], __ldg(L.rel_proj + c * REL + d), a);
                    dst[d] = a;
                }
                if (tid == 0 && d0) dst[REL] = 0.f;
                __syncthreads();
            }
        }
        if (tsv) P.ts[3] = gtime();
        gsync(P.bar, gen);
        if (tsv) P.ts[4] = gtime();

        // D: SWA split-K flash attention, 8 KV heads x 16 splits, 3 Q heads per KV head
        for (int it = blockIdx.x; it < NKV * NSP; it += gridDim.x) {
            const int kh = it / NSP, sp = it % NSP;
            const float sc = 0.08838834764831845f;
            float4 qv[3]; const float* rt[3];
#pragma unroll
            for (int r = 0; r < 3; ++r) {
                qv[r] = __ldcg(reinterpret_cast<const float4*>(P.qc + (3 * kh + r) * HD) + lane);
                rt[r] = P.rel + (size_t)(3 * kh + r) * (REL + 1);
            }
            float mw[3], lw[3], a[3][4] = {};
#pragma unroll
            for (int r = 0; r < 3; ++r) {
                const bool f = (sp == 0 && warp == 0);
                mw[r] = f ? ll_get(myB + OS + 3 * kh + r, ep) : -1e30f;
                lw[r] = f ? 1.f : 0.f;
            }
            const int k_start = pos - WIN + 1, per = (WIN + NSP - 1) / NSP, s_beg = k_start + sp * per, s_end = min(k_start + WIN, s_beg + per);
            for (int kp = s_beg + warp; kp < s_end; kp += NWARP) {
                const size_t o = ((size_t)(kp % RING) * NKV + kh) * HD + lane * 4;
                const uint2 kr = __ldcg(reinterpret_cast<const uint2*>(Vi.k + o));
                const uint2 vr = __ldcg(reinterpret_cast<const uint2*>(Vi.v + o));
                const float2 k01 = __half22float2(*reinterpret_cast<const __half2*>(&kr.x));
                const float2 k23 = __half22float2(*reinterpret_cast<const __half2*>(&kr.y));
                const float2 v01 = __half22float2(*reinterpret_cast<const __half2*>(&vr.x));
                const float2 v23 = __half22float2(*reinterpret_cast<const __half2*>(&vr.y));
                const int dist = min(max(pos - kp, 0), REL);
#pragma unroll
                for (int r = 0; r < 3; ++r) {
                    const float s = wsum(qv[r].x * k01.x + qv[r].y * k01.y + qv[r].z * k23.x + qv[r].w * k23.y) * sc + __ldcg(rt[r] + dist);
                    const float mn = fmaxf(mw[r], s), al = __expf(mw[r] - mn), p = __expf(s - mn);
                    lw[r] = lw[r] * al + p;
                    a[r][0] = a[r][0] * al + p * v01.x; a[r][1] = a[r][1] * al + p * v01.y;
                    a[r][2] = a[r][2] * al + p * v23.x; a[r][3] = a[r][3] * al + p * v23.y;
                    mw[r] = mn;
                }
            }
            float* sa = buf;   // [8][128]
#pragma unroll
            for (int r = 0; r < 3; ++r) {
                if (lane == 0) { sm_m[warp] = mw[r]; sm_l[warp] = lw[r]; }
#pragma unroll
                for (int q = 0; q < 4; ++q) sa[warp * HD + lane * 4 + q] = a[r][q];
                __syncthreads();
                if (tid < HD) {
                    float mg = -1e30f;
                    for (int w = 0; w < NWARP; ++w) mg = fmaxf(mg, sm_m[w]);
                    float lt = 0.f, ac = 0.f;
                    for (int w = 0; w < NWARP; ++w) {
                        const float wt = sm_l[w] > 0.f ? __expf(sm_m[w] - mg) : 0.f;
                        lt += sm_l[w] * wt; ac = fmaf(sa[w * HD + tid], wt, ac);
                    }
                    const size_t ix = (size_t)(3 * kh + r) * NSP + sp;
                    if (tid == 0) { P.pm[ix] = mg; P.pl[ix] = lt; }
                    P.po[ix * HD + tid] = ac;
                }
                __syncthreads();
            }
        }
        if (tsv) P.ts[5] = gtime();
        gsync(P.bar, gen);
        if (tsv) P.ts[6] = gtime();

        // E: merge splits * sigmoid(gate_attn)
        for (int h = blockIdx.x; h < NH; h += gridDim.x) {
            if (tid < HD) {
                const size_t b = (size_t)h * NSP;
                float mg = -1e30f;
                for (int s = 0; s < NSP; ++s) if (__ldcg(P.pl + b + s) > 0.f) mg = fmaxf(mg, __ldcg(P.pm + b + s));
                float lt = 0.f, ac = 0.f;
                for (int s = 0; s < NSP; ++s) {
                    const float ls = __ldcg(P.pl + b + s);
                    if (ls > 0.f) { const float wt = __expf(__ldcg(P.pm + b + s) - mg); lt += ls * wt; ac = fmaf(__ldcg(P.po + (b + s) * HD + tid), wt, ac); }
                }
                const float g = ll_get(myB + OG + h * HD + tid, ep);
                P.ao[h * HD + tid] = ac / lt / (1.f + __expf(-g));
            }
        }
        if (tsv) P.ts[7] = gtime();
        gsync(P.bar, gen);
        if (tsv) P.ts[8] = gtime();

        // F: wo gemv over this rank's 768 output rows -> LL all-gather
        for (int i = tid; i < QD / 4; i += NT) reinterpret_cast<float4*>(buf)[i] = __ldcg(reinterpret_cast<const float4*>(P.ao) + i);
        __syncthreads();
        for (int row = gw; row < WO_ROWS; row += nw) {
            const uint4* wr = reinterpret_cast<const uint4*>(L.wo + (size_t)row * QD);
            uint4 wv[12];
#pragma unroll
            for (int j = 0; j < 12; ++j) wv[j] = __ldg(wr + lane + 32 * j);
            float s = 0.f;
#pragma unroll
            for (int j = 0; j < 12; ++j) s += dot8_bf16(wv[j], buf + (lane + 32 * j) * 8);
            s = wsum(s);
            if (lane < W) ll_put(P.rF[lane] + (size_t)slot * H + R * WO_ROWS + row, s, ep);
        }
        if (tsv) P.ts[9] = gtime();

        // G: attention-output ShortConv4 + residual (replicated; history written by its owner CTA)
        for (int c = tid; c < H; c += NT) {
            const float x = ll_get(myF + c, ep);
            const float* hr = Vi.ha + (size_t)par * 3 * H;
            const float h0 = hr[c], h1 = hr[H + c], h2 = hr[2 * H + c];
            const float4 sw = __ldg(L.sw_a + c);
            s_cur[c] += (x + (h0 * sw.x + h1 * sw.y + h2 * sw.z + x * sw.w)) * 0.44721359549995793f;
            if (c % gridDim.x == blockIdx.x) { float* hw = Vi.ha + (size_t)(par ^ 1) * 3 * H; hw[c] = h1; hw[H + c] = h2; hw[2 * H + c] = x; }
        }
        __syncthreads();
        if (tsv) P.ts[10] = gtime();

        // H: MLP depth mix + RMSNorm -> buf[0..H) (FP32) and buf[H..) (half2 copy for the experts)
        depth_mix(P.bstate, s_cur, L.q_mlp, L.nw_m, buf, red);
        __half2* xh = reinterpret_cast<__half2*>(buf + H);
        for (int i = tid; i < H / 2; i += NT)
            xh[i] = __floats2half2_rn(fminf(fmaxf(buf[2 * i], -65504.f), 65504.f), fminf(fmaxf(buf[2 * i + 1], -65504.f), 65504.f));
        __syncthreads();
        if (tsv) P.ts[11] = gtime();

        // I: router logits, one CTA per expert (8 warps split the dot)
        for (int e = blockIdx.x; e < NE; e += gridDim.x) {
            const float4* rw = reinterpret_cast<const float4*>(L.rw + (size_t)e * H) + warp * 72;
            const float4* xv = reinterpret_cast<const float4*>(buf) + warp * 72;
            float a = 0.f;
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                const int i = lane + 32 * j;
                if (i < 72) { const float4 w4 = __ldg(rw + i), x4 = xv[i]; a = fmaf(w4.x, x4.x, a); a = fmaf(w4.y, x4.y, a); a = fmaf(w4.z, x4.z, a); a = fmaf(w4.w, x4.w, a); }
            }
            float t[1] = {a};
            block_reduce<1>(t, red);
            if (tid == 0) P.lg[e] = t[0];
        }
        if (tsv) P.ts[12] = gtime();
        gsync(P.bar, gen);
        if (tsv) P.ts[13] = gtime();

        // J: top-8 (sigmoid + bias select, normalized * route_scale), replicated
        if (warp == 0) {
            float sc[4], pr[4]; int id[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) { const int e = lane + 32 * j; const float p = 1.f / (1.f + expf(-__ldcg(P.lg + e))); id[j] = e; pr[j] = p; sc[j] = p + __ldg(L.rb + e); }
            float psum = 0.f; int bid = 0; float bp = 0.f;
            for (int k = 0; k < TOPK; ++k) {
                float mv = -1e30f; int mi = 1000000; float mp = 0.f;
#pragma unroll
                for (int j = 0; j < 4; ++j) if (sc[j] > mv || (sc[j] == mv && id[j] < mi)) { mv = sc[j]; mi = id[j]; mp = pr[j]; }
#pragma unroll
                for (int o = 16; o >= 1; o >>= 1) {
                    const float ov = __shfl_xor_sync(0xFFFFFFFFu, mv, o); const int oi = __shfl_xor_sync(0xFFFFFFFFu, mi, o); const float op = __shfl_xor_sync(0xFFFFFFFFu, mp, o);
                    if (ov > mv || (ov == mv && oi < mi)) { mv = ov; mi = oi; mp = op; }
                }
                if (lane == k) { bid = mi; bp = mp; }
                psum += mp;
#pragma unroll
                for (int j = 0; j < 4; ++j) if (id[j] == mi) sc[j] = -1e30f;
            }
            if (lane < TOPK) { s_id[lane] = bid; s_w[lane] = bp * (8.0f / (psum + 1e-20f)); }
        }
        __syncthreads();
        if (tsv) P.ts[14] = gtime();

        // K: SF4 gate/up on this rank's 384-feature shard of the 8 experts + SITU (2 features per warp in flight)
        for (int it0 = gw; it0 < TOPK * FS; it0 += 2 * nw) {
            uint4 gv[2][3], uv[2][3]; uint16_t gsr[2][3], usr[2][3]; float wout[2];
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int it = min(it0 + u * nw, TOPK * FS - 1);
                const int k = it / FS, f = it % FS;
                const uint8_t* body = L.ex + (size_t)s_id[k] * EXPB;
                const float2 h01 = *reinterpret_cast<const float2*>(body);
                wout[u] = h01.x / h01.y;
                const uint8_t* bp = body + 16;
                const uint8_t* bs = bp + (size_t)(2 * FS) * GU_RP;
                const uint4* gp = reinterpret_cast<const uint4*>(bp + (size_t)(2 * f) * GU_RP);
                const uint4* up = reinterpret_cast<const uint4*>(bp + (size_t)(2 * f + 1) * GU_RP);
                const uint16_t* gs = reinterpret_cast<const uint16_t*>(bs + (size_t)(2 * f) * GU_RS);
                const uint16_t* us = reinterpret_cast<const uint16_t*>(bs + (size_t)(2 * f + 1) * GU_RS);
#pragma unroll
                for (int j = 0; j < 3; ++j) {
                    const int i = min(lane + 32 * j, 71);
                    gv[u][j] = __ldg(gp + i); uv[u][j] = __ldg(up + i); gsr[u][j] = __ldg(gs + i); usr[u][j] = __ldg(us + i);
                }
            }
#pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int it = it0 + u * nw;
                float ag = 0.f, au = 0.f;
#pragma unroll
                for (int j = 0; j < 3; ++j) {
                    const int i = lane + 32 * j;
                    if (i < 72) {
                        const float2 sg = e4m3x2(gsr[u][j]), su = e4m3x2(usr[u][j]);
                        const __half2* xx = xh + i * 16;
                        __half2 g8[8], u8[8];
                        sf4_decode8_h2_prmt(gv[u][j].x, g8); sf4_decode8_h2_prmt(gv[u][j].y, g8 + 4);
                        sf4_decode8_h2_prmt(uv[u][j].x, u8); sf4_decode8_h2_prmt(uv[u][j].y, u8 + 4);
                        ag = fmaf(dot16_h2(g8, xx), sg.x, ag); au = fmaf(dot16_h2(u8, xx), su.x, au);
                        sf4_decode8_h2_prmt(gv[u][j].z, g8); sf4_decode8_h2_prmt(gv[u][j].w, g8 + 4);
                        sf4_decode8_h2_prmt(uv[u][j].z, u8); sf4_decode8_h2_prmt(uv[u][j].w, u8 + 4);
                        ag = fmaf(dot16_h2(g8, xx + 8), sg.y, ag); au = fmaf(dot16_h2(u8, xx + 8), su.y, au);
                    }
                }
                ag = wsum(ag); au = wsum(au);
                if (lane == 0 && it < TOPK * FS) P.hb[it] = situ(ag * wout[u], au * wout[u]);
            }
        }
        if (tsv) P.ts[15] = gtime();
        gsync(P.bar, gen);
        if (tsv) P.ts[16] = gtime();

        // L: SF4 down: one warp per output row, lane = (expert k = lane/4, quarter q = lane%4), fixed-order sum
        {
            __half2* hh = reinterpret_cast<__half2*>(buf);   // [8][192]
            for (int i = tid; i < TOPK * FS / 2; i += NT) {
                const float2 f = __ldcg(reinterpret_cast<const float2*>(P.hb) + i);
                hh[i] = __floats2half2_rn(fminf(fmaxf(f.x, -65504.f), 65504.f), fminf(fmaxf(f.y, -65504.f), 65504.f));
            }
            __syncthreads();
            const int k = lane >> 2, q = lane & 3;
            const uint8_t* body = L.ex + (size_t)s_id[k] * EXPB;
            const float2 h23 = *reinterpret_cast<const float2*>(body + 8);
            const float scale = h23.x / h23.y * s_w[k];
            const uint8_t* bp = body + DN_OFF;
            const uint8_t* bs = bp + (size_t)H * DN_RP;
            for (int row0 = gw; row0 < H; row0 += 2 * nw) {
                uint4 vv[2][3]; uint16_t sr2[2][3];
#pragma unroll
                for (int u = 0; u < 2; ++u) {
                    const int row = min(row0 + u * nw, H - 1);
                    const uint4* p = reinterpret_cast<const uint4*>(bp + (size_t)row * DN_RP);
                    const uint16_t* scp = reinterpret_cast<const uint16_t*>(bs + (size_t)row * DN_RS);
#pragma unroll
                    for (int j = 0; j < 3; ++j) { vv[u][j] = __ldg(p + q + 4 * j); sr2[u][j] = __ldg(scp + q + 4 * j); }
                }
#pragma unroll
                for (int u = 0; u < 2; ++u) {
                    const int row = row0 + u * nw;
                    float y = 0.f;
#pragma unroll
                    for (int j = 0; j < 3; ++j) {
                        const int i = q + 4 * j;
                        const float2 s = e4m3x2(sr2[u][j]);
                        const __half2* h2 = hh + k * (FS / 2) + i * 16;
                        __half2 w8[8];
                        sf4_decode8_h2_prmt(vv[u][j].x, w8); sf4_decode8_h2_prmt(vv[u][j].y, w8 + 4);
                        y = fmaf(dot16_h2(w8, h2), s.x, y);
                        sf4_decode8_h2_prmt(vv[u][j].z, w8); sf4_decode8_h2_prmt(vv[u][j].w, w8 + 4);
                        y = fmaf(dot16_h2(w8, h2 + 8), s.y, y);
                    }
                    y += __shfl_xor_sync(0xFFFFFFFFu, y, 2);
                    y += __shfl_xor_sync(0xFFFFFFFFu, y, 1);
                    y *= scale;
                    float tot = 0.f;
#pragma unroll
                    for (int e = 0; e < TOPK; ++e) tot += __shfl_sync(0xFFFFFFFFu, y, 4 * e);
                    if (lane < W && row < H) ll_put(P.rM[lane] + (size_t)slot * W * H + R * H + row, tot, ep);
                }
            }
        }
        if (tsv) P.ts[17] = gtime();

        // M: all-reduce in rank order + MLP ShortConv4 + residual (replicated)
        for (int c = tid; c < H; c += NT) {
            float x = 0.f;
#pragma unroll
            for (int r = 0; r < W; ++r) x += ll_get(myM + r * H + c, ep);
            const float* hr = Vi.hm + (size_t)par * 3 * H;
            const float h0 = hr[c], h1 = hr[H + c], h2 = hr[2 * H + c];
            const float4 sw = __ldg(L.sw_m + c);
            s_cur[c] += (x + (h0 * sw.x + h1 * sw.y + h2 * sw.z + x * sw.w)) * 0.44721359549995793f;
            if (c % gridDim.x == blockIdx.x) { float* hw = Vi.hm + (size_t)(par ^ 1) * 3 * H; hw[c] = h1; hw[H + c] = h2; hw[2 * H + c] = x; }
        }
        __syncthreads();
        if (tsv) P.ts[18] = gtime();
    }
    if (blockIdx.x == 0) for (int c = tid; c < H; c += NT) P.out[c] = s_cur[c];
}

// ---------------- host ----------------
__global__ void k_fill_u32(uint32_t* p, size_t n, uint32_t seed, uint32_t mask, uint32_t orv) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 2654435761u ^ seed; x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
        p[i] = (x & mask) | orv;
    }
}
__global__ void k_fill_f32(float* p, size_t n, uint32_t seed, float scale, float bias) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 2654435761u ^ seed; x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
        p[i] = bias + scale * ((float)(x & 0xFFFFFF) / 16777216.0f - 0.5f);
    }
}
__global__ void k_fill_bf16(uint16_t* p, size_t n, size_t idx0, uint32_t seed, float scale) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const size_t g = idx0 + i;
        uint32_t x = (uint32_t)g * 2654435761u ^ (uint32_t)(g >> 32) ^ seed; x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
        const float f = scale * ((float)(x & 0xFFFFFF) / 16777216.0f - 0.5f);
        p[i] = (uint16_t)(__float_as_uint(f) >> 16);
    }
}
__global__ void k_fix_experts(uint8_t* ex) {   // headers {1,1,1,1}; E4M3 scales in a sane range (no NaN code)
    const int e = blockIdx.x;
    uint8_t* b = ex + (size_t)e * EXPB;
    if (threadIdx.x == 0) { float* h = reinterpret_cast<float*>(b); h[0] = h[1] = h[2] = h[3] = 1.f; }
    uint8_t* gs = b + 16 + (size_t)(2 * FS) * GU_RP;
    for (size_t i = threadIdx.x; i < (size_t)(2 * FS) * GU_RS; i += blockDim.x) gs[i] = 0x30 | (gs[i] & 0x07);
    uint8_t* ds = b + DN_OFF + (size_t)H * DN_RP;
    for (size_t i = threadIdx.x; i < (size_t)H * DN_RS; i += blockDim.x) ds[i] = 0x30 | (ds[i] & 0x07);
}

template <class T> T* dmal(size_t n) { T* p; CK(cudaMalloc(&p, n * sizeof(T))); return p; }
static void fill_f(float* p, size_t n, uint32_t seed, float sc, float b = 0.f) { k_fill_f32<<<256, 256>>>(p, n, seed, sc, b); }

int main(int argc, char** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 50;
    int ndev = 0; CK(cudaGetDeviceCount(&ndev));
    if (ndev < W) { fprintf(stderr, "need 3 GPUs\n"); return 1; }
    for (int i = 0; i < W; ++i) {
        cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, i));
        printf("rank %d: %s sm_%d%d %d SMs coop=%d\n", i, pr.name, pr.major, pr.minor, pr.multiProcessorCount, pr.cooperativeLaunch);
        CK(cudaSetDevice(i));
        for (int j = 0; j < W; ++j) if (j != i) { int ok = 0; CK(cudaDeviceCanAccessPeer(&ok, i, j)); if (!ok) { fprintf(stderr, "no P2P %d->%d\n", i, j); return 1; } CK(cudaDeviceEnablePeerAccess(j, 0)); }
    }
    Params P[W]; cudaStream_t st[W]; int grid = 0;
    for (int r = 0; r < W; ++r) {
        CK(cudaSetDevice(r)); CK(cudaStreamCreateWithFlags(&st[r], cudaStreamNonBlocking));
        Params& p = P[r]; memset(&p, 0, sizeof(p));
        const int rows = qkv_row0(r + 1) - qkv_row0(r);
        for (int l = 0; l < NL; ++l) {
            Layer& L = p.L[l];
            uint16_t* wd = dmal<uint16_t>((size_t)rows * H);
            k_fill_bf16<<<512, 256>>>(wd, (size_t)rows * H, (size_t)qkv_row0(r) * H, 1000u + l, 0.04f);
            uint16_t* wo = dmal<uint16_t>((size_t)WO_ROWS * QD);
            k_fill_bf16<<<512, 256>>>(wo, (size_t)WO_ROWS * QD, (size_t)r * WO_ROWS * QD, 2000u + l, 0.03f);
            float* rp = dmal<float>(DREL * REL); fill_f(rp, DREL * REL, 3000u + l, 0.2f);
            float* sw = dmal<float>(4 * (2 * KVD + 2 * H)); fill_f(sw, 4 * (2 * KVD + 2 * H), 4000u + l, 0.5f);
            float* vec = dmal<float>(4 * H); fill_f(vec, 2 * H, 5000u + l, 0.2f); fill_f(vec + 2 * H, 2 * H, 5100u + l, 0.2f, 1.f);
            float* rw = dmal<float>((size_t)NE * H + NE); fill_f(rw, (size_t)NE * H, 6000u + l, 0.05f); fill_f(rw + (size_t)NE * H, NE, 6100u + l, 0.1f);
            uint8_t* ex = dmal<uint8_t>((size_t)NE * EXPB);
            k_fill_u32<<<1024, 256>>>(reinterpret_cast<uint32_t*>(ex), (size_t)NE * EXPB / 4, 7000u + 97u * l + 13u * r, 0xFFFFFFFFu, 0u);
            k_fix_experts<<<NE, 256>>>(ex);
            L.wd = wd; L.wo = wo; L.rel_proj = rp;
            L.sw_k = reinterpret_cast<float4*>(sw); L.sw_v = L.sw_k + KVD; L.sw_a = L.sw_v + KVD; L.sw_m = L.sw_a + H;
            L.q_res = vec; L.q_mlp = vec + H; L.nw_a = vec + 2 * H; L.nw_m = vec + 3 * H;
            L.rw = rw; L.rb = rw + (size_t)NE * H; L.ex = ex;
        }
        for (int v = 0; v < NV; ++v) {
            Visit& V = p.V[v];
            V.k = dmal<__half>((size_t)RING * KVD); V.v = dmal<__half>((size_t)RING * KVD);
            k_fill_u32<<<256, 256>>>(reinterpret_cast<uint32_t*>(V.k), (size_t)RING * KVD / 2, 8000u + v, 0x37FF37FFu, 0u);
            k_fill_u32<<<256, 256>>>(reinterpret_cast<uint32_t*>(V.v), (size_t)RING * KVD / 2, 8500u + v, 0x37FF37FFu, 0u);
            float* hs = dmal<float>((size_t)2 * 3 * (2 * KVD + 2 * H)); fill_f(hs, (size_t)2 * 3 * (2 * KVD + 2 * H), 9000u + v, 0.2f);
            V.hk = hs; V.hv = hs + 6 * KVD; V.ha = hs + 12 * KVD; V.hm = V.ha + 6 * H;
        }
        float* bs = dmal<float>(3 * H); fill_f(bs, 3 * H, 11u, 2.f); p.bstate = bs;
        p.cur0 = dmal<float>(H); fill_f(p.cur0, H, 12u, 2.f);
        p.qc = dmal<float>(QD); p.rel = dmal<float>(NH * (REL + 1)); p.pm = dmal<float>(NH * NSP); p.pl = dmal<float>(NH * NSP);
        p.po = dmal<float>((size_t)NH * NSP * HD); p.ao = dmal<float>(QD); p.lg = dmal<float>(NE); p.hb = dmal<float>(TOPK * FS);
        p.out = dmal<float>(H); p.bar = dmal<unsigned>(2); CK(cudaMemset(p.bar, 0, 8)); p.ts = dmal<long long>(32);
        p.rB[r] = dmal<uint2>((size_t)4 * NQKV); p.rF[r] = dmal<uint2>((size_t)4 * H); p.rM[r] = dmal<uint2>((size_t)4 * W * H);
        CK(cudaMemset(p.rB[r], 0, (size_t)4 * NQKV * 8)); CK(cudaMemset(p.rF[r], 0, (size_t)4 * H * 8)); CK(cudaMemset(p.rM[r], 0, (size_t)4 * W * H * 8));
        p.rank = r;
        int occ = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_mega, NT, 0));
        cudaDeviceProp pr; CK(cudaGetDeviceProperties(&pr, r));
        grid = occ * pr.multiProcessorCount;
        if (r == 0) printf("occupancy %d CTAs/SM -> grid %d x %d threads\n", occ, grid, NT);
        CK(cudaDeviceSynchronize());
    }
    for (int r = 0; r < W; ++r) for (int q = 0; q < W; ++q) { P[r].rB[q] = P[q].rB[q]; P[r].rF[q] = P[q].rF[q]; P[r].rM[q] = P[q].rM[q]; }

    auto run_token = [&](int tok) {
        for (int r = 0; r < W; ++r) {
            CK(cudaSetDevice(r));
            P[r].tok = tok; P[r].pos = 4096 + tok; P[r].ep0 = (unsigned)tok * NV;
            void* args[] = {&P[r]};
            CK(cudaLaunchCooperativeKernel((void*)k_mega, grid, NT, args, 0, st[r]));
        }
    };
    int tok = 0;
    for (int i = 0; i < 5; ++i) run_token(tok++);
    for (int r = 0; r < W; ++r) { CK(cudaSetDevice(r)); CK(cudaStreamSynchronize(st[r])); }
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) run_token(tok++);
    for (int r = 0; r < W; ++r) { CK(cudaSetDevice(r)); CK(cudaStreamSynchronize(st[r])); }
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
    printf("tokens %d: %.1f us/token (48 SWA visits) -> %.1f us/visit\n", iters, us / iters, us / iters / NV);

    // consistency: every rank must hold a bit-identical residual
    std::vector<float> o[W];
    for (int r = 0; r < W; ++r) { CK(cudaSetDevice(r)); o[r].resize(H); CK(cudaMemcpy(o[r].data(), P[r].out, H * 4, cudaMemcpyDeviceToHost)); }
    int diff = 0, nonfin = 0;
    for (int c = 0; c < H; ++c) { if (memcmp(&o[0][c], &o[1][c], 4) || memcmp(&o[0][c], &o[2][c], 4)) ++diff; if (!std::isfinite(o[0][c])) ++nonfin; }
    printf("ranks bit-identical: %s (%d differing, %d non-finite), cur[0..3] = %g %g %g %g\n", diff ? "NO" : "yes", diff, nonfin, o[0][0], o[0][1], o[0][2], o[0][3]);

    long long ts[32]; CK(cudaSetDevice(0)); CK(cudaMemcpy(ts, P[0].ts, sizeof(ts), cudaMemcpyDeviceToHost));
    const char* nm[] = {"", "A mix+rms", "B qkv gemv", "C post-qkv+rel", "bar1", "D flash", "bar2", "E merge", "bar3",
                        "F wo gemv", "G sconv+res (wait F)", "H mlp mix", "I router", "bar4", "J top8", "K gate/up", "bar5", "L down", "M allreduce+res"};
    printf("rank0 CTA0 phase end times, visit 20 (us):");
    for (int i = 1; i <= 18; ++i) printf(" %s=%.1f", nm[i], (ts[i] - ts[i - 1]) / 1000.0);
    printf("\n");
    return 0;
}
