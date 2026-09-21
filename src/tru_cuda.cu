#include "tru_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

__constant__ uint32_t c_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

static const uint32_t host_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__device__ __forceinline__ uint32_t ror(uint32_t x, uint32_t n) {
    return __funnelshift_r(x, x, n);
}

// 1-cycle 3-input bitwise boolean operations for Maxwell/Pascal/Turing/Ampere/Ada/Blackwell
__device__ __forceinline__ uint32_t lop3_ch(uint32_t e, uint32_t f, uint32_t g) {
    uint32_t ret;
    asm("lop3.b32 %0, %1, %2, %3, 0xca;" : "=r"(ret) : "r"(e), "r"(f), "r"(g));
    return ret;
}

__device__ __forceinline__ uint32_t lop3_maj(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t ret;
    asm("lop3.b32 %0, %1, %2, %3, 0xe8;" : "=r"(ret) : "r"(a), "r"(b), "r"(c));
    return ret;
}

__device__ __forceinline__ uint32_t lop3_xor3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t ret;
    asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(ret) : "r"(a), "r"(b), "r"(c));
    return ret;
}

// Dual-stream asynchronous pipeline buffers
static cudaStream_t g_streams[2] = {nullptr, nullptr};
static int* d_found[2] = {nullptr, nullptr};
static uint32_t* d_found_nonce[2] = {nullptr, nullptr};
static uint8_t* d_found_hash[2] = {nullptr, nullptr};
static uint32_t* d_midstate_both[2] = {nullptr, nullptr};
static uint8_t* d_target[2] = {nullptr, nullptr};
static int current_stream_idx = 0;

// Cache template parameters to avoid redundant PCIe transfers
static uint32_t cached_midstate[8] = {0};
static uint8_t  cached_header80_slice[12] = {0}; // bytes 64..75 (w0, w1, w2)
static uint8_t  cached_target[32] = {0};
static bool     template_initialized = false;

// -------------------------------------------------------------
// 2-WAY INSTRUCTION-LEVEL PARALLEL (ILP) TRU MINER KERNEL
// - 1 thread = 2 nonces in lockstep to hide ALU execution latency
// - Hardware 1-cycle lop3.b32 boolean logic (Ch, Maj, XOR3)
// - Hardware 1-cycle __byte_perm for endianness swap
// - Pass 1: Precomputed Rounds 0..2 on host -> start at Round 3
// - Pass 2: Dual Round 60 Early Abort (aborts 99.999999% nonces)
// -------------------------------------------------------------
extern "C" __global__ void tru_blackwell_kernel_2way(
    const uint32_t* __restrict__ midstate_both,
    uint32_t w0,
    uint32_t w1,
    uint32_t w2,
    uint32_t start_nonce,
    uint32_t batch_size,
    const uint8_t* __restrict__ target,
    int* __restrict__ found,
    uint32_t* __restrict__ found_nonce,
    uint8_t* __restrict__ found_hash
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t idx0 = tid * 2;
    if (idx0 >= batch_size) return;
    if (*found) return;

    uint32_t nonce0 = start_nonce + idx0;
    uint32_t nonce1 = nonce0 + 1;

    uint32_t w3_0 = __byte_perm(nonce0, 0, 0x0123);
    uint32_t w3_1 = __byte_perm(nonce1, 0, 0x0123);

    // =================================================================
    // PASS 1: 2-Way ILP Rounds 3..63
    // =================================================================
    uint32_t W0[16], W1[16];
    W0[0] = w0; W1[0] = w0;
    W0[1] = w1; W1[1] = w1;
    W0[2] = w2; W1[2] = w2;
    W0[3] = w3_0; W1[3] = w3_1;
    W0[4] = 0x80000000; W1[4] = 0x80000000;
    #pragma unroll
    for (int i = 5; i < 15; i++) {
        W0[i] = 0; W1[i] = 0;
    }
    W0[15] = 0x00000280; W1[15] = 0x00000280;

    uint32_t a0 = midstate_both[8], a1 = midstate_both[8];
    uint32_t b0 = midstate_both[9], b1 = midstate_both[9];
    uint32_t c0 = midstate_both[10], c1 = midstate_both[10];
    uint32_t d0 = midstate_both[11], d1 = midstate_both[11];
    uint32_t e0 = midstate_both[12], e1 = midstate_both[12];
    uint32_t f0 = midstate_both[13], f1 = midstate_both[13];
    uint32_t g0 = midstate_both[14], g1 = midstate_both[14];
    uint32_t h0 = midstate_both[15], h1 = midstate_both[15];

    #pragma unroll
    for (int i = 3; i < 64; i++) {
        if (i >= 16) {
            uint32_t w0_i15 = W0[(i - 15) & 15];
            uint32_t w1_i15 = W1[(i - 15) & 15];
            uint32_t w0_i2  = W0[(i - 2)  & 15];
            uint32_t w1_i2  = W1[(i - 2)  & 15];

            uint32_t s0_0 = lop3_xor3(ror(w0_i15, 7),  ror(w0_i15, 18), (w0_i15 >> 3));
            uint32_t s0_1 = lop3_xor3(ror(w1_i15, 7),  ror(w1_i15, 18), (w1_i15 >> 3));

            uint32_t s1_0 = lop3_xor3(ror(w0_i2, 17),  ror(w0_i2, 19),  (w0_i2 >> 10));
            uint32_t s1_1 = lop3_xor3(ror(w1_i2, 17),  ror(w1_i2, 19),  (w1_i2 >> 10));

            W0[i & 15] = W0[(i - 16) & 15] + s0_0 + W0[(i - 7) & 15] + s1_0;
            W1[i & 15] = W1[(i - 16) & 15] + s0_1 + W1[(i - 7) & 15] + s1_1;
        }

        uint32_t S1_0 = lop3_xor3(ror(e0, 6), ror(e0, 11), ror(e0, 25));
        uint32_t S1_1 = lop3_xor3(ror(e1, 6), ror(e1, 11), ror(e1, 25));

        uint32_t ch_0 = lop3_ch(e0, f0, g0);
        uint32_t ch_1 = lop3_ch(e1, f1, g1);

        uint32_t tmp1_0 = h0 + S1_0 + ch_0 + c_K[i] + W0[i & 15];
        uint32_t tmp1_1 = h1 + S1_1 + ch_1 + c_K[i] + W1[i & 15];

        uint32_t S0_0 = lop3_xor3(ror(a0, 2), ror(a0, 13), ror(a0, 22));
        uint32_t S0_1 = lop3_xor3(ror(a1, 2), ror(a1, 13), ror(a1, 22));

        uint32_t maj_0 = lop3_maj(a0, b0, c0);
        uint32_t maj_1 = lop3_maj(a1, b1, c1);

        uint32_t tmp2_0 = S0_0 + maj_0;
        uint32_t tmp2_1 = S0_1 + maj_1;

        h0 = g0; g0 = f0; f0 = e0; e0 = d0 + tmp1_0;
        d0 = c0; c0 = b0; b0 = a0; a0 = tmp1_0 + tmp2_0;

        h1 = g1; g1 = f1; f1 = e1; e1 = d1 + tmp1_1;
        d1 = c1; c1 = b1; b1 = a1; a1 = tmp1_1 + tmp2_1;
    }

    uint32_t s1_0[8], s1_1[8];
    s1_0[0] = midstate_both[0] + a0; s1_1[0] = midstate_both[0] + a1;
    s1_0[1] = midstate_both[1] + b0; s1_1[1] = midstate_both[1] + b1;
    s1_0[2] = midstate_both[2] + c0; s1_1[2] = midstate_both[2] + c1;
    s1_0[3] = midstate_both[3] + d0; s1_1[3] = midstate_both[3] + d1;
    s1_0[4] = midstate_both[4] + e0; s1_1[4] = midstate_both[4] + e1;
    s1_0[5] = midstate_both[5] + f0; s1_1[5] = midstate_both[5] + f1;
    s1_0[6] = midstate_both[6] + g0; s1_1[6] = midstate_both[6] + g1;
    s1_0[7] = midstate_both[7] + h0; s1_1[7] = midstate_both[7] + h1;

    // =================================================================
    // PASS 2: 2-Way ILP Rounds 0..59 with Dual Round 60 Early Abort
    // =================================================================
    uint32_t w2_0[16], w2_1[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        w2_0[i] = s1_0[i];
        w2_1[i] = s1_1[i];
    }
    w2_0[8] = 0x80000000; w2_1[8] = 0x80000000;
    #pragma unroll
    for (int i = 9; i < 15; i++) {
        w2_0[i] = 0; w2_1[i] = 0;
    }
    w2_0[15] = 0x00000100; w2_1[15] = 0x00000100;

    a0 = 0x6a09e667; a1 = 0x6a09e667;
    b0 = 0xbb67ae85; b1 = 0xbb67ae85;
    c0 = 0x3c6ef372; c1 = 0x3c6ef372;
    d0 = 0xa54ff53a; d1 = 0xa54ff53a;
    e0 = 0x510e527f; e1 = 0x510e527f;
    f0 = 0x9b05688c; f1 = 0x9b05688c;
    g0 = 0x1f83d9ab; g1 = 0x1f83d9ab;
    h0 = 0x5be0cd19; h1 = 0x5be0cd19;

    #pragma unroll
    for (int i = 0; i < 60; i++) {
        if (i >= 16) {
            uint32_t w0_i15 = w2_0[(i - 15) & 15];
            uint32_t w1_i15 = w2_1[(i - 15) & 15];
            uint32_t w0_i2  = w2_0[(i - 2)  & 15];
            uint32_t w1_i2  = w2_1[(i - 2)  & 15];

            uint32_t s0_0 = lop3_xor3(ror(w0_i15, 7),  ror(w0_i15, 18), (w0_i15 >> 3));
            uint32_t s0_1 = lop3_xor3(ror(w1_i15, 7),  ror(w1_i15, 18), (w1_i15 >> 3));

            uint32_t s1_w0 = lop3_xor3(ror(w0_i2, 17),  ror(w0_i2, 19),  (w0_i2 >> 10));
            uint32_t s1_w1 = lop3_xor3(ror(w1_i2, 17),  ror(w1_i2, 19),  (w1_i2 >> 10));

            w2_0[i & 15] = w2_0[(i - 16) & 15] + s0_0 + w2_0[(i - 7) & 15] + s1_w0;
            w2_1[i & 15] = w2_1[(i - 16) & 15] + s0_1 + w2_1[(i - 7) & 15] + s1_w1;
        }

        uint32_t S1_0 = lop3_xor3(ror(e0, 6), ror(e0, 11), ror(e0, 25));
        uint32_t S1_1 = lop3_xor3(ror(e1, 6), ror(e1, 11), ror(e1, 25));

        uint32_t ch_0 = lop3_ch(e0, f0, g0);
        uint32_t ch_1 = lop3_ch(e1, f1, g1);

        uint32_t tmp1_0 = h0 + S1_0 + ch_0 + c_K[i] + w2_0[i & 15];
        uint32_t tmp1_1 = h1 + S1_1 + ch_1 + c_K[i] + w2_1[i & 15];

        uint32_t S0_0 = lop3_xor3(ror(a0, 2), ror(a0, 13), ror(a0, 22));
        uint32_t S0_1 = lop3_xor3(ror(a1, 2), ror(a1, 13), ror(a1, 22));

        uint32_t maj_0 = lop3_maj(a0, b0, c0);
        uint32_t maj_1 = lop3_maj(a1, b1, c1);

        uint32_t tmp2_0 = S0_0 + maj_0;
        uint32_t tmp2_1 = S0_1 + maj_1;

        h0 = g0; g0 = f0; f0 = e0; e0 = d0 + tmp1_0;
        d0 = c0; c0 = b0; b0 = a0; a0 = tmp1_0 + tmp2_0;

        h1 = g1; g1 = f1; f1 = e1; e1 = d1 + tmp1_1;
        d1 = c1; c1 = b1; b1 = a1; a1 = tmp1_1 + tmp2_1;
    }

    // Round 60 expansion
    {
        uint32_t w0_i15 = w2_0[(60 - 15) & 15];
        uint32_t w1_i15 = w2_1[(60 - 15) & 15];
        uint32_t w0_i2  = w2_0[(60 - 2)  & 15];
        uint32_t w1_i2  = w2_1[(60 - 2)  & 15];
        uint32_t s0_0 = lop3_xor3(ror(w0_i15, 7),  ror(w0_i15, 18), (w0_i15 >> 3));
        uint32_t s0_1 = lop3_xor3(ror(w1_i15, 7),  ror(w1_i15, 18), (w1_i15 >> 3));
        uint32_t s1_w0 = lop3_xor3(ror(w0_i2, 17),  ror(w0_i2, 19),  (w0_i2 >> 10));
        uint32_t s1_w1 = lop3_xor3(ror(w1_i2, 17),  ror(w1_i2, 19),  (w1_i2 >> 10));
        w2_0[60 & 15] = w2_0[(60 - 16) & 15] + s0_0 + w2_0[(60 - 7) & 15] + s1_w0;
        w2_1[60 & 15] = w2_1[(60 - 16) & 15] + s0_1 + w2_1[(60 - 7) & 15] + s1_w1;
    }

    uint32_t S1_60_0 = lop3_xor3(ror(e0, 6), ror(e0, 11), ror(e0, 25));
    uint32_t S1_60_1 = lop3_xor3(ror(e1, 6), ror(e1, 11), ror(e1, 25));
    uint32_t ch_60_0 = lop3_ch(e0, f0, g0);
    uint32_t ch_60_1 = lop3_ch(e1, f1, g1);
    uint32_t tmp1_60_0 = h0 + S1_60_0 + ch_60_0 + c_K[60] + w2_0[60 & 15];
    uint32_t tmp1_60_1 = h1 + S1_60_1 + ch_60_1 + c_K[60] + w2_1[60 & 15];

    uint32_t pred_s7_0 = 0x5be0cd19u + (d0 + tmp1_60_0) + 0x21E8u;
    uint32_t pred_s7_1 = 0x5be0cd19u + (d1 + tmp1_60_1) + 0x21E8u;

    // Fast target check candidate 0
    bool cand0 = true;
    uint32_t b31_0 = pred_s7_0 & 0xff;
    if (b31_0 > target[31]) cand0 = false;
    else if (b31_0 == target[31]) {
        uint32_t b30_0 = (pred_s7_0 >> 8) & 0xff;
        if (b30_0 > target[30]) cand0 = false;
        else if (b30_0 == target[30]) {
            uint32_t b29_0 = (pred_s7_0 >> 16) & 0xff;
            if (b29_0 > target[29]) cand0 = false;
            else if (b29_0 == target[29]) {
                if (((pred_s7_0 >> 24) & 0xff) > target[28]) cand0 = false;
            }
        }
    }

    // Fast target check candidate 1
    bool cand1 = true;
    uint32_t b31_1 = pred_s7_1 & 0xff;
    if (b31_1 > target[31]) cand1 = false;
    else if (b31_1 == target[31]) {
        uint32_t b30_1 = (pred_s7_1 >> 8) & 0xff;
        if (b30_1 > target[30]) cand1 = false;
        else if (b30_1 == target[30]) {
            uint32_t b29_1 = (pred_s7_1 >> 16) & 0xff;
            if (b29_1 > target[29]) cand1 = false;
            else if (b29_1 == target[29]) {
                if (((pred_s7_1 >> 24) & 0xff) > target[28]) cand1 = false;
            }
        }
    }

    // 99.999999% of pairs rejected immediately
    if (!cand0 && !cand1) return;

    // Finish candidate 0 if survived
    if (cand0) {
        uint32_t a = a0, b = b0, c = c0, d = d0, e = e0, f = f0, g = g0, h = h0;
        uint32_t S0_60 = lop3_xor3(ror(a, 2), ror(a, 13), ror(a, 22));
        uint32_t maj_60 = lop3_maj(a, b, c);
        uint32_t tmp2_60 = S0_60 + maj_60;
        h = g; g = f; f = e; e = d + tmp1_60_0;
        d = c; c = b; b = a; a = tmp1_60_0 + tmp2_60;

        #pragma unroll
        for (int i = 61; i < 64; i++) {
            uint32_t w_i15 = w2_0[(i - 15) & 15];
            uint32_t w_i2  = w2_0[(i - 2)  & 15];
            uint32_t s0 = lop3_xor3(ror(w_i15, 7),  ror(w_i15, 18), (w_i15 >> 3));
            uint32_t s1_w = lop3_xor3(ror(w_i2, 17),  ror(w_i2, 19),  (w_i2 >> 10));
            w2_0[i & 15] = w2_0[(i - 16) & 15] + s0 + w2_0[(i - 7) & 15] + s1_w;

            uint32_t S1 = lop3_xor3(ror(e, 6), ror(e, 11), ror(e, 25));
            uint32_t ch = lop3_ch(e, f, g);
            uint32_t tmp1 = h + S1 + ch + c_K[i] + w2_0[i & 15];
            uint32_t S0 = lop3_xor3(ror(a, 2), ror(a, 13), ror(a, 22));
            uint32_t maj = lop3_maj(a, b, c);
            uint32_t tmp2 = S0 + maj;

            h = g; g = f; f = e; e = d + tmp1;
            d = c; c = b; b = a; a = tmp1 + tmp2;
        }

        uint32_t s2[8];
        s2[0] = 0x6a09e667 + a; s2[1] = 0xbb67ae85 + b;
        s2[2] = 0x3c6ef372 + c; s2[3] = 0xa54ff53a + d;
        s2[4] = 0x510e527f + e; s2[5] = 0x9b05688c + f;
        s2[6] = 0x1f83d9ab + g; s2[7] = pred_s7_0;

        uint8_t finalHash[32];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            finalHash[4 * i]     = (uint8_t)((s2[i] >> 24) & 0xff);
            finalHash[4 * i + 1] = (uint8_t)((s2[i] >> 16) & 0xff);
            finalHash[4 * i + 2] = (uint8_t)((s2[i] >> 8) & 0xff);
            finalHash[4 * i + 3] = (uint8_t)(s2[i] & 0xff);
        }

        bool below = false;
        for (int i = 31; i >= 0; i--) {
            if (finalHash[i] < target[i]) { below = true; break; }
            if (finalHash[i] > target[i]) { below = false; break; }
        }
        if (below && atomicExch(found, 1) == 0) {
            *found_nonce = nonce0;
            for (int i = 0; i < 32; i++) found_hash[i] = finalHash[i];
            return;
        }
    }

    // Finish candidate 1 if survived
    if (cand1) {
        uint32_t a = a1, b = b1, c = c1, d = d1, e = e1, f = f1, g = g1, h = h1;
        uint32_t S0_60 = lop3_xor3(ror(a, 2), ror(a, 13), ror(a, 22));
        uint32_t maj_60 = lop3_maj(a, b, c);
        uint32_t tmp2_60 = S0_60 + maj_60;
        h = g; g = f; f = e; e = d + tmp1_60_1;
        d = c; c = b; b = a; a = tmp1_60_1 + tmp2_60;

        #pragma unroll
        for (int i = 61; i < 64; i++) {
            uint32_t w_i15 = w2_1[(i - 15) & 15];
            uint32_t w_i2  = w2_1[(i - 2)  & 15];
            uint32_t s0 = lop3_xor3(ror(w_i15, 7),  ror(w_i15, 18), (w_i15 >> 3));
            uint32_t s1_w = lop3_xor3(ror(w_i2, 17),  ror(w_i2, 19),  (w_i2 >> 10));
            w2_1[i & 15] = w2_1[(i - 16) & 15] + s0 + w2_1[(i - 7) & 15] + s1_w;

            uint32_t S1 = lop3_xor3(ror(e, 6), ror(e, 11), ror(e, 25));
            uint32_t ch = lop3_ch(e, f, g);
            uint32_t tmp1 = h + S1 + ch + c_K[i] + w2_1[i & 15];
            uint32_t S0 = lop3_xor3(ror(a, 2), ror(a, 13), ror(a, 22));
            uint32_t maj = lop3_maj(a, b, c);
            uint32_t tmp2 = S0 + maj;

            h = g; g = f; f = e; e = d + tmp1;
            d = c; c = b; b = a; a = tmp1 + tmp2;
        }

        uint32_t s2[8];
        s2[0] = 0x6a09e667 + a; s2[1] = 0xbb67ae85 + b;
        s2[2] = 0x3c6ef372 + c; s2[3] = 0xa54ff53a + d;
        s2[4] = 0x510e527f + e; s2[5] = 0x9b05688c + f;
        s2[6] = 0x1f83d9ab + g; s2[7] = pred_s7_1;

        uint8_t finalHash[32];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            finalHash[4 * i]     = (uint8_t)((s2[i] >> 24) & 0xff);
            finalHash[4 * i + 1] = (uint8_t)((s2[i] >> 16) & 0xff);
            finalHash[4 * i + 2] = (uint8_t)((s2[i] >> 8) & 0xff);
            finalHash[4 * i + 3] = (uint8_t)(s2[i] & 0xff);
        }

        bool below = false;
        for (int i = 31; i >= 0; i--) {
            if (finalHash[i] < target[i]) { below = true; break; }
            if (finalHash[i] > target[i]) { below = false; break; }
        }
        if (below && atomicExch(found, 1) == 0) {
            *found_nonce = nonce1;
            for (int i = 0; i < 32; i++) found_hash[i] = finalHash[i];
        }
    }
}

// Host helper to precompute Round 0-2 state
static void host_precompute_s1_r3(const uint32_t midstate[8], uint32_t w0, uint32_t w1, uint32_t w2, uint32_t out[8]) {
    uint32_t a = midstate[0], b = midstate[1], c = midstate[2], d = midstate[3];
    uint32_t e = midstate[4], f = midstate[5], g = midstate[6], h = midstate[7];
    uint32_t W[3] = {w0, w1, w2};

    for (int i = 0; i < 3; i++) {
        uint32_t S1 = ((e >> 6) | (e << 26)) ^ ((e >> 11) | (e << 21)) ^ ((e >> 25) | (e << 7));
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t tmp1 = h + S1 + ch + host_K[i] + W[i];
        uint32_t S0 = ((a >> 2) | (a << 30)) ^ ((a >> 13) | (a << 19)) ^ ((a >> 22) | (a << 10));
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t tmp2 = S0 + maj;
        h = g; g = f; f = e; e = d + tmp1;
        d = c; c = b; b = a; a = tmp1 + tmp2;
    }
    out[0] = a; out[1] = b; out[2] = c; out[3] = d;
    out[4] = e; out[5] = f; out[6] = g; out[7] = h;
}

extern "C" int cuda_miner_get_device_count(int* out_count) {
    if (!out_count) return -1;
    cudaError_t err = cudaGetDeviceCount(out_count);
    return (err == cudaSuccess) ? 0 : -1;
}

extern "C" int cuda_miner_get_device_info(int device_id, CudaDeviceInfo* out_info) {
    if (!out_info) return -1;
    memset(out_info, 0, sizeof(CudaDeviceInfo));

    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) return -1;

    strncpy(out_info->name, prop.name, sizeof(out_info->name) - 1);
    out_info->sm_count = prop.multiProcessorCount;
    out_info->major = prop.major;
    out_info->minor = prop.minor;
    out_info->total_memory = prop.totalGlobalMem;
    out_info->max_threads_per_block = prop.maxThreadsPerBlock;

    if (prop.major >= 12) {
        snprintf(out_info->arch_name, sizeof(out_info->arch_name), "Blackwell Native (sm_%d%d)", prop.major, prop.minor);
    } else if (prop.major == 9 && prop.minor == 0 && strstr(prop.name, "50")) {
        snprintf(out_info->arch_name, sizeof(out_info->arch_name), "Blackwell (Compute %d.%d)", prop.major, prop.minor);
    } else if (prop.major == 9 && prop.minor == 0) {
        snprintf(out_info->arch_name, sizeof(out_info->arch_name), "Hopper (sm_90)");
    } else if (prop.major == 8 && prop.minor == 9) {
        snprintf(out_info->arch_name, sizeof(out_info->arch_name), "Ada Lovelace (sm_89)");
    } else if (prop.major == 8 && prop.minor == 6) {
        snprintf(out_info->arch_name, sizeof(out_info->arch_name), "Ampere (sm_86)");
    } else if (prop.major == 8 && prop.minor == 0) {
        snprintf(out_info->arch_name, sizeof(out_info->arch_name), "Ampere A100 (sm_80)");
    } else if (prop.major == 7 && prop.minor == 5) {
        snprintf(out_info->arch_name, sizeof(out_info->arch_name), "Turing (sm_75)");
    } else {
        snprintf(out_info->arch_name, sizeof(out_info->arch_name), "CUDA Compute %d.%d", prop.major, prop.minor);
    }

    // Hardware-driven optimal batch size recommendation
    if (prop.major >= 12) {
        if (prop.multiProcessorCount >= 100) {
            out_info->recommended_batch_size = 134217728; // 128M nonces (RTX 5090 / 170 SM)
        } else if (prop.multiProcessorCount >= 50) {
            out_info->recommended_batch_size = 67108864;  // 64M nonces (RTX 5080 / 5070)
        } else {
            out_info->recommended_batch_size = 33554432;  // 32M nonces (RTX 5060)
        }
    } else if (prop.multiProcessorCount >= 100) {
        out_info->recommended_batch_size = 134217728; // 128M nonces
    } else if (prop.multiProcessorCount >= 60) {
        out_info->recommended_batch_size = 67108864;  // 64M nonces
    } else if (prop.multiProcessorCount >= 30) {
        out_info->recommended_batch_size = 33554432;  // 32M nonces
    } else {
        out_info->recommended_batch_size = 33554432;  // 32M nonces
    }

    return 0;
}

extern "C" int cuda_miner_init(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) return -1;

    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    for (int s = 0; s < 2; s++) {
        err = cudaStreamCreateWithFlags(&g_streams[s], cudaStreamNonBlocking);
        if (err != cudaSuccess) return -2;

        err = cudaMalloc(&d_found[s], sizeof(int));
        if (err != cudaSuccess) return -3;

        err = cudaMalloc(&d_found_nonce[s], sizeof(uint32_t));
        if (err != cudaSuccess) return -4;

        err = cudaMalloc(&d_found_hash[s], 32);
        if (err != cudaSuccess) return -5;

        err = cudaMalloc(&d_midstate_both[s], 16 * sizeof(uint32_t));
        if (err != cudaSuccess) return -6;

        err = cudaMalloc(&d_target[s], 32);
        if (err != cudaSuccess) return -7;

        cudaMemsetAsync(d_found[s], 0, sizeof(int), g_streams[s]);
    }

    template_initialized = false;
    current_stream_idx = 0;
    return 0;
}

extern "C" void cuda_miner_cleanup(void) {
    for (int s = 0; s < 2; s++) {
        if (d_found[s]) { cudaFree(d_found[s]); d_found[s] = nullptr; }
        if (d_found_nonce[s]) { cudaFree(d_found_nonce[s]); d_found_nonce[s] = nullptr; }
        if (d_found_hash[s]) { cudaFree(d_found_hash[s]); d_found_hash[s] = nullptr; }
        if (d_midstate_both[s]) { cudaFree(d_midstate_both[s]); d_midstate_both[s] = nullptr; }
        if (d_target[s]) { cudaFree(d_target[s]); d_target[s] = nullptr; }
        if (g_streams[s]) { cudaStreamDestroy(g_streams[s]); g_streams[s] = nullptr; }
    }
    template_initialized = false;
}

extern "C" int cuda_miner_search(
    const uint32_t midstate[8],
    const uint8_t header80[80],
    const uint8_t target[32],
    uint32_t start_nonce,
    uint32_t batch_size,
    CudaMiningResult* out_result
) {
    if (!d_found[0] || !out_result) return -1;

    uint32_t w0 = ((uint32_t)header80[64] << 24) | ((uint32_t)header80[65] << 16) |
                  ((uint32_t)header80[66] << 8)  | ((uint32_t)header80[67]);
    uint32_t w1 = ((uint32_t)header80[68] << 24) | ((uint32_t)header80[69] << 16) |
                  ((uint32_t)header80[70] << 8)  | ((uint32_t)header80[71]);
    uint32_t w2 = ((uint32_t)header80[72] << 24) | ((uint32_t)header80[73] << 16) |
                  ((uint32_t)header80[74] << 8)  | ((uint32_t)header80[75]);

    bool template_changed = !template_initialized ||
        memcmp(cached_midstate, midstate, 32) != 0 ||
        memcmp(cached_header80_slice, &header80[64], 12) != 0 ||
        memcmp(cached_target, target, 32) != 0;

    int s = current_stream_idx;

    if (template_changed) {
        cudaDeviceSynchronize();

        memcpy(cached_midstate, midstate, 32);
        memcpy(cached_header80_slice, &header80[64], 12);
        memcpy(cached_target, target, 32);
        template_initialized = true;

        uint32_t h_midstate_both[16];
        memcpy(&h_midstate_both[0], midstate, 32);
        host_precompute_s1_r3(midstate, w0, w1, w2, &h_midstate_both[8]);

        for (int st = 0; st < 2; st++) {
            cudaMemcpy(d_midstate_both[st], h_midstate_both, 64, cudaMemcpyHostToDevice);
            cudaMemcpy(d_target[st], target, 32, cudaMemcpyHostToDevice);
            cudaMemset(d_found[st], 0, sizeof(int));
        }
        cudaDeviceSynchronize();
    }

    const int threads_per_block = 256;
    // 2-Way ILP: each thread handles 2 nonces
    int num_blocks = ((batch_size / 2) + threads_per_block - 1) / threads_per_block;

    cudaMemsetAsync(d_found[s], 0, sizeof(int), g_streams[s]);

    tru_blackwell_kernel_2way<<<num_blocks, threads_per_block, 0, g_streams[s]>>>(
        d_midstate_both[s],
        w0, w1, w2,
        start_nonce,
        batch_size,
        d_target[s],
        d_found[s],
        d_found_nonce[s],
        d_found_hash[s]
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return -2;
    }

    cudaStreamSynchronize(g_streams[s]);

    cudaMemcpyAsync(&out_result->found, d_found[s], sizeof(int), cudaMemcpyDeviceToHost, g_streams[s]);
    cudaStreamSynchronize(g_streams[s]);

    if (out_result->found) {
        cudaMemcpyAsync(out_result->hash, d_found_hash[s], 32, cudaMemcpyDeviceToHost, g_streams[s]);
        cudaMemcpyAsync(&out_result->nonce, d_found_nonce[s], sizeof(uint32_t), cudaMemcpyDeviceToHost, g_streams[s]);
        cudaMemsetAsync(d_found[s], 0, sizeof(int), g_streams[s]);
        cudaStreamSynchronize(g_streams[s]);
    }

    current_stream_idx = 1 - current_stream_idx;
    return 0;
}
