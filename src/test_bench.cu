#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <chrono>

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

__constant__ uint32_t c_midstate[8];
__constant__ uint32_t c_s1_r3[8];
__constant__ uint8_t  c_target[32];
__constant__ uint32_t c_target_w7;

__device__ __forceinline__ uint32_t ror(uint32_t x, uint32_t n) {
    return __funnelshift_r(x, x, n);
}

// -------------------------------------------------------------
// 1. BASELINE KERNEL (Current Implementation)
// -------------------------------------------------------------
__device__ __forceinline__ void sha256_compress_baseline(uint32_t s[8], uint32_t W[16]) {
    uint32_t a = s[0], b = s[1], c = s[2], d = s[3];
    uint32_t e = s[4], f = s[5], g = s[6], h = s[7];

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        if (i >= 16) {
            uint32_t s0 = ror(W[(i - 15) & 15], 7) ^ ror(W[(i - 15) & 15], 18) ^ (W[(i - 15) & 15] >> 3);
            uint32_t s1 = ror(W[(i - 2) & 15], 17) ^ ror(W[(i - 2) & 15], 19) ^ (W[(i - 2) & 15] >> 10);
            W[i & 15] = W[(i - 16) & 15] + s0 + W[(i - 7) & 15] + s1;
        }
        uint32_t S1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t tmp1 = h + S1 + ch + c_K[i] + W[i & 15];
        uint32_t S0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t tmp2 = S0 + maj;

        h = g; g = f; f = e; e = d + tmp1;
        d = c; c = b; b = a; a = tmp1 + tmp2;
    }

    s[0] += a; s[1] += b; s[2] += c; s[3] += d;
    s[4] += e; s[5] += f; s[6] += g; s[7] += h;
}

__global__ void __launch_bounds__(256, 4) kernel_baseline(
    const uint32_t* __restrict__ midstate,
    uint32_t w0, uint32_t w1, uint32_t w2,
    uint32_t start_nonce,
    uint32_t nonce_count,
    const uint8_t* __restrict__ target,
    int* __restrict__ found,
    uint32_t* __restrict__ found_nonce,
    uint8_t* __restrict__ found_hash
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nonce_count || *found != 0) return;

    uint32_t nonce = start_nonce + gid;
    uint32_t nonce_le = ((nonce & 0xff) << 24) |
                        (((nonce >> 8) & 0xff) << 16) |
                        (((nonce >> 16) & 0xff) << 8) |
                        ((nonce >> 24) & 0xff);

    uint32_t s1[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) s1[i] = midstate[i];

    uint32_t w_chunk1[16];
    w_chunk1[0] = w0;
    w_chunk1[1] = w1;
    w_chunk1[2] = w2;
    w_chunk1[3] = nonce_le;
    w_chunk1[4] = 0x80000000;
    #pragma unroll
    for (int i = 5; i < 15; i++) w_chunk1[i] = 0;
    w_chunk1[15] = 0x00000280;

    sha256_compress_baseline(s1, w_chunk1);

    uint32_t s2[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    uint32_t w_chunk2[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) w_chunk2[i] = s1[i];
    w_chunk2[8] = 0x80000000;
    #pragma unroll
    for (int i = 9; i < 15; i++) w_chunk2[i] = 0;
    w_chunk2[15] = 0x00000100;

    sha256_compress_baseline(s2, w_chunk2);
    s2[7] += 0x21E8u;

    if ((s2[7] & 0xff) > target[31]) return;

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
    if (below) {
        if (atomicExch(found, 1) == 0) {
            *found_nonce = nonce;
            for (int i = 0; i < 32; i++) found_hash[i] = finalHash[i];
        }
    }
}

// -------------------------------------------------------------
// 2. ULTRA-OPTIMIZED KERNEL (Round 0-2 Skip + Round 60 Abort)
// -------------------------------------------------------------
__global__ void __launch_bounds__(256, 4) kernel_optimized(
    uint32_t w0, uint32_t w1, uint32_t w2,
    uint32_t start_nonce,
    uint32_t nonce_count,
    int* __restrict__ found,
    uint32_t* __restrict__ found_nonce,
    uint8_t* __restrict__ found_hash
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nonce_count || *found != 0) return;

    uint32_t nonce = start_nonce + gid;
    uint32_t nonce_le = ((nonce & 0xff) << 24) |
                        (((nonce >> 8) & 0xff) << 16) |
                        (((nonce >> 16) & 0xff) << 8) |
                        ((nonce >> 24) & 0xff);

    // Pass 1: Start directly from Round 3 (Rounds 0-2 precomputed into c_s1_r3)
    uint32_t a = c_s1_r3[0], b = c_s1_r3[1], c = c_s1_r3[2], d = c_s1_r3[3];
    uint32_t e = c_s1_r3[4], f = c_s1_r3[5], g = c_s1_r3[6], h = c_s1_r3[7];

    uint32_t W[16];
    W[0] = w0;
    W[1] = w1;
    W[2] = w2;
    W[3] = nonce_le;
    W[4] = 0x80000000;
    #pragma unroll
    for (int i = 5; i < 15; i++) W[i] = 0;
    W[15] = 0x00000280;

    #pragma unroll
    for (int i = 3; i < 64; i++) {
        if (i >= 16) {
            uint32_t s0 = ror(W[(i - 15) & 15], 7) ^ ror(W[(i - 15) & 15], 18) ^ (W[(i - 15) & 15] >> 3);
            uint32_t s1_val = ror(W[(i - 2) & 15], 17) ^ ror(W[(i - 2) & 15], 19) ^ (W[(i - 2) & 15] >> 10);
            W[i & 15] = W[(i - 16) & 15] + s0 + W[(i - 7) & 15] + s1_val;
        }
        uint32_t S1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t tmp1 = h + S1 + ch + c_K[i] + W[i & 15];
        uint32_t S0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t tmp2 = S0 + maj;

        h = g; g = f; f = e; e = d + tmp1;
        d = c; c = b; b = a; a = tmp1 + tmp2;
    }

    uint32_t s1[8] = {
        c_midstate[0] + a, c_midstate[1] + b, c_midstate[2] + c, c_midstate[3] + d,
        c_midstate[4] + e, c_midstate[5] + f, c_midstate[6] + g, c_midstate[7] + h
    };

    // Pass 2: Rounds 0 to 60 with Early Abort
    a = 0x6a09e667; b = 0xbb67ae85; c = 0x3c6ef372; d = 0xa54ff53a;
    e = 0x510e527f; f = 0x9b05688c; g = 0x1f83d9ab; h = 0x5be0cd19;

    uint32_t W2[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) W2[i] = s1[i];
    W2[8] = 0x80000000;
    #pragma unroll
    for (int i = 9; i < 15; i++) W2[i] = 0;
    W2[15] = 0x00000100;

    #pragma unroll
    for (int i = 0; i <= 60; i++) {
        if (i >= 16) {
            uint32_t s0 = ror(W2[(i - 15) & 15], 7) ^ ror(W2[(i - 15) & 15], 18) ^ (W2[(i - 15) & 15] >> 3);
            uint32_t s1_val = ror(W2[(i - 2) & 15], 17) ^ ror(W2[(i - 2) & 15], 19) ^ (W2[(i - 2) & 15] >> 10);
            W2[i & 15] = W2[(i - 16) & 15] + s0 + W2[(i - 7) & 15] + s1_val;
        }
        uint32_t S1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t tmp1 = h + S1 + ch + c_K[i] + W2[i & 15];
        uint32_t S0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t tmp2 = S0 + maj;

        h = g; g = f; f = e; e = d + tmp1;
        d = c; c = b; b = a; a = tmp1 + tmp2;
    }

    // Round 60 Early Abort:
    // e is identically e_61 which will become h_final after round 63!
    uint32_t s2_7 = 0x5be0cd19u + e + 0x21E8u;
    uint32_t hash_w7 = ((s2_7 & 0xff) << 24) | (((s2_7 >> 8) & 0xff) << 16) | (((s2_7 >> 16) & 0xff) << 8) | ((s2_7 >> 24) & 0xff);
    if (hash_w7 > c_target_w7) return;

    // Remaining rounds 61..63 (executes for < 0.000001% of nonces)
    #pragma unroll
    for (int i = 61; i < 64; i++) {
        uint32_t s0 = ror(W2[(i - 15) & 15], 7) ^ ror(W2[(i - 15) & 15], 18) ^ (W2[(i - 15) & 15] >> 3);
        uint32_t s1_val = ror(W2[(i - 2) & 15], 17) ^ ror(W2[(i - 2) & 15], 19) ^ (W2[(i - 2) & 15] >> 10);
        W2[i & 15] = W2[(i - 16) & 15] + s0 + W2[(i - 7) & 15] + s1_val;

        uint32_t S1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t tmp1 = h + S1 + ch + c_K[i] + W2[i & 15];
        uint32_t S0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t tmp2 = S0 + maj;

        h = g; g = f; f = e; e = d + tmp1;
        d = c; c = b; b = a; a = tmp1 + tmp2;
    }

    uint32_t s2[8] = {
        0x6a09e667u + a, 0xbb67ae85u + b, 0x3c6ef372u + c, 0xa54ff53au + d,
        0x510e527fu + e, 0x9b05688cu + f, 0x1f83d9abu + g, s2_7
    };

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
        if (finalHash[i] < c_target[i]) { below = true; break; }
        if (finalHash[i] > c_target[i]) { below = false; break; }
    }
    if (below) {
        if (atomicExch(found, 1) == 0) {
            *found_nonce = nonce;
            for (int i = 0; i < 32; i++) found_hash[i] = finalHash[i];
        }
    }
}

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

// Precompute s1 after rounds 0, 1, 2 on host:
void precompute_s1_r3(const uint32_t midstate[8], uint32_t w0, uint32_t w1, uint32_t w2, uint32_t out[8]) {
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

int main() {
    printf("=================================================================\n");
    printf("🚀 TRU RTX 5060 Benchmark: Baseline vs Ultra-Optimized Kernel\n");
    printf("=================================================================\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU Device: %s (SMs: %d, Max Clocks: %d MHz)\n", prop.name, prop.multiProcessorCount, prop.clockRate / 1000);

    uint32_t h_midstate[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    uint32_t w0 = 0x66E61122, w1 = 0x1C00FFFF, w2 = 0x12345678;
    uint8_t h_target[32];
    memset(h_target, 0, 32);
    h_target[26] = 0x1c; h_target[25] = 0x44;

    uint32_t target_w7 = ((uint32_t)h_target[31] << 24) | ((uint32_t)h_target[30] << 16) |
                         ((uint32_t)h_target[29] << 8) | ((uint32_t)h_target[28]);

    uint32_t h_s1_r3[8];
    precompute_s1_r3(h_midstate, w0, w1, w2, h_s1_r3);

    // Alloc device buffers
    uint32_t *d_midstate, *d_found_nonce;
    uint8_t *d_target, *d_found_hash;
    int *d_found;
    cudaMalloc(&d_midstate, 32);
    cudaMalloc(&d_target, 32);
    cudaMalloc(&d_found, sizeof(int));
    cudaMalloc(&d_found_nonce, sizeof(uint32_t));
    cudaMalloc(&d_found_hash, 32);

    cudaMemcpy(d_midstate, h_midstate, 32, cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, h_target, 32, cudaMemcpyHostToDevice);
    cudaMemset(d_found, 0, sizeof(int));

    cudaMemcpyToSymbol(c_midstate, h_midstate, 32);
    cudaMemcpyToSymbol(c_s1_r3, h_s1_r3, 32);
    cudaMemcpyToSymbol(c_target, h_target, 32);
    cudaMemcpyToSymbol(c_target_w7, &target_w7, sizeof(uint32_t));

    const int threads = 256;
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    // 1. BENCHMARK BASELINE (4M nonces)
    uint32_t batch_baseline = 4194304;
    int blocks_baseline = (batch_baseline + threads - 1) / threads;

    kernel_baseline<<<blocks_baseline, threads>>>(
        d_midstate, w0, w1, w2, 0, batch_baseline, d_target, d_found, d_found_nonce, d_found_hash
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("❌ Baseline launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    printf("\n[1] Testing Baseline Kernel (4M nonces/batch)...\n");
    int runs = 10;
    cudaEventRecord(ev_start);
    for (int i = 0; i < runs; i++) {
        kernel_baseline<<<blocks_baseline, threads>>>(
            d_midstate, w0, w1, w2, i * batch_baseline, batch_baseline, d_target, d_found, d_found_nonce, d_found_hash
        );
    }
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float total_ms_base = 0.0f;
    cudaEventElapsedTime(&total_ms_base, ev_start, ev_stop);
    double ms_base = total_ms_base / runs;
    double hr_base = (batch_baseline / (ms_base / 1000.0)) / 1e9;
    printf("   ⏱️  Avg GPU execution time: %.2f ms / batch\n", ms_base);
    printf("   ⚡ Baseline Hashrate:       %.2f GH/s (%.2f MH/s)\n", hr_base, hr_base * 1000.0);

    // 2. BENCHMARK OPTIMIZED (32M nonces)
    uint32_t batch_opt = 33554432; // 32M nonces
    int blocks_opt = (batch_opt + threads - 1) / threads;

    kernel_optimized<<<blocks_opt, threads>>>(
        w0, w1, w2, 0, batch_opt, d_found, d_found_nonce, d_found_hash
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("❌ Optimized launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    printf("\n[2] Testing Ultra-Optimized Kernel (32M nonces + R0-2 Skip + R60 Early Abort)...\n");
    cudaEventRecord(ev_start);
    for (int i = 0; i < runs; i++) {
        kernel_optimized<<<blocks_opt, threads>>>(
            w0, w1, w2, i * batch_opt, batch_opt, d_found, d_found_nonce, d_found_hash
        );
    }
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float total_ms_opt = 0.0f;
    cudaEventElapsedTime(&total_ms_opt, ev_start, ev_stop);
    double ms_opt = total_ms_opt / runs;
    double hr_opt = (batch_opt / (ms_opt / 1000.0)) / 1e9;
    printf("   ⏱️  Avg GPU execution time: %.2f ms / batch (32M)\n", ms_opt);
    printf("   ⚡ Optimized Hashrate:      %.2f GH/s (%.2f MH/s)\n", hr_opt, hr_opt * 1000.0);

    double speedup = hr_opt / hr_base;
    printf("\n=================================================================\n");
    printf("🎯 SPEEDUP RESULT: %.2fx FASTER (%.2f GH/s -> %.2f GH/s)\n", speedup, hr_base, hr_opt);
    printf("=================================================================\n");

    cudaFree(d_midstate);
    cudaFree(d_target);
    cudaFree(d_found);
    cudaFree(d_found_nonce);
    cudaFree(d_found_hash);
    return 0;
}
