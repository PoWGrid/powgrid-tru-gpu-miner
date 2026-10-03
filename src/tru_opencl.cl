// PowGrid TRU ($TRU) High-Performance OpenCL Kernel for AMD / Intel / NVIDIA GPUs
// Algorithm: TRUHash (Dual SHA-256 with 0x21E8 Custom Modifier)

__constant uint K[64] = {
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

#define ror(x, n) rotate((uint)(x), (uint)(32u - (n)))

inline void sha256_compress(uint* s, uint* W) {
    uint a = s[0], b = s[1], c = s[2], d = s[3];
    uint e = s[4], f = s[5], g = s[6], h = s[7];

    #pragma unroll
    for (int i = 0; i < 64; i++) {
        if (i >= 16) {
            uint s0 = ror(W[(i - 15) & 15], 7) ^ ror(W[(i - 15) & 15], 18) ^ (W[(i - 15) & 15] >> 3);
            uint s1 = ror(W[(i - 2) & 15], 17) ^ ror(W[(i - 2) & 15], 19) ^ (W[(i - 2) & 15] >> 10);
            W[i & 15] = W[(i - 16) & 15] + s0 + W[(i - 7) & 15] + s1;
        }
        uint S1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
        uint ch = (e & f) ^ ((~e) & g);
        uint tmp1 = h + S1 + ch + K[i] + W[i & 15];
        uint S0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
        uint maj = (a & b) ^ (a & c) ^ (b & c);
        uint tmp2 = S0 + maj;

        h = g; g = f; f = e; e = d + tmp1;
        d = c; c = b; b = a; a = tmp1 + tmp2;
    }

    s[0] += a; s[1] += b; s[2] += c; s[3] += d;
    s[4] += e; s[5] += f; s[6] += g; s[7] += h;
}

__kernel void tru_opencl_mine(
    __global const uint* restrict midstate,
    uint w0,
    uint w1,
    uint w2,
    uint start_nonce,
    uint batch_size,
    __global const uchar* restrict target,
    __global volatile int* restrict found,
    __global volatile uint* restrict found_nonce,
    __global uchar* restrict found_hash
) {
    uint gid = get_global_id(0);
    if (gid >= batch_size || *found != 0) return;

    uint nonce = start_nonce + gid;
    uint nonce_le = ((nonce & 0xff) << 24) |
                    (((nonce >> 8) & 0xff) << 16) |
                    (((nonce >> 16) & 0xff) << 8) |
                    ((nonce >> 24) & 0xff);

    uint s1[8];
    for (int i = 0; i < 8; i++) s1[i] = midstate[i];

    uint w_chunk1[16];
    w_chunk1[0] = w0;
    w_chunk1[1] = w1;
    w_chunk1[2] = w2;
    w_chunk1[3] = nonce_le;
    w_chunk1[4] = 0x80000000;
    for (int i = 5; i < 15; i++) w_chunk1[i] = 0;
    w_chunk1[15] = 0x00000280;

    sha256_compress(s1, w_chunk1);

    uint s2[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    uint w_chunk2[16];
    for (int i = 0; i < 8; i++) w_chunk2[i] = s1[i];
    w_chunk2[8] = 0x80000000;
    for (int i = 9; i < 15; i++) w_chunk2[i] = 0;
    w_chunk2[15] = 0x00000100;

    sha256_compress(s2, w_chunk2);

    // TRUHash custom modifier
    s2[7] += 0x21E8u;

    // Fast reject check against target highest byte
    if ((s2[7] & 0xff) > target[31]) return;

    uchar finalHash[32];
    for (int i = 0; i < 8; i++) {
        finalHash[4 * i]     = (uchar)((s2[i] >> 24) & 0xff);
        finalHash[4 * i + 1] = (uchar)((s2[i] >> 16) & 0xff);
        finalHash[4 * i + 2] = (uchar)((s2[i] >> 8) & 0xff);
        finalHash[4 * i + 3] = (uchar)(s2[i] & 0xff);
    }

    bool below = false;
    for (int i = 31; i >= 0; i--) {
        if (finalHash[i] < target[i]) { below = true; break; }
        if (finalHash[i] > target[i]) { below = false; break; }
    }

    if (below) {
        if (atomic_xchg(found, 1) == 0) {
            *found_nonce = nonce;
            for (int i = 0; i < 32; i++) found_hash[i] = finalHash[i];
        }
    }
}
