#ifndef TRU_CUDA_H
#define TRU_CUDA_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char name[256];
    char arch_name[64];
    int sm_count;
    int major;
    int minor;
    uint64_t total_memory;
    uint32_t recommended_batch_size;
    uint32_t max_threads_per_block;
} CudaDeviceInfo;

typedef struct {
    int found;
    uint32_t nonce;
    uint8_t hash[32];
} CudaMiningResult;

// Query number of CUDA devices
int cuda_miner_get_device_count(int* out_count);

// Query device properties and optimal tuning recommendation
int cuda_miner_get_device_info(int device_id, CudaDeviceInfo* out_info);

// Initialize CUDA device and allocate continuous pipeline streams (returns 0 on success)
int cuda_miner_init(int device_id);

// Free CUDA device buffers and streams
void cuda_miner_cleanup(void);

// Run a batch of nonces on GPU with continuous streaming
int cuda_miner_search(
    const uint32_t midstate[8],
    const uint8_t header80[80],
    const uint8_t target[32],
    uint32_t start_nonce,
    uint32_t batch_size,
    CudaMiningResult* out_result
);

#ifdef __cplusplus
}
#endif

#endif // TRU_CUDA_H
