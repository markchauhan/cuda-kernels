#include "cuda_kernels/kernels.h"

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace cuda_kernels {

namespace {
    constexpr int kBlockSize = 256;  // threads per block

    // v1: one element per thread, then tree reduction in shared memory
    __global__ void reduceSumKernelV1(const float *data, float *total, int n) {
        __shared__ float s[kBlockSize];
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        s[threadIdx.x] = (i < n) ? data[i] : 0.0f; // pad past the end with zeros
        __syncthreads();

        // tree reduction in shared memory
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                s[threadIdx.x] += s[threadIdx.x + stride];
            }
            __syncthreads(); // make sure all adds at one stride are done before the next
        }

        if (threadIdx.x == 0) {
            atomicAdd(total, s[0]); // add the block's total to the global total
        }
    }

    // v2: multiple elements per thread, then tree reduction in shared memory
    __global__ void reduceSumKernelV2(const float *data, float *total, int n) {
        __shared__ float s[kBlockSize];

        // Jump by total thread count
        float sum = 0.0f;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
            sum += data[i];
        }

        s[threadIdx.x] = sum;
        __syncthreads();

        // tree reduction in shared memory as v1
        for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
            if (threadIdx.x < stride) {
                s[threadIdx.x] += s[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            atomicAdd(total, s[0]);
        }
    }

    void throwIfError(cudaError_t err, const char* msg) {
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
        }
    }

} // namespace

void reduceSumV1(const float* d_data, float* d_total, int n) {
    // Launch kernel with enough blocks to cover all elements
    throwIfError(cudaMemset(d_total, 0, sizeof(float)), "reduceSumV1 memset failed");
    if (n <= 0) return;

    const int blocks = (n + kBlockSize - 1) / kBlockSize; // number of blocks needed
    reduceSumKernelV1<<<blocks, kBlockSize>>>(d_data, d_total, n);
    throwIfError(cudaGetLastError(), "reduceSumKernelV1 launch failed");
}

void reduceSumV2(const float* d_data, float* d_total, int n, int blocks) {
    if (blocks <= 0) {
        throw std::invalid_argument("reduceSumV2: blocks must be positive");
    }
    throwIfError(cudaMemset(d_total, 0, sizeof(float)), "reduceSumV2 memset failed");
    if (n <= 0) return;

    // Launch kernel with specified number of blocks
    reduceSumKernelV2<<<blocks, kBlockSize>>>(d_data, d_total, n);
    throwIfError(cudaGetLastError(), "reduceSumV2 launch");
    
}

}  // namespace cuda_kernels
