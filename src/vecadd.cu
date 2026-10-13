#include "cuda_kernels/kernels.h"

#include <stdexcept>
#include <string>

namespace cuda_kernels {

namespace { 

__global__ void vecAddKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

}  // namespace

void vecAdd(const float* d_a, const float* d_b, float* d_c, int n) {
    if (n <= 0) {
        throw std::invalid_argument("vecAdd: n must be positive");
    }
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    vecAddKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("vecAdd launch failed: ") +
                                 cudaGetErrorString(err));
    }
}

}  // namespace cuda_kernels