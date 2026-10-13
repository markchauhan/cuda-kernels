#include "cuda_kernels/kernels.h"

namespace cuda_kernels {

// TODO: __global__ reduce kernels (v1 and v2)

void reduceSumV1(const float* d_data, float* d_total, int n) {
    // TODO: __global__ kernel + launch
}

void reduceSumV2(const float* d_data, float* d_total, int n, int blocks) {
    // TODO: __global__ kernel + launch
}

}  // namespace cuda_kernels
