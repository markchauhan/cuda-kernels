#pragma once

namespace cuda_kernels {

// Element-wise add of two device arrays: d_c[i] = d_a[i] + d_b[i] for i in [0, n).
void vecAdd(const float* d_a, const float* d_b, float* d_c, int n);

// Sum n device floats into *d_total using the baseline (v1) reduction.
void reduceSumV1(const float* d_data, float* d_total, int n);

// Sum n device floats into *d_total using the optimized (v2) reduction over `blocks` blocks.
void reduceSumV2(const float* d_data, float* d_total, int n, int blocks);

}  // namespace cuda_kernels
