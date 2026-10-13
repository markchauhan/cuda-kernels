#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <random>
#include <stdexcept>
#include <vector>

#include "cuda_kernels/DeviceBuffer.h"
#include "cuda_kernels/kernels.h"

namespace {

bool hasCudaDevice() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

}  // namespace

// GTEST_SKIP() only returns from the function it is expanded in, so the check
// has to be a macro used directly in the test body.
#define SKIP_IF_NO_GPU()                      \
    do {                                      \
        if (!hasCudaDevice())                 \
            GTEST_SKIP() << "no CUDA device"; \
    } while (0)

TEST(VecAdd, MatchesCpuReference) {
    SKIP_IF_NO_GPU();

    const int n = (1 << 20) + 1;  // 1M elements
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; i++) {
        a[i] = static_cast<float>(i);
        b[i] = 2.0f * i;
    }

    cuda_kernels::DeviceBuffer<float> d_a(n), d_b(n), d_c(n);
    d_a.copyFromHost(a.data());
    d_b.copyFromHost(b.data());

    cuda_kernels::vecAdd(d_a.get(), d_b.get(), d_c.get(), n);
    d_c.copyToHost(c.data());

    for (int i = 0; i < n; i++) {
        ASSERT_EQ(c[i], a[i] + b[i]) << "mismatch at index " << i;
    }
}

TEST(VecAdd, EmptyInputIsNoOp) {
    EXPECT_NO_THROW(cuda_kernels::vecAdd(nullptr, nullptr, nullptr, 0));
}

TEST(Reduce, V1MatchesCpuReference) {
    SKIP_IF_NO_GPU();
    // TODO: fill a host array, sum it on the CPU as the reference, run
    // reduceSumV1 on the GPU, and check the relative error is below 1e-4.
    GTEST_SKIP() << "not implemented yet";
}

TEST(Reduce, V2MatchesCpuReference) {
    SKIP_IF_NO_GPU();
    // TODO: fill a host array, sum it on the CPU as the reference, run
    // reduceSumV2 on the GPU, and check the relative error is below 1e-4.
    GTEST_SKIP() << "not implemented yet";
}
