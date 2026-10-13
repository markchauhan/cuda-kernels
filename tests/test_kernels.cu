#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <cmath>
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

    const int n = (1 << 24) + 1;
    std::vector<float> h(n);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (auto& x : h) {
        x = dist(rng);
    }

    double expected = 0.0;
    for (float x : h) {
        expected += x;
    }

    cuda_kernels::DeviceBuffer<float> d_data(n);
    cuda_kernels::DeviceBuffer<float> d_total(1);

    d_data.copyFromHost(h.data());
    cuda_kernels::reduceSumV1(d_data.get(), d_total.get(), n);

    float got = 0.0f;
    d_total.copyToHost(&got);
    EXPECT_NEAR(got, expected, 1e-4 * std::abs(expected));

}

TEST(Reduce, V2MatchesCpuReference) {
    SKIP_IF_NO_GPU();

    const int n = (1 << 24) + 1;

    std::vector<float> h(n);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (auto& x : h) {
        x = dist(rng);
    }

    double expected = 0.0;
    for (float x : h) {
        expected += x;
    }

    cuda_kernels::DeviceBuffer<float> d_data(n);
    cuda_kernels::DeviceBuffer<float> d_total(1);

    d_data.copyFromHost(h.data());
    cuda_kernels::reduceSumV2(d_data.get(), d_total.get(), n, 64);

    float got = 0.0f;
    d_total.copyToHost(&got);
    EXPECT_NEAR(got, expected, 1e-4 * std::abs(expected));
}
