#pragma once

#include <cstddef>        
#include <stdexcept>     
#include <cuda_runtime.h>

namespace cuda_kernels {

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t n) : n_(n) {
        if (cudaMalloc(reinterpret_cast<void**>(&data_), n * sizeof(T)) != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed");
        }
    }

    ~DeviceBuffer() { cudaFree(data_); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : data_(other.data_), n_(other.n_) {
        other.data_ = nullptr;
        other.n_ = 0;
    }

    DeviceBuffer &operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            cudaFree(data_);
            data_ = other.data_;
            n_ = other.n_;
            other.data_ = nullptr;
            other.n_ = 0;
        }
        return *this;
    }

    void copyFromHost(const T *host_data) {
        if (cudaMemcpy(data_, host_data, n_ * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy from host to device failed");
        }
    }

    void copyToHost(T *host_data) const {
        if (cudaMemcpy(host_data, data_, n_ * sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy from device to host failed");
        }
    }

    void zero() {
        if (cudaMemset(data_, 0, n_ * sizeof(T)) != cudaSuccess) {
            throw std::runtime_error("cudaMemset failed");
        }
    }
    
    T *get()    { return data_; }
    const T* get() const { return data_; }
    size_t size() const { return n_; }

private:
    T *data_ = nullptr;
    size_t n_ = 0;
};

}  // namespace cuda_kernels
