# cuda-kernels

CUDA kernels in C++17 with RAII device memory, tested against CPU references.

## Kernels

- **Vector add** (`src/vecadd.cu`): one thread per element, `c[i] = a[i] + b[i]`.
  Neighboring threads touch neighboring addresses, so every load and store is
  coalesced. A bounds check handles sizes that are not a multiple of the block size.
- **Parallel reduction v1** (`src/reduce.cu`): each thread loads one element into
  shared memory, then the block folds the top half into the bottom half each round
  (`s[t] += s[t + stride]`) with a `__syncthreads()` between rounds, until `s[0]`
  holds the block's sum. Thread 0 adds it to the global total with one `atomicAdd`.
- **Parallel reduction v2** (`src/reduce.cu`): each thread first sums many elements
  privately in a register using a grid-stride loop, then joins the same tree. The
  grid is sized to the GPU (about 4 blocks of 256 threads per SM) instead of the
  data, so the tree, its barriers, and the atomics run hundreds of times less often.

Device memory is managed by `DeviceBuffer<T>` (`include/cuda_kernels/DeviceBuffer.h`),
a move-only RAII wrapper: allocation in the constructor, `cudaFree` in the destructor,
so early returns and exceptions never leak GPU memory.

## Results

Summing 2^24 floats (64 MB) on an NVIDIA Tesla T4 (40 SMs, about 320 GB/s peak
memory bandwidth). Best of 5 timed runs after a warm-up, measured with CUDA events.
All results match a double-precision CPU reference to a relative error below 1e-4.

| Kernel | Config | Time (ms) | Bandwidth (GB/s) | % of T4 peak |
| ------ | ------ | --------- | ---------------- | ------------ |
| Reduce v1 | 65,536 blocks × 256 | 1.146 | 59 | 18% |
| Reduce v2 | 40 blocks (1 per SM) | 0.849 | 79 | 25% |
| Reduce v2 | 80 blocks (2 per SM) | 0.457 | 147 | 46% |
| Reduce v2 | 160 blocks (4 per SM) | 0.306 | 219 | 68% |
| Reduce v2 | 1,280 blocks (32 per SM) | 0.295 | 227 | 71% |

**What the numbers show:**

- **v1 → v2 is 3.8× faster.** In v1, each thread does one useful load and then spends
  most of its time in eight synchronized tree rounds, so the memory system sits idle.
  In v2, threads stream through memory with no synchronization, and the tree runs once
  per block instead of once per 256 elements.
- **One block per SM is not enough.** With only 8 warps per SM, there is little other
  work to switch to while a warp waits on memory, so latency is not hidden.
- **Four blocks per SM fills the T4** (1,024 resident threads per SM). Past that,
  extra blocks only queue, and the kernel is limited by memory bandwidth.
- **Results vary in the last digit between runs.** Blocks finish in a different order
  each run, so the `atomicAdd`s happen in a different order, and float addition is
  not associative. Tests compare against the CPU with a tolerance, never `==`.

These measurements came from the development notebook. `bench/bench_reduce` will
reproduce them from this repo.

## Build and test

Requires CMake 3.24+, a C++17 compiler, and the CUDA toolkit.

```sh
cmake -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The default target architecture is `75` (Tesla T4). Override it with
`-DCMAKE_CUDA_ARCHITECTURES=<arch>`. Tests and benchmarks can be turned off with
`-DBUILD_TESTS=OFF` and `-DBUILD_BENCH=OFF`.

## Local compile check

On a machine without the CUDA toolkit (for example a Mac), you can still check
that everything compiles. With Docker running:

```sh
./scripts/compile_check.sh
```

This builds the project inside the same `nvidia/cuda` image CI uses, as
`linux/amd64`. The build tree stays in `/tmp/build` inside the container, so
nothing is written to the repo, and the script exits non-zero if the build
fails. It only compiles and links: no GPU is involved and no tests are run. On
Apple Silicon the image runs under emulation, so expect the first run to be slow.

## Running on Kaggle

In a notebook with the T4 GPU accelerator and Internet enabled:

```sh
!pip install -q cmake
!git clone https://github.com/markchauhan/cuda-kernels.git
%cd cuda-kernels
!cmake -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
!cmake --build build -j
!ctest --test-dir build --output-on-failure
!./build/bench_reduce
```

## Notes

GitHub-hosted runners have no GPU, so CI only builds the project. Tests are run
manually on a Tesla T4, and each GPU test skips cleanly when no device is present.
