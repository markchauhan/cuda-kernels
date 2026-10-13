# cuda-kernels

CUDA kernels in C++17 with RAII device memory, tested against CPU references.

## Kernels

- **Vector add**: TODO: describe.
- **Parallel reduction v1**: TODO: describe.
- **Parallel reduction v2**: TODO: describe.

## Results

TODO: fill in after benchmarking on a Tesla T4.

| Kernel | Config | Time (ms) | Bandwidth (GB/s) | % of T4 peak |
| ------ | ------ | --------- | ---------------- | ------------ |
|        |        |           |                  |              |

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

In a notebook with a T4 GPU accelerator enabled:

```sh
!git clone https://github.com/<your-username>/cuda-kernels.git
%cd cuda-kernels
!cmake -B build
!cmake --build build -j
!ctest --test-dir build --output-on-failure
!./build/bench_reduce
```

## Notes

CI only compiles; GPU tests are run manually on a Tesla T4.
