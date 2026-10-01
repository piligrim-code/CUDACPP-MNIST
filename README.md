# CUDACPP-MNIST

Small CUDA/C++ MNIST experiment: convolution, ReLU, max pooling, a linear
classifier and softmax. **Training updates only the final linear layer.**
The convolution filters remain randomly initialized and fixed. This is a
random-feature classifier experiment, not end-to-end CNN training.

## CPU checks: no CUDA, dataset or OpenCV required

Requires CMake 3.18+ and a C++17 compiler:

```console
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel 2
ctest --test-dir build -C Release --output-on-failure
```

With MinGW on Windows, add `-G "MinGW Makefiles"` to the configure command.
Use a fresh build directory when changing generators/toolchains.

The executable runs 31 regression cases under one CTest entry. It creates
tiny synthetic IDX files in an owned temporary directory and removes them
afterwards. No real handwriting data is bundled or downloaded. Cases cover
big-endian decoding, normalization, malformed/truncated/oversized files,
count mismatches, unchanged outputs on failure and network input validation.

Linux GCC/Clang sanitizer build:

```console
cmake -S . -B build-sanitize -DENABLE_SANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-sanitize --parallel 2
ctest --test-dir build-sanitize --output-on-failure
```

## Optional CUDA demo

Requires a CUDA toolkit, a compatible host compiler and an NVIDIA GPU with
runtime support for the selected architecture. Architecture 75 is the build
default, not automatic GPU detection; override it for your hardware/toolkit.
CUDA and sanitizer builds use separate directories.

```console
cmake -S . -B build-cuda -DBUILD_CUDA_DEMO=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --config Release --parallel 2
./build-cuda/cuda_cnn --help
./build-cuda/cuda_cnn ./data
```

With a multi-configuration Windows generator, the executable is normally
`build-cuda/Release/cuda_cnn.exe`. A supported CUDA host compiler is required;
the CPU-only MinGW check is not qualification of a Windows CUDA toolchain.

The data directory must contain these **uncompressed** IDX files:

```text
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

Obtain the MNIST data separately from an authorized source and review its
terms. Input is restricted to 28x28 images, labels 0..9 and 1..100000 records
per file. Incorrect magic, size, count and trailing bytes are rejected before
publishing results to the caller. A failing dataset load preserves all four
caller-owned output containers. This is a bounded MNIST reader, not a generic
IDX or compressed-file library.

The demo trains on at most the first 1000 training records for five epochs,
then prints predictions and labels for at most ten test records. Small inputs
are not enlarged with empty images. This display is not an accuracy benchmark.
No model checkpoint is saved and no full convolution backpropagation exists.

For optional interactive windows, configure with `-DENABLE_OPENCV=ON` and
install OpenCV core/imgproc/highgui, then pass `--show` to the executable.
Headless execution is the default. The GUI path needs separate qualification.

## Validation boundaries

CI checks CPU builds on Linux and Windows, runs Linux address/undefined-
behavior sanitizers, and compiles the headless CUDA executable in a CUDA 12.6
development container. Its `--help` check does not initialize a GPU.

These checks do **not** establish GPU numerical correctness, successful real
MNIST training, classification accuracy or a performance improvement. GPU
kernel/reference comparisons, device error/resource tests and a held-out
evaluation are still required. CPU validation functions are tested directly;
device allocation cleanup and runtime failures require a GPU to exercise.

The kernel path now checks launch/synchronization/copy results; GPU-buffer
owners cannot be copied, and partial constructor allocations are cleaned on
failure. The final-layer update copies the pooled vector once per sample,
instead of issuing one device-to-host transfer per weight. No speedup is claimed.

No license file was present in this historical snapshot. This repair does
not assign a license or grant rights to third-party datasets or dependencies.
