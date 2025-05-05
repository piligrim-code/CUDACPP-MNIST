cmake_minimum_required(VERSION 3.10)
project(cuda_cnn_mnist LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} src)

# nvcc flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_50 -O2")

cuda_add_executable(cuda_cnn
    src/main.cu
    src/cnn.cu
    src/mnist.cpp
    src/utils.cpp
)

target_link_libraries(cuda_cnn ${OpenCV_LIBS})
