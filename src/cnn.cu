#include "cnn.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cassert>
#define N_FILT 8
#define K_SIZE 3

#define CUDA_CHECK(err) \
  if(err!=cudaSuccess){printf("CUDA %s\n",cudaGetErrorString(err));abort();}

CNN::CNN(int img_w_, int img_h_)
    : img_w(img_w_), img_h(img_h_) {
    conv_out_w = img_w - K_SIZE + 1;
    conv_out_h = img_h - K_SIZE + 1;
    pool_out_w = conv_out_w / 2;
    pool_out_h = conv_out_h / 2;
    // выделяем Unified Memory
    CUDA_CHECK(cudaMallocManaged(&d_conv_w, N_FILT * K_SIZE * K_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_conv_b, N_FILT * sizeof(float)));
    int fc_in = N_FILT * pool_out_w * pool_out_h;
    CUDA_CHECK(cudaMallocManaged(&d_fc_w, 10 * fc_in * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_fc_b, 10 * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_x, img_w * img_h * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_conv_out, N_FILT * conv_out_w * conv_out_h * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_pool_out, N_FILT * pool_out_w * pool_out_h * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_fc_out, 10 * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&d_prob, 10 * sizeof(float)));
    init_weights();
}

CNN::~CNN() {
    cudaFree(d_conv_w); cudaFree(d_conv_b);
    cudaFree(d_fc_w);   cudaFree(d_fc_b);
    cudaFree(d_x); cudaFree(d_conv_out);
    cudaFree(d_pool_out); cudaFree(d_fc_out); cudaFree(d_prob);
}

__global__ void conv_kernel(const float* x, float* out,
    const float* w, const float* b,
    int img_w, int img_h,
    int out_w, int out_h) {
    int f = blockIdx.z;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    if (ox < out_w && oy < out_h) {
        float sum = b[f];
        for (int ky = 0;ky < K_SIZE;++ky)for (int kx = 0;kx < K_SIZE;++kx) {
            int ix = ox + kx, iy = oy + ky;
            sum += x[iy * img_w + ix] * w[f * K_SIZE * K_SIZE + ky * K_SIZE + kx];
        }
        out[f * out_w * out_h + oy * out_w + ox] = sum;
    }
}

__global__ void relu_kernel(float* x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && x[i] < 0) x[i] = 0;
}

__global__ void pool_kernel(const float* in, float* out,
    int in_w, int in_h) {
    int f = blockIdx.z;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int ow = in_w / 2, oh = in_h / 2;
    if (ox < ow && oy < oh) {
        float m = 0;
        int base = f * in_w * in_h + (2 * oy) * in_w + 2 * ox;
        for (int dy = 0;dy < 2;++dy)for (int dx = 0;dx < 2;++dx) {
            m = fmaxf(m, in[base + dy * in_w + dx]);
        }
        out[f * ow * oh + oy * ow + ox] = m;
    }
}

__global__ void fc_kernel(const float* in, float* out,
    const float* w, const float* b,
    int in_n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < 10) {
        float sum = b[j];
        for (int i = 0;i < in_n;++i)
            sum += in[i] * w[j * in_n + i];
        out[j] = sum;
    }
}

__global__ void softmax_kernel(float* x) {
    float m = x[0];
    for (int i = 1;i < 10;++i) m = fmaxf(m, x[i]);
    float sum = 0;
    for (int i = 0;i < 10;++i) {
        x[i] = expf(x[i] - m);
        sum += x[i];
    }
    for (int i = 0;i < 10;++i) x[i] /= sum;
}

void CNN::init_weights() {
    int conv_sz = N_FILT * K_SIZE * K_SIZE;
    for (int i = 0;i < conv_sz;++i) d_conv_w[i] = (rand() / float(RAND_MAX) - 0.5f) * 0.1f;
    for (int i = 0;i < N_FILT;++i) d_conv_b[i] = 0;
    int fc_in = N_FILT * pool_out_w * pool_out_h;
    for (int i = 0;i < 10 * fc_in;++i) d_fc_w[i] = (rand() / float(RAND_MAX) - 0.5f) * 0.1f;
    for (int i = 0;i < 10;++i) d_fc_b[i] = 0;
}

void CNN::forward(float* x) {
    // conv
    dim3 bs(16, 16, 1), gs((conv_out_w + 15) / 16, (conv_out_h + 15) / 16, N_FILT);
    conv_kernel << <gs, bs >> > (x, d_conv_out, d_conv_w, d_conv_b,
        img_w, img_h, conv_out_w, conv_out_h);
    cudaDeviceSynchronize();
    // ReLU
    int N1 = N_FILT * conv_out_w * conv_out_h;
    relu_kernel << <(N1 + 255) / 256, 256 >> > (d_conv_out, N1);
    cudaDeviceSynchronize();
    // pool
    dim3 gs2((pool_out_w + 15) / 16, (pool_out_h + 15) / 16, N_FILT);
    pool_kernel << <gs2, bs >> > (d_conv_out, d_pool_out, conv_out_w, conv_out_h);
    cudaDeviceSynchronize();
    // fc
    int fc_in = N_FILT * pool_out_w * pool_out_h;
    fc_kernel << <(10 + 31) / 32, 32 >> > (d_pool_out, d_fc_out, d_fc_w, d_fc_b, fc_in);
    cudaDeviceSynchronize();
    // softmax
    softmax_kernel << <1, 1 >> > (d_fc_out);
    cudaDeviceSynchronize();
}

void CNN::backward(float* x, int label, float lr) {
    // dL/dz = p - y
    float h_prob[10];
    cudaMemcpy(h_prob, d_fc_out, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    int fc_in = N_FILT * pool_out_w * pool_out_h;
    for (int j = 0;j < 10;++j) {
        float grad = h_prob[j] - (j == label ? 1.0f : 0.0f);
        // update bias
        d_fc_b[j] -= lr * grad;
        // update weights
        for (int i = 0;i < fc_in;++i) {
            float a;
            cudaMemcpy(&a, d_pool_out + i, sizeof(float), cudaMemcpyDeviceToHost);
            d_fc_w[j * fc_in + i] -= lr * grad * a;
        }
    }
}

void CNN::train(const std::vector<std::vector<float>>& imgs,
    const std::vector<uint8_t>& labs,
    int epochs, float lr) {
    int N = imgs.size();
    for (int e = 0;e < epochs;++e) {
        for (int i = 0;i < N;++i) {
            cudaMemcpy(d_x, imgs[i].data(), img_w * img_h * sizeof(float), cudaMemcpyHostToDevice);
            forward(d_x);
            backward(d_x, labs[i], lr);
        }
        printf("Epoch %d done\n", e + 1);
    }
}

int CNN::predict(const std::vector<float>& img) {
    cudaMemcpy(d_x, img.data(), img_w * img_h * sizeof(float), cudaMemcpyHostToDevice);
    forward(d_x);
    float h[10];
    cudaMemcpy(h, d_fc_out, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    int best = 0; for (int i = 1;i < 10;++i) if (h[i] > h[best]) best = i;
    return best;
}
