#pragma once
#include <vector>
#include <cstdint>

class CNN {
public:
    CNN(int img_w, int img_h);
    ~CNN();
    CNN(const CNN&) = delete;
    CNN& operator=(const CNN&) = delete;
    void train(const std::vector<std::vector<float>>& images,
        const std::vector<uint8_t>& labels,
        int epochs, float lr);
    int  predict(const std::vector<float>& image);
private:
    int img_w, img_h;
    int conv_out_w, conv_out_h;
    int pool_out_w, pool_out_h;
    float* d_conv_w = nullptr;
    float* d_conv_b = nullptr;
    float* d_fc_w = nullptr;
    float* d_fc_b = nullptr;

    float* d_x = nullptr;
    float* d_conv_out = nullptr;
    float* d_pool_out = nullptr;
    float* d_fc_out = nullptr;

    void release() noexcept;
    void init_weights();
    void forward(float* x);
    void backward(float* x, int label, float lr);
};
