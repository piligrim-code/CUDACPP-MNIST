#pragma once
#include <vector>

class CNN {
public:
    CNN(int img_w, int img_h);
    ~CNN();
    void train(const std::vector<std::vector<float>>& images,
        const std::vector<uint8_t>& labels,
        int epochs, float lr);
    int  predict(const std::vector<float>& image);
private:
    int img_w, img_h;
    int conv_out_w, conv_out_h;
    int pool_out_w, pool_out_h;
    float* d_conv_w;    
    float* d_conv_b;    
    float* d_fc_w;     
    float* d_fc_b;  

    float* d_x;        
    float* d_conv_out;
    float* d_pool_out;
    float* d_fc_out;
    float* d_prob;

    void init_weights();
    void forward(float* x);
    void backward(float* x, int label, float lr);
};
