#pragma once
#include <vector>
#include <cstdint>
#include <string>

// Strict, uncompressed 28x28 MNIST IDX input; outputs change only on success.
void read_images(const std::string& file, std::vector<std::vector<float>>& images);
void read_labels(const std::string& file, std::vector<uint8_t>& labels);
void load_mnist(const std::string& path,
    std::vector<std::vector<float>>& tr_imgs,
    std::vector<uint8_t>& tr_lbls,
    std::vector<std::vector<float>>& te_imgs,
    std::vector<uint8_t>& te_lbls);
