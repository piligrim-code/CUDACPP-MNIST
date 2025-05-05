#pragma once
#include <vector>
#include <cstdint>
void load_mnist(const std:string& path,
    std::vector<std::vector<float>>& tr_imgs,
    std::vector<uint8_t>& tr_lbls,
    std::vector<std::vector<float>>& te_imgs,
    std::vector<uint8_t>& te_lbls);
