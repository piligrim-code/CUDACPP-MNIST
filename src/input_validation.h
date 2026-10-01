#pragma once
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

inline void validate_image(const std::vector<float>& image) {
    if (image.size() != 784) throw std::invalid_argument("Expected 784 image pixels");
    for (float pixel : image)
        if (!std::isfinite(pixel) || pixel < 0.f || pixel > 1.f)
            throw std::invalid_argument("Image pixels must be finite and normalized to [0,1]");
}

inline void validate_training_input(const std::vector<std::vector<float>>& images,
                                    const std::vector<std::uint8_t>& labels,
                                    int epochs, float learning_rate) {
    if (images.empty() || images.size() != labels.size())
        throw std::invalid_argument("Training requires nonempty matching image/label counts");
    if (epochs <= 0 || !std::isfinite(learning_rate) || learning_rate <= 0.f)
        throw std::invalid_argument("Epochs and learning rate must be positive and finite");
    for (const auto& image : images) validate_image(image);
    for (auto label : labels)
        if (label > 9) throw std::invalid_argument("Labels must be between 0 and 9");
}
