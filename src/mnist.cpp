#include "mnist.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace {
constexpr std::uint32_t max_records = 100000;

std::uint32_t read_u32(std::istream& stream) {
    std::array<unsigned char, 4> bytes{};
    if (!stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size()))
        throw std::runtime_error("Truncated IDX header");
    return (std::uint32_t(bytes[0]) << 24) | (std::uint32_t(bytes[1]) << 16)
        | (std::uint32_t(bytes[2]) << 8) | std::uint32_t(bytes[3]);
}

void check_count(std::uint32_t count) {
    if (count == 0 || count > max_records)
        throw std::runtime_error("IDX record count must be between 1 and 100000");
}

void check_payload(std::ifstream& stream, std::uint64_t expected) {
    const auto start = stream.tellg();
    stream.seekg(0, std::ios::end);
    const auto end = stream.tellg();
    if (start < 0 || end < start || std::uint64_t(end - start) != expected)
        throw std::runtime_error("IDX payload length does not match its header");
    stream.seekg(start);
    if (!stream) throw std::runtime_error("Unable to seek IDX input");
}
}

void read_images(const std::string& file, std::vector<std::vector<float>>& images) {
    std::ifstream stream(file, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open image IDX file: " + file);
    if (read_u32(stream) != 2051) throw std::runtime_error("Invalid image IDX magic");
    const auto count = read_u32(stream);
    const auto rows = read_u32(stream);
    const auto columns = read_u32(stream);
    check_count(count);
    if (rows != 28 || columns != 28)
        throw std::runtime_error("Expected 28x28 MNIST images");
    check_payload(stream, std::uint64_t(count) * 784);

    std::vector<std::vector<float>> result(count, std::vector<float>(784));
    std::array<unsigned char, 784> pixels{};
    for (auto& image : result) {
        if (!stream.read(reinterpret_cast<char*>(pixels.data()), pixels.size()))
            throw std::runtime_error("Truncated image payload");
        for (std::size_t i = 0; i < pixels.size(); ++i) image[i] = pixels[i] / 255.f;
    }
    images.swap(result);
}

void read_labels(const std::string& file, std::vector<std::uint8_t>& labels) {
    std::ifstream stream(file, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open label IDX file: " + file);
    if (read_u32(stream) != 2049) throw std::runtime_error("Invalid label IDX magic");
    const auto count = read_u32(stream);
    check_count(count);
    check_payload(stream, count);
    std::vector<std::uint8_t> result(count);
    if (!stream.read(reinterpret_cast<char*>(result.data()), count))
        throw std::runtime_error("Truncated label payload");
    for (auto label : result)
        if (label > 9) throw std::runtime_error("MNIST labels must be between 0 and 9");
    labels.swap(result);
}

void load_mnist(const std::string& directory,
                std::vector<std::vector<float>>& train_images,
                std::vector<std::uint8_t>& train_labels,
                std::vector<std::vector<float>>& test_images,
                std::vector<std::uint8_t>& test_labels) {
    const std::filesystem::path root(directory);
    std::vector<std::vector<float>> tr_x, te_x;
    std::vector<std::uint8_t> tr_y, te_y;
    read_images((root / "train-images-idx3-ubyte").string(), tr_x);
    read_labels((root / "train-labels-idx1-ubyte").string(), tr_y);
    read_images((root / "t10k-images-idx3-ubyte").string(), te_x);
    read_labels((root / "t10k-labels-idx1-ubyte").string(), te_y);
    if (tr_x.size() != tr_y.size() || te_x.size() != te_y.size())
        throw std::runtime_error("Image and label counts do not match");
    train_images.swap(tr_x);
    train_labels.swap(tr_y);
    test_images.swap(te_x);
    test_labels.swap(te_y);
}
