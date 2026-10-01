#include "mnist.h"
#include "input_validation.h"
#include "cnn.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace fs = std::filesystem;
using Bytes = std::vector<std::uint8_t>;
using Images = std::vector<std::vector<float>>;
static_assert(!std::is_copy_constructible<CNN>::value, "GPU owner must not be copied");
static_assert(!std::is_copy_assignable<CNN>::value, "GPU owner must not be copy assigned");

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class Exception = std::runtime_error, class Function>
void rejects(Function action) {
    bool rejected = false;
    try { action(); } catch (const Exception&) { rejected = true; }
    require(rejected, "Expected rejection did not occur");
}

void u32(Bytes& bytes, std::uint32_t value) {
    for (int shift : {24, 16, 8, 0}) bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

Bytes image_header(std::uint32_t count = 1, std::uint32_t rows = 28,
                   std::uint32_t cols = 28, std::uint32_t magic = 2051) {
    Bytes result;
    for (auto value : {magic, count, rows, cols}) u32(result, value);
    return result;
}

Bytes image_data(std::uint32_t count = 1) {
    auto result = image_header(count);
    result.resize(16 + count * 784, 0);
    result[16] = 255;
    result[17] = 128;
    return result;
}

Bytes label_data(Bytes labels = {3}, std::uint32_t magic = 2049) {
    Bytes result;
    u32(result, magic);
    u32(result, static_cast<std::uint32_t>(labels.size()));
    result.insert(result.end(), labels.begin(), labels.end());
    return result;
}

struct Fixture {
    fs::path directory;
    Fixture() {
        const auto unique = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        directory = fs::temp_directory_path() / ("synthetic-mnist-" + std::to_string(unique));
        require(fs::create_directory(directory), "Could not create owned fixture directory");
    }
    ~Fixture() {
        std::error_code error;
        for (fs::directory_iterator it(directory, error), end; it != end && !error; it.increment(error))
            fs::remove(it->path(), error);
        fs::remove(directory, error);
    }
    std::string write(const std::string& name, const Bytes& bytes) {
        const auto file = directory / name;
        std::ofstream stream(file, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        require(bool(stream), "Fixture write failed");
        return file.string();
    }
    void complete() {
        write("train-images-idx3-ubyte", image_data());
        write("train-labels-idx1-ubyte", label_data());
        write("t10k-images-idx3-ubyte", image_data());
        write("t10k-labels-idx1-ubyte", label_data());
    }
};

int main() {
    int passed = 0, failed = 0;
    auto run = [&](const char* name, const std::function<void()>& test) {
        try { test(); ++passed; std::cout << "PASS " << name << '\n'; }
        catch (const std::exception& error) {
            ++failed; std::cerr << "FAIL " << name << ": " << error.what() << '\n';
        }
    };
    run("images: endian decoding and normalization", [] {
        Fixture f; Images images;
        read_images(f.write("images", image_data(2)), images);
        require(images.size() == 2 && images[0].size() == 784, "Wrong dimensions");
        require(images[0][0] == 1.f && images[0][1] == 128.f / 255.f && images[1][0] == 0.f,
                "Wrong pixel normalization");
    });
    run("labels: exact values", [] {
        Fixture f; Bytes labels;
        read_labels(f.write("labels", label_data({0, 9, 4})), labels);
        require(labels == Bytes({0, 9, 4}), "Wrong labels");
    });
    auto bad_images = [&](const char* name, const Bytes& payload) {
        run(name, [&] {
            Fixture f; Images images{{0.25f}};
            const auto file = f.write("images", payload);
            rejects([&] { read_images(file, images); });
            require(images == Images({{0.25f}}), "Failed image read changed output");
        });
    };
    bad_images("images: empty file", {});
    bad_images("images: truncated header", Bytes(12, 0));
    bad_images("images: wrong magic", image_header(1, 28, 28, 2049));
    bad_images("images: zero count", image_header(0));
    bad_images("images: count bound before allocation", image_header(100001));
    bad_images("images: unsigned count bound", image_header(0xffffffffu));
    bad_images("images: invalid rows", image_header(1, 0, 28));
    bad_images("images: invalid columns", image_header(1, 28, 29));
    bad_images("images: hostile dimensions", image_header(1, 0xffffffffu, 0xffffffffu));
    auto short_images = image_data(); short_images.pop_back();
    bad_images("images: truncated payload", short_images);
    auto extra_images = image_data(); extra_images.push_back(0);
    bad_images("images: trailing data", extra_images);

    auto bad_labels = [&](const char* name, const Bytes& payload) {
        run(name, [&] {
            Fixture f; Bytes labels{7};
            const auto file = f.write("labels", payload);
            rejects([&] { read_labels(file, labels); });
            require(labels == Bytes({7}), "Failed label read changed output");
        });
    };
    bad_labels("labels: empty file", {});
    bad_labels("labels: wrong magic", label_data({1}, 2051));
    bad_labels("labels: zero count", label_data({}));
    auto huge_labels = label_data(); huge_labels[4] = 0xff;
    bad_labels("labels: excessive count", huge_labels);
    auto short_labels = label_data(); short_labels.pop_back();
    bad_labels("labels: truncated payload", short_labels);
    auto extra_labels = label_data(); extra_labels.push_back(0);
    bad_labels("labels: trailing data", extra_labels);
    bad_labels("labels: invalid digit", label_data({10}));
    bad_labels("labels: unsigned invalid digit", label_data({255}));

    run("missing files fail without altering outputs", [] {
        Fixture f; Images images{{0.f}}; Bytes labels{4};
        rejects([&] { read_images((f.directory / "missing").string(), images); });
        rejects([&] { read_labels((f.directory / "missing").string(), labels); });
        require(images == Images({{0.f}}) && labels == Bytes({4}), "Output changed");
    });
    run("directory joins with and without separator", [] {
        Fixture f; f.complete(); Images tr, te; Bytes tr_y, te_y;
        for (auto path : {f.directory.string(), f.directory.string() + "/"}) {
            load_mnist(path, tr, tr_y, te, te_y);
            require(tr.size() == 1 && te.size() == 1 && tr_y == Bytes({3}) && te_y == Bytes({3}),
                    "Wrong dataset load");
        }
    });
    auto failed_dataset = [&](const char* name, const std::string& file, const Bytes& payload) {
        run(name, [&] {
            Fixture f; f.complete(); f.write(file, payload);
            Images tr{{0.1f}}, te{{0.2f}}; Bytes tr_y{1}, te_y{2};
            rejects([&] { load_mnist(f.directory.string(), tr, tr_y, te, te_y); });
            require(tr == Images({{0.1f}}) && te == Images({{0.2f}})
                    && tr_y == Bytes({1}) && te_y == Bytes({2}), "Dataset update was not atomic");
        });
    };
    failed_dataset("training count mismatch", "train-labels-idx1-ubyte", label_data({0, 1}));
    failed_dataset("test count mismatch", "t10k-labels-idx1-ubyte", label_data({0, 1}));
    failed_dataset("late parse failure keeps all outputs", "t10k-labels-idx1-ubyte", {});

    run("valid network inputs", [] {
        Images images(2, std::vector<float>(784, 0.5f));
        validate_image(images[0]);
        validate_training_input(images, {0, 9}, 1, 0.01f);
    });
    run("image length guard", [] {
        rejects<std::invalid_argument>([] { validate_image({}); });
        rejects<std::invalid_argument>([] { validate_image(std::vector<float>(783)); });
        rejects<std::invalid_argument>([] { validate_image(std::vector<float>(785)); });
    });
    run("pixel finite and range guards", [] {
        for (float bad : {-0.1f, 1.1f, std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()}) {
            std::vector<float> pixels(784); pixels[5] = bad;
            rejects<std::invalid_argument>([&] { validate_image(pixels); });
        }
    });
    run("training shape and label guards", [] {
        Images images(1, std::vector<float>(784));
        rejects<std::invalid_argument>([&] { validate_training_input({}, {}, 1, 0.01f); });
        rejects<std::invalid_argument>([&] { validate_training_input(images, {}, 1, 0.01f); });
        rejects<std::invalid_argument>([&] { validate_training_input(images, {10}, 1, 0.01f); });
    });
    run("training hyperparameter guards", [] {
        Images images(1, std::vector<float>(784));
        for (int epochs : {0, -1})
            rejects<std::invalid_argument>([&] { validate_training_input(images, {0}, epochs, 0.01f); });
        for (float lr : {0.f, -1.f, std::numeric_limits<float>::infinity(),
                         std::numeric_limits<float>::quiet_NaN()})
            rejects<std::invalid_argument>([&] { validate_training_input(images, {0}, 1, lr); });
    });
    std::cout << passed << " cases passed; " << failed << " failed\n";
    return failed ? 1 : 0;
}
