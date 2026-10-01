#include "cnn.h"
#include "mnist.h"
#ifdef MNIST_WITH_OPENCV
#include "utils.h"
#endif
#include <algorithm>
#include <exception>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "Usage: cuda_cnn DATA_DIRECTORY [--show]\n"
                     "Trains only the final layer over fixed random convolution features.\n";
        return 0;
    }
    if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--show")) {
        std::cerr << "Usage: cuda_cnn DATA_DIRECTORY [--show]\n";
        return 2;
    }
    const bool show = argc == 3;
#ifndef MNIST_WITH_OPENCV
    if (show) {
        std::cerr << "--show requires a build with ENABLE_OPENCV=ON\n";
        return 2;
    }
#endif
    try {
        std::vector<std::vector<float>> train_x, test_x;
        std::vector<uint8_t> train_y, test_y;
        load_mnist(argv[1], train_x, train_y, test_x, test_y);
        const auto count = std::min<std::size_t>(train_x.size(), 1000);
        train_x.resize(count);
        train_y.resize(count);
        CNN network(28, 28);
        network.train(train_x, train_y, 5, 0.01f);
        for (std::size_t i = 0; i < std::min<std::size_t>(test_x.size(), 10); ++i) {
            const int prediction = network.predict(test_x[i]);
            std::cout << "Sample " << i << ": prediction=" << prediction
                      << ", label=" << static_cast<int>(test_y[i]) << '\n';
#ifdef MNIST_WITH_OPENCV
            if (show) visualize(test_x[i], prediction);
#endif
        }
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
