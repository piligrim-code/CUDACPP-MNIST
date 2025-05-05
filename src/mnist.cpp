#include "mnist.h"
#include <fstream>
#include <stdexcept>

static int rev(int x) {
    uint8_t a = x & 0xFF, b = (x >> 8) & 0xFF, c = (x >> 16) & 0xFF, d = (x >> 24) & 0xFF;
    return (a << 24) | (b << 16) | (c << 8) | d;
}

void read_images(const std::string& fn, std::vector<std::vector<float>>& M) {
    std::ifstream f(fn, std::ios::binary);
    if (!f) throw std::runtime_error("no " + fn);
    int m, n, r, c; f.read((char*)&m, 4); f.read((char*)&n, 4);
    m = rev(m); n = rev(n); r = rev(r); c = rev(c);
    M.resize(n, std::vector<float>(r * c));
    for (int i = 0;i < n;++i)for (int j = 0;j < r * c;++j) {
        uint8_t p; f.read((char*)&p, 1);
        M[i][j] = p / 255.f;
    }
}

void read_labels(const std::string& fn, std::vector<uint8_t>& L) {
    std::ifstream f(fn, std::ios::binary);
    if (!f) throw std::runtime_error("no " + fn);
    int m, n; f.read((char*)&m, 4); f.read((char*)&n, 4);
    n = rev(n); L.resize(n);
    f.read((char*)L.data(), n);
}

void load_mnist(const std::string& path,
    std::vector<std::vector<float>>& A,
    std::vector<uint8_t>& B,
    std::vector<std::vector<float>>& C,
    std::vector<uint8_t>& D) {
    read_images(path + "train-images-idx3-ubyte", A);
    read_labels(path + "train-labels-idx1-ubyte", B);
    read_images(path + "t10k-images-idx3-ubyte", C);
    read_labels(path + "t10k-labels-idx1-ubyte", D);
}
