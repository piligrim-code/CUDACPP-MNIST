#include "cnn.h"
#include "mnist.h"
#include "utils.h"
#include <iostream>

int main() {
    std::vector<std::vector<float>> trX, teX;
    std::vector<uint8_t> trY, teY;
    load_mnist("data/", trX, trY, teX, teY);
    // возьмём по 1000 для скорости
    trX.resize(1000); trY.resize(1000);
    teX.resize(10);   teY.resize(10);

    CNN net(28, 28);
    net.train(trX, trY, 5, 0.01f);

    for (int i = 0;i < 10;++i) {
        int p = net.predict(teX[i]);
        visualize(teX[i], p);
    }
    return 0;
}
