#include "utils.h"
#include <opencv2/opencv.hpp>
#include <iostream>

void visualize(const std::vector<float>& img, int pred) {
    cv::Mat m(28, 28, CV_32F);
    for (int i = 0;i < 28 * 28;++i) m.at<float>(i / 28, i % 28) = img[i];
    cv::resize(m, m, cv::Size(280, 280), 0, 0, cv::INTER_NEAREST);
    cv::imshow("digit", m);
    std::cout << "Predicted: " << pred << "\n";
    cv::waitKey(0);
}
