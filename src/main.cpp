#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

int main() {

    std::cout << "Hello" << std::endl;
    cv::Mat img = cv::imread("test/data/lena.png", cv::IMREAD_COLOR);
    cv::imshow("lena", img);
    cv::waitKey();
    return 0;
}