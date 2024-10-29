
// GTest
#include <gtest/gtest.h>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <blur_detection/blur_detection.h>


TEST(BlurDetectionUsingHaarWaveletTransforms, testBlurriness) {

    const auto original_image_path = "test/data/lena.png";
    cv::Mat img = cv::imread(original_image_path, cv::IMREAD_COLOR);
    static constexpr float threshold = 35;
    static constexpr float min_zero = 0.05;


    auto output = blur_detection::is_blur(img, threshold, min_zero);
    ASSERT_FALSE(output.has_value());
}
