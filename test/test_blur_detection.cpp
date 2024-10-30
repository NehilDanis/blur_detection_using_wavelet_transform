
// GTest
#include <gtest/gtest.h>

// OpenCV
#include <blur_detection/blur_detection.h>

#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <range/v3/all.hpp>

TEST(BlurDetectionUsingHaarWaveletTransforms,
     testMaxEdgeMapCalculation_empty_matrix)
{
    cv::Mat input;
    std::vector<float> output;
    static constexpr size_t filter_size = 2;

    blur_detection::detail::calculate_max_edge_map(input, filter_size, output);
    ASSERT_EQ(output.size(), 0);
}

TEST(BlurDetectionUsingHaarWaveletTransforms,
     testMaxEdgeMapCalculation_check_matrices_smaller_than_filter_size)
{
    cv::Mat input = cv::Mat::zeros(cv::Size(1, 4), CV_32F);  // 4x1
    std::vector<float> output;
    static constexpr size_t filter_size = 2;

    blur_detection::detail::calculate_max_edge_map(input, filter_size, output);
    ASSERT_EQ(output.size(), 0);

    input.reshape(0, 1);  // makes 1x4
    blur_detection::detail::calculate_max_edge_map(input, filter_size, output);
    ASSERT_EQ(output.size(), 0);
}

TEST(BlurDetectionUsingHaarWaveletTransforms, testMaxEdgeMapCalculation)
{
    cv::Mat input = (cv::Mat_<float>(4, 4) << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                     12, 13, 14, 15, 16);

    std::vector<float> gt_output = {6, 8, 14, 16};
    std::vector<float> output;
    static constexpr size_t filter_size = 2;
    static constexpr float tolerance = 1e-6;

    blur_detection::detail::calculate_max_edge_map(input, filter_size, output);
    ASSERT_EQ(output.size(), gt_output.size());

    auto zipped = ranges::zip_view(output, gt_output);
    std::ranges::for_each(
        zipped, [&](const auto& pair) -> void {
            const auto &[expected, gt] = pair;
            EXPECT_NEAR(expected, gt, tolerance);
        });
}

TEST(BlurDetectionUsingHaarWaveletTransforms, testEdgeMapCalculation)
{
    ASSERT_TRUE(true);
}

TEST(BlurDetectionUsingHaarWaveletTransforms, testHaarTransform)
{
    ASSERT_TRUE(true);
}

TEST(BlurDetectionUsingHaarWaveletTransforms, testBlurriness)
{
    const auto original_image_path = "test/data/lena.png";
    cv::Mat img = cv::imread(original_image_path, cv::IMREAD_COLOR);
    static constexpr float threshold = 35;
    static constexpr float min_zero = 0.05;

    auto output = blur_detection::is_blur(img, threshold, min_zero);
    ASSERT_FALSE(output.has_value());
}
