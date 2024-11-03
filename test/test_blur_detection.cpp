// GTest
#include <gtest/gtest.h>

// OpenCV
#include <blur_detection/blur_detection.h>

#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <range/v3/all.hpp>

namespace
{
static constexpr float TOLERANCE = 1e-6;
}

TEST(BlurDetectionUsingHaarWaveletTransforms,
     testMaxEdgeMapCalculationMatricesSmallerThanFilterSsize)
{
    cv::Mat input =
        cv::Mat::zeros(cv::Size(1, 4), cv::DataType<float>::type);  // 4x1
    std::vector<float> output;
    static constexpr size_t filter_size = 2;

    blur_detection::detail::calculate_max_edge_map<float>(input, filter_size,
                                                          output);
    ASSERT_EQ(output.size(), 0);

    input.reshape(0, 1);  // makes 1x4
    blur_detection::detail::calculate_max_edge_map<float>(input, filter_size,
                                                          output);
    ASSERT_EQ(output.size(), 0);
}

TEST(BlurDetectionUsingHaarWaveletTransforms, testMaxEdgeMapCalculation)
{
    cv::Mat input = (cv::Mat_<float>(4, 4) << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                     12, 13, 14, 15, 16);

    std::vector<float> gt_output = {6, 8, 14, 16};
    std::vector<float> output;
    constexpr size_t filter_size = 2;

    blur_detection::detail::calculate_max_edge_map<float>(input, filter_size,
                                                          output);
    ASSERT_EQ(output.size(), gt_output.size());

    auto zipped = ranges::zip_view(output, gt_output);
    std::ranges::for_each(zipped,
                          [&](const auto& pair) -> void
                          {
                              const auto& [expected, gt] = pair;
                              EXPECT_NEAR(expected, gt, TOLERANCE);
                          });
}

TEST(BlurDetectionUsingHaarWaveletTransforms, testEdgeMapCalculation)
{
    constexpr int row_col_length = 3;
    cv::Mat LH{cv::Mat::zeros(cv::Size(row_col_length, row_col_length),
                              cv::DataType<float>::type)};
    cv::Mat HL{cv::Mat::zeros(cv::Size(row_col_length, row_col_length),
                              cv::DataType<float>::type)};
    cv::Mat HH{cv::Mat::zeros(cv::Size(row_col_length, row_col_length),
                              cv::DataType<float>::type)};
    cv::Mat edge_map;

    blur_detection::detail::calculate_edge_map<float>(LH, HL, HH, edge_map);

    ASSERT_EQ(edge_map.rows, row_col_length);
    ASSERT_EQ(edge_map.cols, row_col_length);

    std::for_each(edge_map.begin<float>(), edge_map.end<float>(),
                  [&](auto& elem) { ASSERT_NEAR(elem, 0, TOLERANCE); });

    constexpr float multiplier = 2;

    LH = (LH + 1) * multiplier;
    HL = (HL + 1) * multiplier * multiplier;
    HH = (HH + 1) * multiplier * multiplier * multiplier;

    auto output = std::sqrt(std::pow(multiplier, 6) + std::pow(multiplier, 4) +
                            std::pow(multiplier, 2));

    blur_detection::detail::calculate_edge_map<float>(LH, HL, HH, edge_map);

    ASSERT_EQ(edge_map.rows, row_col_length);
    ASSERT_EQ(edge_map.cols, row_col_length);

    std::for_each(edge_map.begin<float>(), edge_map.end<float>(),
                  [&](auto& elem) { ASSERT_NEAR(elem, output, TOLERANCE); });
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

    auto output = blur_detection::is_blur<float>(img, threshold, min_zero);
    ASSERT_FALSE(output.has_value());
}
