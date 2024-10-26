
// example_test.cpp
#include <gtest/gtest.h>
#include <blur_detection/blur_detection.h>

TEST(BlurDetectionUsingHaarWaveletTransforms, testBlurriness) {
    auto output = blur_detection::is_blur();
    ASSERT_FALSE(output.has_value());
}
