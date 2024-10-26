#ifndef BLUR_DETECTION_H    // Check if MY_HEADER_H is not defined
#define BLUR_DETECTION_H    // Define MY_HEADER_H to prevent future inclusions

// STL
#include <optional>

// OpenCV
#include <opencv2/core.hpp>

namespace blur_detection {
    inline auto is_blur() -> std::optional<float> {
        return std::nullopt;
    }
} // namespace blur_detection

#endif // BLUR_DETECTION_H