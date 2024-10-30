#ifndef BLUR_DETECTION_H  // Check if MY_HEADER_H is not defined
#define BLUR_DETECTION_H  // Define MY_HEADER_H to prevent future inclusions

// STL
#include <optional>

// OpenCV
#include <opencv2/core.hpp>

namespace blur_detection
{
namespace detail
{

/**
 * @brief This function takes a source image and applies one level of the haar
 * filter and outputs low and high spatial resolution outputs.
 *
 * @param LL
 * @param LH
 * @param HL
 * @param HH
 */
inline auto haar_transform(cv::InputArray src, cv::InputOutputArray LL,
                           cv::InputOutputArray LH, cv::InputOutputArray HL,
                           cv::InputOutputArray HH) -> void
{
}

/**
 * @brief This functions takes the high vertical, horizontal and diagonal
 * information images, and constructs an edge map. All three matrices, LH, HL,
 * and HH, should have the same shape. The formula used to consturct edge map is
 * as follows sqrt(LH^2 + HL^2 + HH^2)
 *
 * @param LH
 * @param HL
 * @param HH
 * @param edge_map
 */
inline auto calculate_edge_map(cv::InputArray LH, cv::InputArray HL,
                               cv::InputArray HH,
                               std::vector<float>& edge_map) -> void
{
}

/**
 * @brief This function calculates the maximum values in the edge map, using a
 * filter_size x filter_size filter. The step size is equal to the filter_size.
 * So we get a coaser map than the input edge_map. The maximum edge_map will be
 * the size of floor(edge_map/filter_size)
 *
 * @param edge_map
 * @param filter_size
 * @param max_edge_map
 */
inline auto calculate_max_edge_map(cv::InputArray edge_map, size_t filter_size,
                                   std::vector<float>& max_edge_map) -> void
{
    size_t stride = filter_size;
    const auto& edge_map_mat = edge_map.getMat();
    auto num_row_pass = (edge_map_mat.rows / filter_size);
    auto num_col_pass = (edge_map_mat.cols / filter_size);
    size_t output_size =
        (edge_map_mat.rows / filter_size) * (edge_map_mat.cols / filter_size);
    max_edge_map.resize(output_size);

    for (auto r = 0U; r < num_row_pass; r++)
    {
        for (auto c = 0U; c < num_col_pass; c++) {
            auto start_point_x = c * filter_size;
            auto start_point_y = r * filter_size; 
            const cv::Mat& tmp = edge_map_mat(cv::Rect(start_point_x, start_point_y, filter_size, filter_size));
            auto max_itr = std::max_element(tmp.begin<float>(), tmp.end<float>());
            max_edge_map.at(r * num_col_pass + c )  = *max_itr;
        }
    }
}

}  // namespace detail

/**
 * @brief This function checks if the image is blurry or not. If the image is
 * blurry the blur extent will be returned.
 *
 * @param img Image to check the blurriness
 * @param threshold This value can be in [0, 255]. But since the human eye
 * cannot distinguish gray scale value below 30, providing a value above that is
 * reasonable.
 * @param min_zero This value cannot be lower than 0. This value is used to
 * compare against the ratio of number of dirac and astep stuctures in all the
 * edges. If the ratio is larger than the min_zero value than the image is
 * considered un-blurred. In the blurred version there should not be any dirac
 * or astep structures.
 *
 * @returns The blur extent in case of a blurry image, otherwise nullopt.
 */
inline auto is_blur(cv::InputArray img, float threshold,
                    float min_zero) -> std::optional<float>
{
    // conver the image to gray scale, and the type to float

    // 3-Level Haar Transform

    // find edge_maps on these 3 levels

    // find maximum edge_map for all levels using 8x8, 4x4, and 2x2 filters,
    // from largest image to smallest

    // Apply algorithm 2 to see if the image is blurry or not
    // if yes to what extent

    return std::nullopt;
}
}  // namespace blur_detection

#endif  // BLUR_DETECTION_H