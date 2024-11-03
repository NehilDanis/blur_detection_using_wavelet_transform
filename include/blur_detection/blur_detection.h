#ifndef BLUR_DETECTION_H  // BLUR_DETECTION_H
#define BLUR_DETECTION_H  // BLUR_DETECTION_H

// STL
#include <algorithm>
#include <concepts>
#include <optional>
#include <range/v3/all.hpp>
#include <ranges>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/traits.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace blur_detection
{
/**
 * This algorithm needs the floating point precision to be able to work
 * correctly. And since there is no direct type for long double in OpenCV,
 * std::is_floating_point is not used here.
 */
template <typename T>
concept Floating_point_type =
    std::is_same_v<T, float> || std::is_same_v<T, double>;

namespace detail
{

/**
 * @brief This function takes a source image and applies one level of the haar
 * filter and outputs low and high spatial resolution outputs.
 *
 * @param LL weighted average
 * @param LH vertical detail
 * @param HL horizontal detail
 * @param HH diagonal detail
 */
template <Floating_point_type T>
inline auto haar_transform(const cv::Mat& src, cv::OutputArray LL,
                           cv::OutputArray LH, cv::OutputArray HL,
                           cv::OutputArray HH) -> void
{
    CV_Assert(src.type() == cv::traits::Type<T>::value);
    constexpr size_t side = 2;
    CV_Assert(src.rows >= side && src.cols >= 2);
    auto filter_size = cv::Size(side, side);
    T coeff = static_cast<T>(0.5f);

    cv::Mat ll_kernel =
        (cv::Mat_<float>(filter_size) << coeff, coeff, coeff, coeff);
    cv::Mat lh_kernel =
        (cv::Mat_<float>(filter_size) << coeff, -coeff, coeff, -coeff);
    cv::Mat hl_kernel =
        (cv::Mat_<float>(filter_size) << coeff, coeff, -coeff, -coeff);
    cv::Mat hh_kernel =
        (cv::Mat_<float>(filter_size) << coeff, -coeff, -coeff, coeff);

    int num_row_pass = src.rows / side;
    int num_col_pass = src.cols / side;

    LL.create(num_row_pass, num_col_pass, src.type());
    LH.create(num_row_pass, num_col_pass, src.type());
    HL.create(num_row_pass, num_col_pass, src.type());
    HH.create(num_row_pass, num_col_pass, src.type());

    auto ll_tmp = LL.getMat();
    auto lh_tmp = LH.getMat();
    auto hl_tmp = HL.getMat();
    auto hh_tmp = HH.getMat();

    for (auto r = 0U; r < num_row_pass; r++)
    {
        for (auto c = 0U; c < num_col_pass; c++)
        {
            auto start_row = r * side;
            auto start_col = c * side;
            const auto& rect_tmp =
                src(cv::Rect(start_col, start_row, side, side));
            // element wise multiplication and summation of all elements.
            ll_tmp.at<T>(start_row, start_col) =
                cv::sum(rect_tmp.mul(ll_kernel)).val[0];
            lh_tmp.at<T>(start_row, start_col) =
                cv::sum(rect_tmp.mul(lh_kernel)).val[0];
            hl_tmp.at<T>(start_row, start_col) =
                cv::sum(rect_tmp.mul(hl_kernel)).val[0];
            hh_tmp.at<T>(start_row, start_col) =
                cv::sum(rect_tmp.mul(hh_kernel)).val[0];
        }
    }
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
template <Floating_point_type T>
inline auto calculate_edge_map(const cv::Mat& LH, const cv::Mat& HL,
                               const cv::Mat& HH,
                               cv::OutputArray edge_map) -> void
{
    CV_Assert(LH.type() == HL.type() && HL.type() == HH.type() &&
              LH.type() == cv::traits::Type<T>::value &&
              LH.size() == HL.size() && HL.size() == HH.size());

    int height = LH.rows;
    int width = LH.cols;

    edge_map.create(cv::Size(width, height), LH.type());

    auto edge_map_mat = edge_map.getMat();

    auto lh_range = std::ranges::subrange(LH.begin<T>(), LH.end<T>());
    auto hl_range = std::ranges::subrange(HL.begin<T>(), HL.end<T>());
    auto hh_range = std::ranges::subrange(HH.begin<T>(), HH.end<T>());
    auto edge_map_range =
        std::ranges::subrange(edge_map_mat.begin<T>(), edge_map_mat.end<T>());

    auto zipped = ranges::views::zip(lh_range, hl_range, hh_range);
    std::ranges::transform(zipped.begin(), zipped.end(), edge_map_range.begin(),
                           [](const auto tuple_elem)
                           {
                               const auto [lh, hl, hh] = tuple_elem;
                               return std::sqrt(std::pow(lh, 2) +
                                                std::pow(hl, 2) +
                                                std::pow(hh, 2));
                           });
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
template <Floating_point_type T>
inline auto calculate_max_edge_map(const cv::Mat& edge_map, size_t filter_size,
                                   std::vector<T>& max_edge_map) -> void
{
    CV_Assert(edge_map.type() == cv::traits::Type<T>::value);
    size_t stride = filter_size;
    auto num_row_pass = (edge_map.rows / filter_size);
    auto num_col_pass = (edge_map.cols / filter_size);
    size_t output_size =
        (edge_map.rows / filter_size) * (edge_map.cols / filter_size);
    max_edge_map.resize(output_size);

    for (auto r = 0U; r < num_row_pass; r++)
    {
        for (auto c = 0U; c < num_col_pass; c++)
        {
            auto start_point_x = c * filter_size;
            auto start_point_y = r * filter_size;
            const cv::Mat& tmp = edge_map(cv::Rect(start_point_x, start_point_y,
                                                   filter_size, filter_size));
            auto max_itr = std::max_element(tmp.begin<T>(), tmp.end<T>());
            max_edge_map.at(r * num_col_pass + c) = *max_itr;
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
template <Floating_point_type T>
inline auto is_blur(const cv::Mat& img, T threshold,
                    T min_zero) -> std::optional<T>
{
    cv::Mat gray_img;
    img.copyTo(gray_img);
    // convert the image to gray scale
    if (img.channels() == 3)
    {
        cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    }

    // convert to floating point number but keep the intensity values between
    // [0, 255]
    if (std::is_same_v<T, float>)
    {  // in case T is float
        gray_img.convertTo(gray_img, CV_32F);
    }
    else
    {  // in case T is double
        gray_img.convertTo(gray_img, CV_64F);
    }

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