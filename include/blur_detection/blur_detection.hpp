#ifndef BLUR_DETECTION_H  // BLUR_DETECTION_HPP
#define BLUR_DETECTION_H  // BLUR_DETECTION_HPP

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
concept floating_point_type =
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
template <floating_point_type T>
inline auto haar_transform(const cv::Mat& src, cv::OutputArray LL,
                           cv::OutputArray LH, cv::OutputArray HL,
                           cv::OutputArray HH) -> void
{
    CV_Assert(src.type() == cv::traits::Type<T>::value);
    constexpr size_t side = 2;
    CV_Assert(src.rows >= side && src.cols >= 2);

    const auto filter_size = cv::Size(side, side);
    const auto coeff = static_cast<T>(0.5f);

    cv::Mat ll_kernel =
        (cv::Mat_<T>(filter_size) << coeff, coeff, coeff, coeff);
    cv::Mat lh_kernel =
        (cv::Mat_<T>(filter_size) << coeff, -coeff, coeff, -coeff);
    cv::Mat hl_kernel =
        (cv::Mat_<T>(filter_size) << coeff, coeff, -coeff, -coeff);
    cv::Mat hh_kernel =
        (cv::Mat_<T>(filter_size) << coeff, -coeff, -coeff, coeff);

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

    for (int r = 0; r < num_row_pass; r++)
    {
        for (int c = 0; c < num_col_pass; c++)
        {
            auto start_row = r * side;
            auto start_col = c * side;
            const auto& rect_tmp =
                src(cv::Rect(start_col, start_row, side, side));
            cv::Mat tmp = rect_tmp.mul(ll_kernel);
            // element wise multiplication and summation of all elements.
            ll_tmp.at<T>(r, c) = cv::sum(rect_tmp.mul(ll_kernel)).val[0];
            lh_tmp.at<T>(r, c) = cv::sum(rect_tmp.mul(lh_kernel)).val[0];
            hl_tmp.at<T>(r, c) = cv::sum(rect_tmp.mul(hl_kernel)).val[0];
            hh_tmp.at<T>(r, c) = cv::sum(rect_tmp.mul(hh_kernel)).val[0];
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
template <floating_point_type T>
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
template <floating_point_type T>
inline auto calculate_max_edge_map(const cv::Mat& edge_map, size_t filter_size,
                                   std::vector<T>& max_edge_map) -> void
{
    CV_Assert(edge_map.type() == cv::traits::Type<T>::value);

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

template <floating_point_type T>
inline auto check_blur_extent(std::span<T> max_edge_map_1,
                              std::span<T> max_edge_map_2,
                              std::span<T> max_edge_map_3, T threshold,
                              T min_zero) -> std::pair<bool, T>
{
    assert(max_edge_map_1.size() == max_edge_map_2.size() &&
           max_edge_map_2.size() == max_edge_map_3.size());

    auto zipped =
        ranges::views::zip(max_edge_map_1, max_edge_map_2, max_edge_map_3);

    auto edge_count = 0U;
    auto n_da = 0U;
    auto n_rg = 0U;
    auto n_brg = 0U;

    auto rule_1 = [threshold](auto zipped)
    {
        const auto& [emax_1, emax_2, emax_3] = zipped;
        return emax_1 > threshold || emax_2 > threshold || emax_3 > threshold;
    };

    auto rule_2 = [](auto zipped)
    {
        const auto& [emax_1, emax_2, emax_3] = zipped;
        return emax_1 > emax_2 && emax_2 > emax_3;
    };

    auto rule_3_4 = [](auto zipped)
    {
        const auto& [emax_1, emax_2, emax_3] = zipped;
        return (emax_1 < emax_2 && emax_2 < emax_3) ||
               (emax_2 > emax_1 && emax_2 > emax_3);
    };

    auto rule_5 = [threshold](auto zipped)
    {
        const auto& [emax_1, emax_2, emax_3] = zipped;
        return emax_1 < threshold;
    };

    auto edges = zipped | std::views::filter(rule_1);

    auto dirac_astep_structures = edges | std::views::filter(rule_2);

    edge_count = std::ranges::distance(edges);
    n_da = std::ranges::distance(dirac_astep_structures);

    auto per = static_cast<T>(n_da) / static_cast<T>(edge_count);
    auto is_blurry = true;
    if (per > min_zero) { is_blurry = false; }

    auto roof_gstep_structures = edges | std::views::filter(rule_3_4);
    auto blurred_roof_gstep_structures =
        roof_gstep_structures | std::views::filter(rule_5);

    n_rg = std::ranges::distance(roof_gstep_structures);
    n_brg = std::ranges::distance(blurred_roof_gstep_structures);

    // calculate the blur extent
    const auto blur_extent = static_cast<T>(n_brg) / static_cast<T>(n_rg);

    return std::make_pair(is_blurry, blur_extent);
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
template <floating_point_type T>
inline auto is_blur(const cv::Mat& img, T threshold,
                    T min_zero) -> std::pair<bool, T>
{
    cv::Mat gray_img = img.clone();
    // convert the image to gray scale
    if (img.channels() == 3)
    {
        cv::cvtColor(img, gray_img, cv::COLOR_BGR2GRAY);
    }

    const auto alpha = static_cast<T>(1.0f / 255.0f);
    //  the data type of the gray image is set to either CV_32F or CV_64F using
    //  T the values in the gray image scaled using 1 / 255 to bring all the
    //  values to [0, 1] range
    gray_img.convertTo(gray_img, std::is_same_v<T, float> ? CV_32F : CV_64F,
                       alpha);

    // since image values are now in [0, 1] range threshold is also scaled using
    // 1 / 255
    threshold *= alpha;

    // 3-Level Haar Transform
    cv::Mat LL1{}, LH1{}, HL1{}, HH1{};
    detail::haar_transform<T>(gray_img, LL1, LH1, HL1,
                              HH1);  // apply transform on gray img

    cv::Mat LL2{}, LH2{}, HL2{}, HH2{};
    detail::haar_transform<T>(LL1, LL2, LH2, HL2,
                              HH2);  // apply transform on LL1

    cv::Mat LL3{}, LH3{}, HL3{}, HH3{};
    detail::haar_transform<T>(LL2, LL3, LH3, HL3,
                              HH3);  // apply transform on LL2

    // find edge_maps on these 3 levels
    std::array<cv::Mat, 3> edge_maps{};

    blur_detection::detail::calculate_edge_map<T>(LH1, HL1, HH1,
                                                  edge_maps.at(0));
    blur_detection::detail::calculate_edge_map<T>(LH2, HL2, HH2,
                                                  edge_maps.at(1));
    blur_detection::detail::calculate_edge_map<T>(LH3, HL3, HH3,
                                                  edge_maps.at(2));

    // find maximum edge_map for all levels using 8x8, 4x4, and 2x2 filters,
    // from largest image to smallest
    std::array<std::vector<T>, 3> max_edge_maps{};
    constexpr size_t filter_size = 2;
    blur_detection::detail::calculate_max_edge_map<T>(
        edge_maps.at(0), std::pow(filter_size, 3),
        max_edge_maps.at(0));  // 8 x 8
    blur_detection::detail::calculate_max_edge_map<T>(
        edge_maps.at(1), std::pow(filter_size, 2),
        max_edge_maps.at(1));  // 4 x 4
    blur_detection::detail::calculate_max_edge_map<T>(
        edge_maps.at(2), filter_size,
        max_edge_maps.at(2));  // 2 x 2

    return detail::check_blur_extent<T>(
        max_edge_maps.at(0), max_edge_maps.at(1), max_edge_maps.at(2),
        threshold, min_zero);
}
}  // namespace blur_detection

#endif  // BLUR_DETECTION_HPP