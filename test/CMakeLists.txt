# Locate gtest
find_package(GTest REQUIRED)
enable_testing()

# Define a test target
add_executable(test_blur_detection main.cpp
                    test_blur_detection.cpp)
target_link_libraries(test_blur_detection PRIVATE blur_detection_using_haar_wavelet_transform GTest::gtest_main)

add_test(NAME BlurDetectionTests COMMAND test_blur_detection)