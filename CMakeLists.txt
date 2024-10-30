cmake_minimum_required(VERSION 3.25)
project(blur_detection_using_haar_wavelet_transform LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)
option(BUILD_TESTING "Build the tests" ON)

find_package(OpenCV REQUIRED)
find_package(range-v3 REQUIRED)

add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} INTERFACE opencv::opencv range-v3::range-v3)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Only include tests if this is the test context
if(BUILD_TESTING)
    add_subdirectory(test)
endif()