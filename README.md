# Blur Detection using Haar Wavelet Transform

| Build | Tests |
| :---: | :----: |
| [![Build Status](https://github.com/NehilDanis/blur_detection_using_wavelet_transform/actions/workflows/build.yml/badge.svg)](https://github.com/NehilDanis/blur_detection_using_wavelet_transform/actions/workflows/build.yml) | [![Tests](https://github.com/NehilDanis/blur_detection_using_wavelet_transform/actions/workflows/test.yml/badge.svg)](https://github.com/NehilDanis/blur_detection_using_wavelet_transform/actions/workflows/test.yml) |

This repository includes the implementation of the [Blur Detection for Digital Images Using Wavelet Transform](http://tonghanghang.org/pdfs/icme04_blur.pdf) paper using C++20.

conan install . -sbuild_type=Debug -of=conan/debug --build=missing

cmake -DCMAKE_TOOLCHAIN_FILE=conan/debug/build/Debug/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug -B build/debug -S .
cmake --build build/debug -j4
