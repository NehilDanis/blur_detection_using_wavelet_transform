# Blur Detection using Haar Wavelet Transform
This repository includes the implementation of the [Blur Detection for Digital Images Using Wavelet Transform](http://tonghanghang.org/pdfs/icme04_blur.pdf) paper using C++20.

conan install . -sbuild_type=Debug -of=conan/debug --build=missing

cmake -DCMAKE_TOOLCHAIN_FILE=conan/debug/build/Debug/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug -B build/debug -S .
cmake --build build/debug -j4
