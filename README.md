# Blur Detection using Haar Wavelet Transform

| Build Status | Test Status |
| :---: | :----: |
| ![Build Status](https://github.com/NehilDanis/blur_detection_using_wavelet_transform/actions/workflows/build_test.yml/badge.svg?job=build) | ![Test Status](https://github.com/NehilDanis/blur_detection_using_wavelet_transform/actions/workflows/build_test.yml/badge.svg?job=test) |

This repository includes the implementation of the [Blur Detection for Digital Images Using Wavelet Transform](http://tonghanghang.org/pdfs/icme04_blur.pdf) paper using C++20.

## Step by step usage guide

After cloning this repository to your local machine follow the steps below.

* [Optional] Do this if you do not have conan installed. Create a python environment and install the requirements.

```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements
```

* Install the required packages to compile this software. You can set the build type as Debug or as Release, remember to then change the output file name from conan/debug to conan/release for more redability.
```bash
$ conan install . -sbuild_type=Debug -of=conan/debug --build=missing
```

Ps. if does not work you might need to detect your conan profile, you can do this simply running the following command:
```bash
$ conan profile detect
```
This basically go through your computer settings (architecture, build type, OS, the compiler etc.) and creates a conan profile. This way conan can obtain the correct packages installed according to your system.

* Run the following line to set up the compilation configuration. Please adapt the following command depending on you want to compile in debug or in release mode.
```bash
$ cmake -DCMAKE_TOOLCHAIN_FILE=conan/debug/build/Debug/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Debug -B build/debug -S .
```
* Run the following command for compilation:
```bash
$ cmake --build build/debug -j4
```
