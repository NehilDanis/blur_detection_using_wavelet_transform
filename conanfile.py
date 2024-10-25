from conan import ConanFile
from conan.tools.cmake import cmake_layout

class BlurDetectionusingWaveletTransforms(ConanFile):
    generators = ("CMakeDeps", "CMakeToolchain")
    settings = ("os", "build_type", "arch", "compiler")
    def requirements(self):
        self.requires("opencv/4.10.0")
    
    def build_requirements(self):
        self.tool_requires("cmake/[>=3.25]")
        self.test_requires("gtest/1.15.0")

    def configure(self):
        pass

    def layout(self):
        cmake_layout(self)