cmake_minimum_required(VERSION 3.16)

project(KakuTag VERSION 0.0.1 LANGUAGES CXX)

option(KAKUTAG_BUILD_COMPILE_CHECK "Build a tiny KakuTag compile-check target" ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d objdetect)

add_library(KakuTag INTERFACE)
add_library(KakuTag::KakuTag ALIAS KakuTag)

target_compile_features(KakuTag INTERFACE cxx_std_20)
target_include_directories(KakuTag INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
target_link_libraries(KakuTag INTERFACE ${OpenCV_LIBS})

if(KAKUTAG_BUILD_COMPILE_CHECK)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/kakutag_compile_check.cpp" [=[
#include "kakutag.hpp"

#include <opencv2/objdetect/aruco_detector.hpp>

int main() {
    auto dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    kakutag::DetectorParameters params;
    params.dicts = {dict};

    kakutag::ArucoDetector detector(params);
    (void)detector;
    return 0;
}
]=])

    add_executable(kakutag_compile_check
        "${CMAKE_CURRENT_BINARY_DIR}/kakutag_compile_check.cpp"
    )
    target_link_libraries(kakutag_compile_check PRIVATE KakuTag::KakuTag)
endif()
