cmake_minimum_required(VERSION 3.16)

project(KakuTag VERSION 0.0.1 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(KAKUTAG_MIN_OPENCV_VERSION 4.7)

option(KAKUTAG_BUILD_COMPILE_CHECK "Build a tiny KakuTag compile-check target" ON)
option(KAKUTAG_BUILD_TESTS "Build KakuTag smoke tests" OFF)
option(KAKUTAG_BUILD_BENCHMARKS "Build KakuTag benchmark tools" OFF)

find_package(OpenCV ${KAKUTAG_MIN_OPENCV_VERSION} REQUIRED COMPONENTS core imgproc imgcodecs calib3d objdetect)

add_library(KakuTag INTERFACE)
add_library(KakuTag::KakuTag ALIAS KakuTag)

target_compile_features(KakuTag INTERFACE cxx_std_20)
target_include_directories(KakuTag INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(TARGET opencv_core)
    target_link_libraries(KakuTag INTERFACE
        opencv_core
        opencv_imgproc
        opencv_calib3d
        opencv_objdetect
    )
else()
    target_link_libraries(KakuTag INTERFACE ${OpenCV_LIBS})
endif()

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

include(CTest)
if(KAKUTAG_BUILD_TESTS)
    add_executable(kakutag_smoke_tests tests/smoke.cpp)
    target_link_libraries(kakutag_smoke_tests PRIVATE KakuTag::KakuTag)
    add_test(NAME kakutag_smoke_tests COMMAND kakutag_smoke_tests)
endif()

if(KAKUTAG_BUILD_BENCHMARKS)
    add_executable(kakutag_shadow_aruco_bench bench/shadow_aruco_bench.cpp)
    target_compile_definitions(kakutag_shadow_aruco_bench PRIVATE
        KAKUTAG_HAS_RECALL_PROFILES=1
    )
    target_link_libraries(kakutag_shadow_aruco_bench PRIVATE KakuTag::KakuTag)
    if(TARGET opencv_imgcodecs)
        target_link_libraries(kakutag_shadow_aruco_bench PRIVATE opencv_imgcodecs)
    endif()
endif()

install(TARGETS KakuTag
    EXPORT KakuTagTargets
)
install(FILES kakutag.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(EXPORT KakuTagTargets
    NAMESPACE KakuTag::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KakuTag
)

configure_package_config_file(
    cmake/KakuTagConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/KakuTagConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KakuTag
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/KakuTagConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/KakuTagConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/KakuTagConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/KakuTag
)
