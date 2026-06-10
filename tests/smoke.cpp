#include "kakutag.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <iostream>
#include <vector>

namespace {

bool contains_id(const std::vector<kakutag::Marker>& markers, int expected_id) {
    for (const auto& marker : markers) {
        if (marker.id == expected_id) return true;
    }
    return false;
}

cv::Mat make_scene(int marker_id) {
    auto dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Mat marker;
    cv::aruco::generateImageMarker(dict, marker_id, 180, marker, 1);

    cv::Mat scene(360, 360, CV_8UC1, cv::Scalar(255));
    marker.copyTo(scene(cv::Rect(90, 90, marker.cols, marker.rows)));
    return scene;
}

int require(bool condition, const char* message) {
    if (condition) return 0;
    std::cerr << "kakutag smoke test failed: " << message << "\n";
    return 1;
}

} // namespace

int main() {
    constexpr int kMarkerId = 23;
    cv::Mat scene = make_scene(kMarkerId);

    kakutag::DetectorParameters params;
    params.mode = kakutag::Mode::Full;
    params.refine_corners = true;

    kakutag::ArucoDetector detector(params);
    const auto& first = detector.detect(scene);
    if (int rc = require(contains_id(first, kMarkerId), "native detect() missed generated marker")) {
        return rc;
    }

    std::vector<kakutag::Marker> copied;
    detector.detect(scene, copied);
    if (int rc = require(contains_id(copied, kMarkerId), "caller-owned detect() output missed generated marker")) {
        return rc;
    }

    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    detector.detectMarkers(scene, corners, ids);
    if (int rc = require(!ids.empty() && ids[0] == kMarkerId, "detectMarkers() missed generated marker")) {
        return rc;
    }

    cv::Mat blank(scene.size(), scene.type(), cv::Scalar(255));
    const auto& negative = detector.detect(blank);
    if (int rc = require(negative.empty(), "blank image produced false marker")) {
        return rc;
    }

    kakutag::DetectorParameters inverted_params = params;
    inverted_params.detect_inverted_marker = true;
    kakutag::ArucoDetector inverted_detector(inverted_params);
    cv::Mat inverted;
    cv::bitwise_not(scene, inverted);
    const auto& inverted_markers = inverted_detector.detect(inverted);
    if (int rc = require(contains_id(inverted_markers, kMarkerId), "inverted marker rescue missed generated marker")) {
        return rc;
    }

    return 0;
}
