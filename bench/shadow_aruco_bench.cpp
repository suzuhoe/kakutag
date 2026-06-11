#include "kakutag.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct MarkerLabel {
    int id = -1;
    std::array<cv::Point2f, 4> corners{};
};

struct RunStats {
    std::string name;
    std::vector<double> ms;
    std::size_t tp = 0;
    std::size_t fp = 0;
    std::size_t fn = 0;
    double corner_error_sum = 0.0;
};

struct Options {
    fs::path dataset_root;
    int max_frames = 300;
    int stride = 1;
    double corner_threshold_px = 20.0;
    std::string dict_name = "4x4_250";
    std::string profile = "default";
};

[[noreturn]] void usage(const char* argv0) {
    std::cerr
        << "usage: " << argv0 << " DATASET_ROOT [--max-frames N] [--stride N]"
        << " [--corner-threshold PX] [--dict NAME] [--profile default|balanced|high]\n"
        << "dict names: 4x4_50, 4x4_100, 4x4_250, 4x4_1000,"
        << " 5x5_50, 5x5_100, 5x5_250, 5x5_1000,"
        << " 6x6_50, 6x6_100, 6x6_250, 6x6_1000,"
        << " 7x7_50, 7x7_100, 7x7_250, 7x7_1000,"
        << " aruco_original, aruco_mip_36h12\n";
    std::exit(2);
}

Options parse_options(int argc, char** argv) {
    if (argc < 2) usage(argv[0]);
    Options opt;
    opt.dataset_root = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) usage(argv[0]);
            return argv[++i];
        };
        if (arg == "--max-frames") {
            opt.max_frames = std::stoi(need_value("--max-frames"));
        } else if (arg == "--stride") {
            opt.stride = std::max(1, std::stoi(need_value("--stride")));
        } else if (arg == "--corner-threshold") {
            opt.corner_threshold_px = std::stod(need_value("--corner-threshold"));
        } else if (arg == "--dict") {
            opt.dict_name = need_value("--dict");
        } else if (arg == "--profile") {
            opt.profile = need_value("--profile");
        } else {
            usage(argv[0]);
        }
    }
    return opt;
}

int dictionary_id_from_name(const std::string& name) {
    if (name == "4x4_50") return cv::aruco::DICT_4X4_50;
    if (name == "4x4_100") return cv::aruco::DICT_4X4_100;
    if (name == "4x4_250") return cv::aruco::DICT_4X4_250;
    if (name == "4x4_1000") return cv::aruco::DICT_4X4_1000;
    if (name == "5x5_50") return cv::aruco::DICT_5X5_50;
    if (name == "5x5_100") return cv::aruco::DICT_5X5_100;
    if (name == "5x5_250") return cv::aruco::DICT_5X5_250;
    if (name == "5x5_1000") return cv::aruco::DICT_5X5_1000;
    if (name == "6x6_50") return cv::aruco::DICT_6X6_50;
    if (name == "6x6_100") return cv::aruco::DICT_6X6_100;
    if (name == "6x6_250") return cv::aruco::DICT_6X6_250;
    if (name == "6x6_1000") return cv::aruco::DICT_6X6_1000;
    if (name == "7x7_50") return cv::aruco::DICT_7X7_50;
    if (name == "7x7_100") return cv::aruco::DICT_7X7_100;
    if (name == "7x7_250") return cv::aruco::DICT_7X7_250;
    if (name == "7x7_1000") return cv::aruco::DICT_7X7_1000;
    if (name == "aruco_original") return cv::aruco::DICT_ARUCO_ORIGINAL;
    if (name == "aruco_mip_36h12") return cv::aruco::DICT_ARUCO_MIP_36h12;
    return -1;
}

std::string read_text(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::size_t skip_ws(const std::string& s, std::size_t pos) {
    while (pos < s.size() && std::isspace(static_cast<unsigned char>(s[pos]))) ++pos;
    return pos;
}

bool parse_number_after_colon(const std::string& s, std::size_t key_pos,
                              double& value, std::size_t& end_pos) {
    const std::size_t colon = s.find(':', key_pos);
    if (colon == std::string::npos) return false;
    const std::size_t begin = skip_ws(s, colon + 1);
    const char* start = s.c_str() + begin;
    char* end = nullptr;
    value = std::strtod(start, &end);
    if (end == start) return false;
    end_pos = static_cast<std::size_t>(end - s.c_str());
    return true;
}

bool parse_object_corner(const std::string& s, std::size_t& pos, cv::Point2f& p) {
    const std::size_t x_key = s.find("\"x\"", pos);
    const std::size_t y_key = s.find("\"y\"", pos);
    if (x_key == std::string::npos || y_key == std::string::npos) return false;
    double x = 0.0, y = 0.0;
    std::size_t x_end = 0, y_end = 0;
    if (!parse_number_after_colon(s, x_key, x, x_end)) return false;
    if (!parse_number_after_colon(s, y_key, y, y_end)) return false;
    p = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
    pos = y_end;
    return true;
}

std::vector<MarkerLabel> parse_shadow_annotation(const fs::path& path) {
    const std::string s = read_text(path);
    std::vector<MarkerLabel> labels;
    std::size_t pos = 0;
    while (true) {
        const std::size_t id_key = s.find("\"id\"", pos);
        if (id_key == std::string::npos) break;

        double id_value = -1.0;
        std::size_t id_end = 0;
        if (!parse_number_after_colon(s, id_key, id_value, id_end)) break;

        const std::size_t corners_key = s.find("\"corners\"", id_end);
        if (corners_key == std::string::npos) break;
        pos = corners_key;

        MarkerLabel label;
        label.id = static_cast<int>(std::llround(id_value));
        bool ok = true;
        for (int i = 0; i < 4; ++i) {
            cv::Point2f corner;
            if (!parse_object_corner(s, pos, corner)) {
                ok = false;
                break;
            }
            label.corners[i] = corner;
        }
        if (ok) labels.push_back(label);
    }
    return labels;
}

bool is_image_file(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png";
}

bool has_corrected_annotations_part(const fs::path& path) {
    for (const auto& part : path) {
        if (part == "corrected_annotations") return true;
    }
    return false;
}

std::vector<fs::path> collect_images(const fs::path& root) {
    std::vector<fs::path> images;
    for (const auto& entry : fs::recursive_directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;
        const fs::path path = entry.path();
        if (has_corrected_annotations_part(path)) continue;
        if (is_image_file(path)) images.push_back(path);
    }
    std::sort(images.begin(), images.end());
    return images;
}

fs::path annotation_for_image(const fs::path& image) {
    return image.parent_path() / "corrected_annotations" / (image.stem().string() + ".json");
}

double corner_rmse(const std::array<cv::Point2f, 4>& gt,
                   const std::array<cv::Point2f, 4>& pred) {
    double best = std::numeric_limits<double>::infinity();
    for (int reversed = 0; reversed < 2; ++reversed) {
        for (int shift = 0; shift < 4; ++shift) {
            double sum_sq = 0.0;
            for (int i = 0; i < 4; ++i) {
                const int j = reversed ? ((shift - i + 4) & 3) : ((shift + i) & 3);
                const cv::Point2f d = gt[i] - pred[j];
                sum_sq += static_cast<double>(d.x) * d.x + static_cast<double>(d.y) * d.y;
            }
            best = std::min(best, std::sqrt(sum_sq / 4.0));
        }
    }
    return best;
}

void score_frame(const std::vector<MarkerLabel>& gt,
                 const std::vector<MarkerLabel>& pred,
                 RunStats& stats,
                 double corner_threshold_px) {
    std::vector<char> used(pred.size(), 0);
    for (const auto& g : gt) {
        double best_err = std::numeric_limits<double>::infinity();
        int best_idx = -1;
        for (std::size_t i = 0; i < pred.size(); ++i) {
            if (used[i] || pred[i].id != g.id) continue;
            const double err = corner_rmse(g.corners, pred[i].corners);
            if (err < best_err) {
                best_err = err;
                best_idx = static_cast<int>(i);
            }
        }
        if (best_idx >= 0 && best_err <= corner_threshold_px) {
            used[static_cast<std::size_t>(best_idx)] = 1;
            ++stats.tp;
            stats.corner_error_sum += best_err;
        } else {
            ++stats.fn;
        }
    }
    for (std::size_t i = 0; i < pred.size(); ++i) {
        if (!used[i]) ++stats.fp;
    }
}

std::vector<MarkerLabel> from_kakutag(const std::vector<kakutag::Marker>& markers) {
    std::vector<MarkerLabel> out;
    out.reserve(markers.size());
    for (const auto& m : markers) {
        MarkerLabel label;
        label.id = m.id;
        for (int i = 0; i < 4; ++i) label.corners[i] = m.corners[i];
        out.push_back(label);
    }
    return out;
}

std::vector<MarkerLabel> from_opencv(const std::vector<std::vector<cv::Point2f>>& corners,
                                     const std::vector<int>& ids) {
    std::vector<MarkerLabel> out;
    const std::size_t n = std::min(corners.size(), ids.size());
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (corners[i].size() != 4) continue;
        MarkerLabel label;
        label.id = ids[i];
        for (int k = 0; k < 4; ++k) label.corners[k] = corners[i][k];
        out.push_back(label);
    }
    return out;
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double idx = (static_cast<double>(values.size() - 1) * p);
    const auto lo = static_cast<std::size_t>(std::floor(idx));
    const auto hi = static_cast<std::size_t>(std::ceil(idx));
    if (lo == hi) return values[lo];
    const double t = idx - static_cast<double>(lo);
    return values[lo] * (1.0 - t) + values[hi] * t;
}

void print_stats(const RunStats& stats, std::size_t frames, std::size_t gt_total) {
    const double sum = std::accumulate(stats.ms.begin(), stats.ms.end(), 0.0);
    const double mean = stats.ms.empty() ? 0.0 : sum / static_cast<double>(stats.ms.size());
    const double recall = gt_total == 0 ? 0.0 : static_cast<double>(stats.tp) / static_cast<double>(gt_total);
    const double precision_den = static_cast<double>(stats.tp + stats.fp);
    const double precision = precision_den == 0.0 ? 0.0 : static_cast<double>(stats.tp) / precision_den;
    const double corner = stats.tp == 0 ? 0.0 : stats.corner_error_sum / static_cast<double>(stats.tp);

    std::cout << stats.name
              << ",frames=" << frames
              << ",gt=" << gt_total
              << ",tp=" << stats.tp
              << ",fp=" << stats.fp
              << ",fn=" << stats.fn
              << ",recall=" << recall
              << ",precision=" << precision
              << ",corner_rmse_mean_px=" << corner
              << ",mean_ms=" << mean
              << ",p50_ms=" << percentile(stats.ms, 0.50)
              << ",p90_ms=" << percentile(stats.ms, 0.90)
              << ",p99_ms=" << percentile(stats.ms, 0.99)
              << "\n";
}

int main(int argc, char** argv) {
    const Options opt = parse_options(argc, argv);
    if (!fs::exists(opt.dataset_root)) {
        std::cerr << "dataset root does not exist: " << opt.dataset_root << "\n";
        return 1;
    }

    std::vector<fs::path> images = collect_images(opt.dataset_root);
    if (images.empty()) {
        std::cerr << "no images found under: " << opt.dataset_root << "\n";
        return 1;
    }

    const int dict_id = dictionary_id_from_name(opt.dict_name);
    if (dict_id < 0) {
        std::cerr << "unknown dictionary: " << opt.dict_name << "\n";
        return 2;
    }
    auto dict = cv::aruco::getPredefinedDictionary(dict_id);
    cv::aruco::DetectorParameters cv_params;
    cv::aruco::ArucoDetector cv_detector(dict, cv_params);

    kakutag::DetectorParameters kt_params;
#ifdef KAKUTAG_HAS_RECALL_PROFILES
    if (opt.profile == "balanced") {
        kt_params = kakutag::make_balanced_recall_parameters(dict);
    } else if (opt.profile == "high") {
        kt_params = kakutag::make_high_recall_parameters(dict);
    } else if (opt.profile == "default") {
        kt_params.dicts = {dict};
    } else {
        std::cerr << "unknown profile: " << opt.profile << "\n";
        return 2;
    }
#else
    if (opt.profile != "default") {
        std::cerr << "this KakuTag header does not expose recall profiles\n";
        return 2;
    }
    kt_params.dicts = {dict};
#endif
    kakutag::ArucoDetector kt_detector(kt_params);

    RunStats kt_stats{"kakutag"};
    RunStats cv_stats{"opencv_aruco"};
    std::size_t frames = 0;
    std::size_t missing_annotations = 0;
    std::size_t gt_total = 0;

    for (std::size_t i = 0; i < images.size(); i += static_cast<std::size_t>(opt.stride)) {
        if (opt.max_frames > 0 && static_cast<int>(frames) >= opt.max_frames) break;

        const fs::path ann_path = annotation_for_image(images[i]);
        if (!fs::exists(ann_path)) {
            ++missing_annotations;
            continue;
        }

        cv::Mat gray = cv::imread(images[i].string(), cv::IMREAD_GRAYSCALE);
        if (gray.empty()) continue;

        const std::vector<MarkerLabel> gt = parse_shadow_annotation(ann_path);
        gt_total += gt.size();

        auto t0 = std::chrono::steady_clock::now();
        const auto& kt_markers = kt_detector.detect(gray);
        auto t1 = std::chrono::steady_clock::now();
        kt_stats.ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        score_frame(gt, from_kakutag(kt_markers), kt_stats, opt.corner_threshold_px);

        std::vector<std::vector<cv::Point2f>> cv_corners;
        std::vector<int> cv_ids;
        t0 = std::chrono::steady_clock::now();
        cv_detector.detectMarkers(gray, cv_corners, cv_ids);
        t1 = std::chrono::steady_clock::now();
        cv_stats.ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        score_frame(gt, from_opencv(cv_corners, cv_ids), cv_stats, opt.corner_threshold_px);

        ++frames;
    }

    std::cout << "dataset_root=" << opt.dataset_root.string()
              << ",max_frames=" << opt.max_frames
              << ",stride=" << opt.stride
              << ",dict=" << opt.dict_name
              << ",profile=" << opt.profile
              << ",corner_threshold_px=" << opt.corner_threshold_px
              << ",missing_annotations=" << missing_annotations
              << "\n";
    print_stats(kt_stats, frames, gt_total);
    print_stats(cv_stats, frames, gt_total);
    return 0;
}
