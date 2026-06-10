# KakuTag ✨

<p align="center">
  <b>A tiny C++20 header for fast square-marker detection.</b><br/>
  <sub>Drop in one file, use OpenCV dictionaries, get marker IDs and corners.</sub>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README.ja.md">日本語</a>
</p>

KakuTag is a single-header fiducial marker detector built for small C++ projects,
camera loops, embedded devices, and quick experiments.

It works with OpenCV's `cv::aruco::Dictionary` format, so you can use standard
OpenCV ArUco dictionaries and the AprilTag dictionaries exposed by OpenCV.

> 🚧 Status: experimental pre-alpha. The API, defaults, and benchmark results may
> change. Validate on your own camera, CPU, compiler, OpenCV version, and marker
> family before production use.

## 📊 Reference Benchmark

KakuTag-only local synthetic single-frame benchmark on Raspberry Pi Zero 2 W:

- Raspberry Pi Zero 2 W, Cortex-A53, 4 cores, aarch64
- Raspberry Pi OS kernel 6.12.75, GCC 14.2.0, OpenCV 4.10.0
- Build flags: `-std=c++20 -O3 -DNDEBUG -mcpu=cortex-a53 -mtune=cortex-a53`
- Dictionary: `cv::aruco::DICT_6X6_250`
- 300 measured iterations after 30 warmup iterations
- Single-core run with `taskset -c 2`
- CPU governor: `ondemand`
- Temperature: 49.9 C before, 53.2 C after
- `vcgencmd get_throttled`: `0x0` before and after

The benchmark reused one `kakutag::ArucoDetector` instance and used the native
`detector.detect(image)` API.

| Scene | Mean ms | p50 ms | p90 ms | p99 ms | Detected |
| --- | ---: | ---: | ---: | ---: | ---: |
| `vga_clean` | 1.537 | 1.527 | 1.580 | 1.602 | 300/300 |
| `vga_small` | 10.335 | 10.334 | 10.381 | 10.465 | 300/300 |
| `vga_rotated` | 1.561 | 1.558 | 1.603 | 1.620 | 300/300 |
| `vga_clutter` | 14.295 | 14.264 | 14.401 | 14.461 | 300/300 |
| `hd_clean` | 4.167 | 4.161 | 4.180 | 4.288 | 300/300 |

These are local synthetic-scene results, not a universal guarantee. Real
performance depends on camera resolution, marker size, scene texture, CPU,
compiler flags, OpenCV build, and selected dictionary. Benchmark on your target
device before production use.

## ✨ Why KakuTag

- 📦 One header: copy `kakutag.hpp` and include it.
- ⚡ Simple API: create a detector, call `detect(image)`.
- 🤝 OpenCV-friendly: use `cv::aruco::Dictionary` and optional OpenCV-style output.
- ♻️ Low overhead: reuse one detector instance to reuse internal scratch buffers.
- 🧩 Small-device friendly: includes scalar and compile-time SIMD paths.

## 🚀 Quick Start

For the simplest setup, put `kakutag.hpp` in the same directory as `main.cpp`.

```text
your-project/
  main.cpp
  kakutag.hpp
```

When building from that directory, `-I .` lets the compiler find
`kakutag.hpp`.

Include KakuTag and the OpenCV headers you use:

```cpp
#include "kakutag.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
```

Load or provide an image. KakuTag accepts `CV_8UC1`, `CV_8UC3`, or `CV_8UC4`.
Grayscale input is the most direct path.

```cpp
cv::Mat image = cv::imread("frame.png", cv::IMREAD_GRAYSCALE);
```

Choose a dictionary and create a detector:

```cpp
auto dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

kakutag::DetectorParameters params;
params.dicts = {dict};

kakutag::ArucoDetector detector(params);
```

Detect markers from the image:

```cpp
const std::vector<kakutag::Marker>& markers = detector.detect(image);

for (const kakutag::Marker& marker : markers) {
    std::cout << "id=" << marker.id << "\n";
}
```

Each `kakutag::Marker` contains:

- `id`: decoded marker ID.
- `corners[4]`: corner positions as `cv::Point2f`.
- `dict_index`: index of the matched dictionary when multiple dictionaries are used.

The returned vector belongs to the detector. Copy the markers if you need to keep
them after the next `detect()` call.

Build from the same directory:

```bash
g++ -std=c++20 -O3 -DNDEBUG -I . main.cpp -o app \
  $(pkg-config --cflags --libs opencv4)
```

## 🧪 Complete Example

```cpp
#include "kakutag.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>

#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: app frame.png\n";
        return 1;
    }

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "failed to read image\n";
        return 1;
    }

    auto dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

    kakutag::DetectorParameters params;
    params.dicts = {dict};

    kakutag::ArucoDetector detector(params);
    const std::vector<kakutag::Marker>& markers = detector.detect(image);

    for (const kakutag::Marker& marker : markers) {
        std::cout << "id=" << marker.id << " corners=";
        for (const cv::Point2f& p : marker) {
            std::cout << "(" << p.x << "," << p.y << ")";
        }
        std::cout << "\n";
    }

    return 0;
}
```

Build with `pkg-config`:

```bash
g++ -std=c++20 -O3 -DNDEBUG -I . your_app.cpp -o your_app \
  $(pkg-config --cflags --libs opencv4)
```

Minimal CMake setup:

```cmake
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d objdetect)

add_executable(app main.cpp)
target_include_directories(app PRIVATE path/to/kakutag)
target_compile_features(app PRIVATE cxx_std_20)
target_link_libraries(app PRIVATE ${OpenCV_LIBS})
```

Quick CMake compile check:

```bash
cmake -S . -B build
cmake --build build
```

This also defines the local header-only target `KakuTag::KakuTag`.

## 🧭 Dictionaries

Use any OpenCV dictionary exposed as `cv::aruco::Dictionary`.

```cpp
params.dicts = {
    cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50),
    cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250),
};
```

OpenCV-provided AprilTag dictionaries can also be used:

```cpp
params.dicts = {
    cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11),
};
```

KakuTag supports AprilTag dictionaries through OpenCV's ArUco dictionary API. It
is not a full AprilTag 3 detector and does not implement native AprilTag 3
families such as `tagStandard41h12`.

## 🔁 OpenCV-Style Output

If your existing code expects OpenCV-style output containers, use
`detectMarkers(...)`:

```cpp
std::vector<std::vector<cv::Point2f>> corners;
std::vector<int> ids;

detector.detectMarkers(image, corners, ids);
```

For tight frame loops, prefer the native `detect(image)` API because it avoids
the extra output conversion.

## 🎥 Frame Loop Tips

- Reuse one `kakutag::ArucoDetector` instance across frames.
- Use grayscale `CV_8UC1` input when possible.
- Use one detector instance per thread for concurrent detection.
- For AVX2 builds, pass `-mavx2` on GCC/Clang or `/arch:AVX2` on MSVC.
- On ARM/AArch64, NEON paths are selected at compile time when available.

Input images should be `CV_8UC1`, `CV_8UC3`, or `CV_8UC4`.

## ✅ Requirements

- C++20 compiler.
- OpenCV `core`, `imgproc`, `calib3d`, and `objdetect`.

## 📄 License

MIT. See `LICENSE`.

KakuTag is independent software and is not affiliated with, endorsed by, or
sponsored by OpenCV, the original ArUco project, or AprilTag.
