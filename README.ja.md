# KakuTag ✨

<p align="center">
  <b>小さく入れて、すぐ使える C++20 single-header marker detector。</b><br/>
  <sub>1ファイルを置いて、OpenCV辞書を渡して、marker ID と corners を受け取ります。</sub>
</p>

<p align="center">
  <a href="README.md">English</a> | <a href="README.ja.md">日本語</a>
</p>

KakuTag は、小さなC++ project、camera loop、embedded device、実験用prototypeで使いやすい
single-header fiducial marker detector です。

OpenCV の `cv::aruco::Dictionary` 形式を使います。標準の OpenCV ArUco 辞書と、
OpenCV が提供する AprilTag 辞書を扱えます。

> 🚧 状態: experimental pre-alpha。API、デフォルト値、ベンチマーク結果は変更される
> 可能性があります。production 用途では、自分の camera、CPU、compiler、OpenCV
> version、marker family で必ず検証してください。

## 📊 Reference Benchmark

Raspberry Pi Zero 2 W での KakuTag のみの synthetic single-frame benchmark です。

- Raspberry Pi Zero 2 W, Cortex-A53, 4 cores, aarch64
- Raspberry Pi OS kernel 6.12.75, GCC 14.2.0, OpenCV 4.10.0
- Build flags: `-std=c++20 -O3 -DNDEBUG -mcpu=cortex-a53 -mtune=cortex-a53`
- Dictionary: `cv::aruco::DICT_6X6_250`
- warmup 30回の後、300回測定
- `taskset -c 2` による single-core run
- CPU governor: `ondemand`
- 温度: 実行前 49.9 C、実行後 53.2 C
- `vcgencmd get_throttled`: 実行前後とも `0x0`

benchmarkでは、1つの `kakutag::ArucoDetector` instanceを再利用し、native APIの
`detector.detect(image)` を使っています。

| Scene | Mean ms | p50 ms | p90 ms | p99 ms | Detected |
| --- | ---: | ---: | ---: | ---: | ---: |
| `vga_clean` | 1.537 | 1.527 | 1.580 | 1.602 | 300/300 |
| `vga_small` | 10.335 | 10.334 | 10.381 | 10.465 | 300/300 |
| `vga_rotated` | 1.561 | 1.558 | 1.603 | 1.620 | 300/300 |
| `vga_clutter` | 14.295 | 14.264 | 14.401 | 14.461 | 300/300 |
| `hd_clean` | 4.167 | 4.161 | 4.180 | 4.288 | 300/300 |

これはlocalなsynthetic sceneでの結果であり、普遍的な保証ではありません。実性能は
camera resolution、marker size、scene texture、CPU、compiler flags、OpenCV build、
dictionary に依存します。production用途では、必ずtarget deviceでbenchmarkしてください。

## ✨ Why KakuTag

- 📦 1ファイル: `kakutag.hpp` をコピーして `#include "kakutag.hpp"` するだけです。
- ⚡ simple API: detectorを作って `detect(image)` を呼びます。
- 🤝 OpenCV-friendly: `cv::aruco::Dictionary` とOpenCV風の出力に対応しています。
- ♻️ low overhead: detectorを再利用すると内部scratch bufferも再利用されます。
- 🧩 small-device friendly: scalar path と compile-time SIMD path を持ちます。

## 🚀 Quick Start

いちばん簡単な例では、`main.cpp` と同じフォルダに `kakutag.hpp` を置きます。

```text
your-project/
  main.cpp
  kakutag.hpp
```

このフォルダでビルドするなら、`-I .` を付けるとコンパイラが `kakutag.hpp` を見つけられます。

KakuTagと、使うOpenCVヘッダをincludeします。

```cpp
#include "kakutag.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
```

画像を読み込むか、cameraから取得した `cv::Mat` を用意します。KakuTagの入力は
`CV_8UC1`, `CV_8UC3`, `CV_8UC4` を想定しています。grayscaleの `CV_8UC1` が
いちばん直接的です。

```cpp
cv::Mat image = cv::imread("frame.png", cv::IMREAD_GRAYSCALE);
```

dictionaryを選んで、detectorを作ります。

```cpp
auto dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

kakutag::DetectorParameters params;
params.dicts = {dict};

kakutag::ArucoDetector detector(params);
```

画像からmarkerを検出します。

```cpp
const std::vector<kakutag::Marker>& markers = detector.detect(image);

for (const kakutag::Marker& marker : markers) {
    std::cout << "id=" << marker.id << "\n";
}
```

`kakutag::Marker` には主に次の情報が入ります。

- `id`: decoded marker ID。
- `corners[4]`: `cv::Point2f` のcorner positions。
- `dict_index`: 複数辞書を使った場合に、どの辞書でmatchしたか。

戻り値のvectorはdetectorが所有します。次の `detect()` 呼び出し後も使いたい場合は、
markerをcopyしてください。

同じディレクトリからビルドする例です。

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

`pkg-config` でビルドする例です。

```bash
g++ -std=c++20 -O3 -DNDEBUG -I . your_app.cpp -o your_app \
  $(pkg-config --cflags --libs opencv4)
```

最小のCMake例です。

```cmake
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d objdetect)

add_executable(app main.cpp)
target_include_directories(app PRIVATE path/to/kakutag)
target_compile_features(app PRIVATE cxx_std_20)
target_link_libraries(app PRIVATE ${OpenCV_LIBS})
```

CMakeでcompile checkもできます。

```bash
cmake -S . -B build
cmake --build build
```

local header-only target `KakuTag::KakuTag` も定義されます。

## 🧭 Dictionaries

`cv::aruco::Dictionary` として公開されているOpenCV辞書を使えます。

```cpp
params.dicts = {
    cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50),
    cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250),
};
```

OpenCV提供のAprilTag辞書も使えます。

```cpp
params.dicts = {
    cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11),
};
```

KakuTag は OpenCV の ArUco dictionary API 経由で AprilTag 辞書を扱います。
AprilTag 3 の完全なdetectorではなく、`tagStandard41h12` などのnative AprilTag 3
familyは実装していません。

## 🔁 OpenCV風の出力

既存コードがOpenCV風のcontainerを期待している場合は、`detectMarkers(...)` を使えます。

```cpp
std::vector<std::vector<cv::Point2f>> corners;
std::vector<int> ids;

detector.detectMarkers(image, corners, ids);
```

frame loopでは、余分な出力変換を避けられるnative APIの `detect(image)` を推奨します。

## 🎥 Frame Loop Tips

- 1つの `kakutag::ArucoDetector` instanceをframe間で再利用してください。
- 可能ならgrayscaleの `CV_8UC1` を入力してください。
- 並列検出ではthreadごとにdetector instanceを用意してください。
- AVX2 buildではGCC/Clangなら `-mavx2`、MSVCなら `/arch:AVX2` を指定してください。
- ARM/AArch64では、利用可能な場合にNEON pathがcompile時に選ばれます。

入力画像は `CV_8UC1`, `CV_8UC3`, `CV_8UC4` を想定しています。

## ✅ Requirements

- C++20 compiler。
- OpenCV `core`, `imgproc`, `calib3d`, `objdetect`。

## 📄 License

MIT. See `LICENSE`.

KakuTag は独立したsoftwareであり、OpenCV、original ArUco project、AprilTagとは
提携・承認・スポンサー関係にありません。
