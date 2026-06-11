// SPDX-License-Identifier: MIT
// KakuTag: header-only ArUco-compatible marker detector.
//
//   #include "kakutag.hpp"
//
// Depends on: OpenCV (objdetect, imgproc, calib3d, core).
// Detects standard OpenCV ArUco dictionaries and OpenCV-provided AprilTag
// dictionaries with low single-frame latency. Default parameters use one-pass
// contour hysteresis and conservative border tolerance for better real-video
// recall without enabling slow ECC by default.
//
// Main entry points:
//   1) kakutag::detect_single(gray)          -- stateless full detector with auto pyramid.
//   2) kakutag::detect_full(gray)            -- single-frame, no pyramid (worst-case path).
//   3) kakutag::ArucoDetector d; d.detect(gray)
//                                         -- reusable stateless detector with persistent scratch.
//
// Compiles to AVX2 on x86_64 with __AVX2__, to NEON on ARMv8 with __ARM_NEON,
// otherwise falls back to a scalar reference. The dispatch is purely compile-time.
//
// A detector instance owns mutable scratch buffers and is not safe for
// concurrent detect() calls. Use one detector instance per thread.
#ifndef KAKUTAG_KAKUTAG_HPP_
#define KAKUTAG_KAKUTAG_HPP_

#define KAKUTAG_VERSION_MAJOR 0
#define KAKUTAG_VERSION_MINOR 0
#define KAKUTAG_VERSION_PATCH 1

// Suzuki85 neighbor-probe optimization.
// 1 = use SWAR (SIMD Within A Register) 8-neighbor probe.
// 0 = use scalar early-break probe.
// SWAR is correctness-equivalent for KakuTag binary states {0,100,255}, but its
// speed benefit is workload-dependent because scalar often breaks after 1-2 probes.
#ifndef KAKUTAG_USE_SWAR_PROBE
#define KAKUTAG_USE_SWAR_PROBE 1
#endif

// Tile popcount gate: rejects 8x8 tiles whose black-pixel count is outside
// [KAKUTAG_TILE_GATE_LO, KAKUTAG_TILE_GATE_HI], skipping Suzuki85 seed search there.
// Conservative defaults preserve detection: a marker contour always spans
// boundary tiles with mixed black/white counts in the surviving range.
// Disabled by default: in current test scenes the popcount+dilate overhead
// outweighs the contour-skip savings on dense scenes. Re-enable for sparse
// scenes with many large empty regions.
#ifndef KAKUTAG_USE_TILE_GATE
#define KAKUTAG_USE_TILE_GATE 0
#endif
#ifndef KAKUTAG_TILE_GATE_LO
#define KAKUTAG_TILE_GATE_LO 1
#endif
#ifndef KAKUTAG_TILE_GATE_HI
#define KAKUTAG_TILE_GATE_HI 63
#endif

// Threshold-bounds seed-scan narrowing. This fuses binary-threshold emission
// with foreground-bounding-box collection and scans only that bbox in Suzuki85.
// Default OFF: on dense/noisy test scenes the extra bookkeeping costs more
// than it saves. Enable for sparse scenes / ROI-heavy real camera workloads.
#ifndef KAKUTAG_USE_THRESHOLD_BOUNDS
#define KAKUTAG_USE_THRESHOLD_BOUNDS 0
#endif

// Run-tile seed gate: mark only tiles containing a horizontal foreground run
// long enough to be a Suzuki85 seed. Unlike popcount gating, this uses the
// detector's own seed condition and is therefore safer as an optional sparse-
// scene accelerator. Default OFF until real-camera A/B proves it wins broadly.
#ifndef KAKUTAG_USE_RUN_TILE_GATE
#define KAKUTAG_USE_RUN_TILE_GATE 0
#endif


// Small-marker contour seed strictness. A fully permissive detector accepts any foreground
// seed; increasing this rejects false-candidate clutter faster but can miss
// tiny/tilted marker borders. Tuned by practical validation.
#ifndef KAKUTAG_SMALL_MARKER_SEED_MIN_RUN
#define KAKUTAG_SMALL_MARKER_SEED_MIN_RUN 6
#endif

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <limits>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <iostream>

#if defined(_MSC_VER)
  #include <intrin.h>
#endif
#if defined(__AVX2__)
  #include <immintrin.h>
#endif
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  #include <arm_neon.h>
#endif

namespace kakutag {

namespace detail {
struct PackedDict;
}

// =============================================================================
// Public types
// =============================================================================

struct Marker {
    int           id        = -1;
    int           rotation  = 0;        // 0..3 CCW (uses canonical ordering)
    cv::Point2f   corners[4]{};
    int           dict_index = 0;
    int           dict      = 0;        // compatibility alias

    // ---- Convenience API ----
    cv::Point2f&       operator[](size_t i)       { return corners[i]; }
    const cv::Point2f& operator[](size_t i) const { return corners[i]; }
    size_t             size()  const              { return 4; }
    cv::Point2f*       begin()                    { return corners; }
    cv::Point2f*       end()                      { return corners + 4; }
    const cv::Point2f* begin() const              { return corners; }
    const cv::Point2f* end()   const              { return corners + 4; }

    std::vector<cv::Point2f> as_vector() const {
        return { corners[0], corners[1], corners[2], corners[3] };
    }


    inline void draw(cv::Mat& image,
                     const cv::Scalar& color = cv::Scalar(0, 0, 255)) const;

    inline std::pair<cv::Mat, cv::Mat>
    estimatePose(const cv::Mat& cameraMatrix,
                 const cv::Mat& distCoeffs,
                 double markerSize = 1.0) const;
};

struct DetectorOptions {
    // Dictionaries to scan. Default = DICT_6X6_250 (OpenCV DICT_6X6_250 default).
    std::vector<cv::aruco::Dictionary> dicts = {
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250)
    };
    int   min_size              = 10;     // pixel-side minimum alias
    int   max_attempts          = 7;      // homography jitter retries; small-marker robust
    bool  refine_corners        = true;   // cornerSubPix
    int   refine_window         = 5;
    int   max_hamming           = 0;      // 0 = exact; positive = ECC tolerated
    bool  need_ids              = true;   // false -> return marker-like quads without dictionary decode
    // Pyramid:
    //   if true, single-frame detect runs candidate generation at 1/2 scale
    //   and refines on full resolution. Safe: if pyramid finds nothing, we
    //   fall back to native full-resolution detection automatically.
    bool  pyramid_enabled       = true;
    // Streaming threshold parameters.
    int   threshold_offset      = 5;      // sparse-first adaptive threshold (compatibility API)
    // Hysteresis (ternary) candidate generation. When >= 0 and < threshold_offset,
    // the candidate generator emits 3-state binaries (STRONG/WEAK/BG): pixels
    // satisfying threshold_offset act as STRONG seeds, pixels satisfying
    // threshold_offset_low act as WEAK continuation. This recovers small/low-
    // contrast marker borders without paying for a separate dual-threshold pass.
    // -1 disables hysteresis and uses the legacy single-threshold path.
    int   threshold_offset_low  = 2;
    // ---- decode controls ----
    int    marker_border_bits             = 1;
    double error_correction_rate          = 0.0;
    double max_erroneous_bits_in_border_rate = 0.10;
    bool   collect_rejected = false;

    // Internal detector-owned packed dictionary view. Public callers normally
    // leave these null; ArucoDetector fills them from its persistent cache.
    const detail::PackedDict* packed_dicts = nullptr;
    size_t packed_dict_count = 0;
};

// =============================================================================
// Implementation details (no compile-time dispatch outside this section)
// =============================================================================
namespace detail {

inline int ctz32_nonzero(uint32_t v) {
#if defined(_MSC_VER) && !defined(__clang__)
    unsigned long bit;
    _BitScanForward(&bit, v);
    return (int)bit;
#else
    return (int)__builtin_ctz(v);
#endif
}

constexpr int kBoxR = 7;
constexpr int kBoxK = 15;
constexpr int kBoxArea  = kBoxK * kBoxK;
constexpr int kBoxRound = kBoxArea / 2;

// Persistent scratch for the adaptive-threshold streaming passes. The original
// header-only prototype allocated these rings every call; ArucoDetector now
// keeps them across frames, which removes a large hidden cost especially on Pi
// and ROI-heavy video verification.
struct ThresholdScratch {
    std::vector<uint16_t> hsum_ring;
    std::vector<uint16_t> vsum;
    std::vector<uint16_t> scratch_rem;
    std::vector<uint8_t>  down_ring;
};

struct BinaryBounds {
    int minx = 0, miny = 0, maxx = -1, maxy = -1;
    void reset() { minx = 0; miny = 0; maxx = -1; maxy = -1; }
    bool empty() const { return maxx < minx || maxy < miny; }
    cv::Rect rect() const {
        if (empty()) return cv::Rect();
        return cv::Rect(minx, miny, maxx - minx + 1, maxy - miny + 1);
    }
    void add_span(int y, int x0, int x1) {
        if (x1 < x0) return;
        if (empty()) { minx = x0; maxx = x1; miny = maxy = y; return; }
        minx = std::min(minx, x0); maxx = std::max(maxx, x1);
        miny = std::min(miny, y);  maxy = std::max(maxy, y);
    }
};

struct RunTileGate {
    uint8_t* tile_mask = nullptr;
    int tw = 0;
    int th = 0;
    int min_run = 6;
    bool any = false;
};

inline void update_run_tile_gate_from_row(const uint8_t* row, int y, int w,
                                          RunTileGate* gate,
                                          int valid_l = 0, int valid_r = -1) {
#if KAKUTAG_USE_RUN_TILE_GATE
    if (!gate || !gate->tile_mask || gate->tw <= 0 || gate->th <= 0 || !row || w <= 0) return;
    if (valid_r < 0 || valid_r >= w) valid_r = w - 1;
    if (valid_l < 0) valid_l = 0;
    if (valid_l > valid_r) return;
    const int ty = y >> 3;
    if (ty < 0 || ty >= gate->th) return;
    uint8_t* mrow = gate->tile_mask + (size_t)ty * gate->tw;
    int x = valid_l;
    while (x <= valid_r) {
        while (x <= valid_r && row[x] != 255) ++x;
        if (x > valid_r) break;
        const int rs = x;
        while (x <= valid_r && row[x] == 255) ++x;
        const int re = x;
        if (re - rs >= gate->min_run) {
            int tx0 = rs >> 3;
            int tx1 = (re - 1) >> 3;
            tx0 = std::max(0, std::min(tx0, gate->tw - 1));
            tx1 = std::max(0, std::min(tx1, gate->tw - 1));
            for (int tx = tx0; tx <= tx1; ++tx) mrow[tx] = 1;
            gate->any = true;
        }
    }
#else
    (void)row; (void)y; (void)w; (void)gate; (void)valid_l; (void)valid_r;
#endif
}

inline void hsum_row_scalar(const uint8_t* row, int w, uint16_t* hr) {
    int s = int(row[0]) * (kBoxR + 1);
    for (int k = 1; k <= kBoxR; ++k) s += row[std::min(k, w - 1)];
    hr[0] = (uint16_t)s;
    for (int x = 1; x < w; ++x) {
        int remove_x = x - kBoxR - 1;
        int add_x    = x + kBoxR;
        int rv = row[(remove_x < 0) ? 0 : remove_x];
        int av = row[(add_x >= w) ? (w - 1) : add_x];
        s += av - rv;
        hr[x] = (uint16_t)s;
    }
}

#if defined(__AVX2__)
inline void hsum_row_simd(const uint8_t* row, int w, uint16_t* hr) {
    // Head
    {
        int s = int(row[0]) * (kBoxR + 1);
        for (int k = 1; k <= kBoxR; ++k) s += row[std::min(k, w - 1)];
        hr[0] = (uint16_t)s;
        for (int x = 1; x < std::min(kBoxR, w); ++x) {
            int remove_x = x - kBoxR - 1;
            int add_x    = x + kBoxR;
            int rv = row[(remove_x < 0) ? 0 : remove_x];
            int av = row[(add_x >= w) ? (w - 1) : add_x];
            s += av - rv;
            hr[x] = (uint16_t)s;
        }
    }
    int x = kBoxR;
    const int body_end = w - kBoxR;
    if (body_end > kBoxR) {
        for (; x + 8 <= body_end && (x - kBoxR + 22) <= w; x += 8) {
            __m128i b0 = _mm_loadu_si128((const __m128i*)(row + x - kBoxR));
            __m128i b1 = _mm_loadu_si128((const __m128i*)(row + x - kBoxR + 16));
            __m128i lo0 = _mm_cvtepu8_epi16(b0);
            __m128i hi0 = _mm_cvtepu8_epi16(_mm_srli_si128(b0, 8));
            __m128i lo1 = _mm_cvtepu8_epi16(b1);
            __m128i acc = lo0;
            acc = _mm_add_epi16(acc, _mm_alignr_epi8(hi0, lo0, 2));
            acc = _mm_add_epi16(acc, _mm_alignr_epi8(hi0, lo0, 4));
            acc = _mm_add_epi16(acc, _mm_alignr_epi8(hi0, lo0, 6));
            acc = _mm_add_epi16(acc, _mm_alignr_epi8(hi0, lo0, 8));
            acc = _mm_add_epi16(acc, _mm_alignr_epi8(hi0, lo0, 10));
            acc = _mm_add_epi16(acc, _mm_alignr_epi8(hi0, lo0, 12));
            acc = _mm_add_epi16(acc, _mm_alignr_epi8(hi0, lo0, 14));
            acc = _mm_add_epi16(acc, hi0);
            acc = _mm_add_epi16(acc, _mm_alignr_epi8(lo1, hi0, 2));
            acc = _mm_add_epi16(acc, _mm_alignr_epi8(lo1, hi0, 4));
            acc = _mm_add_epi16(acc, _mm_alignr_epi8(lo1, hi0, 6));
            acc = _mm_add_epi16(acc, _mm_alignr_epi8(lo1, hi0, 8));
            acc = _mm_add_epi16(acc, _mm_alignr_epi8(lo1, hi0, 10));
            acc = _mm_add_epi16(acc, _mm_alignr_epi8(lo1, hi0, 12));
            _mm_storeu_si128((__m128i*)(hr + x), acc);
        }
        for (; x < body_end; ++x) {
            int sum = 0;
            const uint8_t* p = row + x - kBoxR;
            for (int k = 0; k < kBoxK; ++k) sum += p[k];
            hr[x] = (uint16_t)sum;
        }
    }
    int tail_start = std::max(x, kBoxR);
    for (int xx = tail_start; xx < w; ++xx) {
        int sum = 0;
        for (int dx = -kBoxR; dx <= kBoxR; ++dx) {
            int idx = xx + dx;
            if (idx < 0) idx = 0;
            else if (idx >= w) idx = w - 1;
            sum += row[idx];
        }
        hr[xx] = (uint16_t)sum;
    }
}
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
inline void hsum_row_simd(const uint8_t* row, int w, uint16_t* hr) {
    // NEON fast path: 8-lane sliding sum.
    int s = int(row[0]) * (kBoxR + 1);
    for (int k = 1; k <= kBoxR; ++k) s += row[std::min(k, w - 1)];
    hr[0] = (uint16_t)s;
    for (int x = 1; x < std::min(kBoxR, w); ++x) {
        int remove_x = x - kBoxR - 1;
        int add_x    = x + kBoxR;
        int rv = row[(remove_x < 0) ? 0 : remove_x];
        int av = row[(add_x >= w) ? (w - 1) : add_x];
        s += av - rv;
        hr[x] = (uint16_t)s;
    }
    int x = kBoxR;
    const int body_end = w - kBoxR;
    for (; x + 8 <= body_end && (x - kBoxR + 16) <= w; x += 8) {
        uint8x16_t b = vld1q_u8(row + x - kBoxR);
        uint16x8_t lo = vmovl_u8(vget_low_u8(b));
        uint16x8_t hi = vmovl_u8(vget_high_u8(b));
        uint16x8_t acc = lo;
        acc = vaddq_u16(acc, vextq_u16(lo, hi, 1));
        acc = vaddq_u16(acc, vextq_u16(lo, hi, 2));
        acc = vaddq_u16(acc, vextq_u16(lo, hi, 3));
        acc = vaddq_u16(acc, vextq_u16(lo, hi, 4));
        acc = vaddq_u16(acc, vextq_u16(lo, hi, 5));
        acc = vaddq_u16(acc, vextq_u16(lo, hi, 6));
        acc = vaddq_u16(acc, vextq_u16(lo, hi, 7));
        acc = vaddq_u16(acc, hi);
        // need k=9..14 from row + x - R + 9 ..
        int base = x - kBoxR + 8;
        for (int k = 9; k <= 14; ++k) {
            // load 8 bytes starting at row[x - R + k]
            uint8x8_t b8 = vld1_u8(row + base + (k - 8));
            acc = vaddq_u16(acc, vmovl_u8(b8));
        }
        vst1q_u16(hr + x, acc);
    }
    for (; x < body_end; ++x) {
        int sum = 0;
        const uint8_t* p = row + x - kBoxR;
        for (int k = 0; k < kBoxK; ++k) sum += p[k];
        hr[x] = (uint16_t)sum;
    }
    int tail_start = std::max(x, kBoxR);
    for (int xx = tail_start; xx < w; ++xx) {
        int sum = 0;
        for (int dx = -kBoxR; dx <= kBoxR; ++dx) {
            int idx = xx + dx;
            if (idx < 0) idx = 0;
            else if (idx >= w) idx = w - 1;
            sum += row[idx];
        }
        hr[xx] = (uint16_t)sum;
    }
}
#else
inline void hsum_row_simd(const uint8_t* row, int w, uint16_t* hr) {
    hsum_row_scalar(row, w, hr);
}
#endif



inline int scan_first_nonzero_byte(const uint8_t* row, int w) {
    int c = 0;
#if defined(__AVX2__)
    const __m256i z = _mm256_setzero_si256();
    for (; c + 32 <= w; c += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(row + c));
        __m256i eq = _mm256_cmpeq_epi8(v, z);
        uint32_t nz = ~((uint32_t)_mm256_movemask_epi8(eq));
        if (nz) return c + ctz32_nonzero(nz);
    }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (; c + 16 <= w; c += 16) {
        uint8x16_t v = vld1q_u8(row + c);
        uint8x16_t eq = vceqq_u8(v, vdupq_n_u8(0));
        if (vminvq_u8(eq) != 0xFF) {
            for (int k = 0; k < 16; ++k) if (row[c + k] != 0) return c + k;
        }
    }
#endif
    while (c < w && row[c] == 0) ++c;
    return c;
}

inline int scan_last_nonzero_byte(const uint8_t* row, int w) {
    int end = w;
#if defined(__AVX2__)
    const __m256i z = _mm256_setzero_si256();
    while (end >= 32) {
        int c = end - 32;
        __m256i v = _mm256_loadu_si256((const __m256i*)(row + c));
        __m256i eq = _mm256_cmpeq_epi8(v, z);
        uint32_t nz = ~((uint32_t)_mm256_movemask_epi8(eq));
        if (nz) {
#if defined(_MSC_VER) && !defined(__clang__)
            unsigned long bit;
            _BitScanReverse(&bit, nz);
            return c + (int)bit;
#else
            return c + 31 - (int)__builtin_clz(nz);
#endif
        }
        end = c;
    }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    while (end >= 16) {
        int c = end - 16;
        uint8x16_t v = vld1q_u8(row + c);
        uint8x16_t eq = vceqq_u8(v, vdupq_n_u8(0));
        if (vminvq_u8(eq) != 0xFF) {
            for (int k = 15; k >= 0; --k) if (row[c + k] != 0) return c + k;
        }
        end = c;
    }
#endif
    for (int i = end - 1; i >= 0; --i) if (row[i] != 0) return i;
    return -1;
}

inline void update_binary_bounds_from_row(const uint8_t* row, int y, int w,
                                          BinaryBounds* bounds, int xbase = 0) {
    if (!bounds || w <= 0) return;
    int x0 = scan_first_nonzero_byte(row, w);
    if (x0 >= w) return;
    int x1 = scan_last_nonzero_byte(row + x0, w - x0);
    if (x1 >= 0) bounds->add_span(y, xbase + x0, xbase + x0 + x1);
}

// Emit one binary row: drow[x] = (vsum[x] >= 225*(src[x]+thres+1) - 112) ? 255 : 0
inline void emit_row(const uint8_t* srow, const uint16_t* vsum, int w,
                     int thres, uint8_t* drow,
                     BinaryBounds* bounds = nullptr, int y_for_bounds = 0,
                     int valid_l = 0, int valid_r = -1)
{
    int x = 0;
#if KAKUTAG_USE_THRESHOLD_BOUNDS
    int first_fg = w;
    int last_fg = -1;
    if (valid_r < 0 || valid_r >= w) valid_r = w - 1;
    auto update_mask = [&](uint32_t mask, int base, int lanes) {
        if (!bounds || mask == 0) return;
        int l = std::max(valid_l - base, 0);
        int r = std::min(valid_r - base, lanes - 1);
        if (r < l) return;
        if (l > 0) mask &= (~0u << l);
        if (r + 1 < 32) mask &= ((1u << (r + 1)) - 1u);
        if (!mask) return;
#if defined(_MSC_VER) && !defined(__clang__)
        unsigned long fbit;
        _BitScanForward(&fbit, mask);
        int f = (int)fbit;
        unsigned long bit;
        _BitScanReverse(&bit, mask);
        int lz = (int)bit;
#else
        int f = (int)__builtin_ctz(mask);
        int lz = 31 - (int)__builtin_clz(mask);
#endif
        first_fg = std::min(first_fg, base + f);
        last_fg = std::max(last_fg, base + lz);
    };
#else
    (void)bounds; (void)y_for_bounds; (void)valid_l; (void)valid_r;
#endif
#if defined(__AVX2__)
    const __m256i v_bias = _mm256_set1_epi16((short)(thres + 1));
    const __m256i v_area = _mm256_set1_epi16((short)kBoxArea);
    const __m256i v_round= _mm256_set1_epi16((short)kBoxRound);
    const __m256i v_one  = _mm256_set1_epi16(1);
    const __m256i v_sign = _mm256_set1_epi16((short)0x8000);
    const __m256i v_mask = _mm256_set1_epi16(0x00FF);
    for (; x + 32 <= w; x += 32) {
        __m256i sb = _mm256_loadu_si256((const __m256i*)(srow + x));
        __m128i lo128 = _mm256_castsi256_si128(sb);
        __m128i hi128 = _mm256_extracti128_si256(sb, 1);
        __m256i s_lo = _mm256_cvtepu8_epi16(lo128);
        __m256i s_hi = _mm256_cvtepu8_epi16(hi128);
        __m256i rhs_lo = _mm256_sub_epi16(_mm256_mullo_epi16(_mm256_add_epi16(s_lo, v_bias), v_area), v_round);
        __m256i rhs_hi = _mm256_sub_epi16(_mm256_mullo_epi16(_mm256_add_epi16(s_hi, v_bias), v_area), v_round);
        __m256i lhs_lo = _mm256_loadu_si256((const __m256i*)(vsum + x));
        __m256i lhs_hi = _mm256_loadu_si256((const __m256i*)(vsum + x + 16));
        // unsigned >= via signed > on (val^0x8000) and (rhs-1)
        __m256i lb_lo = _mm256_xor_si256(lhs_lo, v_sign);
        __m256i lb_hi = _mm256_xor_si256(lhs_hi, v_sign);
        __m256i rb_lo = _mm256_xor_si256(_mm256_sub_epi16(rhs_lo, v_one), v_sign);
        __m256i rb_hi = _mm256_xor_si256(_mm256_sub_epi16(rhs_hi, v_one), v_sign);
        __m256i c_lo = _mm256_cmpgt_epi16(lb_lo, rb_lo);
        __m256i c_hi = _mm256_cmpgt_epi16(lb_hi, rb_hi);
        __m256i packed = _mm256_packus_epi16(
            _mm256_and_si256(c_lo, v_mask),
            _mm256_and_si256(c_hi, v_mask));
        packed = _mm256_permute4x64_epi64(packed, 0xD8);
        _mm256_storeu_si256((__m256i*)(drow + x), packed);
#if KAKUTAG_USE_THRESHOLD_BOUNDS
        update_mask((uint32_t)_mm256_movemask_epi8(packed), x, 32);
#endif
    }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    const uint16x8_t v_bias = vdupq_n_u16((uint16_t)(thres + 1));
    const uint16x8_t v_area = vdupq_n_u16((uint16_t)kBoxArea);
    const uint16x8_t v_round= vdupq_n_u16((uint16_t)kBoxRound);
    for (; x + 16 <= w; x += 16) {
        uint8x16_t sb = vld1q_u8(srow + x);
        uint16x8_t s_lo = vmovl_u8(vget_low_u8(sb));
        uint16x8_t s_hi = vmovl_u8(vget_high_u8(sb));
        uint16x8_t rhs_lo = vsubq_u16(vmulq_u16(vaddq_u16(s_lo, v_bias), v_area), v_round);
        uint16x8_t rhs_hi = vsubq_u16(vmulq_u16(vaddq_u16(s_hi, v_bias), v_area), v_round);
        uint16x8_t lhs_lo = vld1q_u16(vsum + x);
        uint16x8_t lhs_hi = vld1q_u16(vsum + x + 8);
        uint16x8_t c_lo = vcgeq_u16(lhs_lo, rhs_lo);
        uint16x8_t c_hi = vcgeq_u16(lhs_hi, rhs_hi);
        uint8x8_t  p_lo = vmovn_u16(c_lo);
        uint8x8_t  p_hi = vmovn_u16(c_hi);
        uint8x16_t outv = vcombine_u8(p_lo, p_hi);
        vst1q_u8(drow + x, outv);
#if KAKUTAG_USE_THRESHOLD_BOUNDS
        if (bounds && vmaxvq_u8(outv) != 0) {
            for (int k = 0; k < 16; ++k) {
                int xx = x + k;
                if (xx >= valid_l && xx <= valid_r && drow[xx]) {
                    first_fg = std::min(first_fg, xx);
                    break;
                }
            }
            for (int k = 15; k >= 0; --k) {
                int xx = x + k;
                if (xx >= valid_l && xx <= valid_r && drow[xx]) {
                    last_fg = std::max(last_fg, xx);
                    break;
                }
            }
        }
#endif
    }
#endif
    for (; x < w; ++x) {
        int lhs = vsum[x];
        int rhs = kBoxArea * (int(srow[x]) + thres + 1) - kBoxRound;
        drow[x] = (lhs >= rhs) ? 255 : 0;
#if KAKUTAG_USE_THRESHOLD_BOUNDS
        if (bounds && x >= valid_l && x <= valid_r && drow[x]) {
            first_fg = std::min(first_fg, x);
            last_fg = std::max(last_fg, x);
        }
#endif
    }
#if KAKUTAG_USE_THRESHOLD_BOUNDS
    if (bounds && last_fg >= first_fg) bounds->add_span(y_for_bounds, first_fg, last_fg);
#endif
}

// vsum[x] += add[x] - rem[x]   (uint16 modular)
inline void vsum_slide(uint16_t* vsum, const uint16_t* rem,
                       const uint16_t* add, int w)
{
    int x = 0;
#if defined(__AVX2__)
    for (; x + 16 <= w; x += 16) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(vsum + x));
        __m256i a = _mm256_loadu_si256((const __m256i*)(add  + x));
        __m256i r = _mm256_loadu_si256((const __m256i*)(rem  + x));
        v = _mm256_sub_epi16(_mm256_add_epi16(v, a), r);
        _mm256_storeu_si256((__m256i*)(vsum + x), v);
    }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (; x + 8 <= w; x += 8) {
        uint16x8_t v = vld1q_u16(vsum + x);
        uint16x8_t a = vld1q_u16(add + x);
        uint16x8_t r = vld1q_u16(rem + x);
        vst1q_u16(vsum + x, vsubq_u16(vaddq_u16(v, a), r));
    }
#endif
    for (; x < w; ++x) {
        vsum[x] = (uint16_t)(int(vsum[x]) + int(add[x]) - int(rem[x]));
    }
}

inline void streaming_threshold_to_buffer(
    const uint8_t* src, int w, int h, int stride, int thres,
    uint8_t* dst, int dst_stride, ThresholdScratch* scratch = nullptr,
    BinaryBounds* bounds = nullptr, RunTileGate* run_gate = nullptr)
{
    if (!src || !dst || w <= 0 || h <= 0) return;
    if (bounds) bounds->reset();
#if KAKUTAG_USE_RUN_TILE_GATE
    if (run_gate) run_gate->any = false;
#else
    (void)run_gate;
#endif

    std::vector<uint16_t> local_ring;
    std::vector<uint16_t> local_vsum;
    std::vector<uint16_t> local_scratch_rem;
    std::vector<uint16_t>& ring = scratch ? scratch->hsum_ring : local_ring;
    std::vector<uint16_t>& vsum = scratch ? scratch->vsum : local_vsum;
    std::vector<uint16_t>& scratch_rem = scratch ? scratch->scratch_rem : local_scratch_rem;
    const size_t ring_need = (size_t)kBoxK * (size_t)w;
    if (ring.size() < ring_need) ring.resize(ring_need);
    if (vsum.size() < (size_t)w) vsum.resize((size_t)w);
    if (scratch_rem.size() < (size_t)w) scratch_rem.resize((size_t)w);

    auto slot = [&](int idx) -> uint16_t* {
        int m = idx % kBoxK;
        if (m < 0) m += kBoxK;
        return ring.data() + (size_t)m * w;
    };
    int rows_to_init = std::min(kBoxR + 1, h);
    for (int r = 0; r < rows_to_init; ++r) {
        hsum_row_simd(src + r * stride, w, slot(r));
    }
    const uint16_t* init_rows[kBoxK];
    for (int dy = -kBoxR, i = 0; dy <= kBoxR; ++dy, ++i) {
        int phys = dy;
        if (phys < 0) phys = 0;
        if (phys >= h) phys = h - 1;
        init_rows[i] = slot(phys);
    }
    for (int x = 0; x < w; ++x) {
        int s = 0;
        for (int k = 0; k < kBoxK; ++k) s += int(init_rows[k][x]);
        vsum[(size_t)x] = (uint16_t)s;
    }
    for (int y = 0; y < h; ++y) {
        const uint8_t* srow = src + y * stride;
        uint8_t* drow = dst + y * dst_stride;
        emit_row(srow, vsum.data(), w, thres, drow,
                 (bounds && y > 0 && y + 1 < h && w > 2) ? bounds : nullptr,
                 y, 1, w - 2);
#if KAKUTAG_USE_RUN_TILE_GATE
        if (run_gate && y > 0 && y + 1 < h && w > 2)
            update_run_tile_gate_from_row(drow, y, w, run_gate, 1, w - 2);
#endif
        if (y + 1 >= h) break;
        int leave_phys = y - kBoxR;
        int enter_phys = y + 1 + kBoxR;
        const uint16_t* rem_ptr;
        if (leave_phys < 0) rem_ptr = slot(0);
        else if (leave_phys >= h) rem_ptr = slot(h - 1);
        else rem_ptr = slot(leave_phys);
        if (enter_phys < h) {
            uint16_t* dst_slot = slot(enter_phys);
            if (dst_slot == rem_ptr) {
                std::memcpy(scratch_rem.data(), rem_ptr, (size_t)w * sizeof(uint16_t));
                hsum_row_simd(src + enter_phys * stride, w, dst_slot);
                vsum_slide(vsum.data(), scratch_rem.data(), dst_slot, w);
            } else {
                hsum_row_simd(src + enter_phys * stride, w, dst_slot);
                vsum_slide(vsum.data(), rem_ptr, dst_slot, w);
            }
        } else {
            const uint16_t* add_ptr = slot(h - 1);
            if (add_ptr != rem_ptr) {
                vsum_slide(vsum.data(), rem_ptr, add_ptr, w);
            }
        }
    }
}

// =============================================================================
// Ternary (hysteresis) variant
// -----------------------------------------------------------------------------
// Emits 3-state binary into dst:
//   STRONG (255) when local_mean - pixel >= thres_high + 0.5
//   WEAK   ( 64) when local_mean - pixel >= thres_low  + 0.5
//   BG     (  0) otherwise
// trace_runlen_streaming() seeds only on FOREGROUND (255), and run length is
// measured via scan_end_fg() which counts only 255s, so only STRONG pixels can
// seed. The 8-neighbor probe (probe8_*) on the other hand traverses STRONG,
// WEAK, and VISITED bytes alike because it stops at the first byte != 0. This
// is the Canny-style hysteresis rule: high-confidence pixels seed, low-
// confidence pixels are walked only when adjacent to a strong border.
// =============================================================================

inline void emit_row_ternary(const uint8_t* srow, const uint16_t* vsum, int w,
                             int thres_high, int thres_low, uint8_t* drow)
{
    int x = 0;
#if defined(__AVX2__)
    // rhs_l = rhs_h + (low - high) * area, so we mullo only for the high
    // threshold and add a precomputed delta for the low side.
    const __m256i v_bias_h = _mm256_set1_epi16((short)(thres_high + 1));
    const __m256i v_delta  = _mm256_set1_epi16((short)((thres_low - thres_high) * kBoxArea));
    const __m256i v_area   = _mm256_set1_epi16((short)kBoxArea);
    const __m256i v_round  = _mm256_set1_epi16((short)kBoxRound);
    const __m256i v_one    = _mm256_set1_epi16(1);
    const __m256i v_sign   = _mm256_set1_epi16((short)0x8000);
    const __m256i v_mask   = _mm256_set1_epi16(0x00FF);
    const __m256i v_weak   = _mm256_set1_epi16((short)64);
    const __m256i v_strong = _mm256_set1_epi16((short)(255-64));
    for (; x + 32 <= w; x += 32) {
        __m256i sb = _mm256_loadu_si256((const __m256i*)(srow + x));
        __m128i lo128 = _mm256_castsi256_si128(sb);
        __m128i hi128 = _mm256_extracti128_si256(sb, 1);
        __m256i s_lo = _mm256_cvtepu8_epi16(lo128);
        __m256i s_hi = _mm256_cvtepu8_epi16(hi128);
        __m256i lhs_lo = _mm256_loadu_si256((const __m256i*)(vsum + x));
        __m256i lhs_hi = _mm256_loadu_si256((const __m256i*)(vsum + x + 16));
        __m256i lb_lo  = _mm256_xor_si256(lhs_lo, v_sign);
        __m256i lb_hi  = _mm256_xor_si256(lhs_hi, v_sign);
        __m256i rhs_h_lo = _mm256_sub_epi16(_mm256_mullo_epi16(_mm256_add_epi16(s_lo, v_bias_h), v_area), v_round);
        __m256i rhs_h_hi = _mm256_sub_epi16(_mm256_mullo_epi16(_mm256_add_epi16(s_hi, v_bias_h), v_area), v_round);
        __m256i rb_h_lo  = _mm256_xor_si256(_mm256_sub_epi16(rhs_h_lo, v_one), v_sign);
        __m256i rb_h_hi  = _mm256_xor_si256(_mm256_sub_epi16(rhs_h_hi, v_one), v_sign);
        __m256i ch_lo    = _mm256_cmpgt_epi16(lb_lo, rb_h_lo);
        __m256i ch_hi    = _mm256_cmpgt_epi16(lb_hi, rb_h_hi);
        __m256i rhs_l_lo = _mm256_add_epi16(rhs_h_lo, v_delta);
        __m256i rhs_l_hi = _mm256_add_epi16(rhs_h_hi, v_delta);
        __m256i rb_l_lo  = _mm256_xor_si256(_mm256_sub_epi16(rhs_l_lo, v_one), v_sign);
        __m256i rb_l_hi  = _mm256_xor_si256(_mm256_sub_epi16(rhs_l_hi, v_one), v_sign);
        __m256i cl_lo    = _mm256_cmpgt_epi16(lb_lo, rb_l_lo);
        __m256i cl_hi    = _mm256_cmpgt_epi16(lb_hi, rb_l_hi);
        __m256i weak_lo   = _mm256_and_si256(cl_lo, v_weak);
        __m256i weak_hi   = _mm256_and_si256(cl_hi, v_weak);
        __m256i strong_lo = _mm256_and_si256(ch_lo, v_strong);
        __m256i strong_hi = _mm256_and_si256(ch_hi, v_strong);
        __m256i out_lo    = _mm256_add_epi16(weak_lo, strong_lo);
        __m256i out_hi    = _mm256_add_epi16(weak_hi, strong_hi);
        __m256i packed    = _mm256_packus_epi16(
            _mm256_and_si256(out_lo, v_mask),
            _mm256_and_si256(out_hi, v_mask));
        packed = _mm256_permute4x64_epi64(packed, 0xD8);
        _mm256_storeu_si256((__m256i*)(drow + x), packed);
    }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    const uint16x8_t v_bias_h = vdupq_n_u16((uint16_t)(thres_high + 1));
    const uint16x8_t v_bias_l = vdupq_n_u16((uint16_t)(thres_low  + 1));
    const uint16x8_t v_area   = vdupq_n_u16((uint16_t)kBoxArea);
    const uint16x8_t v_round  = vdupq_n_u16((uint16_t)kBoxRound);
    const uint8x8_t  v_weak   = vdup_n_u8(64);
    const uint8x8_t  v_str    = vdup_n_u8(255-64);
    for (; x + 16 <= w; x += 16) {
        uint8x16_t sb = vld1q_u8(srow + x);
        uint16x8_t s_lo = vmovl_u8(vget_low_u8(sb));
        uint16x8_t s_hi = vmovl_u8(vget_high_u8(sb));
        uint16x8_t rhs_h_lo = vsubq_u16(vmulq_u16(vaddq_u16(s_lo, v_bias_h), v_area), v_round);
        uint16x8_t rhs_h_hi = vsubq_u16(vmulq_u16(vaddq_u16(s_hi, v_bias_h), v_area), v_round);
        uint16x8_t rhs_l_lo = vsubq_u16(vmulq_u16(vaddq_u16(s_lo, v_bias_l), v_area), v_round);
        uint16x8_t rhs_l_hi = vsubq_u16(vmulq_u16(vaddq_u16(s_hi, v_bias_l), v_area), v_round);
        uint16x8_t lhs_lo = vld1q_u16(vsum + x);
        uint16x8_t lhs_hi = vld1q_u16(vsum + x + 8);
        uint16x8_t ch_lo = vcgeq_u16(lhs_lo, rhs_h_lo);
        uint16x8_t ch_hi = vcgeq_u16(lhs_hi, rhs_h_hi);
        uint16x8_t cl_lo = vcgeq_u16(lhs_lo, rhs_l_lo);
        uint16x8_t cl_hi = vcgeq_u16(lhs_hi, rhs_l_hi);
        uint8x8_t  ch_b_lo = vmovn_u16(ch_lo);
        uint8x8_t  ch_b_hi = vmovn_u16(ch_hi);
        uint8x8_t  cl_b_lo = vmovn_u16(cl_lo);
        uint8x8_t  cl_b_hi = vmovn_u16(cl_hi);
        uint8x8_t  weak_lo_b = vand_u8(cl_b_lo, v_weak);
        uint8x8_t  weak_hi_b = vand_u8(cl_b_hi, v_weak);
        uint8x8_t  str_lo_b  = vand_u8(ch_b_lo, v_str);
        uint8x8_t  str_hi_b  = vand_u8(ch_b_hi, v_str);
        uint8x8_t  out_lo = vadd_u8(weak_lo_b, str_lo_b);
        uint8x8_t  out_hi = vadd_u8(weak_hi_b, str_hi_b);
        vst1q_u8(drow + x, vcombine_u8(out_lo, out_hi));
    }
#endif
    for (; x < w; ++x) {
        int lhs = vsum[x];
        int rhs_h = kBoxArea * (int(srow[x]) + thres_high + 1) - kBoxRound;
        int rhs_l = kBoxArea * (int(srow[x]) + thres_low  + 1) - kBoxRound;
        uint8_t v = 0;
        if (lhs >= rhs_l) v = 64;
        if (lhs >= rhs_h) v = 255;
        drow[x] = v;
    }
}

inline void streaming_threshold_ternary_to_buffer(
    const uint8_t* src, int w, int h, int stride,
    int thres_high, int thres_low,
    uint8_t* dst, int dst_stride, ThresholdScratch* scratch = nullptr)
{
    if (!src || !dst || w <= 0 || h <= 0) return;
    if (thres_low > thres_high) thres_low = thres_high;
    std::vector<uint16_t> local_ring;
    std::vector<uint16_t> local_vsum;
    std::vector<uint16_t> local_scratch_rem;
    std::vector<uint16_t>& ring = scratch ? scratch->hsum_ring : local_ring;
    std::vector<uint16_t>& vsum = scratch ? scratch->vsum : local_vsum;
    std::vector<uint16_t>& scratch_rem = scratch ? scratch->scratch_rem : local_scratch_rem;
    const size_t ring_need = (size_t)kBoxK * (size_t)w;
    if (ring.size() < ring_need) ring.resize(ring_need);
    if (vsum.size() < (size_t)w) vsum.resize((size_t)w);
    if (scratch_rem.size() < (size_t)w) scratch_rem.resize((size_t)w);
    auto sl = [&](int idx) -> uint16_t* {
        int m = idx % kBoxK; if (m < 0) m += kBoxK;
        return ring.data() + (size_t)m * w;
    };
    int rows_to_init = std::min(kBoxR + 1, h);
    for (int r = 0; r < rows_to_init; ++r) hsum_row_simd(src + r * stride, w, sl(r));
    const uint16_t* init_rows[kBoxK];
    for (int dy = -kBoxR, i = 0; dy <= kBoxR; ++dy, ++i) {
        int phys = dy; if (phys < 0) phys = 0; if (phys >= h) phys = h - 1;
        init_rows[i] = sl(phys);
    }
    for (int x = 0; x < w; ++x) {
        int s = 0;
        for (int k = 0; k < kBoxK; ++k) s += int(init_rows[k][x]);
        vsum[(size_t)x] = (uint16_t)s;
    }
    for (int y = 0; y < h; ++y) {
        const uint8_t* srow = src + y * stride;
        uint8_t* drow = dst + y * dst_stride;
        emit_row_ternary(srow, vsum.data(), w, thres_high, thres_low, drow);
        if (y + 1 >= h) break;
        int leave_phys = y - kBoxR;
        int enter_phys = y + 1 + kBoxR;
        const uint16_t* rem_ptr;
        if (leave_phys < 0) rem_ptr = sl(0);
        else if (leave_phys >= h) rem_ptr = sl(h - 1);
        else rem_ptr = sl(leave_phys);
        if (enter_phys < h) {
            uint16_t* dst_slot = sl(enter_phys);
            if (dst_slot == rem_ptr) {
                std::memcpy(scratch_rem.data(), rem_ptr, (size_t)w * sizeof(uint16_t));
                hsum_row_simd(src + enter_phys * stride, w, dst_slot);
                vsum_slide(vsum.data(), scratch_rem.data(), dst_slot, w);
            } else {
                hsum_row_simd(src + enter_phys * stride, w, dst_slot);
                vsum_slide(vsum.data(), rem_ptr, dst_slot, w);
            }
        } else {
            const uint16_t* add_ptr = sl(h - 1);
            if (add_ptr != rem_ptr) vsum_slide(vsum.data(), rem_ptr, add_ptr, w);
        }
    }
}

// 2x downsample one row by exact 2x2 area averaging. This is the primitive
// behind the pyramid path below. Unlike cv::resize + threshold, it lets KakuTag
// stream the half-scale image through the threshold/contour pipeline without
// materializing a full temporary small image.
inline void downsample2_row_simd(const uint8_t* r0, const uint8_t* r1,
                                 int sw, uint8_t* dst) {
    int x = 0;
#if defined(__AVX2__)
    const __m256i ones = _mm256_set1_epi8(1);
    const __m256i round = _mm256_set1_epi16(2);
    for (; x + 16 <= sw; x += 16) {
        __m256i a = _mm256_loadu_si256((const __m256i*)(r0 + 2 * x));
        __m256i b = _mm256_loadu_si256((const __m256i*)(r1 + 2 * x));
        __m256i pa = _mm256_maddubs_epi16(a, ones);  // adjacent byte pairs
        __m256i pb = _mm256_maddubs_epi16(b, ones);
        __m256i avg16 = _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(pa, pb), round), 2);
        __m128i lo = _mm256_castsi256_si128(avg16);
        __m128i hi = _mm256_extracti128_si256(avg16, 1);
        __m128i packed = _mm_packus_epi16(lo, hi);
        _mm_storeu_si128((__m128i*)(dst + x), packed);
    }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    const uint16x8_t round = vdupq_n_u16(2);
    for (; x + 8 <= sw; x += 8) {
        uint8x16_t a = vld1q_u8(r0 + 2 * x);
        uint8x16_t b = vld1q_u8(r1 + 2 * x);
        uint16x8_t pa = vpaddlq_u8(a);
        uint16x8_t pb = vpaddlq_u8(b);
        uint16x8_t avg16 = vshrq_n_u16(vaddq_u16(vaddq_u16(pa, pb), round), 2);
        vst1_u8(dst + x, vmovn_u16(avg16));
    }
#endif
    for (; x < sw; ++x) {
        int sx = 2 * x;
        int sum = int(r0[sx]) + int(r0[sx + 1]) + int(r1[sx]) + int(r1[sx + 1]);
        dst[x] = (uint8_t)((sum + 2) >> 2);
    }
}

inline void hsum_down2_row_simd(const uint8_t* src, int src_h, int stride,
                                int y2, int sw, uint8_t* down_slot,
                                uint16_t* hsum_slot) {
    int sy = 2 * y2;
    if (sy < 0) sy = 0;
    if (sy >= src_h) sy = src_h - 1;
    int sy1 = sy + 1;
    if (sy1 >= src_h) sy1 = src_h - 1;
    const uint8_t* r0 = src + sy * stride;
    const uint8_t* r1 = src + sy1 * stride;
    downsample2_row_simd(r0, r1, sw, down_slot);
    hsum_row_simd(down_slot, sw, hsum_slot);
}

// Fused half-scale threshold: source is full resolution, destination is W/2 x H/2.
// It computes exactly the image needed by the pyramid contour generator while
// avoiding the separate cv::resize pass and a full-size small-image scratch.
inline void streaming_threshold_down2_to_buffer(
    const uint8_t* src, int w, int h, int stride, int thres,
    uint8_t* dst, int dst_stride, ThresholdScratch* scratch = nullptr,
    BinaryBounds* bounds = nullptr, RunTileGate* run_gate = nullptr)
{
    if (!src || !dst || w < 2 || h < 2) return;
    if (bounds) bounds->reset();
#if KAKUTAG_USE_RUN_TILE_GATE
    if (run_gate) run_gate->any = false;
#else
    (void)run_gate;
#endif
    const int sw = w / 2;
    const int sh = h / 2;
    if (sw <= 0 || sh <= 0) return;

    std::vector<uint8_t>  local_down_ring;
    std::vector<uint16_t> local_hsum_ring;
    std::vector<uint16_t> local_vsum;
    std::vector<uint16_t> local_scratch_rem;
    std::vector<uint8_t>& down_ring = scratch ? scratch->down_ring : local_down_ring;
    std::vector<uint16_t>& hsum_ring = scratch ? scratch->hsum_ring : local_hsum_ring;
    std::vector<uint16_t>& vsum = scratch ? scratch->vsum : local_vsum;
    std::vector<uint16_t>& scratch_rem = scratch ? scratch->scratch_rem : local_scratch_rem;

    const size_t ring_need = (size_t)kBoxK * (size_t)sw;
    if (down_ring.size() < ring_need) down_ring.resize(ring_need);
    if (hsum_ring.size() < ring_need) hsum_ring.resize(ring_need);
    if (vsum.size() < (size_t)sw) vsum.resize((size_t)sw);
    if (scratch_rem.size() < (size_t)sw) scratch_rem.resize((size_t)sw);

    auto dslot = [&](int idx) -> uint8_t* {
        int m = idx % kBoxK;
        if (m < 0) m += kBoxK;
        return down_ring.data() + (size_t)m * sw;
    };
    auto hslot = [&](int idx) -> uint16_t* {
        int m = idx % kBoxK;
        if (m < 0) m += kBoxK;
        return hsum_ring.data() + (size_t)m * sw;
    };

    int rows_to_init = std::min(kBoxR + 1, sh);
    for (int r = 0; r < rows_to_init; ++r) {
        hsum_down2_row_simd(src, h, stride, r, sw, dslot(r), hslot(r));
    }

    const uint16_t* init_rows[kBoxK];
    for (int dy = -kBoxR, i = 0; dy <= kBoxR; ++dy, ++i) {
        int phys = dy;
        if (phys < 0) phys = 0;
        if (phys >= sh) phys = sh - 1;
        init_rows[i] = hslot(phys);
    }
    for (int x = 0; x < sw; ++x) {
        int s = 0;
        for (int k = 0; k < kBoxK; ++k) s += int(init_rows[k][x]);
        vsum[(size_t)x] = (uint16_t)s;
    }

    for (int y = 0; y < sh; ++y) {
        uint8_t* drow = dst + y * dst_stride;
        emit_row(dslot(y), vsum.data(), sw, thres, drow,
                 (bounds && y > 0 && y + 1 < sh && sw > 2) ? bounds : nullptr,
                 y, 1, sw - 2);
#if KAKUTAG_USE_RUN_TILE_GATE
        if (run_gate && y > 0 && y + 1 < sh && sw > 2)
            update_run_tile_gate_from_row(drow, y, sw, run_gate, 1, sw - 2);
#endif
        if (y + 1 >= sh) break;

        int leave_phys = y - kBoxR;
        int enter_phys = y + 1 + kBoxR;
        const uint16_t* rem_ptr;
        if (leave_phys < 0) rem_ptr = hslot(0);
        else if (leave_phys >= sh) rem_ptr = hslot(sh - 1);
        else rem_ptr = hslot(leave_phys);

        if (enter_phys < sh) {
            uint16_t* dst_hslot = hslot(enter_phys);
            uint8_t*  dst_dslot = dslot(enter_phys);
            if (dst_hslot == rem_ptr) {
                std::memcpy(scratch_rem.data(), rem_ptr, (size_t)sw * sizeof(uint16_t));
                hsum_down2_row_simd(src, h, stride, enter_phys, sw, dst_dslot, dst_hslot);
                vsum_slide(vsum.data(), scratch_rem.data(), dst_hslot, sw);
            } else {
                hsum_down2_row_simd(src, h, stride, enter_phys, sw, dst_dslot, dst_hslot);
                vsum_slide(vsum.data(), rem_ptr, dst_hslot, sw);
            }
        } else {
            const uint16_t* add_ptr = hslot(sh - 1);
            if (add_ptr != rem_ptr) {
                vsum_slide(vsum.data(), rem_ptr, add_ptr, sw);
            }
        }
    }
}


// Fused half-scale ternary threshold. This is the pyramid counterpart of
// streaming_threshold_ternary_to_buffer(): it preserves KakuTag's fast half-
// scale candidate generation while allowing weak shadowed border pixels to be
// walked only from strong seeds. It avoids the old tradeoff of either using a
// strict half-scale binary pass or paying for a full-resolution fallback.
inline void streaming_threshold_down2_ternary_to_buffer(
    const uint8_t* src, int w, int h, int stride,
    int thres_high, int thres_low,
    uint8_t* dst, int dst_stride, ThresholdScratch* scratch = nullptr,
    BinaryBounds* bounds = nullptr, RunTileGate* run_gate = nullptr)
{
    if (!src || !dst || w < 2 || h < 2) return;
    if (thres_low > thres_high) thres_low = thres_high;
    if (bounds) bounds->reset();
#if KAKUTAG_USE_RUN_TILE_GATE
    if (run_gate) run_gate->any = false;
#else
    (void)run_gate;
#endif
    const int sw = w / 2;
    const int sh = h / 2;
    if (sw <= 0 || sh <= 0) return;

    std::vector<uint8_t>  local_down_ring;
    std::vector<uint16_t> local_hsum_ring;
    std::vector<uint16_t> local_vsum;
    std::vector<uint16_t> local_scratch_rem;
    std::vector<uint8_t>& down_ring = scratch ? scratch->down_ring : local_down_ring;
    std::vector<uint16_t>& hsum_ring = scratch ? scratch->hsum_ring : local_hsum_ring;
    std::vector<uint16_t>& vsum = scratch ? scratch->vsum : local_vsum;
    std::vector<uint16_t>& scratch_rem = scratch ? scratch->scratch_rem : local_scratch_rem;

    const size_t ring_need = (size_t)kBoxK * (size_t)sw;
    if (down_ring.size() < ring_need) down_ring.resize(ring_need);
    if (hsum_ring.size() < ring_need) hsum_ring.resize(ring_need);
    if (vsum.size() < (size_t)sw) vsum.resize((size_t)sw);
    if (scratch_rem.size() < (size_t)sw) scratch_rem.resize((size_t)sw);

    auto dslot = [&](int idx) -> uint8_t* {
        int m = idx % kBoxK;
        if (m < 0) m += kBoxK;
        return down_ring.data() + (size_t)m * sw;
    };
    auto hslot = [&](int idx) -> uint16_t* {
        int m = idx % kBoxK;
        if (m < 0) m += kBoxK;
        return hsum_ring.data() + (size_t)m * sw;
    };

    int rows_to_init = std::min(kBoxR + 1, sh);
    for (int r = 0; r < rows_to_init; ++r) {
        hsum_down2_row_simd(src, h, stride, r, sw, dslot(r), hslot(r));
    }

    const uint16_t* init_rows[kBoxK];
    for (int dy = -kBoxR, i = 0; dy <= kBoxR; ++dy, ++i) {
        int phys = dy;
        if (phys < 0) phys = 0;
        if (phys >= sh) phys = sh - 1;
        init_rows[i] = hslot(phys);
    }
    for (int x = 0; x < sw; ++x) {
        int sum = 0;
        for (int k = 0; k < kBoxK; ++k) sum += int(init_rows[k][x]);
        vsum[(size_t)x] = (uint16_t)sum;
    }

    for (int y = 0; y < sh; ++y) {
        uint8_t* drow = dst + y * dst_stride;
        emit_row_ternary(dslot(y), vsum.data(), sw, thres_high, thres_low, drow);
        if (bounds && y > 0 && y + 1 < sh && sw > 2) {
            for (int x = 1; x < sw - 1; ++x) {
                if (drow[x]) bounds->add_span(y, x, x);
            }
        }
#if KAKUTAG_USE_RUN_TILE_GATE
        if (run_gate && y > 0 && y + 1 < sh && sw > 2)
            update_run_tile_gate_from_row(drow, y, sw, run_gate, 1, sw - 2);
#endif
        if (y + 1 >= sh) break;

        int leave_phys = y - kBoxR;
        int enter_phys = y + 1 + kBoxR;
        const uint16_t* rem_ptr;
        if (leave_phys < 0) rem_ptr = hslot(0);
        else if (leave_phys >= sh) rem_ptr = hslot(sh - 1);
        else rem_ptr = hslot(leave_phys);

        if (enter_phys < sh) {
            uint16_t* dst_hslot = hslot(enter_phys);
            uint8_t*  dst_dslot = dslot(enter_phys);
            if (dst_hslot == rem_ptr) {
                std::memcpy(scratch_rem.data(), rem_ptr, (size_t)sw * sizeof(uint16_t));
                hsum_down2_row_simd(src, h, stride, enter_phys, sw, dst_dslot, dst_hslot);
                vsum_slide(vsum.data(), scratch_rem.data(), dst_hslot, sw);
            } else {
                hsum_down2_row_simd(src, h, stride, enter_phys, sw, dst_dslot, dst_hslot);
                vsum_slide(vsum.data(), rem_ptr, dst_hslot, sw);
            }
        } else {
            const uint16_t* add_ptr = hslot(sh - 1);
            if (add_ptr != rem_ptr) vsum_slide(vsum.data(), rem_ptr, add_ptr, sw);
        }
    }
}

// Runlen-prefiltered Suzuki85 tracer.
// Produces external contours. visited-aware: a contour is rejected if its
// revisited-pixel ratio exceeds maxRevisited.

// Scan: find first foreground (255) starting at c, returning index or `cols`.
// Any non-zero byte (FOREGROUND=255 or VISITED=100) stops the scan; caller decides how to handle it.
inline int scan_first_fg(const uint8_t* row_ptr, int c, int cols) {
#if defined(__AVX2__)
    const __m256i v_zero = _mm256_setzero_si256();
    for (; c + 32 <= cols; c += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(row_ptr + c));
        // bytes that are non-zero -> 0xFF, else 0
        __m256i m = _mm256_cmpgt_epi8(v, v_zero); // signed > 0; visited=100 is > 0, fg=255 = -1 is NOT >0
        // Actually FOREGROUND=255 is signed -1, so cmpgt_epi8 misses it.
        // Use cmpeq with 0 and invert via testc? Easier: cmpeq with 0 -> bg mask, find first 0 bit.
        __m256i bg = _mm256_cmpeq_epi8(v, v_zero);
        uint32_t bg_mask = (uint32_t)_mm256_movemask_epi8(bg);
        uint32_t non_bg = ~bg_mask;
        if (non_bg != 0) {
            // First non-background byte. But we need first FOREGROUND (255), not visited (100).
            // Visited bytes are also non-bg; fall through to scalar to confirm.
            int idx = ctz32_nonzero(non_bg);
            // verify with scalar
            for (int k = idx; k < 32; ++k) {
                if (row_ptr[c + k] != 0) return c + k;
            }
        }
    }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (; c + 16 <= cols; c += 16) {
        uint8x16_t v = vld1q_u8(row_ptr + c);
        uint8x16_t bg = vceqq_u8(v, vdupq_n_u8(0));
        // If any non-bg byte exists, fall through to scalar.
        if (vmaxvq_u8(vmvnq_u8(bg)) != 0) {
            for (int k = 0; k < 16; ++k) {
                if (row_ptr[c + k] != 0) return c + k;
            }
        }
    }
#endif
    while (c < cols && !row_ptr[c]) ++c;
    return c;
}

// Scan: find first STRONG foreground byte (exactly 255) starting at c.
// This is the correct seed scan for ternary hysteresis buffers: WEAK(64)
// pixels may be traversed after a STRONG seed is found, but must never hide a
// later STRONG seed in the same non-zero row run.
inline int scan_first_strong(const uint8_t* row_ptr, int c, int cols) {
#if defined(__AVX2__)
    const __m256i v_fg = _mm256_set1_epi8((char)0xFF);
    for (; c + 32 <= cols; c += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(row_ptr + c));
        __m256i fg = _mm256_cmpeq_epi8(v, v_fg);
        uint32_t mask = (uint32_t)_mm256_movemask_epi8(fg);
        if (mask != 0) return c + ctz32_nonzero(mask);
    }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (; c + 16 <= cols; c += 16) {
        uint8x16_t v = vld1q_u8(row_ptr + c);
        uint8x16_t fg = vceqq_u8(v, vdupq_n_u8(255));
        if (vmaxvq_u8(fg) != 0) {
            for (int k = 0; k < 16; ++k) if (row_ptr[c + k] == 255) return c + k;
        }
    }
#endif
    while (c < cols && row_ptr[c] != 255) ++c;
    return c;
}
// Scan: find first non-foreground (i.e., not exactly 255) starting at c.
// Used to find end of a foreground run.
inline int scan_end_fg(const uint8_t* row_ptr, int c, int cols) {
#if defined(__AVX2__)
    const __m256i v_fg = _mm256_set1_epi8((char)0xFF);
    for (; c + 32 <= cols; c += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(row_ptr + c));
        __m256i fg = _mm256_cmpeq_epi8(v, v_fg);
        uint32_t fg_mask = (uint32_t)_mm256_movemask_epi8(fg);
        // First byte that is NOT fg = first 0-bit in fg_mask.
        if (fg_mask != 0xFFFFFFFFu) {
            uint32_t not_fg = ~fg_mask;
            int idx = ctz32_nonzero(not_fg);
            return c + idx;
        }
    }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (; c + 16 <= cols; c += 16) {
        uint8x16_t v = vld1q_u8(row_ptr + c);
        uint8x16_t fg = vceqq_u8(v, vdupq_n_u8(255));
        // If all 16 bytes are fg, vminvq returns 0xFF.
        if (vminvq_u8(fg) != 0xFF) {
            for (int k = 0; k < 16; ++k) {
                if (row_ptr[c + k] != 255) return c + k;
            }
        }
    }
#endif
    while (c < cols && row_ptr[c] == 255) ++c;
    return c;
}

// Scan: skip any non-zero bytes (foreground OR visited) starting at c.
// Used to advance past a region we've already processed.
inline int scan_skip_nonzero(const uint8_t* row_ptr, int c, int cols) {
#if defined(__AVX2__)
    const __m256i v_zero = _mm256_setzero_si256();
    for (; c + 32 <= cols; c += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(row_ptr + c));
        __m256i bg = _mm256_cmpeq_epi8(v, v_zero);
        uint32_t bg_mask = (uint32_t)_mm256_movemask_epi8(bg);
        if (bg_mask != 0) {
            int idx = ctz32_nonzero(bg_mask);
            return c + idx;
        }
    }
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    for (; c + 16 <= cols; c += 16) {
        uint8x16_t v = vld1q_u8(row_ptr + c);
        uint8x16_t bg = vceqq_u8(v, vdupq_n_u8(0));
        if (vmaxvq_u8(bg) != 0) {
            for (int k = 0; k < 16; ++k) {
                if (row_ptr[c + k] == 0) return c + k;
            }
        }
    }
#endif
    while (c < cols && row_ptr[c]) ++c;
    return c;
}



// 8x8 tile popcount gate (byte mask version; supports tw > 64).
// For each 8x8 tile, count the black (==0) pixels and set tile_mask[ty*tw+tx]
// = 1 when in [lo, hi]. Conservative defaults are wide enough that any
// plausible marker boundary tile survives.
inline void tile_popcount_gate(
    const uint8_t* binary, int w, int h, int stride,
    int lo_black, int hi_black,
    uint8_t* tile_mask, int tw, int th)
{
    const uint64_t kLow  = 0x0101010101010101ULL;
    const uint64_t kHigh = 0x8080808080808080ULL;
    for (int ty = 0; ty < th; ++ty) {
        for (int tx = 0; tx < tw; ++tx) {
            int blk = 0;
            for (int yy = 0; yy < 8; ++yy) {
                const uint8_t* row = binary + (ty*8 + yy)*stride + tx*8;
                uint64_t v;
                std::memcpy(&v, row, 8);
                uint64_t zeros = (v - kLow) & ~v & kHigh;
#if defined(_MSC_VER) && !defined(__clang__)
                blk += (int)__popcnt64(zeros);
#else
                blk += __builtin_popcountll(zeros);
#endif
            }
            tile_mask[(size_t)ty * tw + tx] = (blk >= lo_black && blk <= hi_black) ? 1 : 0;
        }
    }
}

// Mark 8x8 tiles that contain at least one horizontal FOREGROUND run with
// length >= min_run. This is exactly the seed condition used by Suzuki85's
// runlen prefilter, lifted to tile granularity. The trace itself is not
// constrained to these tiles; the mask only skips seed search where a seed
// cannot exist.
inline bool tile_horizontal_run_seed_gate(
    const uint8_t* binary, int w, int h, int stride, int min_run,
    uint8_t* tile_mask, int tw, int th, cv::Rect scan_roi = cv::Rect())
{
    if (!binary || !tile_mask || w <= 0 || h <= 0 || tw <= 0 || th <= 0) return false;
    std::fill(tile_mask, tile_mask + (size_t)tw * th, (uint8_t)0);
    cv::Rect sr = scan_roi.area() > 0 ? (scan_roi & cv::Rect(1, 1, w - 2, h - 2))
                                      : cv::Rect(1, 1, w - 2, h - 2);
    if (sr.empty()) return false;
    const int y_end = sr.y + sr.height;
    const int x_end = sr.x + sr.width;
    bool any = false;
    for (int y = sr.y; y < y_end; ++y) {
        const uint8_t* row = binary + (size_t)y * stride;
        int x = sr.x;
        while (x < x_end) {
            x = scan_first_fg(row, x, x_end);
            if (x >= x_end) break;
            if (row[x] != 255) { x = scan_skip_nonzero(row, x, x_end); continue; }
            const int rs = x;
            const int re = scan_end_fg(row, x, x_end);
            if (re - rs >= min_run) {
                const int ty = y >> 3;
                if (ty >= 0 && ty < th) {
                    int tx0 = rs >> 3;
                    int tx1 = (re - 1) >> 3;
                    tx0 = std::max(0, std::min(tx0, tw - 1));
                    tx1 = std::max(0, std::min(tx1, tw - 1));
                    uint8_t* mrow = tile_mask + (size_t)ty * tw;
                    for (int tx = tx0; tx <= tx1; ++tx) mrow[tx] = 1;
                    any = true;
                }
            }
            x = re;
        }
    }
    return any;
}

// Dilate a byte mask by `r` tiles in both axes (in-place via scratch).
inline void dilate_tile_mask(uint8_t* mask, uint8_t* scratch, int tw, int th, int r) {
    // Horizontal dilation
    for (int ty = 0; ty < th; ++ty) {
        const uint8_t* in = mask + (size_t)ty * tw;
        uint8_t* out = scratch + (size_t)ty * tw;
        for (int tx = 0; tx < tw; ++tx) {
            uint8_t v = 0;
            for (int k = -r; k <= r && !v; ++k) {
                int xx = tx + k;
                if (xx >= 0 && xx < tw && in[xx]) v = 1;
            }
            out[tx] = v;
        }
    }
    // Vertical dilation
    for (int ty = 0; ty < th; ++ty) {
        uint8_t* out = mask + (size_t)ty * tw;
        for (int tx = 0; tx < tw; ++tx) {
            uint8_t v = 0;
            for (int k = -r; k <= r && !v; ++k) {
                int yy = ty + k;
                if (yy >= 0 && yy < th && scratch[(size_t)yy * tw + tx]) v = 1;
            }
            out[tx] = v;
        }
    }
}

// SWAR (SIMD Within A Register) 8-neighbor probe.
// Packs the 8 chain-code neighbors of cp into one uint64 in chain order
//   [W, NW, N, NE, E, SE, S, SW]    (matches offsets index 0..7 after rotation)
// then uses Mycroft's "has-zero-byte" trick to find the first non-background
// neighbor in one bitwise operation. ARM/x86 portable, no intrinsics.
//
// Returns the chain-code direction (0..7) of the first non-BG neighbor when
// scanning from `si` clockwise, OR -1 when no neighbor exists.
// Outputs *out_byte = the neighbor's pixel value (for VISITED detection).
inline int probe8_scalar(const uint8_t* cp, int step, int si, uint8_t* out_byte) {
    const int offsets[16] = {
        -1, -step-1, -step, -step+1, 1, step+1, step, step-1,
        -1, -step-1, -step, -step+1, 1, step+1, step, step-1
    };
    for (int i = 0; i < 8; ++i) {
        int idx = si + i;
        uint8_t v = *(cp + offsets[idx]);
        if (v != 0) {
            *out_byte = v;
            return idx & 7;
        }
    }
    return -1;
}

inline int probe8_swar(const uint8_t* cp, int step, int si, uint8_t* out_byte) {
    // Read 3 rows, 3 cols each. Reads are safe because border was zeroed.
    const uint8_t* r0 = cp - step;
    const uint8_t* r1 = cp;
    const uint8_t* r2 = cp + step;
    // Pack in chain order, low byte first:
    //   slot 0 = W   (r1[-1])
    //   slot 1 = NW  (r0[-1])
    //   slot 2 = N   (r0[ 0])
    //   slot 3 = NE  (r0[+1])
    //   slot 4 = E   (r1[+1])
    //   slot 5 = SE  (r2[+1])
    //   slot 6 = S   (r2[ 0])
    //   slot 7 = SW  (r2[-1])
    // ^ matches `offsets[16]` indices 0..7.
    uint64_t p = (uint64_t)r1[-1]
               | ((uint64_t)r0[-1] <<  8)
               | ((uint64_t)r0[ 0] << 16)
               | ((uint64_t)r0[+1] << 24)
               | ((uint64_t)r1[+1] << 32)
               | ((uint64_t)r2[+1] << 40)
               | ((uint64_t)r2[ 0] << 48)
               | ((uint64_t)r2[-1] << 56);

    // Rotate right so that slot `si` becomes the new slot 0.
    int shift = (si & 7) * 8;
    uint64_t pr = (shift == 0) ? p : ((p >> shift) | (p << (64 - shift)));

    // Mycroft has-zero: yields 0x80 in every byte that is zero, 0 otherwise.
    const uint64_t kLow  = 0x0101010101010101ULL;
    const uint64_t kHigh = 0x8080808080808080ULL;
    uint64_t bg = (pr - kLow) & ~pr & kHigh;     // 0x80 wherever byte == 0
    uint64_t nonbg = kHigh & ~bg;                // 0x80 wherever byte != 0

    if (nonbg == 0) return -1;
#if defined(_MSC_VER) && !defined(__clang__)
    unsigned long bit;
    _BitScanForward64(&bit, nonbg);
    int byte_index = (int)bit / 8;
#else
    int byte_index = __builtin_ctzll(nonbg) / 8;
#endif
    int d = (si + byte_index) & 7;
    // Extract the actual pixel value at that slot (for VISITED accounting).
    *out_byte = (uint8_t)((pr >> (byte_index * 8)) & 0xFF);
    return d;
}

// Streaming variant: invokes `on_contour(const std::vector<cv::Point>&)` for
// each contour as soon as it's found. The supplied `buffer` is reused for
// every contour, avoiding per-contour heap allocations. `on_contour` may
// inspect the buffer but must not retain a reference past return.
template <typename F>
inline void trace_runlen_streaming(
    cv::Mat& binary, size_t minSize, float maxRevisited, int MIN_RUN,
    std::vector<cv::Point>& buffer, F&& on_contour,
    cv::Rect scan_roi = cv::Rect())
{
    CV_Assert(binary.type() == CV_8UC1);
    const int rows = binary.rows;
    const int cols = binary.cols;
    const int step = (int)binary.step;
    uint8_t* data = binary.ptr<uint8_t>();
    const int offsets[16] = {
        -1, -step-1, -step, -step+1, 1, step+1, step, step-1,
        -1, -step-1, -step, -step+1, 1, step+1, step, step-1
    };
    const int dx[8] = { -1, -1,  0,  1, 1, 1, 0, -1 };
    const int dy[8] = {  0, -1, -1, -1, 0, 1, 1,  1 };
    const uint8_t FOREGROUND = 255, BACKGROUND = 0, VISITED = 100;
    const bool strong_seed_scan = (MIN_RUN < 0);
    if (MIN_RUN < 0) MIN_RUN = -MIN_RUN;
    cv::Rect sr = scan_roi.area() > 0 ? (scan_roi & cv::Rect(1, 1, cols - 2, rows - 2))
                                      : cv::Rect(1, 1, cols - 2, rows - 2);
    if (sr.empty()) return;
    const int r_end = sr.y + sr.height;
    const int c_end = sr.x + sr.width;
    for (int r = sr.y; r < r_end; ++r) {
        uint8_t* row_ptr = data + r * step;
        int c = sr.x;
        while (c < c_end) {
            c = strong_seed_scan ? scan_first_strong(row_ptr, c, c_end) : scan_first_fg(row_ptr, c, c_end);
            if (c >= c_end) break;
            int rs = c;
            int re = (row_ptr[c] == FOREGROUND) ? scan_end_fg(row_ptr, c, c_end) : (c + 1);
            int rl = re - rs;
            if (row_ptr[c] == FOREGROUND && rl >= MIN_RUN) {
                buffer.clear();
                int cx=c, cy=r, si=1;
                uint8_t* cp=row_ptr+c, *sp=cp;
                size_t nv=0;
                size_t nstrong=0;
                do {
                    buffer.emplace_back(cx, cy);
                    if (*cp == FOREGROUND) ++nstrong;
                    *cp = VISITED;
                    uint8_t nb_val;
#if KAKUTAG_USE_SWAR_PROBE
                    int d = probe8_swar(cp, step, si, &nb_val);
#else
                    int d = probe8_scalar(cp, step, si, &nb_val);
#endif
                    if (d < 0) break;
                    cp += offsets[d];
                    cx += dx[d];
                    cy += dy[d];
                    nv += (size_t)(nb_val == VISITED);
                    si = (d + 5) & 7;
                } while (cp != sp);
                size_t bs = buffer.size();
                if (nv <= float(bs)*maxRevisited && bs >= minSize && nstrong * 5 >= bs * 3) on_contour(buffer);
            } else if (row_ptr[c] == FOREGROUND) {
                for (int x = rs; x < re; ++x) row_ptr[x] = VISITED;
            }
            c = re;
            c = scan_skip_nonzero(row_ptr, c, c_end);
        }
    }
}


// Gated streaming variant: same as trace_runlen_streaming but skips Suzuki85
// seed search in tiles rejected by `tile_mask`. tile_mask is a tw*th byte array
// where mask[ty*tw + tx] = 1 means the (tx,ty) 8x8 tile is allowed.
// Passing tile_mask == nullptr disables gating (equivalent to the plain variant).
template <typename F>
inline void trace_runlen_streaming_gated(
    cv::Mat& binary, size_t minSize, float maxRevisited, int MIN_RUN,
    std::vector<cv::Point>& buffer,
    const uint8_t* tile_mask, int tw, int th,
    F&& on_contour, cv::Rect scan_roi = cv::Rect())
{
    CV_Assert(binary.type() == CV_8UC1);
    const int rows = binary.rows;
    const int cols = binary.cols;
    const int step = (int)binary.step;
    uint8_t* data = binary.ptr<uint8_t>();
    const int offsets[16] = {
        -1, -step-1, -step, -step+1, 1, step+1, step, step-1,
        -1, -step-1, -step, -step+1, 1, step+1, step, step-1
    };
    const int dx[8] = { -1, -1,  0,  1, 1, 1, 0, -1 };
    const int dy[8] = {  0, -1, -1, -1, 0, 1, 1,  1 };
    const uint8_t FOREGROUND = 255, BACKGROUND = 0, VISITED = 100;
    const bool strong_seed_scan = (MIN_RUN < 0);
    if (MIN_RUN < 0) MIN_RUN = -MIN_RUN;
    cv::Rect sr = scan_roi.area() > 0 ? (scan_roi & cv::Rect(1, 1, cols - 2, rows - 2))
                                      : cv::Rect(1, 1, cols - 2, rows - 2);
    if (sr.empty()) return;
    const int r_end = sr.y + sr.height;
    const int c_roi_end = sr.x + sr.width;
    for (int r = sr.y; r < r_end; ++r) {
        uint8_t* row_ptr = data + r * step;
        const int ty = r / 8;
        const uint8_t* mrow =
            (tile_mask && ty < th) ? (tile_mask + (size_t)ty * tw) : nullptr;
        int c = sr.x;
        while (c < c_roi_end) {
            // Skip past non-surviving tiles in O(tiles) at the top of the loop.
            if (mrow) {
                int tx = c / 8;
                const int tx_end = std::min(tw, (c_roi_end + 7) / 8);
                while (tx < tx_end && !mrow[tx]) ++tx;
                int new_c = tx * 8;
                if (new_c >= c_roi_end) break;
                if (new_c > c) c = new_c;
            }
            c = strong_seed_scan ? scan_first_strong(row_ptr, c, c_roi_end) : scan_first_fg(row_ptr, c, c_roi_end);
            if (c >= c_roi_end) break;
            int rs = c;
            int re = (row_ptr[c] == FOREGROUND) ? scan_end_fg(row_ptr, c, c_roi_end) : (c + 1);
            int rl = re - rs;
            if (row_ptr[c] == FOREGROUND && rl >= MIN_RUN) {
                buffer.clear();
                int cx=c, cy=r, si=1;
                uint8_t* cp=row_ptr+c, *sp=cp;
                size_t nv=0;
                size_t nstrong=0;
                do {
                    buffer.emplace_back(cx, cy);
                    if (*cp == FOREGROUND) ++nstrong;
                    *cp = VISITED;
                    uint8_t nb_val;
#if KAKUTAG_USE_SWAR_PROBE
                    int d = probe8_swar(cp, step, si, &nb_val);
#else
                    int d = probe8_scalar(cp, step, si, &nb_val);
#endif
                    if (d < 0) break;
                    cp += offsets[d];
                    cx += dx[d];
                    cy += dy[d];
                    nv += (size_t)(nb_val == VISITED);
                    si = (d + 5) & 7;
                } while (cp != sp);
                size_t bs = buffer.size();
                if (nv <= float(bs)*maxRevisited && bs >= minSize && nstrong * 5 >= bs * 3) on_contour(buffer);
            } else if (row_ptr[c] == FOREGROUND) {
                for (int x = rs; x < re; ++x) row_ptr[x] = VISITED;
            }
            c = re;
            c = scan_skip_nonzero(row_ptr, c, c_roi_end);
        }
    }
}

inline std::vector<std::vector<cv::Point>> trace_runlen(
    cv::Mat& binary, size_t minSize, float maxRevisited, int MIN_RUN)
{
    CV_Assert(binary.type() == CV_8UC1);
    const int rows = binary.rows;
    const int cols = binary.cols;
    const int step = (int)binary.step;
    uint8_t* data = binary.ptr<uint8_t>();
    const int offsets[16] = {
        -1, -step-1, -step, -step+1, 1, step+1, step, step-1,
        -1, -step-1, -step, -step+1, 1, step+1, step, step-1
    };
    const int dx[8] = { -1, -1,  0,  1, 1, 1, 0, -1 };
    const int dy[8] = {  0, -1, -1, -1, 0, 1, 1,  1 };
    std::vector<std::vector<cv::Point>> contours; contours.reserve(1024);
    std::vector<cv::Point> buffer; buffer.reserve(2048);
    const uint8_t FOREGROUND = 255, BACKGROUND = 0, VISITED = 100;
    for (int r = 1; r < rows - 1; ++r) {
        uint8_t* row_ptr = data + r * step;
        for (int c = 1; c < cols - 1;) {
            for (; c < cols && !row_ptr[c]; ++c) ;
            if (c == cols) break;
            int rs = c, re = c;
            while (re < cols && row_ptr[re] == FOREGROUND) re++;
            int rl = re - rs;
            if (row_ptr[c] == FOREGROUND && rl >= MIN_RUN) {
                buffer.clear();
                int cx=c, cy=r, si=1;
                uint8_t* cp=row_ptr+c, *sp=cp;
                size_t nv=0;
                size_t nstrong=0;
                do {
                    buffer.emplace_back(cx, cy);
                    if (*cp == FOREGROUND) ++nstrong;
                    *cp = VISITED;
                    bool st=false;
                    for (int i=0;i<8;i++) {
                        int idx=si+i;
                        uint8_t* nb=cp+offsets[idx];
                        if (*nb!=BACKGROUND) { cp=nb; int d=idx&7; cx+=dx[d]; cy+=dy[d]; nv+=int(*nb==VISITED); si=(d+5)&7; st=true; break; }
                    }
                    if (!st) break;
                } while (cp != sp);
                size_t bs = buffer.size();
                if (nv <= float(bs)*maxRevisited && bs >= minSize) contours.push_back(buffer);
            } else if (row_ptr[c] == FOREGROUND) {
                for (int x = rs; x < re; ++x) row_ptr[x] = VISITED;
            }
            c = re;
            while (c < cols && row_ptr[c]) ++c;
        }
    }
    return contours;
}

// 4-corner sort (top-left -> CCW). Canonical TL, TR, BR, BL ordering.
inline void sort_corners(cv::Point2f c[4]) {
    cv::Point2f cm(0, 0);
    for (int i = 0; i < 4; ++i) cm += c[i];
    cm *= 0.25f;
    std::sort(c, c + 4, [&](const cv::Point2f& a, const cv::Point2f& b) {
        return std::atan2(a.y - cm.y, a.x - cm.x) < std::atan2(b.y - cm.y, b.x - cm.x);
    });
}

// Preserve approxPolyDP contour order and only normalize orientation, matching
// canonical candidate ordering. This is more stable than centroid-angle
// sorting for tiny, nearly degenerate or perspective-skewed quads.
inline void orient_corners_canonical(cv::Point2f c[4]) {
    double dx1 = c[1].x - c[0].x;
    double dy1 = c[1].y - c[0].y;
    double dx2 = c[2].x - c[0].x;
    double dy2 = c[2].y - c[0].y;
    double o = dx1 * dy2 - dy1 * dx2;
    if (o < 0.0) std::swap(c[1], c[3]);
}

// Geometric dedup: same id markers, drop inner-of or strong-overlap pairs.
inline void dedup_markers(std::vector<Marker>& m) {
    if (m.size() <= 1) return;
    std::sort(m.begin(), m.end(), [](const Marker& a, const Marker& b) {
        if (a.id != b.id) return a.id < b.id;
        auto area = [](const Marker& mm) {
            double s = 0.0;
            for (int i = 0; i < 4; ++i) {
                const auto& p = mm.corners[i];
                const auto& q = mm.corners[(i + 1) & 3];
                s += p.x * q.y - p.y * q.x;
            }
            return std::abs(s) * 0.5;
        };
        return area(a) > area(b);
    });
    std::vector<bool> rem(m.size(), false);
    for (size_t i = 0; i < m.size(); ++i) {
        if (rem[i]) continue;
        for (size_t j = i + 1; j < m.size(); ++j) {
            if (rem[j] || m[i].id != m[j].id) break;
            auto bbox = [](const Marker& mm) {
                float minx=mm.corners[0].x,maxx=mm.corners[0].x;
                float miny=mm.corners[0].y,maxy=mm.corners[0].y;
                for(int k=1;k<4;k++){
                    minx=std::min(minx,mm.corners[k].x); maxx=std::max(maxx,mm.corners[k].x);
                    miny=std::min(miny,mm.corners[k].y); maxy=std::max(maxy,mm.corners[k].y);
                }
                return cv::Rect2f(minx,miny,maxx-minx,maxy-miny);
            };
            auto bi = bbox(m[i]); auto bj = bbox(m[j]);
            float inter = (bi & bj).area();
            float uni = bi.area() + bj.area() - inter;
            if (uni > 0 && inter / uni > 0.3f) rem[j] = true;
        }
    }
    std::vector<Marker> out; out.reserve(m.size());
    for (size_t i = 0; i < m.size(); ++i) if (!rem[i]) out.push_back(m[i]);
    m.swap(out);
}

// Internal helper: bilinear sample with safe clamp.
inline float sample_bilinear_safe(const cv::Mat& img, float fx, float fy) {
    const int W = img.cols, H = img.rows;
    if (fx < 0) fx = 0;
    if (fy < 0) fy = 0;
    if (fx > (float)(W - 1)) fx = (float)(W - 1);
    if (fy > (float)(H - 1)) fy = (float)(H - 1);
    int xi = (int)fx, yi = (int)fy;
    if (xi >= W - 1) xi = W - 2;
    if (yi >= H - 1) yi = H - 2;
    if (xi < 0) xi = 0;
    if (yi < 0) yi = 0;
    float ax = fx - xi, ay = fy - yi;
    const uint8_t* p0 = img.ptr<uint8_t>(yi);
    const uint8_t* p1 = img.ptr<uint8_t>(yi + 1);
    float v00 = p0[xi], v01 = p0[xi+1];
    float v10 = p1[xi], v11 = p1[xi+1];
    return (1-ay)*((1-ax)*v00 + ax*v01) + ay*((1-ax)*v10 + ax*v11);
}

// Hot decode path helper. Candidate quads normally lie fully inside the image
// because they come from image contours. When all four corners have enough
// margin for bilinear xi+1/yi+1 access, skip per-sample clamp branches.
inline bool quad_has_unchecked_bilinear_margin(const cv::Point2f c[4], int W, int H) {
    if (W < 2 || H < 2) return false;
    const float max_x = (float)(W - 2);
    const float max_y = (float)(H - 2);
    for (int k = 0; k < 4; ++k) {
        if (c[k].x < 0.0f || c[k].y < 0.0f || c[k].x > max_x || c[k].y > max_y) {
            return false;
        }
    }
    return true;
}

inline float sample_bilinear_unchecked(const cv::Mat& img, float fx, float fy) {
    const int xi = (int)fx;
    const int yi = (int)fy;
    const float ax = fx - (float)xi;
    const float ay = fy - (float)yi;
    const uint8_t* p0 = img.ptr<uint8_t>(yi);
    const uint8_t* p1 = img.ptr<uint8_t>(yi + 1);
    const float v00 = p0[xi],     v01 = p0[xi + 1];
    const float v10 = p1[xi],     v11 = p1[xi + 1];
    return (1.0f - ay) * ((1.0f - ax) * v00 + ax * v01) +
           ay          * ((1.0f - ax) * v10 + ax * v11);
}

inline float sample_bilinear_fast(const cv::Mat& img, float fx, float fy, bool unchecked) {
    return unchecked ? sample_bilinear_unchecked(img, fx, fy)
                     : sample_bilinear_safe(img, fx, fy);
}

inline bool homography_unit_square_to_quad(const cv::Point2f c[4], double h[9]) {
    const double x0 = c[0].x, y0 = c[0].y;
    const double x1 = c[1].x, y1 = c[1].y;
    const double x2 = c[2].x, y2 = c[2].y;
    const double x3 = c[3].x, y3 = c[3].y;
    const double sx = x0 - x1 + x2 - x3;
    const double sy = y0 - y1 + y2 - y3;
    if (std::abs(sx) < 1e-9 && std::abs(sy) < 1e-9) {
        h[0] = x1 - x0; h[1] = x3 - x0; h[2] = x0;
        h[3] = y1 - y0; h[4] = y3 - y0; h[5] = y0;
        h[6] = 0.0;     h[7] = 0.0;     h[8] = 1.0;
        return true;
    }
    const double dx1 = x1 - x2, dx2 = x3 - x2;
    const double dy1 = y1 - y2, dy2 = y3 - y2;
    const double den = dx1 * dy2 - dx2 * dy1;
    if (std::abs(den) < 1e-9) return false;
    const double g = (sx * dy2 - dx2 * sy) / den;
    const double q = (dx1 * sy - sx * dy1) / den;
    h[0] = x1 - x0 + g * x1;
    h[1] = x3 - x0 + q * x3;
    h[2] = x0;
    h[3] = y1 - y0 + g * y1;
    h[4] = y3 - y0 + q * y3;
    h[5] = y0;
    h[6] = g;
    h[7] = q;
    h[8] = 1.0;
    return true;
}

inline int popcount8_u(unsigned v) {
#if defined(_MSC_VER) && !defined(__clang__)
    return (int)__popcnt(v & 0xFFu);
#else
    return __builtin_popcount(v & 0xFFu);
#endif
}

inline int popcount64_u(uint64_t v) {
#if defined(_MSC_VER) && !defined(__clang__)
#if defined(_M_X64) || defined(_M_ARM64)
    return (int)__popcnt64(v);
#else
    return (int)__popcnt((unsigned)v) + (int)__popcnt((unsigned)(v >> 32));
#endif
#else
    return __builtin_popcountll((unsigned long long)v);
#endif
}

struct PackedEntry {
    int id = -1;
    int rot = 0;
    int dict_index = -1;
};

struct PackedDict {
    int dict_index = -1;
    int marker_size = 0;
    int nbits = 0;
    int nbytes = 0;
    std::unordered_map<uint64_t, PackedEntry> exact64;
    std::unordered_set<uint64_t> ambiguous64;
    std::vector<uint64_t> codes64;
    std::vector<PackedEntry> entries;
};

inline size_t dict_row_bytes(const cv::aruco::Dictionary& dict) {
    if (dict.bytesList.empty() || dict.bytesList.depth() != CV_8U) return 0;
    return (size_t)dict.bytesList.cols * dict.bytesList.elemSize();
}

inline bool dict_has_bytes_list_layout(const cv::aruco::Dictionary& dict,
                                       int nbytes) {
    return !dict.bytesList.empty() && dict.bytesList.rows > 0 && nbytes > 0 &&
           dict_row_bytes(dict) >= (size_t)4 * (size_t)nbytes;
}

inline uint64_t packed_bytes_to_u64(const uint8_t* code, int nbytes) {
    uint64_t v = 0;
    for (int j = 0; j < nbytes; ++j) {
        v = (v << 8) | (uint64_t)code[j];
    }
    return v;
}

inline PackedDict build_packed_dict(const cv::aruco::Dictionary& dict,
                                    int dict_index) {
    PackedDict packed;
    packed.dict_index = dict_index;
    packed.marker_size = dict.markerSize;
    packed.nbits = dict.markerSize * dict.markerSize;
    packed.nbytes = (packed.nbits + 7) / 8;
    if (packed.nbits <= 0 || packed.nbits > 64 ||
        !dict_has_bytes_list_layout(dict, packed.nbytes)) {
        return packed;
    }
    const size_t entries = (size_t)dict.bytesList.rows * 4u;
    packed.codes64.reserve(entries);
    packed.entries.reserve(entries);
    packed.exact64.reserve(entries * 2u);
    packed.ambiguous64.reserve(8u);
    for (int id = 0; id < dict.bytesList.rows; ++id) {
        const uint8_t* row = dict.bytesList.ptr<uint8_t>(id);
        for (int rot = 0; rot < 4; ++rot) {
            const uint8_t* code = row + rot * packed.nbytes;
            const uint64_t v = packed_bytes_to_u64(code, packed.nbytes);
            PackedEntry e;
            e.id = id;
            e.rot = rot;
            e.dict_index = dict_index;
            packed.codes64.push_back(v);
            packed.entries.push_back(e);
            auto inserted = packed.exact64.emplace(v, e);
            if (!inserted.second) {
                const PackedEntry& old = inserted.first->second;
                if (old.id != e.id || old.dict_index != e.dict_index) {
                    packed.ambiguous64.insert(v);
                }
            }
        }
    }
    return packed;
}

inline bool packed_code_is_ambiguous64(const PackedDict* packed, uint64_t v) {
    return packed && !packed->ambiguous64.empty() &&
           packed->ambiguous64.find(v) != packed->ambiguous64.end();
}

inline bool identify_packed_exact64(const PackedDict* packed,
                                    const uint8_t* code, int nbytes,
                                    int& out_id, int& out_rot) {
    if (!packed || !code || packed->nbytes != nbytes || packed->nbits <= 0 ||
        packed->nbits > 64 || packed->exact64.empty()) {
        return false;
    }
    const uint64_t v = packed_bytes_to_u64(code, nbytes);
    auto it = packed->exact64.find(v);
    if (it == packed->exact64.end()) return false;
    if (packed_code_is_ambiguous64(packed, v)) return false;
    out_id = it->second.id;
    out_rot = it->second.rot;
    return true;
}

inline bool packed_exact64_available(const PackedDict* packed, int nbytes) {
    return packed && packed->nbytes == nbytes && packed->nbits > 0 &&
           packed->nbits <= 64 && !packed->exact64.empty();
}

inline bool packed_hamming64_available(const PackedDict* packed, int nbytes) {
    return packed && packed->nbytes == nbytes && packed->nbits > 0 &&
           packed->nbits <= 64 && !packed->codes64.empty() &&
           packed->codes64.size() == packed->entries.size();
}

inline bool identify_packed_hamming64(const PackedDict* packed,
                                      const uint8_t* code, int nbytes,
                                      int max_hamming,
                                      int& out_id, int& out_rot,
                                      int* out_min_dist = nullptr) {
    if (!packed_hamming64_available(packed, nbytes) || !code) {
        if (out_min_dist) *out_min_dist = 9999;
        return false;
    }
    const uint64_t v = packed_bytes_to_u64(code, nbytes);
    const int limit = std::max(0, max_hamming);
    int best = limit + 1;
    size_t best_index = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < packed->codes64.size(); ++i) {
        const int dist = popcount64_u(v ^ packed->codes64[i]);
        if (dist < best) {
            best = dist;
            best_index = i;
            if (best == 0) {
                if (packed_code_is_ambiguous64(packed, v)) {
                    if (out_min_dist) *out_min_dist = 0;
                    return false;
                }
                const PackedEntry& e = packed->entries[i];
                if (out_min_dist) *out_min_dist = 0;
                out_id = e.id;
                out_rot = e.rot;
                return true;
            }
        }
    }
    if (out_min_dist) *out_min_dist = best;
    if (best_index != std::numeric_limits<size_t>::max() && best <= limit) {
        const PackedEntry& e = packed->entries[best_index];
        out_id = e.id;
        out_rot = e.rot;
        return true;
    }
    return false;
}

inline bool identify_packed_hamming(const cv::aruco::Dictionary& dict,
                                    const uint8_t* code, int nbytes,
                                    int max_hamming,
                                    int& out_id, int& out_rot,
                                    int* out_min_dist = nullptr) {
    if (!dict_has_bytes_list_layout(dict, nbytes) || !code) {
        if (out_min_dist) *out_min_dist = 9999;
        return false;
    }
    const int limit = std::max(0, max_hamming);
    int best = limit + 1;
    int best_id = -1;
    int best_rot = 0;
    for (int id = 0; id < dict.bytesList.rows; ++id) {
        const uint8_t* row = dict.bytesList.ptr<uint8_t>(id);
        for (int rot = 0; rot < 4; ++rot) {
            int dist = 0;
            for (int j = 0; j < nbytes; ++j) {
                const uint8_t ref = row[rot * nbytes + j];
                dist += popcount8_u((unsigned)(code[j] ^ ref));
                if (dist >= best) break;
            }
            if (dist < best) {
                best = dist;
                best_id = id;
                best_rot = rot;
                if (best == 0) {
                    if (out_min_dist) *out_min_dist = 0;
                    out_id = best_id;
                    out_rot = best_rot;
                    return true;
                }
            }
        }
    }
    if (out_min_dist) *out_min_dist = best;
    if (best_id >= 0 && best <= limit) {
        out_id = best_id;
        out_rot = best_rot;
        return true;
    }
    return false;
}

inline int opencv_packed_bit_position(int bit_index, int nbits) {
    const int nbytes = (nbits + 7) / 8;
    const int byte_index = bit_index >> 3;
    const int bit_in_byte = bit_index & 7;
    if (byte_index == nbytes - 1) {
        const int remaining = nbits - byte_index * 8;
        return remaining - 1 - bit_in_byte;
    }
    return 7 - bit_in_byte;
}

inline void pack_marker_bits_rowmajor(const cv::Mat& bits01,
                                      uint8_t* code, int nbytes) {
    if (!code || nbytes <= 0) return;
    std::memset(code, 0, (size_t)nbytes);
    const int M = bits01.rows;
    const int nbits = M * M;
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < M; ++c) {
            if (bits01.at<uint8_t>(r, c)) {
                const int b = r * M + c;
                code[b >> 3] |= (uint8_t)(1u << opencv_packed_bit_position(b, nbits));
            }
        }
    }
}

inline int attempts_for_candidate(float side_px, int max_attempts) {
    if (side_px >= 80.f) return std::min(max_attempts, 1);
    if (side_px >= 48.f) return std::min(max_attempts, 3);
    return std::min(max_attempts, 5);
}

inline cv::Point2f fixed_jitter_for_attempt(int attempt_index) {
    static const cv::Point2f kJitters[] = {
        { 0.0f,  0.0f},
        { 0.5f,  0.0f},
        {-0.5f,  0.0f},
        { 0.0f,  0.5f},
        { 0.0f, -0.5f},
        { 0.4f,  0.4f},
        {-0.4f, -0.4f},
    };
    const int n = (int)(sizeof(kJitters) / sizeof(kJitters[0]));
    return kJitters[std::max(0, attempt_index) % n];
}

// Sample the marker grid using a 3x3 homography, including the 1-cell border.
// Output: (M+2)x(M+2) uint8_t binarized via Otsu (or adaptive on 3rd attempt).
inline bool sample_and_decode_cv(
    const cv::Mat& full_bw,
    const cv::Point2f img_corners[4],
    const cv::aruco::Dictionary& dict,
    int attempt_index,
    int& out_id, int& out_rot,
    int border_bits = 1,
    double error_correction_rate = 0.0,
    double max_erroneous_bits_in_border_rate = 0.0,
    int max_hamming = 0,
    const PackedDict* packed_dict = nullptr)
{
    (void)attempt_index;
    const int M = dict.markerSize;
    if (M <= 0 || border_bits <= 0) return false;
    const int gs = M + 2 * border_bits;
    if (gs <= 0 || gs > 64) return false;

    // High-recall reference sampler.  The fast path samples one value per cell;
    // that is intentionally cheap, but it is brittle under cast shadows, blur,
    // and small corner bias.  This fallback mirrors OpenCV's ArUco extraction
    // strategy more closely: warp the whole candidate to a canonical image,
    // threshold there, and decide each bit by majority vote inside the cell.
    // It is slower, so callers reach this path only on robust attempts, small
    // candidates, or when ECC/high-recall parameters are enabled.
    constexpr int kCellSize = 4;
    constexpr double kCellMarginRate = 0.13;
    constexpr double kMinStdDevOtsu = 5.0;
    const int canonical_side = gs * kCellSize;
    if (canonical_side <= 0) return false;

    const cv::Point2f dst[4] = {
        {0.0f, 0.0f},
        {(float)canonical_side - 1.0f, 0.0f},
        {(float)canonical_side - 1.0f, (float)canonical_side - 1.0f},
        {0.0f, (float)canonical_side - 1.0f}
    };

    cv::Mat H;
    try {
        H = cv::getPerspectiveTransform(img_corners, dst);
    } catch (...) {
        return false;
    }
    if (H.empty()) return false;

    cv::Mat canonical;
    try {
        cv::warpPerspective(full_bw, canonical, H,
                            cv::Size(canonical_side, canonical_side),
                            cv::INTER_NEAREST);
    } catch (...) {
        return false;
    }
    if (canonical.empty() || canonical.type() != CV_8UC1) return false;

    cv::Mat binary;
    const int inner_margin = std::max(0, kCellSize / 2);
    cv::Rect inner_rect(inner_margin, inner_margin,
                        canonical.cols - 2 * inner_margin,
                        canonical.rows - 2 * inner_margin);
    inner_rect &= cv::Rect(0, 0, canonical.cols, canonical.rows);
    if (inner_rect.empty()) return false;

    cv::Scalar mean, stddev;
    cv::meanStdDev(canonical(inner_rect), mean, stddev);
    if (stddev[0] < kMinStdDevOtsu) {
        binary = cv::Mat(canonical.size(), CV_8UC1,
                         cv::Scalar(mean[0] > 127.0 ? 255 : 0));
    } else {
        cv::threshold(canonical, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    }

    cv::Mat bits(gs, gs, CV_8UC1, cv::Scalar(0));
    const int cell_margin = std::max(0, (int)std::floor(kCellMarginRate * (double)kCellSize));
    const int vote_side = std::max(1, kCellSize - 2 * cell_margin);
    for (int y = 0; y < gs; ++y) {
        uint8_t* brow = bits.ptr<uint8_t>(y);
        for (int x = 0; x < gs; ++x) {
            const int x0 = x * kCellSize + cell_margin;
            const int y0 = y * kCellSize + cell_margin;
            cv::Rect cell(x0, y0, vote_side, vote_side);
            cell &= cv::Rect(0, 0, binary.cols, binary.rows);
            if (cell.empty()) return false;
            const int white = cv::countNonZero(binary(cell));
            if (white > (int)cell.area() / 2) brow[x] = 1;
        }
    }

    int border_errors = 0, border_total = 0;
    for (int br = 0; br < border_bits; ++br) {
        for (int x = 0; x < gs; ++x) {
            border_total += 2;
            if (bits.ptr<uint8_t>(br)[x] != 0) ++border_errors;
            if (bits.ptr<uint8_t>(gs - 1 - br)[x] != 0) ++border_errors;
        }
        for (int y = br + 1; y < gs - 1 - br; ++y) {
            border_total += 2;
            if (bits.ptr<uint8_t>(y)[br] != 0) ++border_errors;
            if (bits.ptr<uint8_t>(y)[gs - 1 - br] != 0) ++border_errors;
        }
    }
    const float border_error_rate = border_total > 0
        ? (float)border_errors / (float)border_total : 0.0f;
    if (border_total > 0 && border_error_rate > (float)max_erroneous_bits_in_border_rate) {
        return false;
    }

    cv::Mat inner = bits.rowRange(border_bits, gs - border_bits)
                        .colRange(border_bits, gs - border_bits);
    const int nbits = M * M;
    const int nbytes = (nbits + 7) / 8;
    std::vector<uint8_t> code;
    const bool has_packed_exact = packed_exact64_available(packed_dict, nbytes);
    const bool has_packed_hamming = packed_hamming64_available(packed_dict, nbytes);
    if (has_packed_exact || max_hamming > 0) {
        code.resize((size_t)nbytes);
        pack_marker_bits_rowmajor(inner, code.data(), nbytes);
    }
    if (max_hamming == 0 && has_packed_exact) {
        if (identify_packed_exact64(packed_dict, code.data(), nbytes, out_id, out_rot)) {
            return true;
        }
        if (error_correction_rate <= 0.0) return false;
    }
    if (max_hamming > 0) {
        bool matched = false;
        if (has_packed_hamming) {
            matched = identify_packed_hamming64(
                packed_dict, code.data(), nbytes, max_hamming, out_id, out_rot);
        } else {
            matched = identify_packed_hamming(
                dict, code.data(), nbytes, max_hamming, out_id, out_rot);
        }
        if (matched || error_correction_rate <= 0.0) return matched;
    }
    if (!dict_has_bytes_list_layout(dict, nbytes)) return false;
    return dict.identify(inner, out_id, out_rot, error_correction_rate);
}

inline uint8_t otsu_threshold_u8_small(const uint8_t* v, int n) {
    int hist[256] = {};
    int sum = 0;
    uint8_t min_v = 255, max_v = 0;
    for (int i = 0; i < n; ++i) {
        const uint8_t x = v[i];
        ++hist[x];
        sum += x;
        if (x < min_v) min_v = x;
        if (x > max_v) max_v = x;
    }
    // Preserve the previous Otsu behavior for uniform samples: threshold 0.
    // This keeps all-white quads rejected by the black-border check.
    if (min_v == max_v) return 0;
    int sumB = 0, wB = 0;
    double best = -1.0;
    int best_t = min_v;
    for (int t = (int)min_v; t <= (int)max_v; ++t) {
        wB += hist[t];
        if (wB == 0) continue;
        const int wF = n - wB;
        if (wF == 0) break;
        sumB += t * hist[t];
        const double mB = (double)sumB / (double)wB;
        const double mF = (double)(sum - sumB) / (double)wF;
        const double d = mB - mF;
        const double between = (double)wB * (double)wF * d * d;
        if (between > best) { best = between; best_t = t; }
    }
    return (uint8_t)best_t;
}

// Fast path for the common exact-dictionary case: exact DICT_* matching.
// It removes two tiny cv::Mat allocations, cv::threshold/adaptiveThreshold,
// and Dictionary::identify's byte-list construction. If anything is outside
// this narrow contract or the packed exact match fails, we fall back to the
// OpenCV-compatible reference path below, so recall is preserved.
inline bool sample_and_decode(
    const cv::Mat& full_bw,
    const cv::Point2f img_corners[4],
    const cv::aruco::Dictionary& dict,
    int attempt_index,
    int& out_id, int& out_rot,
    int border_bits = 1,
    double error_correction_rate = 0.0,
    double max_erroneous_bits_in_border_rate = 0.0,
    int max_hamming = 0,
    const PackedDict* packed_dict = nullptr)
{
    const int M = dict.markerSize;
    const int gs = M + 2 * border_bits;
    if (attempt_index == 2 || M <= 0 || border_bits <= 0 || gs <= 0 || gs > 9) {
        return sample_and_decode_cv(full_bw, img_corners, dict, attempt_index, out_id, out_rot,
            border_bits, error_correction_rate, max_erroneous_bits_in_border_rate,
            max_hamming, packed_dict);
    }

    std::array<uint8_t, 81> vals{};
    double hh[9];
    if (!homography_unit_square_to_quad(img_corners, hh)) return false;
    const double* h = hh;

    const int total = gs * gs;
    const bool unchecked_sample =
        quad_has_unchecked_bilinear_margin(img_corners, full_bw.cols, full_bw.rows);
    const double inv_gs = 1.0 / (double)gs;
    const double u0 = 0.5 * inv_gs;
    int pos = 0;
    for (int rr = 0; rr < gs; ++rr) {
        const double v = ((double)rr + 0.5) * inv_gs;
        double xn = h[0] * u0 + h[1] * v + h[2];
        double yn = h[3] * u0 + h[4] * v + h[5];
        double dn = h[6] * u0 + h[7] * v + h[8];
        for (int cc = 0; cc < gs; ++cc) {
            if (std::abs(dn) < 1e-6) return false;
            const float fx = (float)(xn / dn);
            const float fy = (float)(yn / dn);
            vals[(size_t)pos++] =
                (uint8_t)(sample_bilinear_fast(full_bw, fx, fy, unchecked_sample) + 0.5f);
            xn += h[0] * inv_gs;
            yn += h[3] * inv_gs;
            dn += h[6] * inv_gs;
        }
    }

    const uint8_t thr = otsu_threshold_u8_small(vals.data(), total);
    int border_black = 0, border_total = 0;
    auto is_white = [&](int r, int c) -> bool { return vals[(size_t)r * gs + c] > thr; };
    for (int br = 0; br < border_bits; ++br) {
        for (int x = 0; x < gs; ++x) {
            border_total += 2;
            if (!is_white(br, x)) ++border_black;
            if (!is_white(gs - 1 - br, x)) ++border_black;
        }
        for (int y = br + 1; y < gs - 1 - br; ++y) {
            border_total += 2;
            if (!is_white(y, br)) ++border_black;
            if (!is_white(y, gs - 1 - br)) ++border_black;
        }
    }
    const float border_white_rate =
        border_total > 0 ? 1.0f - (float)border_black / (float)border_total : 0.0f;
    if (border_total > 0 &&
        (float)border_white_rate > (float)max_erroneous_bits_in_border_rate) return false;

    const int nbits = M * M;
    const int nbytes = (nbits + 7) / 8;
    uint8_t code[16] = {};
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < M; ++c) {
            if (is_white(r + border_bits, c + border_bits)) {
                int b = r * M + c;
                code[b >> 3] |= (uint8_t)(1u << opencv_packed_bit_position(b, nbits));
            }
        }
    }
    // Exact decode. With ECC disabled, a packed exact64 miss is
    // definitive and must not fall through to a linear dictionary scan.
    if (max_hamming == 0) {
        if (packed_exact64_available(packed_dict, nbytes)) {
            if (identify_packed_exact64(packed_dict, code, nbytes, out_id, out_rot)) {
                return true;
            }
            if (error_correction_rate <= 0.0) return false;
        } else {
            if (identify_packed_hamming(dict, code, nbytes, 0, out_id, out_rot)) return true;
            if (error_correction_rate <= 0.0) return false;
        }
    } else {
        bool matched = false;
        if (packed_hamming64_available(packed_dict, nbytes)) {
            matched = identify_packed_hamming64(
                packed_dict, code, nbytes, max_hamming, out_id, out_rot);
        } else {
            matched = identify_packed_hamming(
                dict, code, nbytes, max_hamming, out_id, out_rot);
        }
        if (matched) return true;
    }
    // ECC fallback: only when caller explicitly enabled error correction.
    if (error_correction_rate > 0.0) {
        return sample_and_decode_cv(full_bw, img_corners, dict, /*attempt=*/2, out_id, out_rot,
            border_bits, error_correction_rate, max_erroneous_bits_in_border_rate,
            max_hamming, packed_dict);
    }
    return false;
}


struct QuadCand { cv::Point2f c[4]; };

// Persistent per-detector scratch. Holding these buffers in the detector
// removes per-frame container construction in collect/decode for real-scene
// workloads where the number of false quads can dominate.
struct RoiScratch {
    ThresholdScratch            threshold;
    std::vector<cv::Point>      contour_buf;
    std::vector<cv::Point>      approx_fallback;
    std::vector<QuadCand>       candidates;
    std::vector<uint8_t>        candidate_alive;
    std::vector<uint8_t>        tile_mask;
    std::vector<uint8_t>        tile_scratch;
    std::vector<cv::Point2f>    corner_buf;
    std::vector<Marker>         detected;
    std::vector<std::vector<cv::Point2f>> rejected_quads;
    int                         last_candidate_count = 0;

    RoiScratch() {
        contour_buf.reserve(2048);
        approx_fallback.reserve(16);
        candidates.reserve(128);
        candidate_alive.reserve(128);
        tile_mask.reserve(4096);
        tile_scratch.reserve(4096);
        corner_buf.reserve(64);
        detected.reserve(64);
        rejected_quads.reserve(128);
    }
};

inline float quad_area2_abs(const cv::Point2f c[4]) {
    float s = 0.0f;
    for (int i = 0; i < 4; ++i) {
        const auto& a = c[i];
        const auto& b = c[(i + 1) & 3];
        s += a.x * b.y - a.y * b.x;
    }
    return std::abs(s);
}

inline bool quad_is_convex4(const cv::Point2f c[4]) {
    float prev = 0.0f;
    for (int i = 0; i < 4; ++i) {
        const auto& a = c[i];
        const auto& b = c[(i + 1) & 3];
        const auto& d = c[(i + 2) & 3];
        float ux = b.x - a.x, uy = b.y - a.y;
        float vx = d.x - b.x, vy = d.y - b.y;
        float cr = ux * vy - uy * vx;
        if (std::abs(cr) < 1e-3f) return false;
        if (prev != 0.0f && cr * prev < 0.0f) return false;
        prev = cr;
    }
    return true;
}

inline float contour_area2_abs_i(const std::vector<cv::Point>& contour) {
    int64_t s = 0;
    const size_t n = contour.size();
    for (size_t i = 0; i < n; ++i) {
        const auto& a = contour[i];
        const auto& b = contour[(i + 1) % n];
        s += (int64_t)a.x * b.y - (int64_t)a.y * b.x;
    }
    return (float)std::llabs(s);
}

// ArUco-specific 4-corner extraction for Suzuki85 external contours.
// Generic approxPolyDP solves an arbitrary-polyline simplification problem.
// Here the target is always a black square marker outer boundary, so the four
// diagonal-coordinate extrema are the right primitive: one O(n) pass, no heap,
// no recursive DP stack, and deterministic exactly four corners.
inline bool fast_quad_from_contour_extrema(const std::vector<cv::Point>& contour,
                                           QuadCand& q) {
    if (contour.size() < 8) return false;
    int i_min_sum = 0, i_max_sum = 0, i_min_diff = 0, i_max_diff = 0;
    int min_sum = contour[0].x + contour[0].y;
    int max_sum = min_sum;
    int min_diff = contour[0].x - contour[0].y;
    int max_diff = min_diff;
    for (int i = 1, n = (int)contour.size(); i < n; ++i) {
        const int sum = contour[i].x + contour[i].y;
        const int diff = contour[i].x - contour[i].y;
        if (sum < min_sum) { min_sum = sum; i_min_sum = i; }
        if (sum > max_sum) { max_sum = sum; i_max_sum = i; }
        if (diff < min_diff) { min_diff = diff; i_min_diff = i; }
        if (diff > max_diff) { max_diff = diff; i_max_diff = i; }
    }
    int idx[4] = { i_min_sum, i_max_diff, i_max_sum, i_min_diff };
    for (int a = 0; a < 4; ++a) {
        for (int b = a + 1; b < 4; ++b) {
            const auto& pa = contour[idx[a]];
            const auto& pb = contour[idx[b]];
            int dx = pa.x - pb.x, dy = pa.y - pb.y;
            if (dx * dx + dy * dy < 9) return false;
        }
    }
    // The extrema order is already TL/TR/BR/BL under the image coordinate
    // convention used by sort_corners(), so avoid four atan2 calls per contour.
    for (int k = 0; k < 4; ++k) {
        q.c[k] = cv::Point2f((float)contour[idx[k]].x, (float)contour[idx[k]].y);
    }
    if (!quad_is_convex4(q.c)) return false;

    if (quad_area2_abs(q.c) < 32.0f) return false;
    return true;
}

inline void zero_binary_border(cv::Mat& thr) {
    const int w = thr.cols;
    const int h = thr.rows;
    if (w <= 0 || h <= 0) return;
    for (int xx = 0; xx < w; ++xx) {
        thr.ptr<uint8_t>(0)[xx] = 0;
        thr.ptr<uint8_t>(h - 1)[xx] = 0;
    }
    for (int yy = 0; yy < h; ++yy) {
        thr.ptr<uint8_t>(yy)[0] = 0;
        thr.ptr<uint8_t>(yy)[w - 1] = 0;
    }
}

inline void collect_quad_candidates_from_binary(
    cv::Mat& thr, const DetectorOptions& opt, std::vector<QuadCand>& candidates,
    RoiScratch* scratch = nullptr, cv::Rect scan_roi = cv::Rect(),
    const uint8_t* fused_run_tile_mask = nullptr, int fused_tw = 0, int fused_th = 0)
{
    // Stream contours straight into the candidate list to avoid the
    // intermediate std::vector<std::vector<Point>> staging buffer. This keeps
    // each contour in L1 cache between Suzuki85 trace and the ArUco-specific
    // quad extractor. Reuse scratch buffers when called from ArucoDetector.
    candidates.clear();
    std::vector<cv::Point> local_contour_buf;
    std::vector<cv::Point> local_approx_fallback;
    if (!scratch) {
        local_contour_buf.reserve(2048);
        local_approx_fallback.reserve(16);
    }
    std::vector<cv::Point>& contour_buf = scratch ? scratch->contour_buf : local_contour_buf;
    std::vector<cv::Point>& approx_fallback = scratch ? scratch->approx_fallback : local_approx_fallback;
    const int minSizeSq = opt.min_size * opt.min_size;
    const int w = thr.cols, h = thr.rows;
    // ternary mode lowers the noise of contour seeds, so we can drop the
    // 80-pixel safety floor that the binary mode needs to avoid noise contours.
    const bool ternary_mode_for_size = (opt.threshold_offset_low >= 0 && opt.threshold_offset_low < opt.threshold_offset);
    const size_t minContourSize = std::max<size_t>(ternary_mode_for_size ? 40 : 80,
                                                   (size_t)std::max(1, opt.min_size) * 4u);
    // A fully permissive seed scan starts contours from any foreground pixel. KakuTag's MIN_RUN=6 is a
    // major speed win for medium/large markers, but it can skip tiny or tilted
    // marker borders. Use low seed strictness only in small-marker mode.
    // Ternary hysteresis mode: STRONG pixels are already cleaned of noise by
    // the high threshold, so a seed of even 1 STRONG pixel is safe and recovers
    // thin/tilted marker borders that lose a long horizontal STRONG run.
    const bool ternary_mode = (opt.threshold_offset_low >= 0 && opt.threshold_offset_low < opt.threshold_offset);
    // STRONG-only seed with MIN_RUN=2 kills 1-px noise speckles (the dominant
    // false-seed mass at HD) without losing thin/tilted marker borders, which
    // always have at least 2 contiguous STRONG pixels somewhere on their top or
    // bottom edge after adaptive thresholding.
    // Keep the high-recall, fast low seed strictness; ternary mode
    // doesn't need MIN_RUN=1 because dual recovery is still on for the few
    // markers whose STRONG segment is shorter than 6 px (very tilted edges).
    // In ternary mode, use STRONG-only seed scanning (negative MIN_RUN) so
    // WEAK pixels cannot hide a later STRONG seed in the same non-zero run.
    // A 2-pixel STRONG run rejects 1px speckles but recovers thin/tilted
    // small-marker borders that do not contain a 6px horizontal STRONG run.
    const int seedMinRun = ternary_mode ? -2
                           : ((opt.min_size <= 20) ? KAKUTAG_SMALL_MARKER_SEED_MIN_RUN : 6);

#if KAKUTAG_USE_RUN_TILE_GATE
    int tw = fused_tw > 0 ? fused_tw : (w + 7) / 8;
    int th = fused_th > 0 ? fused_th : (h + 7) / 8;
    std::vector<uint8_t> local_tile_mask;
    const uint8_t* run_mask = fused_run_tile_mask;
    if (!run_mask) {
        // Fallback for legacy callers that did not fuse mask generation into
        // threshold emission. ArucoDetector hot path passes a precomputed mask.
        std::vector<uint8_t>& tile_mask = scratch ? scratch->tile_mask : local_tile_mask;
        tile_mask.resize((size_t)tw * th);
        bool any_seed_tile = tile_horizontal_run_seed_gate(thr.ptr<uint8_t>(), w, h, (int)thr.step,
                                                           6, tile_mask.data(), tw, th, scan_roi);
        if (!any_seed_tile) return;
        run_mask = tile_mask.data();
    }
    trace_runlen_streaming_gated(thr, minContourSize, 0.05f, seedMinRun, contour_buf,
        run_mask, tw, th,
        [&](const std::vector<cv::Point>& contour) {
#elif KAKUTAG_USE_TILE_GATE
    int tw = w / 8;
    int th = h / 8;
    std::vector<uint8_t> local_tile_byte_mask;
    std::vector<uint8_t> local_tile_scratch;
    std::vector<uint8_t>& tile_byte_mask = scratch ? scratch->tile_mask : local_tile_byte_mask;
    std::vector<uint8_t>& tile_scratch = scratch ? scratch->tile_scratch : local_tile_scratch;
    tile_byte_mask.assign((size_t)tw * th, 0);
    tile_scratch.assign((size_t)tw * th, 0);
    tile_popcount_gate(thr.ptr<uint8_t>(), w, h, (int)thr.step,
                       KAKUTAG_TILE_GATE_LO, KAKUTAG_TILE_GATE_HI,
                       tile_byte_mask.data(), tw, th);
    dilate_tile_mask(tile_byte_mask.data(), tile_scratch.data(), tw, th, 2);
    trace_runlen_streaming_gated(thr, minContourSize, 0.05f, seedMinRun, contour_buf,
        tile_byte_mask.data(), tw, th,
        [&](const std::vector<cv::Point>& contour) {
#else
    trace_runlen_streaming(thr, minContourSize, 0.05f, seedMinRun, contour_buf,
        [&](const std::vector<cv::Point>& contour) {
#endif
            if (contour.size() < 8) return;
            QuadCand q;
            bool ok_quad = false;
            // Small-marker robustness: the extrema-based quad extractor is very
            // fast for medium/large ArUco borders, but on 28-50 px markers a
            // 1-2 px corner bias is enough to break exact dictionary decoding.
            // For small contours, use approxPolyDP first; for larger
            // contours keep the extrema fast path and only fall back to DP if
            // geometry rejects it.
            constexpr size_t kSmallContourDpFirst = 260;
            if (contour.size() <= kSmallContourDpFirst) {
                cv::approxPolyDP(contour, approx_fallback, double(contour.size()) * 0.03, true);
                if (approx_fallback.size() == 4 && cv::isContourConvex(approx_fallback)) {
                    for (int j = 0; j < 4; ++j)
                        q.c[j] = cv::Point2f((float)approx_fallback[j].x, (float)approx_fallback[j].y);
                    orient_corners_canonical(q.c);
                    ok_quad = true;
                }
            }
            if (!ok_quad) ok_quad = fast_quad_from_contour_extrema(contour, q);
            if (!ok_quad) {
                // Safety net for non-ideal real contours: the fast path is the
                // normal ArUco-square case, but if its strict extrema geometry
                // rejects a contour, fall back to OpenCV's generic DP so recall
                // is not sacrificed for speed.
                cv::approxPolyDP(contour, approx_fallback, double(contour.size()) * 0.03, true);
                if (approx_fallback.size() != 4 || !cv::isContourConvex(approx_fallback)) return;
                for (int j = 0; j < 4; ++j)
                    q.c[j] = cv::Point2f((float)approx_fallback[j].x, (float)approx_fallback[j].y);
                orient_corners_canonical(q.c);
            }
            for (int j = 0; j < 4; ++j) {
                int k = (j + 1) & 3;
                float ddx = q.c[j].x - q.c[k].x;
                float ddy = q.c[j].y - q.c[k].y;
                if (ddx*ddx + ddy*ddy < (float)minSizeSq) return;
            }
            candidates.push_back(q);
        }, scan_roi);
}

// Core: append decoded markers into `out`. Existing entries in `out` are
// preserved so callers can keep a running list across ROIs without per-call
// vector copies. `dedup_markers` is applied only to the *new* tail block.
inline void decode_quad_candidates_into(
    const cv::Mat& full_bw, cv::Rect roi,
    std::vector<QuadCand>& candidates, const DetectorOptions& opt,
    std::vector<Marker>& out, RoiScratch* scratch = nullptr)
{
    std::vector<uint8_t> local_alive;
    std::vector<cv::Point2f> local_corners;
    std::vector<uint8_t>& alive = scratch ? scratch->candidate_alive : local_alive;
    std::vector<cv::Point2f>& corners = scratch ? scratch->corner_buf : local_corners;

    const size_t out_start = out.size();
    alive.assign(candidates.size(), (uint8_t)1);
    auto refine_new_markers = [&]() {
        if (!opt.refine_corners || out.size() <= out_start) return;
        corners.clear();
        const size_t new_n = out.size() - out_start;
        corners.reserve(new_n * 4);
        for (size_t i = out_start; i < out.size(); ++i)
            corners.insert(corners.end(), out[i].corners, out[i].corners + 4);
        // Use a small cornerSubPix window for stable corner precision.
        // 5x5 over a 28-50 px marker can include the adjacent black border and
        // bias the corner toward the bit grid edge, raising corner error.
        int hw = std::min(opt.refine_window, 4);
        cv::cornerSubPix(full_bw, corners, cv::Size(hw, hw), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS, 12, 0.005));
        size_t k = 0;
        for (size_t i = out_start; i < out.size(); ++i)
            for (int c = 0; c < 4; ++c) out[i].corners[c] = corners[k++];
    };

    if (!opt.need_ids) {
        for (size_t ci = 0; ci < candidates.size(); ++ci) {
            const QuadCand& q = candidates[ci];
            Marker m;
            m.id = -1;
            m.rotation = 0;
            m.dict_index = -1;
            m.dict = -1;
            for (int k = 0; k < 4; ++k) {
                m.corners[k].x = q.c[k].x + (float)roi.x;
                m.corners[k].y = q.c[k].y + (float)roi.y;
            }
            out.push_back(m);
            alive[ci] = 0;
        }
        if (out.size() > out_start + 1) {
            std::vector<Marker> tail(out.begin() + out_start, out.end());
            dedup_markers(tail);
            out.resize(out_start);
            out.insert(out.end(), tail.begin(), tail.end());
        }
        refine_new_markers();
        return;
    }
    int dict_index = -1;
    for (auto& dict : opt.dicts) {
        ++dict_index;
        const PackedDict* packed_dict =
            (opt.packed_dicts && (size_t)dict_index < opt.packed_dict_count)
                ? &opt.packed_dicts[dict_index] : nullptr;
        for (size_t ci = 0; ci < candidates.size(); ++ci) {
            if (!alive[ci]) continue;
            const QuadCand& q = candidates[ci];
            float q_side_sum = 0.f;
            for (int sk = 0; sk < 4; ++sk) {
                const cv::Point2f d = q.c[sk] - q.c[(sk + 1) & 3];
                q_side_sum += std::sqrt(d.x * d.x + d.y * d.y);
            }
            const float q_avg_side = q_side_sum * 0.25f;
            const bool small_cv_decode_fallback =
                q_avg_side <= (float)std::max(24, opt.min_size * 8);
            int id = -1, rot = 0;
            bool found = false;
            const int attempts = attempts_for_candidate(q_avg_side, opt.max_attempts);
            for (int ai = 0; ai < attempts && !found; ++ai) {
                const cv::Point2f jitter = fixed_jitter_for_attempt(ai);
                cv::Point2f q2[4];
                for (int k = 0; k < 4; ++k) {
                    q2[k].x = q.c[k].x + (float)roi.x + jitter.x;
                    q2[k].y = q.c[k].y + (float)roi.y + jitter.y;
                }
                if (sample_and_decode(full_bw, q2, dict, ai, id, rot,
                                      opt.marker_border_bits,
                                      opt.error_correction_rate,
                                      opt.max_erroneous_bits_in_border_rate,
                                      opt.max_hamming,
                                      packed_dict)) {
                    found = true;
                } else if (ai == 0 && small_cv_decode_fallback) {
                    // Tiny markers are hypersensitive to the fast packed sampler:
                    // a valid 28-50 px quad can fail exact packed decoding while
                    // the OpenCV-style Mat path decodes it correctly. Pay the
                    // slower cv path only for small quads and only once on the
                    // original, non-jittered corners. This restores stable
                    // small-marker recall without touching FP-heavy large scenes.
                    if (sample_and_decode_cv(full_bw, q2, dict, ai, id, rot,
                                             opt.marker_border_bits,
                                             opt.error_correction_rate,
                                             opt.max_erroneous_bits_in_border_rate,
                                             opt.max_hamming,
                                             packed_dict)) {
                        found = true;
                    }
                }
            }
            if (found) {
                Marker m;
                m.id = id; m.rotation = rot; m.dict_index = dict_index; m.dict = dict_index;
                cv::Point2f rot_c[4];
                for (int k = 0; k < 4; ++k) rot_c[k] = q.c[(k + 4 - rot) & 3];
                for (int k = 0; k < 4; ++k) {
                    m.corners[k].x = rot_c[k].x + (float)roi.x;
                    m.corners[k].y = rot_c[k].y + (float)roi.y;
                }
                out.push_back(m);
                alive[ci] = 0; // no vector erase: stable order, O(1) removal mark.
            }
        }
    }
    if (scratch && opt.collect_rejected) {
        for (size_t ci = 0; ci < candidates.size(); ++ci) {
            if (!alive[ci]) continue;
            const QuadCand& q = candidates[ci];
            std::vector<cv::Point2f> rej(4);
            for (int k = 0; k < 4; ++k) {
                rej[k].x = q.c[k].x + (float)roi.x;
                rej[k].y = q.c[k].y + (float)roi.y;
            }
            scratch->rejected_quads.push_back(std::move(rej));
        }
    }
    // Dedup only the markers we just produced. Existing entries are kept
    // untouched; callers that merge multiple ROIs may decide to run a global
    // dedup later.
    if (out.size() > out_start + 1) {
        std::vector<Marker> tail(out.begin() + out_start, out.end());
        dedup_markers(tail);
        out.resize(out_start);
        out.insert(out.end(), tail.begin(), tail.end());
    }
    refine_new_markers();
}

// Backwards-compatible wrapper that returns by value.
inline std::vector<Marker> decode_quad_candidates(
    const cv::Mat& full_bw, cv::Rect roi,
    std::vector<QuadCand>& candidates, const DetectorOptions& opt,
    RoiScratch* scratch = nullptr)
{
    std::vector<Marker> det;
    decode_quad_candidates_into(full_bw, roi, candidates, opt, det, scratch);
    return det;
}

// Single ROI detect, into-buffer variant. Appends to `out` and returns
// nothing. Existing markers in `out` are preserved. This is the per-frame
// hot path used by ArucoDetector to avoid the final std::vector copy.
inline void detect_on_roi_into(
    const cv::Mat& full_bw, cv::Rect roi, cv::Mat& thr_scratch,
    const DetectorOptions& opt, std::vector<Marker>& out,
    RoiScratch* scratch = nullptr)
{
    roi &= cv::Rect(0, 0, full_bw.cols, full_bw.rows);
    if (roi.width < 32 || roi.height < 32) return;
    cv::Mat sub = full_bw(roi);
    int w = sub.cols, h = sub.rows;
#if KAKUTAG_USE_RUN_TILE_GATE
    const int run_tw = (w + 7) / 8;
    const int run_th = (h + 7) / 8;
    std::vector<uint8_t> local_run_tile_mask;
    std::vector<uint8_t>& run_tile_mask = scratch ? scratch->tile_mask : local_run_tile_mask;
    run_tile_mask.assign((size_t)run_tw * run_th, (uint8_t)0);
    RunTileGate run_gate{ run_tile_mask.data(), run_tw, run_th, 6, false };
#endif
#if KAKUTAG_USE_THRESHOLD_BOUNDS
    BinaryBounds threshold_bounds;
    streaming_threshold_to_buffer(sub.ptr<uint8_t>(), w, h, (int)sub.step,
                                  opt.threshold_offset,
                                  thr_scratch.ptr<uint8_t>(), (int)thr_scratch.step,
                                  scratch ? &scratch->threshold : nullptr,
                                  &threshold_bounds
#if KAKUTAG_USE_RUN_TILE_GATE
                                  , &run_gate
#endif
                                  );
#else
    if (opt.threshold_offset_low >= 0 && opt.threshold_offset_low < opt.threshold_offset) {
        // Ternary hysteresis: single pass encodes STRONG (=255) and WEAK (=64).
        streaming_threshold_ternary_to_buffer(sub.ptr<uint8_t>(), w, h, (int)sub.step,
                                              opt.threshold_offset, opt.threshold_offset_low,
                                              thr_scratch.ptr<uint8_t>(), (int)thr_scratch.step,
                                              scratch ? &scratch->threshold : nullptr);
    } else {
        streaming_threshold_to_buffer(sub.ptr<uint8_t>(), w, h, (int)sub.step,
                                      opt.threshold_offset,
                                      thr_scratch.ptr<uint8_t>(), (int)thr_scratch.step,
                                      scratch ? &scratch->threshold : nullptr,
                                      nullptr
#if KAKUTAG_USE_RUN_TILE_GATE
                                      , &run_gate
#endif
                                      );
    }
#endif
#if KAKUTAG_USE_RUN_TILE_GATE
    if (!run_gate.any) return;
#endif
    cv::Mat thr = thr_scratch(cv::Rect(0, 0, w, h));
    // Guard borders so visited-aware contour tracing cannot escape image bounds.
    zero_binary_border(thr);

    std::vector<QuadCand> local_candidates;
    std::vector<QuadCand>& candidates = scratch ? scratch->candidates : local_candidates;
#if KAKUTAG_USE_THRESHOLD_BOUNDS
    cv::Rect scan_roi = threshold_bounds.rect() & cv::Rect(1, 1, w - 2, h - 2);
    if (scan_roi.empty()) return;
    collect_quad_candidates_from_binary(thr, opt, candidates, scratch, scan_roi
#if KAKUTAG_USE_RUN_TILE_GATE
                                        , run_tile_mask.data(), run_tw, run_th
#endif
                                        );
#else
    collect_quad_candidates_from_binary(thr, opt, candidates, scratch, cv::Rect()
#if KAKUTAG_USE_RUN_TILE_GATE
                                        , run_tile_mask.data(), run_tw, run_th
#endif
                                        );
#endif
    if (scratch) scratch->last_candidate_count = (int)candidates.size();
    decode_quad_candidates_into(full_bw, roi, candidates, opt, out, scratch);
}

// Backwards-compatible: ROI is in image coordinates of `full_bw`.
inline std::vector<Marker> detect_on_roi(
    const cv::Mat& full_bw, cv::Rect roi, cv::Mat& thr_scratch,
    const DetectorOptions& opt, RoiScratch* scratch = nullptr)
{
    std::vector<Marker> det;
    detect_on_roi_into(full_bw, roi, thr_scratch, opt, det, scratch);
    return det;
}

// Pyramid detector, v2 -- into-buffer variant. See halfscale wrapper below.
inline void detect_halfscale_full_decode_into(
    const cv::Mat& gray, const DetectorOptions& opt, cv::Mat& thr_small_buf,
    std::vector<Marker>& out, RoiScratch* scratch = nullptr)
{
    const int sw = std::max(1, gray.cols / 2);
    const int sh = std::max(1, gray.rows / 2);
    if (sw < 32 || sh < 32) return;
    if (thr_small_buf.empty() || thr_small_buf.cols < sw || thr_small_buf.rows < sh)
        thr_small_buf.create(sh, sw, CV_8UC1);

#if KAKUTAG_USE_RUN_TILE_GATE
    const int run_tw = (sw + 7) / 8;
    const int run_th = (sh + 7) / 8;
    std::vector<uint8_t> local_run_tile_mask;
    std::vector<uint8_t>& run_tile_mask = scratch ? scratch->tile_mask : local_run_tile_mask;
    run_tile_mask.assign((size_t)run_tw * run_th, (uint8_t)0);
    RunTileGate run_gate{ run_tile_mask.data(), run_tw, run_th, 6, false };
#endif
#if KAKUTAG_USE_THRESHOLD_BOUNDS
    BinaryBounds threshold_bounds;
    if (opt.threshold_offset_low >= 0 && opt.threshold_offset_low < opt.threshold_offset) {
        streaming_threshold_down2_ternary_to_buffer(gray.ptr<uint8_t>(), gray.cols, gray.rows,
                                                    (int)gray.step, opt.threshold_offset,
                                                    opt.threshold_offset_low,
                                                    thr_small_buf.ptr<uint8_t>(),
                                                    (int)thr_small_buf.step,
                                                    scratch ? &scratch->threshold : nullptr,
                                                    &threshold_bounds
#if KAKUTAG_USE_RUN_TILE_GATE
                                                    , &run_gate
#endif
                                                    );
    } else {
        streaming_threshold_down2_to_buffer(gray.ptr<uint8_t>(), gray.cols, gray.rows,
                                            (int)gray.step, opt.threshold_offset,
                                            thr_small_buf.ptr<uint8_t>(),
                                            (int)thr_small_buf.step,
                                            scratch ? &scratch->threshold : nullptr,
                                            &threshold_bounds
#if KAKUTAG_USE_RUN_TILE_GATE
                                            , &run_gate
#endif
                                            );
    }
#else
    if (opt.threshold_offset_low >= 0 && opt.threshold_offset_low < opt.threshold_offset) {
        streaming_threshold_down2_ternary_to_buffer(gray.ptr<uint8_t>(), gray.cols, gray.rows,
                                                    (int)gray.step, opt.threshold_offset,
                                                    opt.threshold_offset_low,
                                                    thr_small_buf.ptr<uint8_t>(),
                                                    (int)thr_small_buf.step,
                                                    scratch ? &scratch->threshold : nullptr,
                                                    nullptr
#if KAKUTAG_USE_RUN_TILE_GATE
                                                    , &run_gate
#endif
                                                    );
    } else {
        streaming_threshold_down2_to_buffer(gray.ptr<uint8_t>(), gray.cols, gray.rows,
                                            (int)gray.step, opt.threshold_offset,
                                            thr_small_buf.ptr<uint8_t>(),
                                            (int)thr_small_buf.step,
                                            scratch ? &scratch->threshold : nullptr,
                                            nullptr
#if KAKUTAG_USE_RUN_TILE_GATE
                                            , &run_gate
#endif
                                            );
    }
#endif
#if KAKUTAG_USE_RUN_TILE_GATE
    if (!run_gate.any) return;
#endif
    cv::Mat thr = thr_small_buf(cv::Rect(0, 0, sw, sh));
    zero_binary_border(thr);

    DetectorOptions half_opt = opt;
    half_opt.refine_corners = false;
    half_opt.min_size = std::max(20, opt.min_size / 2);
    std::vector<QuadCand> local_candidates;
    std::vector<QuadCand>& candidates = scratch ? scratch->candidates : local_candidates;
#if KAKUTAG_USE_THRESHOLD_BOUNDS
    cv::Rect scan_roi = threshold_bounds.rect() & cv::Rect(1, 1, sw - 2, sh - 2);
    if (scan_roi.empty()) return;
    collect_quad_candidates_from_binary(thr, half_opt, candidates, scratch, scan_roi
#if KAKUTAG_USE_RUN_TILE_GATE
                                        , run_tile_mask.data(), run_tw, run_th
#endif
                                        );
#else
    collect_quad_candidates_from_binary(thr, half_opt, candidates, scratch, cv::Rect()
#if KAKUTAG_USE_RUN_TILE_GATE
                                        , run_tile_mask.data(), run_tw, run_th
#endif
                                        );
#endif

    if (scratch) scratch->last_candidate_count = (int)candidates.size();
    for (auto& q : candidates) {
        for (int c = 0; c < 4; ++c) {
            q.c[c].x *= 2.0f;
            q.c[c].y *= 2.0f;
        }
    }

    DetectorOptions decode_opt = opt;
    decode_opt.refine_corners = false;  // full-image refinement happens once below.
    const size_t out_start = out.size();
    decode_quad_candidates_into(gray, cv::Rect(0, 0, gray.cols, gray.rows),
                                candidates, decode_opt, out, scratch);

    if (opt.refine_corners && out.size() > out_start) {
        std::vector<cv::Point2f> local_corners;
        std::vector<cv::Point2f>& corners = scratch ? scratch->corner_buf : local_corners;
        corners.clear();
        const size_t new_n = out.size() - out_start;
        corners.reserve(new_n * 4);
        for (size_t i = out_start; i < out.size(); ++i)
            corners.insert(corners.end(), out[i].corners, out[i].corners + 4);
        int hw = opt.refine_window;
        cv::cornerSubPix(gray, corners, cv::Size(hw, hw), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::MAX_ITER|cv::TermCriteria::EPS, 12, 0.005));
        size_t k = 0;
        for (size_t i = out_start; i < out.size(); ++i)
            for (int c = 0; c < 4; ++c) out[i].corners[c] = corners[k++];
    }
}

// Backwards-compatible wrapper that returns by value.
inline std::vector<Marker> detect_halfscale_full_decode(
    const cv::Mat& gray, const DetectorOptions& opt, cv::Mat& thr_small_buf,
    RoiScratch* scratch = nullptr)
{
    std::vector<Marker> det;
    detect_halfscale_full_decode_into(gray, opt, thr_small_buf, det, scratch);
    return det;
}

inline cv::Rect marker_bbox(const Marker& m, float margin) {
    float minx = m.corners[0].x, maxx = minx, miny = m.corners[0].y, maxy = miny;
    for (int i = 1; i < 4; ++i) {
        minx = std::min(minx, m.corners[i].x); maxx = std::max(maxx, m.corners[i].x);
        miny = std::min(miny, m.corners[i].y); maxy = std::max(maxy, m.corners[i].y);
    }
    float cx = (minx + maxx) * 0.5f, cy = (miny + maxy) * 0.5f;
    float hw = (maxx - minx) * 0.5f * margin, hh = (maxy - miny) * 0.5f * margin;
    return cv::Rect((int)(cx - hw), (int)(cy - hh), (int)(2 * hw), (int)(2 * hh));
}

inline std::vector<cv::Rect> merge_rects(std::vector<cv::Rect> rects, int W, int H) {
    for (auto& r : rects) r &= cv::Rect(0, 0, W, H);
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 0; i < rects.size() && !changed; ++i) {
            for (size_t j = i + 1; j < rects.size() && !changed; ++j) {
                if ((rects[i] & rects[j]).area() > 0) {
                    rects[i] |= rects[j];
                    rects.erase(rects.begin() + j);
                    changed = true;
                }
            }
        }
    }
    return rects;
}

inline bool same_id_set(const std::vector<Marker>& a, const std::vector<Marker>& b) {
    if (a.size() != b.size()) return false;
    std::set<int> ai, bi;
    for (auto& m : a) ai.insert(m.id);
    for (auto& m : b) bi.insert(m.id);
    return ai == bi;
}

inline std::vector<Marker> detect_full_native(const cv::Mat& gray,
                                              const DetectorOptions& opt,
                                              cv::Mat& thr_scratch)
{
    return detect_on_roi(gray, cv::Rect(0, 0, gray.cols, gray.rows), thr_scratch, opt);
}

inline std::vector<Marker> detect_full_pyramid(const cv::Mat& gray,
                                               const DetectorOptions& opt,
                                               cv::Mat& small_buf,
                                               cv::Mat& thr_small_buf)
{
    (void)small_buf; // kept for ABI/source compatibility with older callers.
    return detect_halfscale_full_decode(gray, opt, thr_small_buf);
}

} // namespace detail

// =============================================================================
// Public API
// =============================================================================

// Selection of the internal processing path.
//   AutoSingle  -- stateless. Pyramid for >=VGA; falls back to native full
//                  automatically if pyramid finds nothing (safe_fallback).
//   Full        -- stateless. Always native full resolution (predictable
//                  latency; used by calibration / photogrammetry).
//   Pyramid     -- stateless. Always pyramid (no safety fallback).
enum class Mode {
    AutoSingle = 0,
    Full       = 1,
    Pyramid    = 2,
};

// Full detector parameter bundle, exposed through a single stateless
// ArucoDetector class.
struct DetectorParameters {
    // Backbone
    std::vector<cv::aruco::Dictionary> dicts = {
        cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250)
    };
    int   min_size          = 10;
    int   max_attempts      = 7;
    int   threshold_offset  = 5;
    // Hysteresis low offset (>=0 enables ternary single-pass candidate generation).
    // Default ON (2): one-pass strong/weak contour hysteresis. This improves
    // low-contrast/shadow recall without running a second threshold pass.
    // Set to -1 for the older strict single-threshold path.
    int   threshold_offset_low = 2;

    // What to compute (skip the rest for speed)
    bool  refine_corners    = true;
    int   refine_window     = 5;
    bool  need_ids          = true;       // false -> skip bit decode

    // Single-frame strategy
    Mode  mode              = Mode::AutoSingle;
    bool  safe_fallback     = true;       // AutoSingle: if pyramid is unsafe,
                                          //   re-run native full.
    bool  robust_small_markers = true; // If pyramid finds a small marker,
                                          //   full re-check catches other small IDs.
    int   small_marker_full_fallback_side = 72; // full-res marker side threshold
    bool  dual_threshold_small_markers = true; // ternary 1-pass + dual safety net
    int   small_marker_expected_min_count = 4; // second pass if tiny count is below this
    int   empty_pyramid_full_candidate_max = 24; // full fallback only if half-scale candidates are plausible
    bool  cv_contour_small_recovery = true; // ternary 1-pass + recovery safety net
    // ---- Decode/identification controls ----
    // Width of the marker external black border, in marker-bit units. The compatibility
    // default is 1. We accept arbitrary positive ints and pass-through to the
    // bit decoder; the bit grid size used for sampling is
    // (dict.markerSize + 2*markerBorderBits).
    int   marker_border_bits = 1;
    // Bit-error correction rate fed to dict.identify(). This defaults to
    // 0 (no ECC) because the OpenCV default 0.6 is FP-prone; KakuTag keeps that conservative default.
    double error_correction_rate = 0.0;
    int    max_hamming = 0;            // absolute bit errors accepted by packed matcher
    // Maximum acceptable rate of erroneous bits in the marker border (the
    // outer black ring). KakuTag keeps this much stricter than OpenCV's
    // 0.35 default, but no longer requires a perfectly black border: real
    // shadowed markers often contain a few damaged border cells.
    double max_erroneous_bits_in_border_rate = 0.10;
    // Detect markers printed in white on a black background by also running a
    // bit-inverted pass when the primary pass finds nothing.
    bool   detect_inverted_marker = false;
    // Populate the `rejected` output of detectMarkers() with quad candidates
    // that failed dictionary identification. Off by default for speed.
    bool   collect_rejected = false;
    // ---- DetectorParameters compatibility aliases ----
    // Sentinel values mean "not explicitly set; keep KakuTag optimized defaults".
    int    boxFilterSize = 15;          // fixed 15x15 box filter, accepted for source compatibility
    int    thres = -1;                  // if >=0, overrides threshold_offset
    int    minSize = -1;                // if >=0, overrides min_size
    int    maxAttemptsPerCandidate = -1;// if >=0, overrides max_attempts
    float  maxTimesRevisited = 0.05f;   // accepted; current tracer uses 0.05 internally
    float  markerBorderBits = -1.f;     // if >=0, overrides marker_border_bits
    double errorCorrectionRate = -1.0;  // if >=0, overrides error_correction_rate
    double maxErroneousBitsInBorderRate = -1.0; // if >=0, overrides max_erroneous_bits_in_border_rate
    bool   detectInvertedMarker = false;// ORed with detect_inverted_marker

};


// Balanced profile: keep the fast AutoSingle backbone, but enable one-pass
// contour hysteresis and a small border-error allowance. This is intended for
// real video with uneven illumination, mild blur, or low contrast without
// invoking OpenCV-style ECC on every candidate.
inline void apply_balanced_recall_profile(DetectorParameters& p) {
    p.mode = Mode::AutoSingle;
    p.threshold_offset = 5;
    p.threshold_offset_low = 2;
    p.max_erroneous_bits_in_border_rate = std::max(p.max_erroneous_bits_in_border_rate, 0.10);
    p.error_correction_rate = 0.0;
    p.max_hamming = std::max(p.max_hamming, 0);
    p.safe_fallback = true;
    p.robust_small_markers = true;
    p.dual_threshold_small_markers = true;
    p.cv_contour_small_recovery = true;
}

inline DetectorParameters make_balanced_recall_parameters(const cv::aruco::Dictionary& dict) {
    DetectorParameters p;
    p.dicts = {dict};
    apply_balanced_recall_profile(p);
    return p;
}

// High-recall profile for difficult illumination: cast shadows, low contrast,
// blur, and heavily slanted markers. It intentionally trades some latency and
// false-positive margin for recall. Validate precision on your own scenes.
inline void apply_high_recall_profile(DetectorParameters& p) {
    p.mode = Mode::Full;
    p.threshold_offset = 5;
    p.threshold_offset_low = 2;
    p.max_attempts = std::max(p.max_attempts, 7);
    p.refine_corners = true;
    p.safe_fallback = true;
    p.robust_small_markers = true;
    p.dual_threshold_small_markers = true;
    p.cv_contour_small_recovery = true;
    p.max_erroneous_bits_in_border_rate = std::max(p.max_erroneous_bits_in_border_rate, 0.30);
    p.max_hamming = std::max(p.max_hamming, 3);
    p.error_correction_rate = std::max(p.error_correction_rate, 0.6);
}

inline DetectorParameters make_high_recall_parameters(const cv::aruco::Dictionary& dict) {
    DetectorParameters p;
    p.dicts = {dict};
    apply_high_recall_profile(p);
    return p;
}

// ============================================================================
// Unified detector. Compatible in spirit with cv::aruco::ArucoDetector:
//   - One class.
//   - detectMarkers(image, corners, ids) signature mirrors OpenCV.
//   - Holds persistent scratch buffers (no per-frame allocation).
//   - Mode::AutoSingle preserves the stateless contract of ArucoDetector.
// ============================================================================
class ArucoDetector {
public:
    explicit ArucoDetector(DetectorParameters params = {})
        : params_(std::move(params)) {
        rebuild_packed_dicts();
        reset();
    }
    // compatibility constructors.
    ArucoDetector(const cv::aruco::Dictionary& dict,
                  const DetectorParameters& params = {})
        : params_(params) {
        params_.dicts = { dict };
        rebuild_packed_dicts();
        reset();
    }
    ArucoDetector(const std::vector<cv::aruco::Dictionary>& dicts,
                  const DetectorParameters& params = {})
        : params_(params) {
        params_.dicts = dicts;
        rebuild_packed_dicts();
        reset();
    }

    // OpenCV ArucoDetector-style: returns corners as vector<vector<Point2f>>
    // and ids as vector<int>. Rejected quads are collected when requested.
    void detectMarkers(
        cv::InputArray image,
        std::vector<std::vector<cv::Point2f>>& corners,
        std::vector<int>& ids,
        std::vector<std::vector<cv::Point2f>>* rejected = nullptr)
    {
        const std::vector<Marker>* m = nullptr;
        const bool old_collect = params_.collect_rejected;
        if (rejected) params_.collect_rejected = true;
        try {
            m = &detect(image);
            params_.collect_rejected = old_collect;
        } catch (...) {
            params_.collect_rejected = old_collect;
            throw;
        }
        corners.clear(); ids.clear();
        if (rejected) rejected->clear();
        corners.reserve(m->size()); ids.reserve(m->size());
        for (const auto& mk : *m) {
            corners.emplace_back(mk.corners, mk.corners + 4);
            ids.push_back(mk.id);
        }
        if (rejected) *rejected = last_rejected_;
    }

    // OpenCV-style multi-dictionary entry point.
    // Returns dictionary index per detection in `dictIndices`, matching
    // legacy detectMarkersMultiDict APIs.
    void detectMarkersMultiDict(
        cv::InputArray image,
        std::vector<std::vector<cv::Point2f>>& corners,
        std::vector<int>& ids,
        std::vector<std::vector<cv::Point2f>>* rejected = nullptr,
        std::vector<int>* dictIndices = nullptr)
    {
        const std::vector<Marker>* m = nullptr;
        const bool old_collect = params_.collect_rejected;
        if (rejected) params_.collect_rejected = true;
        try {
            m = &detect(image);
            params_.collect_rejected = old_collect;
        } catch (...) {
            params_.collect_rejected = old_collect;
            throw;
        }
        corners.clear(); ids.clear();
        if (rejected) rejected->clear();
        if (dictIndices) dictIndices->clear();
        corners.reserve(m->size()); ids.reserve(m->size());
        if (dictIndices) dictIndices->reserve(m->size());
        for (const auto& mk : *m) {
            corners.emplace_back(mk.corners, mk.corners + 4);
            ids.push_back(mk.id);
            if (dictIndices) dictIndices->push_back(mk.dict_index);
        }
        if (rejected) *rejected = last_rejected_;
    }

    // cv::OutputArrayOfArrays overload, compatible with
    // cv::aruco::ArucoDetector.
    void detectMarkers(
        cv::InputArray image,
        cv::OutputArrayOfArrays corners,
        cv::OutputArray ids,
        cv::OutputArrayOfArrays rejectedImgPoints = cv::noArray())
    {
        std::vector<std::vector<cv::Point2f>> cv_corners;
        std::vector<std::vector<cv::Point2f>> cv_rejected;
        std::vector<int> cv_ids;
        detectMarkers(image, cv_corners, cv_ids,
                      rejectedImgPoints.needed() ? &cv_rejected : nullptr);
        copyVector2Output(cv_corners, corners);
        ids.create((int)cv_ids.size(), 1, CV_32SC1);
        cv::Mat(cv_ids).copyTo(ids);
        if (rejectedImgPoints.needed()) copyVector2Output(cv_rejected, rejectedImgPoints);
    }

    void detectMarkersMultiDict(
        cv::InputArray image,
        cv::OutputArrayOfArrays corners,
        cv::OutputArray ids,
        cv::OutputArrayOfArrays rejectedImgPoints = cv::noArray(),
        cv::OutputArray dictIndices = cv::noArray())
    {
        std::vector<std::vector<cv::Point2f>> cv_corners;
        std::vector<std::vector<cv::Point2f>> cv_rejected;
        std::vector<int> cv_ids, cv_dicts;
        detectMarkersMultiDict(image, cv_corners, cv_ids,
            rejectedImgPoints.needed() ? &cv_rejected : nullptr,
            dictIndices.needed() ? &cv_dicts : nullptr);
        copyVector2Output(cv_corners, corners);
        ids.create((int)cv_ids.size(), 1, CV_32SC1);
        cv::Mat(cv_ids).copyTo(ids);
        if (rejectedImgPoints.needed()) copyVector2Output(cv_rejected, rejectedImgPoints);
        if (dictIndices.needed()) {
            dictIndices.create((int)cv_dicts.size(), 1, CV_32SC1);
            cv::Mat(cv_dicts).copyTo(dictIndices);
        }
    }
    // Direct access to the rejected quad list from the last call that collected it.
    const std::vector<std::vector<cv::Point2f>>& rejected() const {
        return last_rejected_;
    }

    // KakuTag-native form: returns Marker structs.
    const std::vector<Marker>& detect(cv::InputArray image_in) {
        cv::Mat image = ensure_gray(image_in);
        out_.clear();
        last_rejected_.clear();
        if (params_.collect_rejected) roi_scratch_.rejected_quads.clear();
        switch (params_.mode) {
            case Mode::Full:       run_full_into(image, out_);        break;
            case Mode::Pyramid:    run_pyramid_into(image, out_);     break;
            case Mode::AutoSingle:
            default:               run_auto_single_into(image, out_); break;
        }
        // Inverted-marker rescue: when the primary detection
        // pass yielded nothing AND the caller opted in, retry once on the
        // bit-inverted image. Markers printed white-on-black are recovered
        // without forcing a second pass on every frame.
        if (out_.empty() && (params_.detect_inverted_marker || params_.detectInvertedMarker)) {
            cv::Mat inv; cv::bitwise_not(image, inv);
            switch (params_.mode) {
                case Mode::Full:       run_full_into(inv, out_);        break;
                case Mode::Pyramid:    run_pyramid_into(inv, out_);     break;
                case Mode::AutoSingle:
                default:               run_auto_single_into(inv, out_); break;
            }
        }
        if (params_.collect_rejected) last_rejected_ = roi_scratch_.rejected_quads;
        return out_;
    }

    // Caller-owned output variants. Prefer these when marker results must live
    // past the next detect() call.
    void detect(cv::InputArray image_in, std::vector<Marker>& markers) {
        const auto& detected = detect(image_in);
        markers.assign(detected.begin(), detected.end());
    }

    std::vector<Marker> detect_copy(cv::InputArray image_in) {
        const auto& detected = detect(image_in);
        return std::vector<Marker>(detected.begin(), detected.end());
    }

    void reset() {
        out_.clear();
        last_rejected_.clear();
    }

    // Replace parameters at runtime (allocates scratch lazily on next call).
    void set_parameters(const DetectorParameters& p) {
        params_ = p;
        rebuild_packed_dicts();
        reset();
    }
    const DetectorParameters& parameters() const { return params_; }

private:
    static void copyVector2Output(const std::vector<std::vector<cv::Point2f>>& vec,
                                  cv::OutputArrayOfArrays out) {
        out.create((int)vec.size(), 1, CV_32FC2);
        if (out.isMatVector()) {
            for (size_t i = 0; i < vec.size(); ++i) {
                out.create(4, 1, CV_32FC2, (int)i);
                cv::Mat& m = out.getMatRef((int)i);
                cv::Mat(cv::Mat(vec[i]).t()).copyTo(m);
            }
        } else if (out.isUMatVector()) {
            for (size_t i = 0; i < vec.size(); ++i) {
                out.create(4, 1, CV_32FC2, (int)i);
                cv::UMat& m = out.getUMatRef((int)i);
                cv::Mat(cv::Mat(vec[i]).t()).copyTo(m);
            }
        } else if (out.kind() == cv::_OutputArray::STD_VECTOR_VECTOR) {
            for (size_t i = 0; i < vec.size(); ++i) {
                out.create(4, 1, CV_32FC2, (int)i);
                cv::Mat m = out.getMat((int)i);
                cv::Mat(cv::Mat(vec[i]).t()).copyTo(m);
            }
        } else {
            CV_Error(cv::Error::StsNotImplemented,
                     "Only Mat vector, UMat vector, and vector<vector<Point2f>> OutputArrays are supported.");
        }
    }

    DetectorOptions to_opt(bool refine_override = true) const {
        DetectorOptions o;
        o.dicts             = params_.dicts;
        o.min_size          = (params_.minSize >= 0) ? params_.minSize : params_.min_size;
        o.max_attempts      = (params_.maxAttemptsPerCandidate >= 0) ? params_.maxAttemptsPerCandidate : params_.max_attempts;
        o.refine_corners    = params_.refine_corners && refine_override;
        o.refine_window     = params_.refine_window;
        o.need_ids          = params_.need_ids;
        o.threshold_offset  = (params_.thres >= 0) ? params_.thres : params_.threshold_offset;
        o.threshold_offset_low = params_.threshold_offset_low;
        o.marker_border_bits = (params_.markerBorderBits >= 0.f)
            ? std::max(1, (int)std::lround(params_.markerBorderBits))
            : std::max(1, params_.marker_border_bits);
        o.error_correction_rate = (params_.errorCorrectionRate >= 0.0)
            ? params_.errorCorrectionRate : params_.error_correction_rate;
        o.max_hamming = params_.max_hamming;
        o.max_erroneous_bits_in_border_rate = (params_.maxErroneousBitsInBorderRate >= 0.0)
            ? params_.maxErroneousBitsInBorderRate : params_.max_erroneous_bits_in_border_rate;
        o.collect_rejected = params_.collect_rejected;
        o.packed_dicts = packed_dicts_.empty() ? nullptr : packed_dicts_.data();
        o.packed_dict_count = packed_dicts_.size();
        return o;
    }

    void rebuild_packed_dicts() {
        packed_dicts_.clear();
        packed_dicts_.reserve(params_.dicts.size());
        for (size_t i = 0; i < params_.dicts.size(); ++i) {
            packed_dicts_.push_back(detail::build_packed_dict(params_.dicts[i], (int)i));
        }
    }

    static cv::Mat ensure_gray(cv::InputArray in) {
        CV_Assert(!in.empty());
        cv::Mat m = in.getMat();
        if (m.type() == CV_8UC1) return m;
        if (m.type() == CV_8UC3) {
            cv::Mat g;
            cv::cvtColor(m, g, cv::COLOR_BGR2GRAY);
            return g;
        }
        if (m.type() == CV_8UC4) {
            cv::Mat g;
            cv::cvtColor(m, g, cv::COLOR_BGRA2GRAY);
            return g;
        }
        CV_Error(cv::Error::StsUnsupportedFormat,
                 "KakuTag expects CV_8UC1, CV_8UC3, or CV_8UC4 input.");
    }

    void ensure_full_scratch(int W, int H) {
        if (thr_full_.empty() || thr_full_.cols != W || thr_full_.rows != H)
            thr_full_.create(H, W, CV_8UC1);
    }
    void ensure_pyramid_scratch(int W, int H) {
        int sw = std::max(1, W / 2);
        int sh = std::max(1, H / 2);
        if (thr_small_.empty() || thr_small_.cols < sw || thr_small_.rows < sh)
            thr_small_.create(sh, sw, CV_8UC1);
    }

    void run_full_once_into(const cv::Mat& gray, std::vector<Marker>& out, int threshold_offset) {
        ensure_full_scratch(gray.cols, gray.rows);
        auto o = to_opt();
        o.threshold_offset = threshold_offset;
        detail::detect_on_roi_into(gray, cv::Rect(0,0,gray.cols,gray.rows),
                                   thr_full_, o, out, &roi_scratch_);
    }

    bool has_small_marker(const std::vector<Marker>& out) const {
        const float th = (float)std::max(params_.small_marker_full_fallback_side,
                                         params_.min_size * 3);
        for (const auto& m : out)
            if (marker_min_side_px(m) <= th) return true;
        return false;
    }

    // Helper: bounding boxes of already-detected markers (slightly inflated)
    // used to skip recovery candidates that fall inside a known TP.
    static bool quad_inside_any_marker(const detail::QuadCand& q,
                                       const std::vector<Marker>& known) {
        if (known.empty()) return false;
        for (const auto& m : known) {
            float minx = m.corners[0].x, maxx = minx;
            float miny = m.corners[0].y, maxy = miny;
            for (int k = 1; k < 4; ++k) {
                minx = std::min(minx, m.corners[k].x);
                maxx = std::max(maxx, m.corners[k].x);
                miny = std::min(miny, m.corners[k].y);
                maxy = std::max(maxy, m.corners[k].y);
            }
            const float pad = 2.f;
            cv::Rect2f bb(minx - pad, miny - pad,
                          (maxx - minx) + 2 * pad, (maxy - miny) + 2 * pad);
            int inside = 0;
            for (int j = 0; j < 4; ++j) if (bb.contains(q.c[j])) ++inside;
            if (inside >= 3) return true;
        }
        return false;
    }

    // Visited-aware contour recovery for small markers.
    //
    // Replaces the previous cv::findContours path AND avoids rebuilding the
    // adaptive threshold. After the first detect pass, thr_full_ already
    // contains:
    //   * 255 for foreground pixels that were never traced (the small markers
    //     we missed because of MIN_RUN=6 / tile-gate / threshold-bounds).
    //   * 100 (VISITED) for borders the first pass already walked.
    //   *   0 for background.
    // Tracing with MIN_RUN=1 over this buffer therefore visits ONLY the
    // contours the first pass missed. This is the fundamental p95/p99 fix:
    // previously, recovery rebuilt the binary from scratch (+2-3 ms) and
    // re-traced every already-discovered contour (+3-5 ms). Now it does
    // neither.
    //
    // The visited-aware tracer treats nonzero pixels (255 or 100) as
    // foreground for the "skip past current row" SIMD scan, but only seeds a
    // new trace when row_ptr[c] == FOREGROUND (255), so visited contours are
    // naturally skipped without any extra bookkeeping.
    //
    // Caveats:
    //   * Must be called immediately after a detect pass on the same gray
    //     image -- caller must not have invalidated thr_full_ in between.
    //   * Skips quad candidates inside a known marker bbox so the dictionary
    //     decode is never paid twice for the same TP.
    //   * max_attempts capped at 3 because contours here have pixel-accurate
    //     corners (no extrema-style 1-2 px bias on tiny quads).
    //   * Uses approxPolyDP first and canonical orientation
    //     so cornerSubPix uses stable initial corners, preserving
    //     0.533 px target corner error.
    void run_cv_contour_recovery_into(const cv::Mat& gray, std::vector<Marker>& out) {
        if (!params_.cv_contour_small_recovery) return;
        if (params_.small_marker_expected_min_count <= 0) return;
        if ((int)out.size() >= params_.small_marker_expected_min_count) return;
        if (!has_small_marker(out)) return;
        if (thr_full_.empty() || thr_full_.cols < gray.cols || thr_full_.rows < gray.rows) {
            // First pass didn't run (shouldn't happen in current call sites).
            // Fall through to the rebuild path for correctness.
            ensure_full_scratch(gray.cols, gray.rows);
            auto otmp = to_opt();
            detail::streaming_threshold_to_buffer(gray.ptr<uint8_t>(), gray.cols, gray.rows,
                (int)gray.step, otmp.threshold_offset,
                thr_full_.ptr<uint8_t>(), (int)thr_full_.step,
                &roi_scratch_.threshold, nullptr);
        }
        auto o = to_opt();
        if (o.max_attempts > 3) o.max_attempts = 3;

        cv::Mat thr = thr_full_(cv::Rect(0, 0, gray.cols, gray.rows));
        detail::zero_binary_border(thr);

        // Snapshot of already-detected markers for skip test on the decode side.
        // We don't need to stamp them in `thr` because the first pass already
        // wrote 100 (VISITED) along their borders, so the visited-aware tracer
        // will not re-seed inside them.
        std::vector<Marker> already = out;

        auto& candidates = roi_scratch_.candidates;
        candidates.clear();
        auto& contour_buf = roi_scratch_.contour_buf;
        auto& approx = roi_scratch_.approx_fallback;
        const int minSideSq = o.min_size * o.min_size;
        const size_t minContourSize = std::max<size_t>(40,
            (size_t)std::max(1, o.min_size) * 4u);

        // MIN_RUN=3: recovery is entered only after a small marker was found.
        // Length-3 seeding skips 1-2 px noise specks while preserving recoverable
        // marker borders. Decode robustness for tiny quads is handled by the
        // small-candidate cv decode fallback in decode_quad_candidates_into().
        // A fully permissive scan uses MIN_RUN=1
        // implicitly, but that then pays for every noise contour; we save that
        // cost without losing recall.
        // VISITED (100) borders from the first pass are skipped for free
        // because trace_runlen_streaming only seeds on row_ptr[c] == FOREGROUND.
        detail::trace_runlen_streaming(thr, minContourSize, 0.05f,
            /*MIN_RUN=*/3, contour_buf,
            [&](const std::vector<cv::Point>& contour) {
                if (contour.size() < 8) return;
                cv::approxPolyDP(contour, approx,
                                 double(contour.size()) * 0.03, true);
                if (approx.size() != 4 || !cv::isContourConvex(approx)) return;
                detail::QuadCand q;
                for (int j = 0; j < 4; ++j)
                    q.c[j] = cv::Point2f((float)approx[j].x, (float)approx[j].y);
                detail::orient_corners_canonical(q.c);
                for (int j = 0; j < 4; ++j) {
                    int k = (j + 1) & 3;
                    float dx = q.c[j].x - q.c[k].x;
                    float dy = q.c[j].y - q.c[k].y;
                    if (dx * dx + dy * dy < (float)minSideSq) return;
                }
                if (quad_inside_any_marker(q, already)) return;
                candidates.push_back(q);
            });

        if (candidates.empty()) return;
        detail::decode_quad_candidates_into(gray, cv::Rect(0,0,gray.cols,gray.rows),
                                            candidates, o, out, &roi_scratch_);
        detail::dedup_markers(out);
    }

    // Recovery at a custom threshold offset. Used as a second pass when the
    // primary offset's recovery still left small markers undetected (e.g. on
    // borderline low-contrast scenes). This pass DOES need to rebuild thr
    // because it uses a different threshold offset than the primary pass.
    // Cost is bounded by also masking out already-detected marker bboxes and
    // capping decode attempts.
    void run_cv_contour_recovery_at_offset(const cv::Mat& gray,
                                           std::vector<Marker>& out, int off) {
        if (params_.small_marker_expected_min_count <= 0) return;
        if ((int)out.size() >= params_.small_marker_expected_min_count) return;
        if (!has_small_marker(out)) return;
        ensure_full_scratch(gray.cols, gray.rows);
        auto o = to_opt();
        if (o.max_attempts > 3) o.max_attempts = 3;
        detail::streaming_threshold_to_buffer(gray.ptr<uint8_t>(), gray.cols, gray.rows,
            (int)gray.step, off,
            thr_full_.ptr<uint8_t>(), (int)thr_full_.step,
            &roi_scratch_.threshold, nullptr);
        cv::Mat thr = thr_full_(cv::Rect(0, 0, gray.cols, gray.rows));
        detail::zero_binary_border(thr);
        std::vector<Marker> already = out;
        // Stamp known marker regions to 0 so this independent threshold pass
        // does not re-trace and re-decode them.
        for (const auto& m : already) {
            float minx = m.corners[0].x, maxx = minx;
            float miny = m.corners[0].y, maxy = miny;
            for (int k = 1; k < 4; ++k) {
                minx = std::min(minx, m.corners[k].x);
                maxx = std::max(maxx, m.corners[k].x);
                miny = std::min(miny, m.corners[k].y);
                maxy = std::max(maxy, m.corners[k].y);
            }
            const float pad = 3.f;
            cv::Rect r((int)std::floor(minx - pad), (int)std::floor(miny - pad),
                       (int)std::ceil((maxx - minx) + 2 * pad),
                       (int)std::ceil((maxy - miny) + 2 * pad));
            r &= cv::Rect(0, 0, thr.cols, thr.rows);
            if (r.area() > 0) thr(r).setTo(cv::Scalar(0));
        }
        auto& candidates = roi_scratch_.candidates;
        candidates.clear();
        auto& contour_buf = roi_scratch_.contour_buf;
        auto& approx = roi_scratch_.approx_fallback;
        const int minSideSq = o.min_size * o.min_size;
        const size_t minContourSize = std::max<size_t>(40,
            (size_t)std::max(1, o.min_size) * 4u);
        detail::trace_runlen_streaming(thr, minContourSize, 0.05f,
            /*MIN_RUN=*/1, contour_buf,
            [&](const std::vector<cv::Point>& contour) {
                if (contour.size() < 8) return;
                cv::approxPolyDP(contour, approx,
                                 double(contour.size()) * 0.03, true);
                if (approx.size() != 4 || !cv::isContourConvex(approx)) return;
                detail::QuadCand q;
                for (int j = 0; j < 4; ++j)
                    q.c[j] = cv::Point2f((float)approx[j].x, (float)approx[j].y);
                detail::orient_corners_canonical(q.c);
                for (int j = 0; j < 4; ++j) {
                    int k = (j + 1) & 3;
                    float dx = q.c[j].x - q.c[k].x;
                    float dy = q.c[j].y - q.c[k].y;
                    if (dx * dx + dy * dy < (float)minSideSq) return;
                }
                if (quad_inside_any_marker(q, already)) return;
                candidates.push_back(q);
            });
        if (candidates.empty()) return;
        detail::decode_quad_candidates_into(gray, cv::Rect(0,0,gray.cols,gray.rows),
                                            candidates, o, out, &roi_scratch_);
        detail::dedup_markers(out);
    }

    void maybe_dual_threshold_small_into(const cv::Mat& gray, std::vector<Marker>& out) {
        if (params_.small_marker_expected_min_count <= 0) return;
        if ((int)out.size() >= params_.small_marker_expected_min_count) return;
        if (!has_small_marker(out)) return;
        // First, run cheap visited-aware recovery on the existing threshold.
        const size_t before_rec = out.size();
        run_cv_contour_recovery_into(gray, out);
        if (!params_.dual_threshold_small_markers) return;
        if ((int)out.size() >= params_.small_marker_expected_min_count) return;
        if (!has_small_marker(out)) return;
        // Skip dual when recovery already found something at this offset.
        if (out.size() > before_rec) return;
        run_cv_contour_recovery_at_offset(gray, out, params_.threshold_offset + 1);
    }

    void run_full_into(const cv::Mat& gray, std::vector<Marker>& out) {
        run_full_once_into(gray, out, params_.threshold_offset);
        maybe_dual_threshold_small_into(gray, out);
    }

    void run_pyramid_into(const cv::Mat& gray, std::vector<Marker>& out) {
        ensure_pyramid_scratch(gray.cols, gray.rows);
        // pass refine_corners through; the half-scale variant performs the
        // single full-resolution cornerSubPix pass internally.
        DetectorOptions oh = to_opt(/*refine_override=*/true);
        detail::detect_halfscale_full_decode_into(gray, oh, thr_small_, out, &roi_scratch_);
    }

    static float marker_min_side_px(const Marker& m) {
        float best = std::numeric_limits<float>::max();
        for (int k = 0; k < 4; ++k) {
            const auto& a = m.corners[k];
            const auto& b = m.corners[(k + 1) & 3];
            float dx = a.x - b.x;
            float dy = a.y - b.y;
            best = std::min(best, std::sqrt(dx * dx + dy * dy));
        }
        return best;
    }

    bool needs_full_recheck_after_pyramid(const std::vector<Marker>& out) const {
        if (!params_.safe_fallback) return false;
        // Empty pyramid output is also the no-marker fast path. Pay a full-frame
        // pass only when half-scale tracing produced a plausible number of quad
        // candidates. This rescues small markers while avoiding FP-heavy empty
        // scenes where full robust tracing is wasted.
        if (out.empty()) {
            int nc = roi_scratch_.last_candidate_count;
            return nc > 0 && nc <= params_.empty_pyramid_full_candidate_max;
        }
        if (!params_.robust_small_markers) return false;
        return has_small_marker(out);
    }

    void run_auto_single_into(const cv::Mat& gray, std::vector<Marker>& out) {
        if (gray.cols < 640 || gray.rows < 480) {
            run_full_into(gray, out);
            return;
        }
        run_pyramid_into(gray, out);
        // Original fallback only fired when pyramid returned zero markers.
        // That was fast, but it lost many small markers: if pyramid found one
        // small marker, the remaining small markers were never searched at full
        // resolution. For a detector that is tuned for both recall and
        // speed, small-marker scenes need a guarded full-resolution re-check.
        if (needs_full_recheck_after_pyramid(out)) {
            out.clear();
            run_full_into(gray, out);
        }
    }

    DetectorParameters params_;
    std::vector<detail::PackedDict> packed_dicts_;
    std::vector<Marker> out_;
    std::vector<std::vector<cv::Point2f>> last_rejected_;
    cv::Mat thr_full_;
    cv::Mat thr_small_;
    detail::RoiScratch roi_scratch_;
};

class MarkerDetector {
public:
    static inline std::vector<Marker> detect(const cv::Mat& img,
                                             const DetectorParameters& params = {},
                                             std::vector<Marker>* candidatesOut = nullptr) {
        DetectorParameters p = params;
        if (candidatesOut) p.collect_rejected = true;
        ArucoDetector d(p);
        auto& det_ref = d.detect(img);
        std::vector<Marker> det(det_ref.begin(), det_ref.end());
        if (candidatesOut) {
            candidatesOut->clear();
            for (const auto& c : d.rejected()) {
                if (c.size() != 4) continue;
                Marker m;
                m.id = -1;
                for (int k = 0; k < 4; ++k) m.corners[k] = c[k];
                candidatesOut->push_back(m);
            }
        }
        return det;
    }
};

namespace detail {
inline DetectorParameters parameters_from_options(const DetectorOptions& opt, Mode mode) {
    DetectorParameters p;
    p.dicts            = opt.dicts;
    p.min_size         = opt.min_size;
    p.max_attempts     = opt.max_attempts;
    p.refine_corners   = opt.refine_corners;
    p.refine_window    = opt.refine_window;
    p.need_ids         = opt.need_ids;
    p.threshold_offset = opt.threshold_offset;
    p.threshold_offset_low = opt.threshold_offset_low;
    p.marker_border_bits = std::max(1, opt.marker_border_bits);
    p.error_correction_rate = opt.error_correction_rate;
    p.max_hamming = opt.max_hamming;
    p.max_erroneous_bits_in_border_rate = opt.max_erroneous_bits_in_border_rate;
    p.collect_rejected = opt.collect_rejected;
    p.mode = mode;
    return p;
}
} // namespace detail

// ----------------------------------------------------------------------------
// Backwards-compatible thin wrappers around ArucoDetector. These mirror the
// original free-function API used during the prototype era; they allocate a
// fresh scratch buffer on every call and therefore are *slower* than reusing
// an ArucoDetector instance. Use ArucoDetector for performance.
// ----------------------------------------------------------------------------
inline std::vector<Marker> detect_full(const cv::Mat& gray,
                                       const DetectorOptions& opt = {}) {
    DetectorParameters p = detail::parameters_from_options(opt, Mode::Full);
    ArucoDetector d(p);
    return d.detect(gray);
}

inline std::vector<Marker> detect_single(const cv::Mat& gray,
                                         const DetectorOptions& opt = {}) {
    DetectorParameters p = detail::parameters_from_options(
        opt, opt.pyramid_enabled ? Mode::AutoSingle : Mode::Full);
    p.safe_fallback    = true;
    ArucoDetector d(p);
    return d.detect(gray);
}

// ============================================================================
// Convenience helpers (Marker::draw, Marker::estimatePose,
// estimatePoseSingleMarkers, drawDetectedMarkers, generateImageMarker,
// detect helpers with inverted-marker / errorCorrectionRate / borderBits).
// Implementations live here so the Marker struct stays a POD-like header.
// ============================================================================

inline void Marker::draw(cv::Mat& in, const cv::Scalar& color) const {
    const float flineWidth = std::max(1.f, std::min(5.f, float(in.cols) / 500.f));
    const int   lineWidth  = (int)std::lround(flineWidth);
    for (int i = 0; i < 4; ++i) {
        cv::line(in, corners[i], corners[(i + 1) & 3], color, lineWidth);
    }
    const cv::Point2f p2(2.f * lineWidth, 2.f * lineWidth);
    // First corner: filled red square (origin marker).
    cv::rectangle(in, corners[0] - p2, corners[0] + p2,
                  cv::Scalar(0, 0, 255, 255), -1);
    cv::rectangle(in, corners[1] - p2, corners[1] + p2,
                  cv::Scalar(0, 255, 0, 255), lineWidth);
    cv::rectangle(in, corners[2] - p2, corners[2] + p2,
                  cv::Scalar(255, 0, 0, 255), lineWidth);
    cv::Point2f cent(0, 0);
    for (int i = 0; i < 4; ++i) cent += corners[i];
    cent *= 0.25f;
    const float fsize = std::min(3.0f, flineWidth * 0.75f);
    std::stringstream ss; ss << id;
    cv::putText(in, ss.str(),
                cent - cv::Point2f(10 * flineWidth, 0),
                cv::FONT_HERSHEY_SIMPLEX, fsize,
                cv::Scalar(255, 255, 255) - color, lineWidth, cv::LINE_AA);
}

inline std::pair<cv::Mat, cv::Mat>
Marker::estimatePose(const cv::Mat& cameraMatrix,
                     const cv::Mat& distCoeffs,
                     double markerSize) const {
    // Corner ordering: TL, TR, BR, BL, in marker local frame
    // with marker plane = Z=0, side length = markerSize.
    const double h = markerSize * 0.5;
    const std::vector<cv::Point3d> objPts = {
        {-h,  h, 0}, { h,  h, 0}, { h, -h, 0}, {-h, -h, 0}
    };
    const std::vector<cv::Point2d> imgPts = {
        corners[0], corners[1], corners[2], corners[3]
    };
    cv::Mat rvec, tvec;
    // IPPE_SQUARE is the analytic 4-coplanar-point solver for a square marker.
    // Fall back to SQPNP if IPPE rejects the input.
    bool ok = false;
    try {
        ok = cv::solvePnP(objPts, imgPts, cameraMatrix, distCoeffs,
                          rvec, tvec, false, cv::SOLVEPNP_IPPE_SQUARE);
    } catch (...) { ok = false; }
    if (!ok) {
        cv::solvePnP(objPts, imgPts, cameraMatrix, distCoeffs,
                     rvec, tvec, false, cv::SOLVEPNP_SQPNP);
    }
    return { rvec, tvec };
}

// OpenCV cv::aruco::estimatePoseSingleMarkers-compatible helper.
// `corners` may be std::vector<std::vector<cv::Point2f>> (one per marker).
inline void estimatePoseSingleMarkers(
    const std::vector<std::vector<cv::Point2f>>& corners,
    float markerLength,
    const cv::Mat& cameraMatrix,
    const cv::Mat& distCoeffs,
    std::vector<cv::Vec3d>& rvecs,
    std::vector<cv::Vec3d>& tvecs)
{
    rvecs.clear(); tvecs.clear();
    rvecs.reserve(corners.size()); tvecs.reserve(corners.size());
    const double h = double(markerLength) * 0.5;
    const std::vector<cv::Point3d> objPts = {
        {-h,  h, 0}, { h,  h, 0}, { h, -h, 0}, {-h, -h, 0}
    };
    for (const auto& c : corners) {
        std::vector<cv::Point2d> img = { c[0], c[1], c[2], c[3] };
        cv::Mat r, t;
        bool ok = false;
        try {
            ok = cv::solvePnP(objPts, img, cameraMatrix, distCoeffs,
                              r, t, false, cv::SOLVEPNP_IPPE_SQUARE);
        } catch (...) { ok = false; }
        if (!ok) {
            cv::solvePnP(objPts, img, cameraMatrix, distCoeffs,
                         r, t, false, cv::SOLVEPNP_SQPNP);
        }
        cv::Vec3d rv(r.at<double>(0), r.at<double>(1), r.at<double>(2));
        cv::Vec3d tv(t.at<double>(0), t.at<double>(1), t.at<double>(2));
        rvecs.push_back(rv);
        tvecs.push_back(tv);
    }
}

// cv::aruco::drawDetectedMarkers-compatible helper.
inline void drawDetectedMarkers(
    cv::Mat& image,
    const std::vector<std::vector<cv::Point2f>>& corners,
    const std::vector<int>& ids = {},
    const cv::Scalar& borderColor = cv::Scalar(0, 255, 0))
{
    for (size_t i = 0; i < corners.size(); ++i) {
        Marker m;
        for (int k = 0; k < 4; ++k) m.corners[k] = corners[i][k];
        m.id = (i < ids.size()) ? ids[i] : -1;
        m.draw(image, borderColor);
    }
}

// Thin wrapper around cv::aruco::generateImageMarker for symmetric API.
inline void generateImageMarker(const cv::aruco::Dictionary& dict,
                                int id, int sidePixels, cv::Mat& out,
                                int borderBits = 1)
{
    cv::aruco::generateImageMarker(dict, id, sidePixels, out, borderBits);
}

// detect() entry point that provides a simple static detection API.
// Honours `detect_inverted_marker`: when true and the image yields no markers,
// the image is bit-inverted and re-detected to support white-on-black marker
// printouts.
inline std::vector<Marker> detect(const cv::Mat& image,
                                  const DetectorParameters& params = {},
                                  std::vector<Marker>* rejected = nullptr)
{
    DetectorParameters p = params;
    if (rejected) p.collect_rejected = true;
    ArucoDetector d(p);
    auto result = d.detect(image);
    if (rejected) {
        rejected->clear();
        for (const auto& c : d.rejected()) {
            if (c.size() != 4) continue;
            Marker m;
            m.id = -1;
            for (int k = 0; k < 4; ++k) m.corners[k] = c[k];
            rejected->push_back(m);
        }
    }
    return result;
}

} // namespace kakutag

#endif // KAKUTAG_KAKUTAG_HPP_
