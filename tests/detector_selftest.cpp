// Board-side check of the YOLO26 detector without a camera.
// Input is a raw BGR file (make one with tools/image_to_bgr.py).
// Usage: [PERF_DETAIL=1] detector_selftest MODEL IMAGE.bgr WIDTH HEIGHT [runs]
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "stopwatch.h"
#include "yolo26_detector.h"

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: " << argv[0] << " MODEL IMAGE.bgr WIDTH HEIGHT [runs]\n";
        return 2;
    }
    try {
        const int width = std::stoi(argv[3]);
        const int height = std::stoi(argv[4]);
        const int runs = argc > 5 ? std::stoi(argv[5]) : 20;

        cv::Mat image(height, width, CV_8UC3);
        std::ifstream file(argv[2], std::ios::binary);
        if (!file.read(reinterpret_cast<char*>(image.data), static_cast<std::streamsize>(image.total() * 3))) {
            throw std::runtime_error("cannot read " + std::to_string(width * height * 3) + " bytes from " + argv[2]);
        }

        DetectorConfig config;
        config.class_filter = -1;
        config.collect_perf = std::getenv("PERF_DETAIL") != nullptr;
        Yolo26Detector detector(argv[1], config);
        std::vector<DetectBox> boxes;
        double pre = 0, npu = 0, post = 0;
        for (int i = 0; i < runs; ++i) {
            boxes = detector.detect(image);
            if (i == 0) continue;  // skip warm-up
            pre += detector.last_timing().preprocess_ms;
            npu += detector.last_timing().inference_ms;
            post += detector.last_timing().postprocess_ms;
        }
        const double n = std::max(1, runs - 1);
        std::cout << "avg ms: preprocess=" << pre / n << " npu=" << npu / n << " postprocess=" << post / n << '\n';
        if (config.collect_perf) std::cout << detector.perf_detail() << '\n';
        for (const auto& b : boxes) {
            std::cout << "class=" << static_cast<int>(b.classID) << " score=" << b.confidence << " box=("
                      << static_cast<int>(b.x1) << ',' << static_cast<int>(b.y1) << ',' << static_cast<int>(b.x2)
                      << ',' << static_cast<int>(b.y2) << ")\n";
        }
        return boxes.empty() ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
