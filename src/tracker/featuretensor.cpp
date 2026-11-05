#include "featuretensor.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <opencv2/imgproc.hpp>

bool FeatureTensor::getRectsFeature(const cv::Mat& image, DETECTIONS& detections) {
    if (input_width() <= 0 || input_height() <= 0 || input_channels() != 3) {
        std::fprintf(stderr, "unsupported ReID input shape: %dx%dx%d\n",
                     input_width(), input_height(), input_channels());
        return false;
    }

    const cv::Rect image_bounds(0, 0, image.cols, image.rows);
    for (auto& detection : detections) {
        cv::Rect roi(static_cast<int>(detection.tlwh(0)), static_cast<int>(detection.tlwh(1)),
                     static_cast<int>(detection.tlwh(2)), static_cast<int>(detection.tlwh(3)));
        roi &= image_bounds;
        if (roi.width < 2 || roi.height < 2) {
            return false;
        }

        cv::Mat resized;
        cv::resize(image(roi), resized, cv::Size(input_width(), input_height()), 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
        if (!resized.isContinuous()) {
            resized = resized.clone();
        }

        const std::size_t bytes = resized.total() * resized.elemSize();
        if (inference(resized.data, bytes) != RKNN_SUCC) {
            std::fprintf(stderr, "ReID inference failed\n");
            return false;
        }

        const auto& values = output(0);
        if (values.size() < 512) {
            std::fprintf(stderr, "ReID output has %zu values, expected 512\n", values.size());
            return false;
        }
        float norm = 0.0f;
        for (int i = 0; i < 512; ++i) {
            norm += values[i] * values[i];
        }
        norm = std::sqrt(std::max(norm, 1e-12f));
        for (int i = 0; i < 512; ++i) {
            detection.feature(i) = values[i] / norm;
        }
    }
    return true;
}
