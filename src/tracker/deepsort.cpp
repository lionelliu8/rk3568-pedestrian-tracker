#include "deepsort.h"

#include <algorithm>
#include <stdexcept>

DeepSort::DeepSort(const std::string& model_path, int track_interval, int max_age, int n_init)
    : track_interval_(std::max(1, track_interval)),
      tracker_(new tracker(0.2f, 100, 0.7f, max_age, n_init)),
      feature_extractor_(new FeatureTensor(model_path.c_str())) {}

DeepSort::~DeepSort() {
    delete feature_extractor_;
    delete tracker_;
}

void DeepSort::sort(cv::Mat& frame, std::vector<DetectBox>& boxes) {
    DETECTIONS detections;
    std::vector<CLSCONF> classes;
    detections.reserve(boxes.size());
    classes.reserve(boxes.size());

    for (const auto& box : boxes) {
        const float x1 = std::clamp(box.x1, 0.0f, static_cast<float>(frame.cols - 1));
        const float y1 = std::clamp(box.y1, 0.0f, static_cast<float>(frame.rows - 1));
        const float x2 = std::clamp(box.x2, 0.0f, static_cast<float>(frame.cols));
        const float y2 = std::clamp(box.y2, 0.0f, static_cast<float>(frame.rows));
        if (x2 - x1 < 2.0f || y2 - y1 < 2.0f) {
            continue;
        }
        DETECTION_ROW row;
        row.tlwh = DETECTBOX(x1, y1, x2 - x1, y2 - y1);
        row.confidence = box.confidence;
        row.feature.setZero();
        detections.push_back(row);
        classes.emplace_back(static_cast<int>(box.classID), box.confidence);
    }

    if (!detections.empty() && !feature_extractor_->getRectsFeature(frame, detections)) {
        throw std::runtime_error("ReID feature extraction failed");
    }

    tracker_->predict();
    tracker_->update(DETECTIONSV2(classes, detections));
    collect_results(boxes, 1);
}

void DeepSort::sort_interval(std::vector<DetectBox>& boxes) {
    tracker_->predict();
    collect_results(boxes, track_interval_ + 1);
}

void DeepSort::collect_results(std::vector<DetectBox>& boxes, int max_age) {
    boxes.clear();
    for (Track& track : tracker_->tracks) {
        if (!track.is_confirmed() || track.time_since_update > max_age) {
            continue;
        }
        const DETECTBOX box = track.to_tlwh();
        DetectBox result(box(0), box(1), box(0) + box(2), box(1) + box(3), track.conf, track.cls,
                         static_cast<float>(track.track_id));
        boxes.push_back(result);
    }
}
