#ifndef PEDESTRIAN_TRACKER_MULTI_TRACKER_H
#define PEDESTRIAN_TRACKER_MULTI_TRACKER_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "detect_box.h"

struct TrackerOptions {
    std::string kind = "bytetrack";  // bytetrack | ocsort | deepsort
    float track_threshold = -1.0f;   // < 0: the tracker's default (see make_tracker)
    int max_age = 30;
    int n_init = 2;                  // deepsort only
    int detect_interval = 1;
    std::string reid_model;          // deepsort only
};

// Common interface over ByteTrack, OC-SORT and DeepSORT.
class MultiTracker {
public:
    virtual ~MultiTracker() = default;
    // One frame with detections. `frame` is only used by DeepSORT (appearance features).
    virtual std::vector<DetectBox> update(const cv::Mat& frame, const std::vector<DetectBox>& detections) = 0;
    // A frame without detections (--detect-interval > 1).
    virtual std::vector<DetectBox> skip_frame() = 0;
    // Lowest detection score the tracker uses; the detector should keep boxes down to this.
    virtual float min_detection_score() const = 0;
    virtual std::string description() const = 0;
};

// ByteTrack: track_threshold = high threshold (default 0.5), new tracks need +0.1, boxes down to 0.1
//            are used to keep existing tracks.
// OC-SORT:   track_threshold = detection threshold (default 0.3).
// DeepSORT:  track_threshold = detection threshold (default 0.25), OSNet ReID on the NPU.
// Defaults were chosen on MOT17-05 (docs/BENCHMARK_20261105.md).
std::unique_ptr<MultiTracker> make_tracker(const TrackerOptions& options);

#endif
