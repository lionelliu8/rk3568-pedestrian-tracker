#include "multi_tracker.h"

#include <sstream>
#include <stdexcept>

#include "bytetrack.h"
#include "deepsort.h"
#include "ocsort.h"

namespace {

class ByteTrackAdapter : public MultiTracker {
public:
    explicit ByteTrackAdapter(const mot::ByteTrackConfig& c) : config_(c), tracker_(c) {}
    std::vector<DetectBox> update(const cv::Mat&, const std::vector<DetectBox>& d) override {
        return last_ = tracker_.update(d);
    }
    std::vector<DetectBox> skip_frame() override { return last_; }
    float min_detection_score() const override { return config_.low_thresh; }
    std::string description() const override {
        std::ostringstream s;
        s << "ByteTrack (high " << config_.high_thresh << ", new " << config_.new_track_thresh << ", low "
          << config_.low_thresh << ", buffer " << config_.track_buffer << ")";
        return s.str();
    }

private:
    mot::ByteTrackConfig config_;
    mot::ByteTrack tracker_;
    std::vector<DetectBox> last_;
};

class OcSortAdapter : public MultiTracker {
public:
    explicit OcSortAdapter(const mot::OcSortConfig& c) : config_(c), tracker_(c) {}
    std::vector<DetectBox> update(const cv::Mat&, const std::vector<DetectBox>& d) override {
        return last_ = tracker_.update(d);
    }
    std::vector<DetectBox> skip_frame() override { return last_; }
    float min_detection_score() const override { return config_.use_byte ? config_.low_thresh : config_.det_thresh; }
    std::string description() const override {
        std::ostringstream s;
        s << "OC-SORT (det " << config_.det_thresh << ", IoU " << config_.iou_thresh << ", max_age " << config_.max_age
          << ")";
        return s.str();
    }

private:
    mot::OcSortConfig config_;
    mot::OcSort tracker_;
    std::vector<DetectBox> last_;
};

class DeepSortAdapter : public MultiTracker {
public:
    DeepSortAdapter(const TrackerOptions& o, float threshold)
        : threshold_(threshold), tracker_(o.reid_model, o.detect_interval, o.max_age, o.n_init) {}
    std::vector<DetectBox> update(const cv::Mat& frame, const std::vector<DetectBox>& d) override {
        std::vector<DetectBox> boxes = d;
        cv::Mat image = frame;  // DeepSort::sort takes a non-const reference
        tracker_.sort(image, boxes);
        return boxes;
    }
    std::vector<DetectBox> skip_frame() override {
        std::vector<DetectBox> boxes;
        tracker_.sort_interval(boxes);
        return boxes;
    }
    float min_detection_score() const override { return threshold_; }
    std::string description() const override { return "DeepSORT (OSNet ReID on the NPU)"; }

private:
    float threshold_;
    DeepSort tracker_;
};

}  // namespace

std::unique_ptr<MultiTracker> make_tracker(const TrackerOptions& o) {
    if (o.kind == "bytetrack") {
        mot::ByteTrackConfig c;
        if (o.track_threshold > 0) c.high_thresh = o.track_threshold;
        c.new_track_thresh = c.high_thresh + 0.1f;
        c.track_buffer = o.max_age;
        return std::make_unique<ByteTrackAdapter>(c);
    }
    if (o.kind == "ocsort") {
        mot::OcSortConfig c;
        c.det_thresh = o.track_threshold > 0 ? o.track_threshold : 0.3f;
        c.max_age = o.max_age;
        return std::make_unique<OcSortAdapter>(c);
    }
    if (o.kind == "deepsort") {
        return std::make_unique<DeepSortAdapter>(o, o.track_threshold > 0 ? o.track_threshold : 0.25f);
    }
    throw std::invalid_argument("unknown tracker: " + o.kind + " (bytetrack, ocsort or deepsort)");
}
