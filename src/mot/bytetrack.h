#ifndef PEDESTRIAN_TRACKER_MOT_BYTETRACK_H
#define PEDESTRIAN_TRACKER_MOT_BYTETRACK_H

#include <vector>

#include "detect_box.h"
#include "kalman_xyah.h"

namespace mot {

struct ByteTrackConfig {
    float high_thresh = 0.5f;   // detections above this start the first association
    float low_thresh = 0.1f;    // detections in [low, high) only extend existing tracks
    float new_track_thresh = 0.6f;
    float match_thresh = 0.8f;  // max (1 - IoU) in the first association
    int track_buffer = 30;      // frames a lost track is kept
    bool fuse_score = true;     // weight IoU by detection score (official setting for MOT17)
};

// ByteTrack (Zhang et al., ECCV 2022), following the official BYTETracker logic.
class ByteTrack {
public:
    explicit ByteTrack(const ByteTrackConfig& config = {}) : config_(config) {}

    // All detections of one frame (any score). Returns the active tracks with trackID set.
    std::vector<DetectBox> update(const std::vector<DetectBox>& detections);

private:
    enum class State { Tracked, Lost, Removed };
    struct Track {
        KalmanXyah kf;
        int id = 0;
        State state = State::Tracked;
        bool activated = false;
        int frame = 0;        // last frame updated
        int start_frame = 0;
        float score = 0;
        float cls = 0;
    };
    struct Det {
        Box box;
        float score;
        float cls;
    };

    void activate(Track& t, const Det& d, bool confirmed);
    void refresh(Track& t, const Det& d);

    ByteTrackConfig config_;
    std::vector<Track> tracked_, lost_;
    int frame_ = 0;
    int next_id_ = 1;
};

}  // namespace mot

#endif
