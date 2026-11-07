#ifndef PEDESTRIAN_TRACKER_MOT_OCSORT_H
#define PEDESTRIAN_TRACKER_MOT_OCSORT_H

#include <map>
#include <vector>

#include <Eigen/Dense>

#include "assignment.h"
#include "detect_box.h"

namespace mot {

struct OcSortConfig {
    float det_thresh = 0.6f;   // detections below this are ignored (unless use_byte)
    float iou_thresh = 0.3f;   // minimum IoU for a match
    int max_age = 30;          // frames a lost track is kept
    int min_hits = 3;          // matched frames before a track is reported
    int delta_t = 3;           // frame gap used to estimate the motion direction
    float inertia = 0.2f;      // weight of the direction-consistency term (OCM)
    bool use_byte = false;     // second association with low-score detections, as in ByteTrack
    float low_thresh = 0.1f;
};

// OC-SORT (Cao et al., CVPR 2023): SORT plus observation-centric re-update (ORU) after
// occlusions, observation-centric momentum (OCM) in the association cost, and
// observation-centric recovery (OCR) for tracks that just lost their detection.
// Follows the official ocsort.py; reported boxes are the matched detections.
class OcSort {
public:
    explicit OcSort(const OcSortConfig& config = {}) : config_(config) {}

    std::vector<DetectBox> update(const std::vector<DetectBox>& detections);

private:
    using Vec7 = Eigen::Matrix<float, 7, 1>;
    using Mat7 = Eigen::Matrix<float, 7, 7>;
    using Vec4 = Eigen::Vector4f;

    struct Kalman {  // state: cx, cy, area, aspect, vcx, vcy, varea (SORT)
        Vec7 x = Vec7::Zero();
        Mat7 p = Mat7::Identity();
        void init(const Vec4& z);
        void predict();
        void update(const Vec4& z);
    };
    struct Obs {
        Box box;
        float score;
    };
    struct Track {
        Kalman kf;
        Kalman saved;           // filter state when the track lost its detection (for ORU)
        bool observed = true;
        int id = 0;
        int age = 0;
        int time_since_update = 0;
        int hit_streak = 0;
        bool has_obs = false;
        Obs last_obs{};
        int last_obs_age = 0;
        std::map<int, Obs> observations;  // age -> observation
        Eigen::Vector2f velocity = Eigen::Vector2f::Zero();  // unit direction (dy, dx)
        bool has_velocity = false;
        float cls = 0;
        Box predicted{};
    };

    static Vec4 to_z(const Box& b);
    static Box to_box(const Vec7& x);
    void match_track(Track& t, const Obs& obs);
    void miss_track(Track& t);
    Obs previous_obs(const Track& t) const;

    OcSortConfig config_;
    std::vector<Track> tracks_;
    int frame_ = 0;
    int next_id_ = 1;
};

}  // namespace mot

#endif
