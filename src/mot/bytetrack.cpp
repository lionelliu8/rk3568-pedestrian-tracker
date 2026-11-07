#include "bytetrack.h"

#include <algorithm>

namespace mot {

namespace {

std::vector<std::vector<float>> iou_cost(const std::vector<Box>& tracks, const std::vector<Box>& dets,
                                         const std::vector<float>* det_scores) {
    auto cost = iou_matrix(tracks, dets);
    for (auto& row : cost)
        for (std::size_t j = 0; j < row.size(); ++j)
            row[j] = 1.0f - (det_scores ? row[j] * (*det_scores)[j] : row[j]);
    return cost;
}

}  // namespace

void ByteTrack::activate(Track& t, const Det& d, bool confirmed) {
    t.kf.initiate(d.box);
    t.id = next_id_++;
    t.state = State::Tracked;
    t.activated = confirmed;
    t.frame = t.start_frame = frame_;
    t.score = d.score;
    t.cls = d.cls;
}

void ByteTrack::refresh(Track& t, const Det& d) {
    t.kf.update(d.box);
    t.state = State::Tracked;
    t.activated = true;
    t.frame = frame_;
    t.score = d.score;
}

std::vector<DetectBox> ByteTrack::update(const std::vector<DetectBox>& detections) {
    ++frame_;
    std::vector<Det> high, low;
    for (const auto& d : detections) {
        const Det det{{d.x1, d.y1, d.x2, d.y2}, d.confidence, d.classID};
        if (d.confidence >= config_.high_thresh) high.push_back(det);
        else if (d.confidence > config_.low_thresh) low.push_back(det);
    }

    // Confirmed tracks and lost tracks are matched first; unconfirmed (seen once) separately.
    std::vector<Track> pool, unconfirmed;
    for (auto& t : tracked_) (t.activated ? pool : unconfirmed).push_back(t);
    for (auto& t : lost_) pool.push_back(t);
    for (auto& t : pool) {
        if (t.state != State::Tracked) t.kf.mean(7) = 0;  // freeze height velocity while lost
        t.kf.predict();
    }
    for (auto& t : unconfirmed) t.kf.predict();

    std::vector<Track> next_tracked, next_lost;
    auto boxes_of = [](const std::vector<Track>& ts) {
        std::vector<Box> b;
        for (const auto& t : ts) b.push_back(t.kf.box());
        return b;
    };
    auto det_boxes = [](const std::vector<Det>& ds, const std::vector<int>& idx) {
        std::vector<Box> b;
        for (int i : idx) b.push_back(ds[i].box);
        return b;
    };
    std::vector<int> high_idx(high.size()), low_idx(low.size());
    for (std::size_t i = 0; i < high.size(); ++i) high_idx[i] = static_cast<int>(i);
    for (std::size_t i = 0; i < low.size(); ++i) low_idx[i] = static_cast<int>(i);

    // 1) pool vs high-score detections
    std::vector<float> high_scores;
    for (const auto& d : high) high_scores.push_back(d.score);
    const auto a1 = linear_assignment(
        iou_cost(boxes_of(pool), det_boxes(high, high_idx), config_.fuse_score ? &high_scores : nullptr),
        static_cast<int>(high.size()), config_.match_thresh);
    for (auto [ti, di] : a1.matches) {
        refresh(pool[ti], high[di]);
        next_tracked.push_back(pool[ti]);
    }

    // 2) remaining *tracked* (not lost) tracks vs low-score detections
    std::vector<Track> remaining;
    for (int ti : a1.unmatched_rows)
        if (pool[ti].state == State::Tracked) remaining.push_back(pool[ti]);
        else next_lost.push_back(pool[ti]);
    const auto a2 = linear_assignment(iou_cost(boxes_of(remaining), det_boxes(low, low_idx), nullptr),
                                      static_cast<int>(low.size()), 0.5f);
    for (auto [ti, di] : a2.matches) {
        refresh(remaining[ti], low[di]);
        next_tracked.push_back(remaining[ti]);
    }
    for (int ti : a2.unmatched_rows) {
        remaining[ti].state = State::Lost;
        next_lost.push_back(remaining[ti]);
    }

    // 3) unconfirmed tracks vs high-score detections left over from step 1
    std::vector<Det> left_high;
    std::vector<float> left_scores;
    for (int di : a1.unmatched_cols) {
        left_high.push_back(high[di]);
        left_scores.push_back(high[di].score);
    }
    std::vector<int> left_idx(left_high.size());
    for (std::size_t i = 0; i < left_high.size(); ++i) left_idx[i] = static_cast<int>(i);
    const auto a3 = linear_assignment(
        iou_cost(boxes_of(unconfirmed), det_boxes(left_high, left_idx), config_.fuse_score ? &left_scores : nullptr),
        static_cast<int>(left_high.size()), 0.7f);
    for (auto [ti, di] : a3.matches) {
        refresh(unconfirmed[ti], left_high[di]);
        next_tracked.push_back(unconfirmed[ti]);
    }
    // unmatched unconfirmed tracks are dropped

    // 4) new tracks from confident unmatched detections
    for (int di : a3.unmatched_cols) {
        if (left_high[di].score < config_.new_track_thresh) continue;
        Track t;
        activate(t, left_high[di], frame_ == 1);
        next_tracked.push_back(t);
    }

    // 5) forget tracks lost for too long; drop lost duplicates of tracked ones
    std::vector<Track> kept_lost;
    for (auto& t : next_lost)
        if (frame_ - t.frame <= config_.track_buffer) kept_lost.push_back(t);
    const auto dup = iou_matrix(boxes_of(next_tracked), boxes_of(kept_lost));
    std::vector<char> drop_tracked(next_tracked.size(), false), drop_lost(kept_lost.size(), false);
    for (std::size_t i = 0; i < next_tracked.size(); ++i)
        for (std::size_t j = 0; j < kept_lost.size(); ++j)
            if (1.0f - dup[i][j] < 0.15f) {
                const int age_t = next_tracked[i].frame - next_tracked[i].start_frame;
                const int age_l = kept_lost[j].frame - kept_lost[j].start_frame;
                (age_t > age_l ? drop_lost[j] : drop_tracked[i]) = true;
            }
    tracked_.clear();
    lost_.clear();
    for (std::size_t i = 0; i < next_tracked.size(); ++i)
        if (!drop_tracked[i]) tracked_.push_back(next_tracked[i]);
    for (std::size_t j = 0; j < kept_lost.size(); ++j)
        if (!drop_lost[j]) lost_.push_back(kept_lost[j]);

    std::vector<DetectBox> out;
    for (const auto& t : tracked_) {
        if (!t.activated) continue;
        const Box b = t.kf.box();
        out.emplace_back(b[0], b[1], b[2], b[3], t.score, t.cls, static_cast<float>(t.id));
    }
    return out;
}

}  // namespace mot
