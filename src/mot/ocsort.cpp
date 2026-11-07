#include "ocsort.h"

#include <algorithm>
#include <cmath>

namespace mot {

namespace {

Eigen::Vector2f direction(const Box& from, const Box& to) {
    const float dy = (to[1] + to[3]) / 2 - (from[1] + from[3]) / 2;
    const float dx = (to[0] + to[2]) / 2 - (from[0] + from[2]) / 2;
    const float n = std::sqrt(dx * dx + dy * dy) + 1e-6f;
    return Eigen::Vector2f(dy / n, dx / n);
}

}  // namespace

OcSort::Vec4 OcSort::to_z(const Box& b) {
    const float w = b[2] - b[0], h = b[3] - b[1];
    return Vec4((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, w * h, w / std::max(h, 1e-3f));
}

Box OcSort::to_box(const Vec7& x) {
    const float s = std::max(x(2), 1e-3f), r = std::max(x(3), 1e-3f);
    const float w = std::sqrt(s * r), h = s / w;
    return {x(0) - w / 2, x(1) - h / 2, x(0) + w / 2, x(1) + h / 2};
}

void OcSort::Kalman::init(const Vec4& z) {
    x.setZero();
    x.head<4>() = z;
    p = Mat7::Identity();
    p.block<3, 3>(4, 4) *= 1000.0f;  // unknown initial velocity
    p *= 10.0f;
}

void OcSort::Kalman::predict() {
    if (x(6) + x(2) <= 0) x(6) = 0;
    Mat7 f = Mat7::Identity();
    f(0, 4) = f(1, 5) = f(2, 6) = 1.0f;
    Mat7 q = Mat7::Identity();
    q.block<3, 3>(4, 4) *= 0.01f;
    q(6, 6) *= 0.01f;
    x = f * x;
    p = f * p * f.transpose() + q;
}

void OcSort::Kalman::update(const Vec4& z) {
    Eigen::Matrix<float, 4, 7> h = Eigen::Matrix<float, 4, 7>::Zero();
    h.block<4, 4>(0, 0).setIdentity();
    Eigen::Matrix4f r = Eigen::Matrix4f::Identity();
    r(2, 2) = r(3, 3) = 10.0f;
    const Eigen::Matrix4f s = h * p * h.transpose() + r;
    const Eigen::Matrix<float, 7, 4> k = p * h.transpose() * s.inverse();
    x += k * (z - h * x);
    const Mat7 ikh = Mat7::Identity() - k * h;
    p = ikh * p * ikh.transpose() + k * r * k.transpose();
}

OcSort::Obs OcSort::previous_obs(const Track& t) const {
    for (int i = 0; i < config_.delta_t; ++i) {
        const auto it = t.observations.find(t.age - (config_.delta_t - i));
        if (it != t.observations.end()) return it->second;
    }
    return t.last_obs;
}

void OcSort::match_track(Track& t, const Obs& obs) {
    if (t.has_obs) {
        t.velocity = direction(previous_obs(t).box, obs.box);
        t.has_velocity = true;
    }
    if (!t.observed && t.has_obs) {
        // ORU: replay the occlusion along a straight line between the two real observations,
        // starting from the filter state saved when the detection was lost.
        const int gap = t.age - t.last_obs_age;
        t.kf = t.saved;  // state predicted one frame after the last observation
        const Box a = t.last_obs.box, b = obs.box;
        for (int k = 1; k < gap; ++k) {
            const float f = static_cast<float>(k) / gap;
            Box v;
            for (int i = 0; i < 4; ++i) v[i] = a[i] + (b[i] - a[i]) * f;
            t.kf.update(to_z(v));
            t.kf.predict();
        }
    }
    t.observed = true;
    t.kf.update(to_z(obs.box));
    t.last_obs = obs;
    t.has_obs = true;
    t.last_obs_age = t.age;
    t.observations[t.age] = obs;
    while (!t.observations.empty() && t.observations.begin()->first < t.age - 2 * config_.delta_t - 2)
        t.observations.erase(t.observations.begin());
    t.time_since_update = 0;
    ++t.hit_streak;
}

void OcSort::miss_track(Track& t) {
    if (t.observed) {
        t.saved = t.kf;  // already predicted to this frame
        t.observed = false;
    }
}

std::vector<DetectBox> OcSort::update(const std::vector<DetectBox>& detections) {
    ++frame_;
    std::vector<Obs> dets, low;
    std::vector<float> dets_cls;
    for (const auto& d : detections) {
        if (d.confidence > config_.det_thresh) {
            dets.push_back({{d.x1, d.y1, d.x2, d.y2}, d.confidence});
            dets_cls.push_back(d.classID);
        } else if (config_.use_byte && d.confidence > config_.low_thresh) {
            low.push_back({{d.x1, d.y1, d.x2, d.y2}, d.confidence});
        }
    }

    for (auto& t : tracks_) {
        t.kf.predict();
        ++t.age;
        if (t.time_since_update > 0) t.hit_streak = 0;
        ++t.time_since_update;
        t.predicted = to_box(t.kf.x);
    }

    const int nd = static_cast<int>(dets.size()), nt = static_cast<int>(tracks_.size());
    std::vector<char> det_done(nd, false), trk_done(nt, false);

    // First association: IoU + direction consistency (OCM).
    if (nd && nt) {
        std::vector<std::vector<float>> cost(nd, std::vector<float>(nt));
        for (int i = 0; i < nd; ++i)
            for (int j = 0; j < nt; ++j) {
                const Track& t = tracks_[j];
                const float o = iou(dets[i].box, t.predicted);
                float angle_cost = 0;
                if (t.has_obs && t.has_velocity) {
                    const Eigen::Vector2f dir = direction(previous_obs(t).box, dets[i].box);
                    const float c = std::clamp(t.velocity.dot(dir), -1.0f, 1.0f);
                    const float diff = (static_cast<float>(M_PI) / 2 - std::fabs(std::acos(c))) / static_cast<float>(M_PI);
                    angle_cost = diff * config_.inertia * dets[i].score;
                }
                cost[i][j] = o >= config_.iou_thresh ? -(o + angle_cost) : 1e5f;
            }
        for (auto [i, j] : linear_assignment(cost, nt, 1e4f).matches) {
            match_track(tracks_[j], dets[i]);
            tracks_[j].cls = dets_cls[i];
            det_done[i] = trk_done[j] = true;
        }
    }

    // Optional BYTE round: low-score detections for still-unmatched tracks.
    if (!low.empty()) {
        std::vector<int> tj;
        for (int j = 0; j < nt; ++j)
            if (!trk_done[j]) tj.push_back(j);
        std::vector<std::vector<float>> cost(tj.size(), std::vector<float>(low.size()));
        for (std::size_t a = 0; a < tj.size(); ++a)
            for (std::size_t b = 0; b < low.size(); ++b) {
                const float o = iou(tracks_[tj[a]].predicted, low[b].box);
                cost[a][b] = o >= config_.iou_thresh ? -o : 1e5f;
            }
        for (auto [a, b] : linear_assignment(cost, static_cast<int>(low.size()), 1e4f).matches) {
            match_track(tracks_[tj[a]], low[b]);
            trk_done[tj[a]] = true;
        }
    }

    // OCR: unmatched detections vs the *last observation* of unmatched tracks.
    {
        std::vector<int> di, tj;
        for (int i = 0; i < nd; ++i)
            if (!det_done[i]) di.push_back(i);
        for (int j = 0; j < nt; ++j)
            if (!trk_done[j] && tracks_[j].has_obs) tj.push_back(j);
        std::vector<std::vector<float>> cost(di.size(), std::vector<float>(tj.size()));
        for (std::size_t a = 0; a < di.size(); ++a)
            for (std::size_t b = 0; b < tj.size(); ++b) {
                const float o = iou(dets[di[a]].box, tracks_[tj[b]].last_obs.box);
                cost[a][b] = o >= config_.iou_thresh ? -o : 1e5f;
            }
        for (auto [a, b] : linear_assignment(cost, static_cast<int>(tj.size()), 1e4f).matches) {
            match_track(tracks_[tj[b]], dets[di[a]]);
            tracks_[tj[b]].cls = dets_cls[di[a]];
            det_done[di[a]] = trk_done[tj[b]] = true;
        }
    }

    for (int j = 0; j < nt; ++j)
        if (!trk_done[j]) miss_track(tracks_[j]);

    for (int i = 0; i < nd; ++i) {
        if (det_done[i]) continue;
        Track t;
        t.kf.init(to_z(dets[i].box));
        t.id = next_id_++;
        t.cls = dets_cls[i];
        t.predicted = dets[i].box;
        // Like the official code, the first detection initialises the filter but is not an
        // observation, so it does not count towards min_hits.
        tracks_.push_back(t);
    }

    std::vector<DetectBox> out;
    for (const auto& t : tracks_) {
        if (t.time_since_update >= 1 || !t.has_obs) continue;
        if (t.hit_streak < config_.min_hits && frame_ > config_.min_hits) continue;
        const Box& b = t.last_obs.box;
        out.emplace_back(b[0], b[1], b[2], b[3], t.last_obs.score, t.cls, static_cast<float>(t.id));
    }
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [&](const Track& t) { return t.time_since_update > config_.max_age; }),
                  tracks_.end());
    return out;
}

}  // namespace mot
