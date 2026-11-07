#ifndef PEDESTRIAN_TRACKER_MOT_KALMAN_XYAH_H
#define PEDESTRIAN_TRACKER_MOT_KALMAN_XYAH_H

#include <Eigen/Dense>

#include "assignment.h"

namespace mot {

// Constant-velocity Kalman filter on (centre x, centre y, aspect w/h, height), as in
// DeepSORT / ByteTrack. State: those four plus their velocities.
class KalmanXyah {
public:
    using Vec8 = Eigen::Matrix<float, 8, 1>;
    using Mat8 = Eigen::Matrix<float, 8, 8>;
    using Vec4 = Eigen::Matrix<float, 4, 1>;

    static Vec4 to_xyah(const Box& b) {
        const float w = b[2] - b[0], h = b[3] - b[1];
        return Vec4((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, w / std::max(h, 1e-3f), h);
    }
    static Box to_box(const Vec8& m) {
        const float h = m(3), w = m(2) * h;
        return {m(0) - w / 2, m(1) - h / 2, m(0) + w / 2, m(1) + h / 2};
    }

    void initiate(const Box& box) {
        const Vec4 z = to_xyah(box);
        mean.setZero();
        mean.head<4>() = z;
        const float h = z(3);
        Vec8 std;
        std << 2 * kPos * h, 2 * kPos * h, 1e-2f, 2 * kPos * h, 10 * kVel * h, 10 * kVel * h, 1e-5f, 10 * kVel * h;
        cov = std.array().square().matrix().asDiagonal();
    }

    void predict() {
        const float h = mean(3);
        Vec8 std;
        std << kPos * h, kPos * h, 1e-2f, kPos * h, kVel * h, kVel * h, 1e-5f, kVel * h;
        Mat8 f = Mat8::Identity();
        for (int i = 0; i < 4; ++i) f(i, i + 4) = 1.0f;
        mean = f * mean;
        cov = f * cov * f.transpose() + Mat8(std.array().square().matrix().asDiagonal());
    }

    void update(const Box& box) {
        const Vec4 z = to_xyah(box);
        const float h = mean(3);
        Vec4 std(kPos * h, kPos * h, 1e-1f, kPos * h);
        const Eigen::Matrix<float, 4, 8> hm = Eigen::Matrix<float, 4, 8>::Identity();
        const Eigen::Matrix4f s = hm * cov * hm.transpose() + Eigen::Matrix4f(std.array().square().matrix().asDiagonal());
        const Eigen::Matrix<float, 8, 4> gain = cov * hm.transpose() * s.ldlt().solve(Eigen::Matrix4f::Identity());
        mean += gain * (z - hm * mean);
        cov -= gain * s * gain.transpose();
    }

    Box box() const { return to_box(mean); }

    Vec8 mean = Vec8::Zero();
    Mat8 cov = Mat8::Identity();

private:
    static constexpr float kPos = 1.0f / 20;
    static constexpr float kVel = 1.0f / 160;
};

}  // namespace mot

#endif
