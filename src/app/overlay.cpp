#include "overlay.h"

#include <algorithm>
#include <fstream>

#include <opencv2/imgproc.hpp>

namespace {

cv::Scalar track_color(int id) {
    return cv::Scalar((37 * id + 67) % 205 + 50, (17 * id + 113) % 205 + 50, (29 * id + 31) % 205 + 50);
}

}  // namespace

std::vector<std::string> load_labels(const std::string& path) {
    std::ifstream file(path);
    std::vector<std::string> labels;
    for (std::string line; std::getline(file, line);) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        labels.push_back(line);
    }
    return labels;
}

void draw_tracks(cv::Mat& frame, const std::vector<DetectBox>& tracks,
                 const std::vector<std::string>& labels, double fps) {
    const cv::Rect bounds(0, 0, frame.cols, frame.rows);
    for (const auto& track : tracks) {
        const int id = static_cast<int>(track.trackID);
        const int class_id = static_cast<int>(track.classID);
        const cv::Rect box = cv::Rect(cv::Point(static_cast<int>(track.x1), static_cast<int>(track.y1)),
                                      cv::Point(static_cast<int>(track.x2), static_cast<int>(track.y2))) &
                             bounds;
        if (box.empty()) continue;

        const cv::Scalar color = track_color(id);
        const std::string name = class_id >= 0 && class_id < static_cast<int>(labels.size())
                                     ? labels[class_id]
                                     : std::to_string(class_id);
        cv::rectangle(frame, box, color, 2);
        cv::putText(frame, name + " ID:" + std::to_string(id), cv::Point(box.x, std::max(18, box.y - 6)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv::LINE_AA);
    }
    cv::putText(frame, "FPS " + std::to_string(static_cast<int>(fps + 0.5)), cv::Point(12, 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(40, 230, 40), 2, cv::LINE_AA);
}
