#ifndef PEDESTRIAN_TRACKER_OVERLAY_H
#define PEDESTRIAN_TRACKER_OVERLAY_H

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "detect_box.h"

// One label per line; returns an empty list if the file is missing.
std::vector<std::string> load_labels(const std::string& path);

// Draws track boxes with "<class> ID:<n>" captions and an FPS counter.
void draw_tracks(cv::Mat& frame, const std::vector<DetectBox>& tracks,
                 const std::vector<std::string>& labels, double fps);

#endif
