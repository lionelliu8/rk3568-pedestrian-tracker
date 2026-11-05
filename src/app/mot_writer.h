#ifndef PEDESTRIAN_TRACKER_MOT_WRITER_H
#define PEDESTRIAN_TRACKER_MOT_WRITER_H

#include <fstream>
#include <string>
#include <vector>

#include "detect_box.h"

// Writes boxes in MOTChallenge text format: frame,id,x,y,w,h,conf,-1,-1,-1 (frame is 1-based).
// Tracks carry their track ID; raw detections are written with id -1.
class MotWriter {
public:
    explicit MotWriter(const std::string& path);
    void write(int frame, const std::vector<DetectBox>& boxes);

private:
    std::ofstream file_;
};

#endif
