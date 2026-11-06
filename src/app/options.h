#ifndef PEDESTRIAN_TRACKER_OPTIONS_H
#define PEDESTRIAN_TRACKER_OPTIONS_H

#include <string>

struct Options {
    std::string input = "/dev/video-camera0";
    std::string yolo_model;   // default: <exe dir>/model/yolo26n_person_rk3568.rknn
    std::string reid_model;   // default: <exe dir>/model/osnet_x0_25_market_rk3568.rknn
    std::string labels;       // default: <exe dir>/model/person_labels.txt
    std::string mot_output;   // tracks in MOTChallenge format (optional)
    std::string det_output;   // raw detections in MOTChallenge format (optional)
    int class_filter = 0;     // COCO 0 = person; -1 = all classes
    int detect_interval = 1;  // run the detector every n frames, Kalman-predict in between
    int camera_width = 1280;
    int camera_height = 720;
    int max_frames = 0;       // 0 = unlimited
    float confidence_threshold = 0.25f;
    float nms_threshold = 0.7f;
    bool use_rga = true;
    bool display = false;
};

void print_usage(const char* program);

// Throws std::invalid_argument on bad input. "--help" prints usage and exits.
Options parse_options(int argc, char** argv);

#endif
