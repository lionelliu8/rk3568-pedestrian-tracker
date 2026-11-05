#ifndef PEDESTRIAN_TRACKER_DETECT_BOX_H
#define PEDESTRIAN_TRACKER_DETECT_BOX_H

// One axis-aligned box in source-image pixels. Shared by the detector (trackID = -1)
// and the tracker output (trackID >= 1). classID/trackID stay float for DeepSORT compatibility.
struct DetectBox {
    DetectBox(float x1_ = 0, float y1_ = 0, float x2_ = 0, float y2_ = 0, float confidence_ = 0,
              float class_id = -1, float track_id = -1)
        : x1(x1_), y1(y1_), x2(x2_), y2(y2_), confidence(confidence_), classID(class_id),
          trackID(track_id) {}

    float width() const { return x2 - x1; }
    float height() const { return y2 - y1; }

    float x1, y1, x2, y2;
    float confidence;
    float classID;
    float trackID;
};

#endif
