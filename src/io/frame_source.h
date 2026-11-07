#ifndef PEDESTRIAN_TRACKER_FRAME_SOURCE_H
#define PEDESTRIAN_TRACKER_FRAME_SOURCE_H

#include <memory>
#include <string>

#include <opencv2/core.hpp>

// A stream of BGR frames: a live camera or a recorded image sequence.
class FrameSource {
public:
    virtual ~FrameSource() = default;

    // Returns false when no frame was produced: at the end of a sequence, or when a camera
    // wait was interrupted by a signal. finished() tells the two apart.
    virtual bool read(cv::Mat& frame) = 0;
    virtual bool finished() const { return false; }
    // Frames dropped by the source since the last call (camera frames older than the newest).
    virtual int take_skipped() { return 0; }
    virtual bool is_live() const = 0;
};

// "--input" value -> source. A directory is read as an image sequence (sorted file names,
// e.g. MOT17 img1/); anything else is opened as a V4L2 camera (index or /dev/video*).
std::unique_ptr<FrameSource> open_frame_source(const std::string& input, int camera_width,
                                               int camera_height);

#endif
