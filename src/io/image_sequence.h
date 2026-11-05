#ifndef PEDESTRIAN_TRACKER_IMAGE_SEQUENCE_H
#define PEDESTRIAN_TRACKER_IMAGE_SEQUENCE_H

#include <cstddef>
#include <string>
#include <vector>

#include "frame_source.h"

// Reads .jpg/.jpeg/.png files of a directory in file-name order.
class ImageSequence : public FrameSource {
public:
    explicit ImageSequence(const std::string& directory);

    bool read(cv::Mat& frame) override;
    bool finished() const override { return next_ >= files_.size(); }
    bool is_live() const override { return false; }

    std::size_t size() const { return files_.size(); }

private:
    std::vector<std::string> files_;
    std::size_t next_ = 0;
};

#endif
