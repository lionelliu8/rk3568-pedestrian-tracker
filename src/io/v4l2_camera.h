#ifndef PEDESTRIAN_TRACKER_V4L2_CAMERA_H
#define PEDESTRIAN_TRACKER_V4L2_CAMERA_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <linux/videodev2.h>
#include <sys/mman.h>

#include <opencv2/core.hpp>

#include "frame_source.h"

// Minimal V4L2 mmap capture (single- or multi-planar) that delivers BGR frames.
// Supported pixel formats: YUYV, NV12, NV21, BGR24, RGB24.
class V4l2Camera : public FrameSource {
public:
    // input: camera index ("0") or device path ("/dev/video-camera0").
    V4l2Camera(const std::string& input, int requested_width, int requested_height);
    ~V4l2Camera();

    V4l2Camera(const V4l2Camera&) = delete;
    V4l2Camera& operator=(const V4l2Camera&) = delete;

    // Blocks for the next frame. Returns false if interrupted by a signal; throws on errors
    // or after 2 s without a frame.
    bool read(cv::Mat& frame) override;
    bool is_live() const override { return true; }

    int width() const { return width_; }
    int height() const { return height_; }

private:
    struct Buffer {
        void* data = MAP_FAILED;
        std::size_t length = 0;
    };

    void convert(void* data, cv::Mat& frame) const;
    void release();

    std::string device_;
    int fd_ = -1;
    int width_ = 0;
    int height_ = 0;
    std::size_t bytes_per_line_ = 0;
    uint32_t pixel_format_ = 0;
    v4l2_buf_type buffer_type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    bool streaming_ = false;
    std::vector<Buffer> buffers_;
};

#endif
