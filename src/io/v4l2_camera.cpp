#include "v4l2_camera.h"

#include <cerrno>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <unistd.h>

#include <opencv2/imgproc.hpp>

#include "posix_util.h"

namespace {

bool is_index(const std::string& input) {
    return !input.empty() && input.find_first_not_of("0123456789") == std::string::npos;
}

std::string fourcc(uint32_t value) {
    const char text[5] = {static_cast<char>(value), static_cast<char>(value >> 8),
                          static_cast<char>(value >> 16), static_cast<char>(value >> 24), 0};
    return text;
}

}  // namespace

V4l2Camera::V4l2Camera(const std::string& input, int requested_width, int requested_height) {
    try {
        device_ = is_index(input) ? "/dev/video" + input : input;
        if (device_.rfind("/dev/video", 0) != 0) {
            throw std::runtime_error("only V4L2 camera input is supported: " + input);
        }

        fd_ = open(device_.c_str(), O_RDWR | O_NONBLOCK);
        if (fd_ < 0) throw std::runtime_error("cannot open " + device_ + ": " + std::strerror(errno));

        v4l2_capability capability{};
        if (xioctl(fd_, VIDIOC_QUERYCAP, &capability) < 0) {
            throw std::runtime_error(device_ + " is not a streaming capture device");
        }
        const uint32_t capabilities = capability.capabilities & V4L2_CAP_DEVICE_CAPS
                                          ? capability.device_caps
                                          : capability.capabilities;
        if (capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE) {
            buffer_type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        } else if (capabilities & V4L2_CAP_VIDEO_CAPTURE) {
            buffer_type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        } else {
            throw std::runtime_error(device_ + " has no video capture capability");
        }
        if (!(capabilities & V4L2_CAP_STREAMING)) {
            throw std::runtime_error(device_ + " has no streaming capability");
        }

        v4l2_format format{};
        format.type = buffer_type_;
        if (buffer_type_ == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) {
            format.fmt.pix_mp.width = requested_width;
            format.fmt.pix_mp.height = requested_height;
            format.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_NV12;
            format.fmt.pix_mp.field = V4L2_FIELD_ANY;
            format.fmt.pix_mp.num_planes = 1;
        } else {
            format.fmt.pix.width = requested_width;
            format.fmt.pix.height = requested_height;
            format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
            format.fmt.pix.field = V4L2_FIELD_ANY;
        }
        if (xioctl(fd_, VIDIOC_S_FMT, &format) < 0) {
            throw std::runtime_error("cannot configure " + device_ + ": " + std::strerror(errno));
        }
        if (buffer_type_ == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) {
            width_ = format.fmt.pix_mp.width;
            height_ = format.fmt.pix_mp.height;
            pixel_format_ = format.fmt.pix_mp.pixelformat;
            bytes_per_line_ = format.fmt.pix_mp.plane_fmt[0].bytesperline;
        } else {
            width_ = format.fmt.pix.width;
            height_ = format.fmt.pix.height;
            pixel_format_ = format.fmt.pix.pixelformat;
            bytes_per_line_ = format.fmt.pix.bytesperline;
        }
        if (pixel_format_ != V4L2_PIX_FMT_YUYV && pixel_format_ != V4L2_PIX_FMT_NV12 &&
            pixel_format_ != V4L2_PIX_FMT_NV21 && pixel_format_ != V4L2_PIX_FMT_BGR24 &&
            pixel_format_ != V4L2_PIX_FMT_RGB24) {
            throw std::runtime_error(device_ + " returned unsupported format " + fourcc(pixel_format_) +
                                     "; configure the camera for YUYV or NV12");
        }

        v4l2_requestbuffers request{};
        request.count = 4;
        request.type = buffer_type_;
        request.memory = V4L2_MEMORY_MMAP;
        if (xioctl(fd_, VIDIOC_REQBUFS, &request) < 0 || request.count < 2) {
            throw std::runtime_error("cannot allocate V4L2 buffers");
        }

        buffers_.resize(request.count);
        for (uint32_t i = 0; i < request.count; ++i) {
            v4l2_buffer buffer{};
            v4l2_plane planes[VIDEO_MAX_PLANES]{};
            buffer.type = buffer_type_;
            buffer.memory = V4L2_MEMORY_MMAP;
            buffer.index = i;
            if (buffer_type_ == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) {
                buffer.m.planes = planes;
                buffer.length = VIDEO_MAX_PLANES;
            }
            if (xioctl(fd_, VIDIOC_QUERYBUF, &buffer) < 0) {
                throw std::runtime_error("cannot query V4L2 buffer");
            }
            const size_t length = buffer_type_ == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
                                      ? planes[0].length
                                      : buffer.length;
            const off_t offset = buffer_type_ == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
                                     ? planes[0].m.mem_offset
                                     : buffer.m.offset;
            buffers_[i].length = length;
            buffers_[i].data =
                mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, offset);
            if (buffers_[i].data == MAP_FAILED) throw std::runtime_error("cannot map V4L2 buffer");
            if (xioctl(fd_, VIDIOC_QBUF, &buffer) < 0) {
                throw std::runtime_error("cannot queue V4L2 buffer");
            }
        }

        v4l2_buf_type type = buffer_type_;
        if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
            throw std::runtime_error("cannot start V4L2 camera");
        }
        streaming_ = true;
        std::cout << "camera " << device_ << ": " << width_ << 'x' << height_ << ' '
                  << fourcc(pixel_format_) << '\n';
    } catch (...) {
        release();
        throw;
    }
}

V4l2Camera::~V4l2Camera() { release(); }

void V4l2Camera::release() {
    if (streaming_) {
        v4l2_buf_type type = buffer_type_;
        xioctl(fd_, VIDIOC_STREAMOFF, &type);
    }
    for (const auto& buffer : buffers_) {
        if (buffer.data != MAP_FAILED) munmap(buffer.data, buffer.length);
    }
    if (fd_ >= 0) close(fd_);
}

bool V4l2Camera::read(cv::Mat& frame) {
    for (;;) {
        fd_set descriptors;
        FD_ZERO(&descriptors);
        FD_SET(fd_, &descriptors);
        timeval timeout{2, 0};
        const int ready = select(fd_ + 1, &descriptors, nullptr, nullptr, &timeout);
        if (ready == 0) throw std::runtime_error("camera frame timeout");
        if (ready < 0) {
            if (errno == EINTR) return false;  // signal: let the caller decide
            throw std::runtime_error("camera wait failed: " + std::string(std::strerror(errno)));
        }

        v4l2_buffer buffer{};
        v4l2_plane planes[VIDEO_MAX_PLANES]{};
        buffer.type = buffer_type_;
        buffer.memory = V4L2_MEMORY_MMAP;
        if (buffer_type_ == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE) {
            buffer.m.planes = planes;
            buffer.length = VIDEO_MAX_PLANES;
        }
        if (xioctl(fd_, VIDIOC_DQBUF, &buffer) < 0) {
            if (errno == EAGAIN) continue;
            throw std::runtime_error("cannot dequeue camera frame");
        }
        convert(buffers_.at(buffer.index).data, frame);
        if (xioctl(fd_, VIDIOC_QBUF, &buffer) < 0) {
            throw std::runtime_error("cannot requeue camera frame");
        }
        return true;
    }
}

void V4l2Camera::convert(void* data, cv::Mat& frame) const {
    if (pixel_format_ == V4L2_PIX_FMT_YUYV) {
        const cv::Mat source(height_, width_, CV_8UC2, data, bytes_per_line_);
        cv::cvtColor(source, frame, cv::COLOR_YUV2BGR_YUY2);
    } else if (pixel_format_ == V4L2_PIX_FMT_NV12 || pixel_format_ == V4L2_PIX_FMT_NV21) {
        const cv::Mat source(height_ * 3 / 2, width_, CV_8UC1, data, bytes_per_line_);
        cv::cvtColor(source, frame, pixel_format_ == V4L2_PIX_FMT_NV12
                                        ? cv::COLOR_YUV2BGR_NV12
                                        : cv::COLOR_YUV2BGR_NV21);
    } else {
        const cv::Mat source(height_, width_, CV_8UC3, data, bytes_per_line_);
        if (pixel_format_ == V4L2_PIX_FMT_RGB24) cv::cvtColor(source, frame, cv::COLOR_RGB2BGR);
        else source.copyTo(frame);
    }
}
