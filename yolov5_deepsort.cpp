#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include <drm/drm.h>
#include <drm/drm_mode.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <unistd.h>

#include "deepsort.h"
#include "detector.h"

namespace fs = std::filesystem;

namespace {
std::atomic<bool> running{true};

struct Options {
    std::string input = "0";
    std::string yolo_model;
    std::string reid_model;
    std::string labels;
    int class_filter = 0;
    int detect_interval = 1;
    int camera_width = 1280;
    int camera_height = 720;
    int max_frames = 0;
    float confidence_threshold = 0.15f;
    bool display = false;
};

void on_signal(int) {
    running = false;
}

void usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --input <index|device>     V4L2 camera index or /dev/videoX (default: 0)\n"
        << "  --model <file>             RK3568 YOLOv5 RKNN model\n"
        << "  --reid-model <file>        RK3568 OSNet RKNN model\n"
        << "  --labels <file>            COCO label file\n"
        << "  --class <id>               COCO class to track; -1 tracks all (default: 0 person)\n"
        << "  --detect-interval <n>      Run YOLO every n frames (default: 1)\n"
        << "  --camera-width <pixels>    Requested USB camera width\n"
        << "  --camera-height <pixels>   Requested USB camera height\n"
        << "  --max-frames <n>           Stop after n frames; 0 means unlimited\n"
        << "  --conf-threshold <0..1>    Detection confidence (default: 0.15)\n"
        << "  --display                  Show directly on DRM DSI screen\n"
        << "  --help                     Show this help\n";
}

int parse_int(const std::string& value, const std::string& option) {
    int result = 0;
    const char* begin = value.data();
    const char* end = begin + value.size();
    const auto parsed = std::from_chars(begin, end, result);
    if (parsed.ec != std::errc() || parsed.ptr != end) {
        throw std::runtime_error("invalid value for " + option + ": " + value);
    }
    return result;
}

float parse_float(const std::string& value, const std::string& option) {
    std::size_t parsed = 0;
    const float result = std::stof(value, &parsed);
    if (parsed != value.size()) throw std::runtime_error("invalid value for " + option + ": " + value);
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    const fs::path executable_dir = fs::absolute(argv[0]).parent_path();
    options.yolo_model = (executable_dir / "model/yolov5_rk3568.rknn").string();
    options.reid_model = (executable_dir / "model/osnet_x0_25_market_rk3568.rknn").string();
    options.labels = (executable_dir / "model/coco_80_labels_list.txt").string();

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) {
                throw std::runtime_error("missing value after " + arg);
            }
            return argv[i];
        };
        if (arg == "--input") options.input = next();
        else if (arg == "--model") options.yolo_model = next();
        else if (arg == "--reid-model") options.reid_model = next();
        else if (arg == "--labels") options.labels = next();
        else if (arg == "--class") options.class_filter = parse_int(next(), arg);
        else if (arg == "--detect-interval") options.detect_interval = parse_int(next(), arg);
        else if (arg == "--camera-width") options.camera_width = parse_int(next(), arg);
        else if (arg == "--camera-height") options.camera_height = parse_int(next(), arg);
        else if (arg == "--max-frames") options.max_frames = parse_int(next(), arg);
        else if (arg == "--conf-threshold") options.confidence_threshold = parse_float(next(), arg);
        else if (arg == "--display") options.display = true;
        else if (arg == "--help") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    if (options.detect_interval < 1 || options.max_frames < 0 || options.camera_width < 1 ||
        options.camera_height < 1 || options.confidence_threshold <= 0.0f ||
        options.confidence_threshold > 1.0f) {
        throw std::runtime_error("invalid numeric option");
    }
    return options;
}

bool is_camera_index(const std::string& input) {
    return !input.empty() && input.find_first_not_of("0123456789") == std::string::npos;
}

int xioctl(int fd, unsigned long request, void* argument) {
    int result;
    do {
        result = ioctl(fd, request, argument);
    } while (result == -1 && errno == EINTR);
    return result;
}

std::string fourcc(uint32_t value) {
    char text[5] = {static_cast<char>(value), static_cast<char>(value >> 8),
                    static_cast<char>(value >> 16), static_cast<char>(value >> 24), 0};
    return text;
}

class V4l2Camera {
  public:
    V4l2Camera(const std::string& input, int requested_width, int requested_height) {
        const std::string device = is_camera_index(input) ? "/dev/video" + input : input;
        if (device.rfind("/dev/video", 0) != 0) {
            throw std::runtime_error("only V4L2 camera input is supported: " + input);
        }

        fd_ = open(device.c_str(), O_RDWR | O_NONBLOCK);
        if (fd_ < 0) throw std::runtime_error("cannot open " + device + ": " + std::strerror(errno));

        v4l2_capability capability{};
        if (xioctl(fd_, VIDIOC_QUERYCAP, &capability) < 0) {
            throw std::runtime_error(device + " is not a streaming capture device");
        }
        const uint32_t capabilities = capability.capabilities & V4L2_CAP_DEVICE_CAPS
                                          ? capability.device_caps
                                          : capability.capabilities;
        if (capabilities & V4L2_CAP_VIDEO_CAPTURE_MPLANE) {
            buffer_type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
        } else if (capabilities & V4L2_CAP_VIDEO_CAPTURE) {
            buffer_type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        } else {
            throw std::runtime_error(device + " has no video capture capability");
        }
        if (!(capabilities & V4L2_CAP_STREAMING)) {
            throw std::runtime_error(device + " has no streaming capability");
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
            throw std::runtime_error("cannot configure " + device + ": " + std::strerror(errno));
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
            throw std::runtime_error(device + " returned unsupported format " + fourcc(pixel_format_) +
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
        std::cout << "camera " << device << ": " << width_ << 'x' << height_ << ' '
                  << fourcc(pixel_format_) << '\n';
    }

    ~V4l2Camera() {
        if (streaming_) {
            v4l2_buf_type type = buffer_type_;
            xioctl(fd_, VIDIOC_STREAMOFF, &type);
        }
        for (const auto& buffer : buffers_) {
            if (buffer.data != MAP_FAILED) munmap(buffer.data, buffer.length);
        }
        if (fd_ >= 0) close(fd_);
    }

    bool read(cv::Mat& frame) {
        while (running) {
            fd_set descriptors;
            FD_ZERO(&descriptors);
            FD_SET(fd_, &descriptors);
            timeval timeout{2, 0};
            const int ready = select(fd_ + 1, &descriptors, nullptr, nullptr, &timeout);
            if (ready == 0) throw std::runtime_error("camera frame timeout");
            if (ready < 0) {
                if (errno == EINTR) continue;
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
        return false;
    }

  private:
    struct Buffer {
        void* data = MAP_FAILED;
        size_t length = 0;
    };

    void convert(void* data, cv::Mat& frame) const {
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

    int fd_ = -1;
    int width_ = 0;
    int height_ = 0;
    size_t bytes_per_line_ = 0;
    uint32_t pixel_format_ = 0;
    v4l2_buf_type buffer_type_ = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    bool streaming_ = false;
    std::vector<Buffer> buffers_;
};

class DrmDisplay {
  public:
    explicit DrmDisplay(const char* device = "/dev/dri/card0") {
        fd_ = open(device, O_RDWR);
        if (fd_ < 0) throw std::runtime_error("cannot open " + std::string(device));

        drm_mode_card_res resources{};
        if (xioctl(fd_, DRM_IOCTL_MODE_GETRESOURCES, &resources) < 0) {
            throw std::runtime_error("cannot query DRM resources");
        }
        std::vector<uint32_t> framebuffer_ids(resources.count_fbs);
        std::vector<uint32_t> crtc_ids(resources.count_crtcs);
        std::vector<uint32_t> connector_ids(resources.count_connectors);
        std::vector<uint32_t> encoder_ids(resources.count_encoders);
        resources.fb_id_ptr = pointer(framebuffer_ids.data());
        resources.crtc_id_ptr = pointer(crtc_ids.data());
        resources.connector_id_ptr = pointer(connector_ids.data());
        resources.encoder_id_ptr = pointer(encoder_ids.data());
        if (xioctl(fd_, DRM_IOCTL_MODE_GETRESOURCES, &resources) < 0) {
            throw std::runtime_error("cannot read DRM resources");
        }

        uint32_t encoder_id = 0;
        for (const uint32_t id : connector_ids) {
            drm_mode_get_connector connector{};
            connector.connector_id = id;
            if (xioctl(fd_, DRM_IOCTL_MODE_GETCONNECTOR, &connector) < 0) continue;
            std::vector<drm_mode_modeinfo> modes(connector.count_modes);
            std::vector<uint32_t> props(connector.count_props);
            std::vector<uint64_t> prop_values(connector.count_props);
            std::vector<uint32_t> encoders(connector.count_encoders);
            connector.modes_ptr = pointer(modes.data());
            connector.props_ptr = pointer(props.data());
            connector.prop_values_ptr = pointer(prop_values.data());
            connector.encoders_ptr = pointer(encoders.data());
            if (xioctl(fd_, DRM_IOCTL_MODE_GETCONNECTOR, &connector) < 0 ||
                connector.connection != 1 || modes.empty()) {
                continue;
            }
            connector_id_ = id;
            encoder_id = connector.encoder_id ? connector.encoder_id : encoders.front();
            mode_ = modes.front();
            for (const auto& mode : modes) {
                if (mode.type & DRM_MODE_TYPE_PREFERRED) mode_ = mode;
            }
            break;
        }
        if (!connector_id_) throw std::runtime_error("no connected DRM display");

        drm_mode_get_encoder encoder{};
        encoder.encoder_id = encoder_id;
        if (xioctl(fd_, DRM_IOCTL_MODE_GETENCODER, &encoder) < 0) {
            throw std::runtime_error("cannot query DRM encoder");
        }
        crtc_id_ = encoder.crtc_id;
        if (!crtc_id_) {
            for (size_t i = 0; i < crtc_ids.size(); ++i) {
                if (encoder.possible_crtcs & (1u << i)) {
                    crtc_id_ = crtc_ids[i];
                    break;
                }
            }
        }
        if (!crtc_id_) throw std::runtime_error("no compatible DRM CRTC");

        saved_crtc_.crtc_id = crtc_id_;
        saved_crtc_valid_ = xioctl(fd_, DRM_IOCTL_MODE_GETCRTC, &saved_crtc_) == 0;

        dumb_.width = mode_.hdisplay;
        dumb_.height = mode_.vdisplay;
        dumb_.bpp = 32;
        if (xioctl(fd_, DRM_IOCTL_MODE_CREATE_DUMB, &dumb_) < 0) {
            throw std::runtime_error("cannot create DRM buffer");
        }
        drm_mode_fb_cmd framebuffer{};
        framebuffer.width = dumb_.width;
        framebuffer.height = dumb_.height;
        framebuffer.pitch = dumb_.pitch;
        framebuffer.bpp = 32;
        framebuffer.depth = 24;
        framebuffer.handle = dumb_.handle;
        if (xioctl(fd_, DRM_IOCTL_MODE_ADDFB, &framebuffer) < 0) {
            throw std::runtime_error("cannot create DRM framebuffer");
        }
        framebuffer_id_ = framebuffer.fb_id;

        drm_mode_map_dumb mapping{};
        mapping.handle = dumb_.handle;
        if (xioctl(fd_, DRM_IOCTL_MODE_MAP_DUMB, &mapping) < 0) {
            throw std::runtime_error("cannot map DRM buffer");
        }
        memory_ = static_cast<uint8_t*>(
            mmap(nullptr, dumb_.size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, mapping.offset));
        if (memory_ == MAP_FAILED) throw std::runtime_error("cannot mmap DRM buffer");
        std::memset(memory_, 0, dumb_.size);

        drm_mode_crtc crtc{};
        crtc.set_connectors_ptr = pointer(&connector_id_);
        crtc.count_connectors = 1;
        crtc.crtc_id = crtc_id_;
        crtc.fb_id = framebuffer_id_;
        crtc.mode_valid = 1;
        crtc.mode = mode_;
        if (xioctl(fd_, DRM_IOCTL_MODE_SETCRTC, &crtc) < 0) {
            throw std::runtime_error("cannot activate DRM display: " +
                                     std::string(std::strerror(errno)));
        }
        std::cout << "DRM display " << device << ": " << mode_.hdisplay << 'x' << mode_.vdisplay
                  << " connector=" << connector_id_ << "\n";
    }

    ~DrmDisplay() {
        if (saved_crtc_valid_) {
            saved_crtc_.set_connectors_ptr = pointer(&connector_id_);
            saved_crtc_.count_connectors = 1;
            xioctl(fd_, DRM_IOCTL_MODE_SETCRTC, &saved_crtc_);
        }
        if (memory_ != MAP_FAILED) munmap(memory_, dumb_.size);
        if (framebuffer_id_) xioctl(fd_, DRM_IOCTL_MODE_RMFB, &framebuffer_id_);
        if (dumb_.handle) {
            drm_mode_destroy_dumb destroy{};
            destroy.handle = dumb_.handle;
            xioctl(fd_, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy);
        }
        if (fd_ >= 0) close(fd_);
    }

    void show(const cv::Mat& frame) {
        cv::Mat oriented;
        if (mode_.hdisplay < mode_.vdisplay && frame.cols > frame.rows) {
            cv::rotate(frame, oriented, cv::ROTATE_90_CLOCKWISE);
        } else {
            oriented = frame;
        }
        const double scale = std::max(static_cast<double>(mode_.hdisplay) / oriented.cols,
                                      static_cast<double>(mode_.vdisplay) / oriented.rows);
        const int width = std::max(static_cast<int>(mode_.hdisplay),
                                   static_cast<int>(std::ceil(oriented.cols * scale)));
        const int height = std::max(static_cast<int>(mode_.vdisplay),
                                    static_cast<int>(std::ceil(oriented.rows * scale)));
        const int crop_left = (width - static_cast<int>(mode_.hdisplay)) / 2;
        const int crop_top = (height - static_cast<int>(mode_.vdisplay)) / 2;
        cv::Mat resized;
        cv::resize(oriented, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);

        for (int y = 0; y < mode_.vdisplay; ++y) {
            uint32_t* target = reinterpret_cast<uint32_t*>(memory_ + y * dumb_.pitch);
            const cv::Vec3b* source = resized.ptr<cv::Vec3b>(y + crop_top) + crop_left;
            for (int x = 0; x < mode_.hdisplay; ++x) {
                target[x] = static_cast<uint32_t>(source[x][0]) |
                            (static_cast<uint32_t>(source[x][1]) << 8) |
                            (static_cast<uint32_t>(source[x][2]) << 16);
            }
        }
    }

  private:
    static uint64_t pointer(const void* value) {
        return reinterpret_cast<uintptr_t>(value);
    }

    int fd_ = -1;
    uint32_t connector_id_ = 0;
    uint32_t crtc_id_ = 0;
    uint32_t framebuffer_id_ = 0;
    drm_mode_modeinfo mode_{};
    drm_mode_crtc saved_crtc_{};
    drm_mode_create_dumb dumb_{};
    bool saved_crtc_valid_ = false;
    uint8_t* memory_ = reinterpret_cast<uint8_t*>(MAP_FAILED);
};

std::vector<std::string> load_labels(const std::string& path) {
    std::ifstream file(path);
    std::vector<std::string> labels;
    std::string label;
    while (std::getline(file, label)) {
        labels.push_back(label);
    }
    return labels;
}

cv::Scalar track_color(int id) {
    return cv::Scalar((37 * id + 67) % 205 + 50, (17 * id + 113) % 205 + 50,
                      (29 * id + 31) % 205 + 50);
}

void draw_tracks(cv::Mat& frame, const std::vector<DetectBox>& tracks,
                 const std::vector<std::string>& labels, double fps) {
    for (const auto& track : tracks) {
        const int id = static_cast<int>(track.trackID);
        const int class_id = static_cast<int>(track.classID);
        const cv::Rect bounds(0, 0, frame.cols, frame.rows);
        cv::Rect box(cv::Point(static_cast<int>(track.x1), static_cast<int>(track.y1)),
                     cv::Point(static_cast<int>(track.x2), static_cast<int>(track.y2)));
        box &= bounds;
        if (box.empty()) continue;

        const cv::Scalar color = track_color(id);
        cv::rectangle(frame, box, color, 2);
        const std::string class_name = class_id >= 0 && class_id < static_cast<int>(labels.size())
                                           ? labels[class_id]
                                           : std::to_string(class_id);
        const std::string text = class_name + " ID:" + std::to_string(id);
        cv::putText(frame, text, cv::Point(box.x, std::max(18, box.y - 6)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv::LINE_AA);
    }
    cv::putText(frame, "FPS " + std::to_string(static_cast<int>(fps + 0.5)), cv::Point(12, 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(40, 230, 40), 2, cv::LINE_AA);
}
}

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);

        V4l2Camera capture(options.input, options.camera_width, options.camera_height);
        cv::Mat frame;
        if (!capture.read(frame) || frame.empty()) throw std::runtime_error("camera produced no frames");

        RknnYolov5Detector detector(options.yolo_model);
        DeepSort tracker(options.reid_model, options.detect_interval);
        const auto labels = load_labels(options.labels);

        std::unique_ptr<DrmDisplay> display;
        if (options.display) {
            if (std::system("/etc/init.d/S50QDesktop stop >/dev/null 2>&1; "
                            "/etc/init.d/S49weston stop >/dev/null 2>&1") == -1) {
                throw std::runtime_error("cannot stop the desktop before DRM takeover");
            }
            sleep(1);
            display = std::make_unique<DrmDisplay>();
        }

        int frame_index = 0;
        int measured_frames = 0;
        double measured_fps = 0.0;
        double detector_ms = 0.0;
        double tracker_ms = 0.0;
        double display_ms = 0.0;
        auto measure_start = std::chrono::steady_clock::now();

        do {
            std::vector<DetectBox> tracks;
            const auto detector_start = std::chrono::steady_clock::now();
            if (frame_index % options.detect_interval == 0) {
                tracks = detector.infer(frame, options.class_filter, options.confidence_threshold);
                const auto tracker_start = std::chrono::steady_clock::now();
                tracker.sort(frame, tracks);
                tracker_ms += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - tracker_start).count();
            } else {
                tracker.sort_interval(tracks);
            }
            detector_ms += std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - detector_start).count();

            ++frame_index;
            ++measured_frames;
            const auto now = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(now - measure_start).count();
            if (elapsed >= 1.0) {
                measured_fps = measured_frames / elapsed;
                std::cout << "frames=" << frame_index << " tracks=" << tracks.size()
                          << " fps=" << std::fixed << std::setprecision(1) << measured_fps
                          << " detector+tracker=" << detector_ms / measured_frames
                          << " tracker=" << tracker_ms / measured_frames
                          << " display=" << display_ms / measured_frames << " ms\n";
                measured_frames = 0;
                detector_ms = tracker_ms = display_ms = 0.0;
                measure_start = now;
            }

            draw_tracks(frame, tracks, labels, measured_fps);
            if (display) {
                const auto display_start = std::chrono::steady_clock::now();
                display->show(frame);
                display_ms += std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - display_start).count();
            }
            if (options.max_frames > 0 && frame_index >= options.max_frames) break;
        } while (running && capture.read(frame) && !frame.empty());

        std::cout << "finished after " << frame_index << " frames\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        usage(argv[0]);
        return 1;
    }
}
