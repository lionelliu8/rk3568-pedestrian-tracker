#include "drm_display.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <opencv2/imgproc.hpp>

#include <cstdlib>
#include <fstream>
#include <string>

#include <im2d.h>
#include <rga.h>

#include "posix_util.h"

namespace {

uint64_t to_u64(const void* value) { return reinterpret_cast<uintptr_t>(value); }

}  // namespace

DrmDisplay::DrmDisplay(const char* device) {
    try {
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
        resources.fb_id_ptr = to_u64(framebuffer_ids.data());
        resources.crtc_id_ptr = to_u64(crtc_ids.data());
        resources.connector_id_ptr = to_u64(connector_ids.data());
        resources.encoder_id_ptr = to_u64(encoder_ids.data());
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
            connector.modes_ptr = to_u64(modes.data());
            connector.props_ptr = to_u64(props.data());
            connector.prop_values_ptr = to_u64(prop_values.data());
            connector.encoders_ptr = to_u64(encoders.data());
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

        drm_prime_handle prime{};
        prime.handle = dumb_.handle;
        prime.flags = DRM_CLOEXEC | DRM_RDWR;
        if (xioctl(fd_, DRM_IOCTL_PRIME_HANDLE_TO_FD, &prime) == 0) {
            dmabuf_fd_ = prime.fd;
            use_rga_ = std::getenv("DISPLAY_RGA") == nullptr || std::string(std::getenv("DISPLAY_RGA")) != "0";
        }

        drm_mode_crtc crtc{};
        crtc.set_connectors_ptr = to_u64(&connector_id_);
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
                  << " connector=" << connector_id_ << (use_rga_ ? ", RGA" : ", CPU") << "\n";
    } catch (...) {
        release();
        throw;
    }
}

DrmDisplay::~DrmDisplay() { release(); }

void DrmDisplay::release() {
    if (saved_crtc_valid_) {
        saved_crtc_.set_connectors_ptr = to_u64(&connector_id_);
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
    if (dmabuf_fd_ >= 0) close(dmabuf_fd_);
    if (fd_ >= 0) close(fd_);
}

void DrmDisplay::show(const cv::Mat& frame) {
    // Portrait panel and landscape frame: rotate 90 degrees clockwise. target_w/h is the screen
    // size seen in the frame's orientation; the frame is scaled to fill it (centre crop).
    const bool rotate = mode_.hdisplay < mode_.vdisplay && frame.cols > frame.rows;
    const int target_w = rotate ? mode_.vdisplay : mode_.hdisplay;
    const int target_h = rotate ? mode_.hdisplay : mode_.vdisplay;
    if (!(use_rga_ && show_rga(frame, rotate, target_w, target_h))) show_cpu(frame, rotate, target_w, target_h);
    dump_once();
}

bool DrmDisplay::show_rga(const cv::Mat& frame, bool rotate, int target_w, int target_h) {
    if (!frame.isContinuous() || frame.cols % 16 != 0) return false;  // RGA RGB888 stride alignment
    const double scale = std::max(static_cast<double>(target_w) / frame.cols,
                                  static_cast<double>(target_h) / frame.rows);
    const int crop_w = std::min(frame.cols, static_cast<int>(target_w / scale)) & ~1;
    const int crop_h = std::min(frame.rows, static_cast<int>(target_h / scale)) & ~1;
    const im_rect src_rect{((frame.cols - crop_w) / 2) & ~1, ((frame.rows - crop_h) / 2) & ~1, crop_w, crop_h};
    const im_rect dst_rect{0, 0, static_cast<int>(mode_.hdisplay), static_cast<int>(mode_.vdisplay)};
    rga_buffer_t src = wrapbuffer_virtualaddr_t(frame.data, frame.cols, frame.rows, frame.cols, frame.rows,
                                                RK_FORMAT_BGR_888);
    // XRGB8888 scan-out = bytes B, G, R, X = RGA BGRA_8888.
    rga_buffer_t dst = wrapbuffer_fd_t(dmabuf_fd_, mode_.hdisplay, mode_.vdisplay,
                                       static_cast<int>(dumb_.pitch / 4), mode_.vdisplay, RK_FORMAT_BGRA_8888);
    rga_buffer_t pat{};
    const im_rect empty{};
    const int usage = (rotate ? IM_HAL_TRANSFORM_ROT_90 : 0) | IM_SYNC;
    const IM_STATUS status = improcess(src, dst, pat, src_rect, dst_rect, empty, usage);
    if (status == IM_STATUS_SUCCESS) return true;
    std::cerr << "display: RGA failed (" << imStrError_t(status) << "), using the CPU from now on\n";
    use_rga_ = false;
    return false;
}

void DrmDisplay::show_cpu(const cv::Mat& frame, bool rotate, int target_w, int target_h) {
    // Scale down first so the remaining steps run on screen-sized images, then rotate and
    // convert BGR -> XRGB straight into the scan-out buffer.
    const double scale = std::max(static_cast<double>(target_w) / frame.cols,
                                  static_cast<double>(target_h) / frame.rows);
    const int width = std::max(target_w, static_cast<int>(std::ceil(frame.cols * scale)));
    const int height = std::max(target_h, static_cast<int>(std::ceil(frame.rows * scale)));
    cv::resize(frame, scaled_, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    const cv::Mat cropped = scaled_(cv::Rect((width - target_w) / 2, (height - target_h) / 2, target_w, target_h));
    cv::Mat screen(mode_.vdisplay, mode_.hdisplay, CV_8UC4, memory_, dumb_.pitch);
    if (rotate) {
        cv::rotate(cropped, rotated_, cv::ROTATE_90_CLOCKWISE);
        cv::cvtColor(rotated_, screen, cv::COLOR_BGR2BGRA);
    } else {
        cv::cvtColor(cropped, screen, cv::COLOR_BGR2BGRA);
    }
}

void DrmDisplay::dump_once() {
    const char* path = std::getenv("DRM_DUMP");
    if (dumped_ || !path) return;
    dumped_ = true;
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << mode_.hdisplay << ' ' << mode_.vdisplay << "\n255\n";
    for (int y = 0; y < mode_.vdisplay; ++y) {
        const uint8_t* row = memory_ + y * dumb_.pitch;
        for (int x = 0; x < mode_.hdisplay; ++x) out.put(row[4 * x + 2]).put(row[4 * x + 1]).put(row[4 * x]);
    }
}

