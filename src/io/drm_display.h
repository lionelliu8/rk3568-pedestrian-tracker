#ifndef PEDESTRIAN_TRACKER_DRM_DISPLAY_H
#define PEDESTRIAN_TRACKER_DRM_DISPLAY_H

#include <cstdint>

#include <drm/drm.h>
#include <drm/drm_mode.h>
#include <sys/mman.h>

#include <opencv2/core.hpp>

// Full-screen output on the first connected DRM connector via a dumb buffer.
// The desktop (weston / QDesktop) must be stopped first; scripts/run.sh does that.
// Frames are rotated for portrait panels and scaled to fill (centre crop).
class DrmDisplay {
public:
    explicit DrmDisplay(const char* device = "/dev/dri/card0");
    ~DrmDisplay();

    DrmDisplay(const DrmDisplay&) = delete;
    DrmDisplay& operator=(const DrmDisplay&) = delete;

    void show(const cv::Mat& bgr);

private:
    void release();

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

#endif
