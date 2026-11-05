#include "frame_source.h"

#include <filesystem>

#include "image_sequence.h"
#include "v4l2_camera.h"

std::unique_ptr<FrameSource> open_frame_source(const std::string& input, int camera_width,
                                               int camera_height) {
    if (std::filesystem::is_directory(input)) return std::make_unique<ImageSequence>(input);
    return std::make_unique<V4l2Camera>(input, camera_width, camera_height);
}
