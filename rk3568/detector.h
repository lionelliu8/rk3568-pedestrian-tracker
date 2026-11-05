#ifndef RK3568_DETECTOR_H
#define RK3568_DETECTOR_H

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "box.h"
#include "rknn_api.h"

struct rknn_app_context_t {
    rknn_context rknn_ctx = 0;
    rknn_input_output_num io_num{};
    rknn_tensor_attr* input_attrs = nullptr;
    rknn_tensor_attr* output_attrs = nullptr;
    int model_channel = 0;
    int model_width = 0;
    int model_height = 0;
    bool is_quant = false;
};

class RknnYolov5Detector {
public:
    explicit RknnYolov5Detector(const std::string& model_path);
    ~RknnYolov5Detector();

    RknnYolov5Detector(const RknnYolov5Detector&) = delete;
    RknnYolov5Detector& operator=(const RknnYolov5Detector&) = delete;

    std::vector<DetectBox> infer(const cv::Mat& bgr, int class_filter = -1,
                                 float confidence_threshold = 0.25f);

private:
    rknn_app_context_t context_;
    bool rga_resize_enabled_ = true;
    bool resize_backend_logged_ = false;
};

#endif
