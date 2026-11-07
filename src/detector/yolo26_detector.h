#ifndef PEDESTRIAN_TRACKER_YOLO26_DETECTOR_H
#define PEDESTRIAN_TRACKER_YOLO26_DETECTOR_H

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "detect_box.h"
#include "rknn_api.h"

struct DetectorConfig {
    float confidence_threshold = 0.25f;
    float nms_threshold = 0.7f;  // YOLO26 is NMS-free; this only removes rare INT8 duplicates
    int class_filter = 0;        // COCO class to keep, -1 keeps all
    int max_detections = 100;
    bool use_rga = true;         // RGA resize with automatic CPU fallback
    bool collect_perf = false;   // per-layer NPU timing (slower; see perf_detail())
};

struct DetectorTiming {
    double preprocess_ms = 0.0;
    double inference_ms = 0.0;
    double postprocess_ms = 0.0;
};

// Letterboxed network input for one frame (output of prepare()).
struct DetectorInput {
    cv::Mat tensor;        // RGB, model input size
    float scale = 1.0f;    // frame -> tensor
    int pad_x = 0;
    int pad_y = 0;
    cv::Size frame_size;
    double preprocess_ms = 0.0;
};

// Raw NPU outputs for one frame (output of infer()).
struct DetectorOutput {
    std::vector<std::vector<int8_t>> int8;  // INT8 models
    std::vector<std::vector<float>> fp;     // float models
    double inference_ms = 0.0;
};

// YOLO26 detector on the RK3568 NPU.
//
// The RKNN model is produced by tools/export_yolo26_onnx.py + tools/convert_yolo26_rknn.py
// and has six outputs: for each stride (8, 16, 32) a box tensor (1, 4, H, W) holding
// left/top/right/bottom distances in grid units and a class tensor (1, C, H, W) of logits.
class Yolo26Detector {
public:
    Yolo26Detector(const std::string& model_path, const DetectorConfig& config);
    ~Yolo26Detector();

    Yolo26Detector(const Yolo26Detector&) = delete;
    Yolo26Detector& operator=(const Yolo26Detector&) = delete;

    // bgr: 8-bit BGR frame of any size. Returned boxes are in frame pixels.
    std::vector<DetectBox> detect(const cv::Mat& bgr);

    // The same work in three steps, so a pipeline can run them on different threads:
    //   prepare()  CPU/RGA letterbox    - one thread at a time
    //   infer()    NPU                  - one thread at a time (owns the RKNN context)
    //   decode()   CPU, read-only       - any thread
    DetectorInput prepare(const cv::Mat& bgr);
    DetectorOutput infer(const DetectorInput& input);
    std::vector<DetectBox> decode(const DetectorOutput& output, const DetectorInput& input) const;

    // Lowest score that decode() keeps (trackers such as ByteTrack want low-score boxes too).
    void set_confidence_threshold(float threshold);
    float confidence_threshold() const { return config_.confidence_threshold; }

    const DetectorTiming& last_timing() const { return timing_; }
    // Per-layer timing table of the last run; needs DetectorConfig::collect_perf.
    std::string perf_detail() const;
    int input_width() const { return input_width_; }
    int input_height() const { return input_height_; }
    int num_classes() const { return num_classes_; }

private:
    struct Level {
        int stride = 0;
        int grid_w = 0;
        int grid_h = 0;
        uint32_t box_index = 0;
        uint32_t cls_index = 0;
    };
    void load_model(const std::string& model_path);
    void resolve_outputs();
    void resize(const cv::Mat& src, cv::Mat& dst);
    template <typename T>
    void decode_level(const Level& level, const T* box, const T* cls, const DetectorInput& input,
                      std::vector<DetectBox>& candidates) const;

    DetectorConfig config_;
    DetectorTiming timing_;
    rknn_context context_ = 0;
    std::vector<rknn_tensor_attr> output_attrs_;
    std::vector<Level> levels_;
    bool int8_outputs_ = false;
    int input_width_ = 0;
    int input_height_ = 0;
    int num_classes_ = 0;
};

#endif
