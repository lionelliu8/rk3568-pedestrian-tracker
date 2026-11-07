#include "yolo26_detector.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <im2d.h>
#include <opencv2/imgproc.hpp>
#include <rga.h>

#include "stopwatch.h"

namespace {

constexpr int kPadValue = 114;

void check(int ret, const char* what) {
    if (ret != RKNN_SUCC) {
        throw std::runtime_error(std::string(what) + " failed: " + std::to_string(ret));
    }
}

float iou(const DetectBox& a, const DetectBox& b) {
    const float w = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    const float h = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    const float inter = w * h;
    const float uni = a.width() * a.height() + b.width() * b.height() - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

std::vector<DetectBox> non_max_suppression(std::vector<DetectBox> boxes, float threshold,
                                           int max_detections) {
    std::sort(boxes.begin(), boxes.end(),
              [](const DetectBox& a, const DetectBox& b) { return a.confidence > b.confidence; });
    std::vector<DetectBox> kept;
    std::vector<bool> removed(boxes.size(), false);
    for (std::size_t i = 0; i < boxes.size() && static_cast<int>(kept.size()) < max_detections; ++i) {
        if (removed[i]) continue;
        kept.push_back(boxes[i]);
        for (std::size_t j = i + 1; j < boxes.size(); ++j) {
            if (!removed[j] && boxes[i].classID == boxes[j].classID &&
                iou(boxes[i], boxes[j]) > threshold) {
                removed[j] = true;
            }
        }
    }
    return kept;
}

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

}  // namespace

Yolo26Detector::Yolo26Detector(const std::string& model_path, const DetectorConfig& config)
    : config_(config) {
    if (config_.confidence_threshold <= 0.0f || config_.confidence_threshold >= 1.0f) {
        throw std::invalid_argument("confidence threshold must be in (0, 1)");
    }
    load_model(model_path);
    resolve_outputs();
    std::cout << "YOLO26 model: " << input_width_ << 'x' << input_height_ << ", " << num_classes_
              << " classes, " << (int8_outputs_ ? "INT8" : "float") << " outputs\n";
}

Yolo26Detector::~Yolo26Detector() {
    if (context_ != 0) rknn_destroy(context_);
}

void Yolo26Detector::load_model(const std::string& model_path) {
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open YOLO26 model: " + model_path);
    std::vector<char> blob(static_cast<std::size_t>(file.tellg()));
    file.seekg(0);
    if (!file.read(blob.data(), static_cast<std::streamsize>(blob.size()))) {
        throw std::runtime_error("cannot read YOLO26 model: " + model_path);
    }
    const uint32_t flags = config_.collect_perf ? RKNN_FLAG_COLLECT_PERF_MASK : 0;
    check(rknn_init(&context_, blob.data(), static_cast<uint32_t>(blob.size()), flags, nullptr),
          "rknn_init(YOLO26)");

    rknn_input_output_num io{};
    check(rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &io, sizeof(io)), "rknn_query(io)");
    if (io.n_input != 1 || io.n_output != 6) {
        throw std::runtime_error("YOLO26 model must have 1 input and 6 outputs, got " +
                                 std::to_string(io.n_input) + "/" + std::to_string(io.n_output) +
                                 " (was it converted with tools/convert_yolo26_rknn.py?)");
    }

    rknn_tensor_attr input{};
    input.index = 0;
    check(rknn_query(context_, RKNN_QUERY_INPUT_ATTR, &input, sizeof(input)), "rknn_query(input)");
    if (input.fmt == RKNN_TENSOR_NCHW) {
        input_height_ = static_cast<int>(input.dims[2]);
        input_width_ = static_cast<int>(input.dims[3]);
    } else {
        input_height_ = static_cast<int>(input.dims[1]);
        input_width_ = static_cast<int>(input.dims[2]);
    }

    output_attrs_.resize(io.n_output);
    for (uint32_t i = 0; i < io.n_output; ++i) {
        output_attrs_[i].index = i;
        check(rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(rknn_tensor_attr)),
              "rknn_query(output)");
    }
}

// Pair box/class tensors by grid size; the stride follows from the input size.
void Yolo26Detector::resolve_outputs() {
    int8_outputs_ = true;
    for (uint32_t i = 0; i < output_attrs_.size(); ++i) {
        const auto& attr = output_attrs_[i];
        if (attr.n_dims != 4) throw std::runtime_error("YOLO26 outputs must be 4-D");
        const int channels = static_cast<int>(attr.dims[1]);
        const int grid_h = static_cast<int>(attr.dims[2]);
        const int grid_w = static_cast<int>(attr.dims[3]);
        int8_outputs_ = int8_outputs_ && attr.type == RKNN_TENSOR_INT8 &&
                        attr.qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC;

        auto level = std::find_if(levels_.begin(), levels_.end(),
                                  [&](const Level& l) { return l.grid_h == grid_h && l.grid_w == grid_w; });
        if (level == levels_.end()) {
            Level created;
            created.grid_h = grid_h;
            created.grid_w = grid_w;
            created.stride = input_height_ / grid_h;
            created.box_index = created.cls_index = UINT32_MAX;
            levels_.push_back(created);
            level = levels_.end() - 1;
        }
        if (channels == 4) {
            level->box_index = i;
        } else {
            level->cls_index = i;
            if (num_classes_ != 0 && num_classes_ != channels) {
                throw std::runtime_error("YOLO26 class tensors disagree on class count");
            }
            num_classes_ = channels;
        }
    }
    for (const auto& level : levels_) {
        if (level.box_index == UINT32_MAX || level.cls_index == UINT32_MAX ||
            level.stride * level.grid_w != input_width_) {
            throw std::runtime_error("unexpected YOLO26 output layout");
        }
    }
    std::sort(levels_.begin(), levels_.end(),
              [](const Level& a, const Level& b) { return a.stride < b.stride; });
    if (config_.class_filter >= num_classes_) {
        throw std::invalid_argument("class filter " + std::to_string(config_.class_filter) +
                                    " is outside the model's " + std::to_string(num_classes_) + " classes");
    }
}

std::string Yolo26Detector::perf_detail() const {
    rknn_perf_detail detail{};
    if (rknn_query(context_, RKNN_QUERY_PERF_DETAIL, &detail, sizeof(detail)) != RKNN_SUCC || !detail.perf_data) {
        return "per-layer timing unavailable (collect_perf off?)";
    }
    return std::string(detail.perf_data, detail.data_len);
}

void Yolo26Detector::resize(const cv::Mat& src, cv::Mat& dst) {
    // RGA needs 16-pixel aligned BGR888 widths; other sizes take the CPU path for that frame.
    if (config_.use_rga && src.cols % 16 == 0 && dst.cols % 16 == 0) {
        const cv::Mat packed = src.isContinuous() ? src : src.clone();
        const auto rga_src = wrapbuffer_virtualaddr_t(packed.data, packed.cols, packed.rows, packed.cols,
                                                      packed.rows, RK_FORMAT_BGR_888);
        const auto rga_dst = wrapbuffer_virtualaddr_t(dst.data, dst.cols, dst.rows, dst.cols, dst.rows,
                                                      RK_FORMAT_BGR_888);
        const IM_STATUS status = imresize_t(rga_src, rga_dst, 0, 0, INTER_LINEAR, 1);
        if (status == IM_STATUS_SUCCESS) return;
        std::cerr << "RGA resize failed (" << imStrError_t(status) << "), using CPU resize from now on\n";
        config_.use_rga = false;
    }
    cv::resize(src, dst, dst.size(), 0, 0, cv::INTER_LINEAR);
}

void Yolo26Detector::set_confidence_threshold(float threshold) {
    if (threshold <= 0.0f || threshold >= 1.0f) throw std::invalid_argument("confidence threshold must be in (0, 1)");
    config_.confidence_threshold = threshold;
}

// One RGA call: scale the BGR frame, convert to RGB and write it into the letterbox area of the
// tensor. Returns false (CPU path) when RGA is off, fails, or the sizes do not meet its
// alignment rules (RGB888 strides must be multiples of 16 pixels, rectangles even).
bool Yolo26Detector::letterbox_rga(const cv::Mat& bgr, DetectorInput& in, int width, int height) {
    if (!config_.use_rga || !bgr.isContinuous() || bgr.cols % 16 != 0 || input_width_ % 16 != 0 ||
        (in.pad_x | in.pad_y | width | height) & 1) {
        return false;
    }
    const auto src = wrapbuffer_virtualaddr_t(bgr.data, bgr.cols, bgr.rows, bgr.cols, bgr.rows, RK_FORMAT_BGR_888);
    const auto dst = wrapbuffer_virtualaddr_t(in.tensor.data, input_width_, input_height_, input_width_,
                                              input_height_, RK_FORMAT_RGB_888);
    rga_buffer_t pat{};
    const im_rect src_rect{0, 0, bgr.cols, bgr.rows};
    const im_rect dst_rect{in.pad_x, in.pad_y, width, height};
    const im_rect empty{};
    const IM_STATUS status = improcess(src, dst, pat, src_rect, dst_rect, empty, IM_SYNC);
    if (status == IM_STATUS_SUCCESS) return true;
    std::cerr << "RGA letterbox failed (" << imStrError_t(status) << "), using the CPU from now on\n";
    config_.use_rga = false;
    return false;
}

DetectorInput Yolo26Detector::prepare(const cv::Mat& bgr) {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::invalid_argument("YOLO26 input must be a non-empty BGR image");
    }
    Stopwatch watch;
    DetectorInput in;
    in.frame_size = bgr.size();
    in.scale = std::min(static_cast<float>(input_width_) / bgr.cols, static_cast<float>(input_height_) / bgr.rows);
    const int width = std::max(1, static_cast<int>(bgr.cols * in.scale));
    const int height = std::max(1, static_cast<int>(bgr.rows * in.scale));
    in.pad_x = (input_width_ - width) / 2;
    in.pad_y = (input_height_ - height) / 2;

    in.tensor.create(input_height_, input_width_, CV_8UC3);
    in.tensor.setTo(cv::Scalar::all(kPadValue));  // grey borders: a 1.2 MB memset
    if (!letterbox_rga(bgr, in, width, height)) {
        cv::Mat resized(height, width, CV_8UC3);
        resize(bgr, resized);
        cv::Mat roi = in.tensor(cv::Rect(in.pad_x, in.pad_y, width, height));
        cv::cvtColor(resized, roi, cv::COLOR_BGR2RGB);
    }
    in.preprocess_ms = watch.elapsed();
    return in;
}

DetectorOutput Yolo26Detector::infer(const DetectorInput& in) {
    Stopwatch watch;
    rknn_input input{};
    input.index = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = RKNN_TENSOR_NHWC;
    input.size = static_cast<uint32_t>(in.tensor.total() * in.tensor.elemSize());
    input.buf = in.tensor.data;
    check(rknn_inputs_set(context_, 1, &input), "rknn_inputs_set(YOLO26)");
    check(rknn_run(context_, nullptr), "rknn_run(YOLO26)");

    // Outputs are written straight into our buffers (is_prealloc), so they stay valid after
    // the next rknn_run on this thread.
    DetectorOutput out;
    std::vector<rknn_output> outputs(output_attrs_.size());
    if (int8_outputs_) out.int8.resize(outputs.size());
    else out.fp.resize(outputs.size());
    for (uint32_t i = 0; i < outputs.size(); ++i) {
        outputs[i].index = i;
        outputs[i].is_prealloc = 1;
        outputs[i].want_float = int8_outputs_ ? 0 : 1;
        if (int8_outputs_) {
            out.int8[i].resize(output_attrs_[i].n_elems);
            outputs[i].buf = out.int8[i].data();
            outputs[i].size = static_cast<uint32_t>(out.int8[i].size());
        } else {
            out.fp[i].resize(output_attrs_[i].n_elems);
            outputs[i].buf = out.fp[i].data();
            outputs[i].size = static_cast<uint32_t>(out.fp[i].size() * sizeof(float));
        }
    }
    check(rknn_outputs_get(context_, static_cast<uint32_t>(outputs.size()), outputs.data(), nullptr),
          "rknn_outputs_get(YOLO26)");
    rknn_outputs_release(context_, static_cast<uint32_t>(outputs.size()), outputs.data());
    out.inference_ms = watch.elapsed();
    return out;
}

template <typename T>
void Yolo26Detector::decode_level(const Level& level, const T* box, const T* cls,
                                  const DetectorInput& lb, std::vector<DetectBox>& candidates) const {
    const cv::Size& frame_size = lb.frame_size;
    const auto& box_attr = output_attrs_[level.box_index];
    const auto& cls_attr = output_attrs_[level.cls_index];
    // Float models use identity quantization so one code path serves both.
    const float box_scale = int8_outputs_ ? box_attr.scale : 1.0f;
    const float box_zp = int8_outputs_ ? static_cast<float>(box_attr.zp) : 0.0f;
    const float cls_scale = int8_outputs_ ? cls_attr.scale : 1.0f;
    const float cls_zp = int8_outputs_ ? static_cast<float>(cls_attr.zp) : 0.0f;

    // Compare raw values against the threshold mapped into tensor units: no exp per anchor.
    const float c = config_.confidence_threshold;
    const float raw_threshold = std::log(c / (1.0f - c)) / cls_scale + cls_zp;

    const int area = level.grid_w * level.grid_h;
    std::vector<float> best(area, -1e30f);
    std::vector<int> best_class(area, -1);
    const int first = config_.class_filter >= 0 ? config_.class_filter : 0;
    const int last = config_.class_filter >= 0 ? config_.class_filter + 1 : num_classes_;
    for (int k = first; k < last; ++k) {
        const T* plane = cls + static_cast<std::size_t>(k) * area;
        for (int p = 0; p < area; ++p) {
            const float v = static_cast<float>(plane[p]);
            if (v >= raw_threshold && v > best[p]) {
                best[p] = v;
                best_class[p] = k;
            }
        }
    }

    const float max_x = static_cast<float>(frame_size.width);
    const float max_y = static_cast<float>(frame_size.height);
    for (int p = 0; p < area; ++p) {
        if (best_class[p] < 0) continue;
        const float anchor_x = static_cast<float>(p % level.grid_w) + 0.5f;
        const float anchor_y = static_cast<float>(p / level.grid_w) + 0.5f;
        auto distance = [&](int side) {
            return (static_cast<float>(box[static_cast<std::size_t>(side) * area + p]) - box_zp) * box_scale;
        };
        const float s = static_cast<float>(level.stride);
        const float x1 = ((anchor_x - distance(0)) * s - lb.pad_x) / lb.scale;
        const float y1 = ((anchor_y - distance(1)) * s - lb.pad_y) / lb.scale;
        const float x2 = ((anchor_x + distance(2)) * s - lb.pad_x) / lb.scale;
        const float y2 = ((anchor_y + distance(3)) * s - lb.pad_y) / lb.scale;
        DetectBox det(std::clamp(x1, 0.0f, max_x), std::clamp(y1, 0.0f, max_y),
                      std::clamp(x2, 0.0f, max_x), std::clamp(y2, 0.0f, max_y),
                      sigmoid((best[p] - cls_zp) * cls_scale), static_cast<float>(best_class[p]));
        if (det.width() >= 2.0f && det.height() >= 2.0f) candidates.push_back(det);
    }
}

std::vector<DetectBox> Yolo26Detector::decode(const DetectorOutput& out, const DetectorInput& in) const {
    std::vector<DetectBox> candidates;
    for (const auto& level : levels_) {
        if (int8_outputs_) {
            decode_level(level, out.int8[level.box_index].data(), out.int8[level.cls_index].data(), in, candidates);
        } else {
            decode_level(level, out.fp[level.box_index].data(), out.fp[level.cls_index].data(), in, candidates);
        }
    }
    return non_max_suppression(std::move(candidates), config_.nms_threshold, config_.max_detections);
}

std::vector<DetectBox> Yolo26Detector::detect(const cv::Mat& bgr) {
    const DetectorInput in = prepare(bgr);
    const DetectorOutput out = infer(in);
    Stopwatch watch;
    auto detections = decode(out, in);
    timing_.preprocess_ms = in.preprocess_ms;
    timing_.inference_ms = out.inference_ms;
    timing_.postprocess_ms = watch.elapsed();
    return detections;
}
