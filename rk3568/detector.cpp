#include "detector.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <im2d.h>
#include <rga.h>

#include "common.h"
#include "postprocess.h"

namespace {
constexpr int kYoloValuesPerBox = 85;

void query_tensor(rknn_context context, rknn_query_cmd command, rknn_tensor_attr& attr) {
    const int ret = rknn_query(context, command, &attr, sizeof(attr));
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("rknn_query failed: " + std::to_string(ret));
    }
}

float intersection_over_union(const DetectBox& a, const DetectBox& b) {
    const float left = std::max(a.x1, b.x1);
    const float top = std::max(a.y1, b.y1);
    const float right = std::min(a.x2, b.x2);
    const float bottom = std::min(a.y2, b.y2);
    const float intersection = std::max(0.0f, right - left) * std::max(0.0f, bottom - top);
    const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    const float union_area = area_a + area_b - intersection;
    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

std::vector<DetectBox> decode_flat_yolov5(const float* output, std::size_t count,
                                           const letterbox_t& letterbox, int image_width,
                                           int image_height, int class_filter,
                                           float confidence_threshold) {
    std::vector<DetectBox> candidates;
    const std::size_t box_count = count / kYoloValuesPerBox;
    candidates.reserve(std::min<std::size_t>(box_count, 1024));
    float max_objectness = 0.0f;
    float max_person_confidence = 0.0f;
    float max_confidence = 0.0f;

    for (std::size_t i = 0; i < box_count; ++i) {
        const float* values = output + i * kYoloValuesPerBox;
        max_objectness = std::max(max_objectness, values[4]);
        max_person_confidence = std::max(max_person_confidence, values[4] * values[5]);
        for (int cls = 0; cls < OBJ_CLASS_NUM; ++cls) {
            max_confidence = std::max(max_confidence, values[4] * values[5 + cls]);
        }
        if (values[4] < confidence_threshold) continue;

        int class_id = 0;
        float class_score = values[5];
        for (int cls = 1; cls < OBJ_CLASS_NUM; ++cls) {
            if (values[5 + cls] > class_score) {
                class_score = values[5 + cls];
                class_id = cls;
            }
        }
        const float confidence = values[4] * class_score;
        if (confidence < confidence_threshold ||
            (class_filter >= 0 && class_id != class_filter)) continue;

        const float x1 = (values[0] - values[2] * 0.5f - letterbox.x_pad) / letterbox.scale;
        const float y1 = (values[1] - values[3] * 0.5f - letterbox.y_pad) / letterbox.scale;
        const float x2 = (values[0] + values[2] * 0.5f - letterbox.x_pad) / letterbox.scale;
        const float y2 = (values[1] + values[3] * 0.5f - letterbox.y_pad) / letterbox.scale;
        DetectBox box(std::clamp(x1, 0.0f, static_cast<float>(image_width - 1)),
                      std::clamp(y1, 0.0f, static_cast<float>(image_height - 1)),
                      std::clamp(x2, 0.0f, static_cast<float>(image_width)),
                      std::clamp(y2, 0.0f, static_cast<float>(image_height)), confidence,
                      static_cast<float>(class_id));
        if (box.x2 - box.x1 >= 2.0f && box.y2 - box.y1 >= 2.0f) candidates.push_back(box);
    }

    static bool logged = false;
    if (!logged) {
        std::cout << "YOLO output: boxes=" << box_count << " max_obj=" << max_objectness
                  << " max_person=" << max_person_confidence << " max_any=" << max_confidence
                  << " candidates=" << candidates.size() << '\n';
        logged = true;
    }

    std::sort(candidates.begin(), candidates.end(), [](const DetectBox& a, const DetectBox& b) {
        return a.confidence > b.confidence;
    });
    std::vector<DetectBox> results;
    std::vector<bool> suppressed(candidates.size(), false);
    for (std::size_t i = 0; i < candidates.size() && results.size() < OBJ_NUMB_MAX_SIZE; ++i) {
        if (suppressed[i]) continue;
        results.push_back(candidates[i]);
        for (std::size_t j = i + 1; j < candidates.size(); ++j) {
            if (!suppressed[j] && candidates[i].classID == candidates[j].classID &&
                intersection_over_union(candidates[i], candidates[j]) > NMS_THRESH) {
                suppressed[j] = true;
            }
        }
    }
    return results;
}
}

RknnYolov5Detector::RknnYolov5Detector(const std::string& model_path) {
    const char* rga_setting = std::getenv("RGA_RESIZE");
    rga_resize_enabled_ = !rga_setting || std::string(rga_setting) != "0";
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("cannot open YOLO model: " + model_path);
    }
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> model(static_cast<std::size_t>(size));
    if (!file.read(model.data(), size)) {
        throw std::runtime_error("cannot read YOLO model: " + model_path);
    }

    int ret = rknn_init(&context_.rknn_ctx, model.data(), static_cast<uint32_t>(model.size()), 0, nullptr);
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("rknn_init YOLO failed: " + std::to_string(ret));
    }
    ret = rknn_query(context_.rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &context_.io_num, sizeof(context_.io_num));
    if (ret != RKNN_SUCC || context_.io_num.n_input != 1 || context_.io_num.n_output < 1) {
        throw std::runtime_error("unexpected YOLO input/output count");
    }

    context_.input_attrs = new rknn_tensor_attr[context_.io_num.n_input]();
    context_.output_attrs = new rknn_tensor_attr[context_.io_num.n_output]();
    for (uint32_t i = 0; i < context_.io_num.n_input; ++i) {
        context_.input_attrs[i].index = i;
        query_tensor(context_.rknn_ctx, RKNN_QUERY_INPUT_ATTR, context_.input_attrs[i]);
    }
    for (uint32_t i = 0; i < context_.io_num.n_output; ++i) {
        context_.output_attrs[i].index = i;
        query_tensor(context_.rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, context_.output_attrs[i]);
    }
    if (context_.io_num.n_output == 1 &&
        context_.output_attrs[0].n_elems % kYoloValuesPerBox != 0) {
        throw std::runtime_error("unsupported single-output YOLO tensor shape");
    }

    const auto& input = context_.input_attrs[0];
    if (input.fmt == RKNN_TENSOR_NCHW) {
        context_.model_channel = input.dims[1];
        context_.model_height = input.dims[2];
        context_.model_width = input.dims[3];
    } else {
        context_.model_height = input.dims[1];
        context_.model_width = input.dims[2];
        context_.model_channel = input.dims[3];
    }
    context_.is_quant = context_.output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC &&
                        context_.output_attrs[0].type != RKNN_TENSOR_FLOAT16;

    std::cout << "YOLO model: input=" << context_.model_width << 'x' << context_.model_height
              << " outputs=" << context_.io_num.n_output << '\n';
    for (uint32_t i = 0; i < context_.io_num.n_output; ++i) {
        const auto& output = context_.output_attrs[i];
        std::cout << "YOLO tensor " << i << ": fmt=" << output.fmt << " type=" << output.type
                  << " dims=";
        for (uint32_t d = 0; d < output.n_dims; ++d) std::cout << output.dims[d] << 'x';
        std::cout << " elements=" << output.n_elems << '\n';
    }
}

RknnYolov5Detector::~RknnYolov5Detector() {
    delete[] context_.input_attrs;
    delete[] context_.output_attrs;
    if (context_.rknn_ctx != 0) {
        rknn_destroy(context_.rknn_ctx);
    }
}

std::vector<DetectBox> RknnYolov5Detector::infer(const cv::Mat& bgr, int class_filter,
                                                 float confidence_threshold) {
    const auto profile_start = std::chrono::steady_clock::now();
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::runtime_error("YOLO input must be a non-empty BGR8 image");
    }

    const float scale = std::min(static_cast<float>(context_.model_width) / bgr.cols,
                                 static_cast<float>(context_.model_height) / bgr.rows);
    const int resized_width = std::max(1, static_cast<int>(bgr.cols * scale));
    const int resized_height = std::max(1, static_cast<int>(bgr.rows * scale));
    const int x_pad = (context_.model_width - resized_width) / 2;
    const int y_pad = (context_.model_height - resized_height) / 2;

    cv::Mat resized;
    if (rga_resize_enabled_) {
        // Own packed buffers until the synchronous RGA operation finishes.
        const cv::Mat packed = bgr.isContinuous() ? bgr : bgr.clone();
        resized.create(resized_height, resized_width, CV_8UC3);
        const auto src = wrapbuffer_virtualaddr_t(packed.data, packed.cols, packed.rows,
                                                  packed.cols, packed.rows, RK_FORMAT_BGR_888);
        const auto dst = wrapbuffer_virtualaddr_t(resized.data, resized.cols, resized.rows,
                                                  resized.cols, resized.rows, RK_FORMAT_BGR_888);
        const IM_STATUS status = imresize_t(src, dst, 0, 0, INTER_LINEAR, 1);
        if (status != IM_STATUS_SUCCESS) {
            std::cerr << "YOLO resize: RGA failed: " << imStrError_t(status)
                      << "; falling back to CPU for this run\n";
            rga_resize_enabled_ = false;
            resize_backend_logged_ = false;
        }
    }
    if (!rga_resize_enabled_) {
        cv::resize(bgr, resized, cv::Size(resized_width, resized_height), 0, 0, cv::INTER_LINEAR);
    }
    if (!resize_backend_logged_) {
        std::cout << "YOLO resize backend: " << (rga_resize_enabled_ ? "RGA" : "CPU") << '\n';
        resize_backend_logged_ = true;
    }
    cv::Mat letterboxed(context_.model_height, context_.model_width, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(letterboxed(cv::Rect(x_pad, y_pad, resized_width, resized_height)));
    cv::cvtColor(letterboxed, letterboxed, cv::COLOR_BGR2RGB);
    const auto profile_preprocessed = std::chrono::steady_clock::now();

    rknn_input input{};
    input.index = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = RKNN_TENSOR_NHWC;
    input.size = static_cast<uint32_t>(letterboxed.total() * letterboxed.elemSize());
    input.buf = letterboxed.data;

    int ret = rknn_inputs_set(context_.rknn_ctx, 1, &input);
    if (ret != RKNN_SUCC || (ret = rknn_run(context_.rknn_ctx, nullptr)) != RKNN_SUCC) {
        throw std::runtime_error("YOLO inference failed: " + std::to_string(ret));
    }
    const auto profile_inferred = std::chrono::steady_clock::now();

    std::vector<rknn_output> outputs(context_.io_num.n_output);
    for (uint32_t i = 0; i < context_.io_num.n_output; ++i) {
        outputs[i].index = i;
        outputs[i].want_float = context_.io_num.n_output == 1 || !context_.is_quant;
    }
    ret = rknn_outputs_get(context_.rknn_ctx, context_.io_num.n_output, outputs.data(), nullptr);
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("cannot get YOLO output: " + std::to_string(ret));
    }
    const auto profile_output = std::chrono::steady_clock::now();

    std::vector<DetectBox> results;
    letterbox_t letterbox{x_pad, y_pad, scale};
    if (context_.io_num.n_output == 1) {
        results = decode_flat_yolov5(static_cast<const float*>(outputs[0].buf),
                                     context_.output_attrs[0].n_elems, letterbox,
                                     bgr.cols, bgr.rows, class_filter, confidence_threshold);
    } else {
        object_detect_result_list raw_results{};
        post_process(&context_, outputs.data(), &letterbox, confidence_threshold, NMS_THRESH,
                     &raw_results);
        results.reserve(raw_results.count);
        for (int i = 0; i < raw_results.count; ++i) {
            const auto& item = raw_results.results[i];
            if (class_filter >= 0 && item.cls_id != class_filter) continue;
            results.emplace_back(item.box.left, item.box.top, item.box.right, item.box.bottom,
                                 item.prop, static_cast<float>(item.cls_id));
        }
    }
    rknn_outputs_release(context_.rknn_ctx, context_.io_num.n_output, outputs.data());
    const auto profile_done = std::chrono::steady_clock::now();
    static int profile_frames = 0;
    static double preprocess_ms = 0.0;
    static double inference_ms = 0.0;
    static double output_ms = 0.0;
    static double postprocess_ms = 0.0;
    const auto elapsed_ms = [](auto begin, auto end) {
        return std::chrono::duration<double, std::milli>(end - begin).count();
    };
    preprocess_ms += elapsed_ms(profile_start, profile_preprocessed);
    inference_ms += elapsed_ms(profile_preprocessed, profile_inferred);
    output_ms += elapsed_ms(profile_inferred, profile_output);
    postprocess_ms += elapsed_ms(profile_output, profile_done);
    if (++profile_frames == 30) {
        std::cout << "YOLO profile ms: preprocess=" << preprocess_ms / profile_frames
                  << " inference=" << inference_ms / profile_frames
                  << " output=" << output_ms / profile_frames
                  << " postprocess=" << postprocess_ms / profile_frames << '\n';
        profile_frames = 0;
        preprocess_ms = inference_ms = output_ms = postprocess_ms = 0.0;
    }
    return results;
}
