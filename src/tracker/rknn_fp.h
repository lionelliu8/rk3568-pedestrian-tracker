#ifndef RKNN_FP_H
#define RKNN_FP_H

#include <cstddef>
#include <vector>

#include "rknn_api.h"

class rknn_fp {
public:
    explicit rknn_fp(const char* model_path);
    virtual ~rknn_fp();

    rknn_fp(const rknn_fp&) = delete;
    rknn_fp& operator=(const rknn_fp&) = delete;

    int inference(const unsigned char* data, std::size_t size);
    const std::vector<float>& output(std::size_t index) const;
    int input_width() const { return input_width_; }
    int input_height() const { return input_height_; }
    int input_channels() const { return input_channels_; }

private:
    static void dump_tensor_attr(const rknn_tensor_attr& attr);

    rknn_context ctx_ = 0;
    rknn_input_output_num io_num_{};
    rknn_tensor_attr input_attr_{};
    std::vector<rknn_tensor_attr> output_attrs_;
    std::vector<std::vector<float>> outputs_;
    int input_width_ = 0;
    int input_height_ = 0;
    int input_channels_ = 0;
};

#endif
