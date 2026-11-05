#include "rknn_fp.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

rknn_fp::rknn_fp(const char* model_path) {
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error(std::string("cannot open RKNN model: ") + model_path);
    }

    const std::streamsize model_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> model(static_cast<std::size_t>(model_size));
    if (!file.read(model.data(), model_size)) {
        throw std::runtime_error(std::string("cannot read RKNN model: ") + model_path);
    }

    int ret = rknn_init(&ctx_, model.data(), static_cast<uint32_t>(model.size()), 0, nullptr);
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("rknn_init failed: " + std::to_string(ret));
    }

    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
    if (ret != RKNN_SUCC || io_num_.n_input != 1 || io_num_.n_output < 1) {
        throw std::runtime_error("unexpected ReID model input/output count");
    }

    input_attr_.index = 0;
    ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attr_, sizeof(input_attr_));
    if (ret != RKNN_SUCC) {
        throw std::runtime_error("cannot query ReID input tensor");
    }
    dump_tensor_attr(input_attr_);

    output_attrs_.resize(io_num_.n_output);
    outputs_.resize(io_num_.n_output);
    for (uint32_t i = 0; i < io_num_.n_output; ++i) {
        output_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(output_attrs_[i]));
        if (ret != RKNN_SUCC) {
            throw std::runtime_error("cannot query ReID output tensor");
        }
        dump_tensor_attr(output_attrs_[i]);
        outputs_[i].resize(output_attrs_[i].n_elems);
    }

    if (input_attr_.fmt == RKNN_TENSOR_NCHW) {
        input_channels_ = input_attr_.dims[1];
        input_height_ = input_attr_.dims[2];
        input_width_ = input_attr_.dims[3];
    } else {
        input_height_ = input_attr_.dims[1];
        input_width_ = input_attr_.dims[2];
        input_channels_ = input_attr_.dims[3];
    }
}

rknn_fp::~rknn_fp() {
    if (ctx_ != 0) {
        rknn_destroy(ctx_);
    }
}

int rknn_fp::inference(const unsigned char* data, std::size_t size) {
    const std::size_t expected = static_cast<std::size_t>(input_width_) * input_height_ * input_channels_;
    if (data == nullptr || size != expected) {
        std::fprintf(stderr, "invalid ReID input size: got %zu, expected %zu\n", size, expected);
        return -1;
    }

    rknn_input input{};
    input.index = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = RKNN_TENSOR_NHWC;
    input.size = static_cast<uint32_t>(size);
    input.buf = const_cast<unsigned char*>(data);

    int ret = rknn_inputs_set(ctx_, 1, &input);
    if (ret != RKNN_SUCC) {
        return ret;
    }
    ret = rknn_run(ctx_, nullptr);
    if (ret != RKNN_SUCC) {
        return ret;
    }

    std::vector<rknn_output> raw_outputs(io_num_.n_output);
    for (uint32_t i = 0; i < io_num_.n_output; ++i) {
        raw_outputs[i].index = i;
        raw_outputs[i].want_float = 1;
    }
    ret = rknn_outputs_get(ctx_, io_num_.n_output, raw_outputs.data(), nullptr);
    if (ret != RKNN_SUCC) {
        return ret;
    }

    for (uint32_t i = 0; i < io_num_.n_output; ++i) {
        const float* src = static_cast<const float*>(raw_outputs[i].buf);
        std::memcpy(outputs_[i].data(), src, outputs_[i].size() * sizeof(float));
    }
    rknn_outputs_release(ctx_, io_num_.n_output, raw_outputs.data());
    return RKNN_SUCC;
}

const std::vector<float>& rknn_fp::output(std::size_t index) const {
    return outputs_.at(index);
}

void rknn_fp::dump_tensor_attr(const rknn_tensor_attr& attr) {
    std::printf("tensor %u: name=%s dims=[%d,%d,%d,%d] elements=%u fmt=%s type=%s\n",
                attr.index, attr.name, attr.dims[0], attr.dims[1], attr.dims[2], attr.dims[3],
                attr.n_elems, get_format_string(attr.fmt), get_type_string(attr.type));
}
