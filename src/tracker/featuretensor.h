#ifndef FEATURETENSOR_H
#define FEATURETENSOR_H

#include <opencv2/core.hpp>

#include "model.hpp"
#include "rknn_fp.h"

class FeatureTensor : public rknn_fp {
public:
    explicit FeatureTensor(const char* model_path) : rknn_fp(model_path) {}
    bool getRectsFeature(const cv::Mat& image, DETECTIONS& detections);
};

#endif
