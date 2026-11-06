#ifndef DEEPSORT_H
#define DEEPSORT_H

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "datatype.h"
#include "featuretensor.h"
#include "tracker.h"

class DeepSort {
public:
    // max_age: frames a lost track is kept (it can be re-identified within this time).
    // n_init: consecutive detections needed before a new track is reported.
    DeepSort(const std::string& model_path, int track_interval = 1, int max_age = 30, int n_init = 2);
    ~DeepSort();

    DeepSort(const DeepSort&) = delete;
    DeepSort& operator=(const DeepSort&) = delete;

    void sort(cv::Mat& frame, std::vector<DetectBox>& detections);
    void sort_interval(std::vector<DetectBox>& detections);

private:
    void collect_results(std::vector<DetectBox>& detections, int max_age);

    int track_interval_;
    tracker* tracker_;
    FeatureTensor* feature_extractor_;
};

#endif
