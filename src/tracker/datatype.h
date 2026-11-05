#ifndef DEEPSORTDATATYPE_H
#define DEEPSORTDATATYPE_H

#include <cstddef>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "detect_box.h"

struct CLSCONF {
    CLSCONF() : cls(-1), conf(-1) {}
    CLSCONF(int cls_value, float conf_value) : cls(cls_value), conf(conf_value) {}
    int cls;
    float conf;
};

using DETECTBOX = Eigen::Matrix<float, 1, 4, Eigen::RowMajor>;
using DETECTBOXSS = Eigen::Matrix<float, -1, 4, Eigen::RowMajor>;
using FEATURE = Eigen::Matrix<float, 1, 512, Eigen::RowMajor>;
using FEATURESS = Eigen::Matrix<float, Eigen::Dynamic, 512, Eigen::RowMajor>;
using KAL_MEAN = Eigen::Matrix<float, 1, 8, Eigen::RowMajor>;
using KAL_COVA = Eigen::Matrix<float, 8, 8, Eigen::RowMajor>;
using KAL_HMEAN = Eigen::Matrix<float, 1, 4, Eigen::RowMajor>;
using KAL_HCOVA = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;
using KAL_DATA = std::pair<KAL_MEAN, KAL_COVA>;
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;
using RESULT_DATA = std::pair<int, DETECTBOX>;
using TRACKER_DATA = std::pair<int, FEATURESS>;
using MATCH_DATA = std::pair<int, int>;

struct TRACHER_MATCHD {
    std::vector<MATCH_DATA> matches;
    std::vector<int> unmatched_tracks;
    std::vector<int> unmatched_detections;
};

using DYNAMICM = Eigen::Matrix<float, -1, -1, Eigen::RowMajor>;

#endif
