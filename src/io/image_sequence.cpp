#include "image_sequence.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;

ImageSequence::ImageSequence(const std::string& directory) {
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") files_.push_back(entry.path().string());
    }
    std::sort(files_.begin(), files_.end());
    if (files_.empty()) throw std::runtime_error("no .jpg/.png images in " + directory);
    std::cout << "image sequence " << directory << ": " << files_.size() << " frames\n";
}

bool ImageSequence::read(cv::Mat& frame) {
    if (finished()) return false;
    const std::string& path = files_[next_++];
    frame = cv::imread(path, cv::IMREAD_COLOR);
    if (frame.empty()) throw std::runtime_error("cannot decode " + path);
    return true;
}
