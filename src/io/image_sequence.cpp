#include "image_sequence.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>

namespace fs = std::filesystem;

ImageSequence::ImageSequence(const std::string& directory, int decoders, int lookahead)
    : lookahead_(static_cast<std::size_t>(std::max(1, lookahead))) {
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") files_.push_back(entry.path().string());
    }
    std::sort(files_.begin(), files_.end());
    if (files_.empty()) throw std::runtime_error("no .jpg/.png images in " + directory);
    for (int i = 0; i < std::max(1, decoders); ++i) workers_.emplace_back(&ImageSequence::decode_loop, this);
    std::cout << "image sequence " << directory << ": " << files_.size() << " frames, " << workers_.size()
              << " decoder threads\n";
}

ImageSequence::~ImageSequence() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
    }
    cv_.notify_all();
    for (auto& t : workers_) t.join();
}

void ImageSequence::decode_loop() {
    for (;;) {
        std::size_t index;
        {
            // Do not run further ahead of the reader than `lookahead_` frames.
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&] { return stopping_ || claimed_ < next_ + lookahead_; });
            if (stopping_ || claimed_ >= files_.size()) return;
            index = claimed_++;
        }
        cv::Mat image;
        std::exception_ptr error;
        try {
            image = cv::imread(files_[index], cv::IMREAD_COLOR);
            if (image.empty()) throw std::runtime_error("cannot decode " + files_[index]);
        } catch (...) {
            error = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (error && !error_) error_ = error;
            ready_[index] = std::move(image);
        }
        cv_.notify_all();
    }
}

bool ImageSequence::read(cv::Mat& frame) {
    if (finished()) return false;
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return error_ || ready_.count(next_); });
    if (error_) std::rethrow_exception(error_);
    frame = std::move(ready_[next_]);
    ready_.erase(next_);
    ++next_;
    lock.unlock();
    cv_.notify_all();
    return true;
}
