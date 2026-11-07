#ifndef PEDESTRIAN_TRACKER_IMAGE_SEQUENCE_H
#define PEDESTRIAN_TRACKER_IMAGE_SEQUENCE_H

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "frame_source.h"

// Reads .jpg/.jpeg/.png files of a directory in file-name order. JPEG decoding is the slow
// part on the RK3568 (about 90 ms for 1080p), so `decoders` threads decode ahead of the reader.
class ImageSequence : public FrameSource {
public:
    explicit ImageSequence(const std::string& directory, int decoders = 2, int lookahead = 6);
    ~ImageSequence() override;

    bool read(cv::Mat& frame) override;
    bool finished() const override { return next_ >= files_.size(); }
    bool is_live() const override { return false; }

    std::size_t size() const { return files_.size(); }

private:
    void decode_loop();

    std::vector<std::string> files_;
    std::size_t next_ = 0;                  // next frame handed to read()
    std::atomic<std::size_t> claimed_{0};   // next frame a decoder will take
    std::size_t lookahead_;
    std::map<std::size_t, cv::Mat> ready_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stopping_ = false;
    std::exception_ptr error_;
    std::vector<std::thread> workers_;
};

#endif
