// RK3568 pedestrian tracker: V4L2 camera or image sequence -> YOLO26 (NPU) -> ByteTrack /
// OC-SORT / DeepSORT -> optional DRM screen output.
//
// By default the work runs as a three-stage pipeline so the NPU never waits for the CPU:
//   reader thread   read frame (camera / JPEG decode) + letterbox      -> queue
//   NPU thread      rknn inference                                     -> queue
//   main thread     decode boxes, track, write results, draw, display
// Throughput is then bounded by the slowest stage (normally the NPU, ~43 ms) instead of the sum.
#include <atomic>
#include <csignal>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>

#include "bounded_queue.h"
#include "drm_display.h"
#include "frame_source.h"
#include "mot_writer.h"
#include "multi_tracker.h"
#include "options.h"
#include "overlay.h"
#include "stopwatch.h"
#include "yolo26_detector.h"

namespace {

std::atomic<bool> g_running{true};

void on_signal(int) { g_running = false; }

struct FrameJob {
    int index = 0;  // 1-based frame number
    cv::Mat frame;
    bool detect = false;
    DetectorInput input;
    DetectorOutput output;
    double read_ms = 0;
};

// Per-second averages printed to stdout.
struct Stats {
    int frames = 0, detected = 0, dropped = 0;
    double read = 0, preprocess = 0, inference = 0, postprocess = 0, tracker = 0, display = 0;
    Stopwatch window;

    void add(const FrameJob& job) {
        read += job.read_ms;
        if (job.detect) {
            ++detected;
            preprocess += job.input.preprocess_ms;
            inference += job.output.inference_ms;
        }
    }

    // Returns the FPS of the finished window, or a negative value while the window is open.
    double maybe_report(int total_frames, std::size_t tracks) {
        const double elapsed_ms = window.elapsed();
        if (elapsed_ms < 1000.0) return -1.0;
        const double fps = frames * 1000.0 / elapsed_ms;
        const double d = std::max(1, detected);
        const double n = std::max(1, frames);
        std::cout << std::fixed << std::setprecision(1) << "frames=" << total_frames << " tracks=" << tracks
                  << " fps=" << fps << " | ms: read=" << read / n << " pre=" << preprocess / d
                  << " npu=" << inference / d << " post=" << postprocess / d << " track=" << tracker / n
                  << " display=" << display / n;
        if (dropped) std::cout << " | dropped=" << dropped;
        std::cout << std::endl;
        *this = Stats{};
        return fps;
    }
};

// Records the first exception of any pipeline thread so main can rethrow it.
class ErrorSlot {
public:
    void set(std::exception_ptr e) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!error_) error_ = e;
        g_running = false;
    }
    void rethrow() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (error_) std::rethrow_exception(error_);
    }

private:
    std::mutex mutex_;
    std::exception_ptr error_;
};

}  // namespace

int main(int argc, char** argv) {
    Options options;
    try {
        options = parse_options(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << "\n\n";
        print_usage(argv[0]);
        return 2;
    }

    try {
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);

        auto source = open_frame_source(options.input, options.camera_width, options.camera_height);

        TrackerOptions tracker_options;
        tracker_options.kind = options.tracker;
        tracker_options.track_threshold = options.confidence_threshold;
        tracker_options.max_age = options.max_age;
        tracker_options.n_init = options.n_init;
        tracker_options.detect_interval = options.detect_interval;
        tracker_options.reid_model = options.reid_model;
        auto tracker = make_tracker(tracker_options);

        DetectorConfig detector_config;
        detector_config.confidence_threshold = tracker->min_detection_score();
        detector_config.nms_threshold = options.nms_threshold;
        detector_config.class_filter = options.class_filter;
        detector_config.use_rga = options.use_rga;
        Yolo26Detector detector(options.yolo_model, detector_config);
        const auto labels = load_labels(options.labels);
        std::cout << "tracker: " << tracker->description() << ", detector keeps scores >= "
                  << detector.confidence_threshold() << (options.pipeline ? ", pipelined" : ", sequential")
                  << std::endl;

        std::unique_ptr<DrmDisplay> display;
        if (options.display) display = std::make_unique<DrmDisplay>();
        std::unique_ptr<MotWriter> track_writer, detection_writer;
        if (!options.mot_output.empty()) track_writer = std::make_unique<MotWriter>(options.mot_output);
        if (!options.det_output.empty()) detection_writer = std::make_unique<MotWriter>(options.det_output);

        // Stage 1: read + letterbox. Returns nullopt at the end of the input.
        int read_index = 0;
        auto read_job = [&]() -> std::optional<FrameJob> {
            FrameJob job;
            bool got = false;
            while (g_running && !got) {
                Stopwatch watch;
                got = source->read(job.frame);
                job.read_ms = watch.elapsed();
                if (!got && source->finished()) return std::nullopt;  // else: interrupted, retry
            }
            if (!got) return std::nullopt;
            job.index = ++read_index;
            job.detect = (job.index - 1) % options.detect_interval == 0;
            if (job.detect) job.input = detector.prepare(job.frame);
            return job;
        };
        // Stage 2: NPU.
        auto infer_job = [&](FrameJob& job) {
            if (job.detect) job.output = detector.infer(job.input);
        };

        Stats stats;
        double fps = 0.0;
        int processed = 0;
        // Stage 3: everything after the NPU. Returns false to stop.
        auto finish_job = [&](FrameJob& job) -> bool {
            stats.add(job);
            std::vector<DetectBox> tracks;
            Stopwatch watch;
            if (job.detect) {
                const auto detections = detector.decode(job.output, job.input);
                stats.postprocess += watch.lap();
                if (detection_writer) detection_writer->write(job.index, detections);
                tracks = tracker->update(job.frame, detections);
            } else {
                tracks = tracker->skip_frame();
            }
            stats.tracker += watch.lap();
            if (track_writer) track_writer->write(job.index, tracks);

            ++processed;
            ++stats.frames;
            const double reported = stats.maybe_report(processed, tracks.size());
            if (reported >= 0.0) fps = reported;
            if (display) {
                watch.lap();
                draw_tracks(job.frame, tracks, labels, fps);
                display->show(job.frame);
                stats.display += watch.lap();
            }
            return !(options.max_frames > 0 && processed >= options.max_frames);
        };

        if (!options.pipeline) {
            while (g_running) {
                auto job = read_job();
                if (!job) break;
                infer_job(*job);
                if (!finish_job(*job)) break;
            }
        } else {
            // A camera gets a 1-deep queue: the frame waiting for the NPU is at most one NPU
            // period old. Files get some slack to absorb JPEG decode jitter.
            BoundedQueue<FrameJob> to_npu(source->is_live() ? 1 : 2), to_main(2);
            ErrorSlot errors;
            std::atomic<int> dropped{0};
            std::thread reader([&] {
                try {
                    while (g_running) {
                        auto job = read_job();
                        if (!job) break;
                        // Blocks while the NPU is busy. A camera keeps capturing meanwhile and
                        // read() then returns only the newest frame, so no stale frame is
                        // converted and latency stays low. Files never drop frames.
                        dropped += source->take_skipped();
                        if (!to_npu.push(std::move(*job))) break;
                    }
                } catch (...) {
                    errors.set(std::current_exception());
                }
                to_npu.close();
            });
            std::thread npu([&] {
                try {
                    while (auto job = to_npu.pop()) {
                        infer_job(*job);
                        if (!to_main.push(std::move(*job))) break;
                    }
                } catch (...) {
                    errors.set(std::current_exception());
                }
                to_main.close();
            });
            try {
                while (auto job = to_main.pop()) {
                    stats.dropped += dropped.exchange(0);
                    if (!finish_job(*job)) break;
                }
            } catch (...) {
                errors.set(std::current_exception());
            }
            g_running = false;
            to_npu.close();
            to_main.close();
            reader.join();
            npu.join();
            errors.rethrow();
        }
        std::cout << "finished after " << processed << " frames" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << std::endl;
        return 1;
    }
}
