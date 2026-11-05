// RK3568 pedestrian tracker: V4L2 camera or image sequence -> YOLO26 (NPU) -> DeepSORT with OSNet ReID (NPU)
// -> optional DRM screen output.
#include <atomic>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "deepsort.h"
#include "drm_display.h"
#include "frame_source.h"
#include "mot_writer.h"
#include "options.h"
#include "overlay.h"
#include "stopwatch.h"
#include "yolo26_detector.h"

namespace {

std::atomic<bool> g_running{true};

void on_signal(int) { g_running = false; }

// Per-second averages printed to stdout.
struct Stats {
    int frames = 0;
    int detected_frames = 0;
    double read = 0, preprocess = 0, inference = 0, postprocess = 0, tracker = 0, display = 0;
    Stopwatch window;

    void add_detection(const DetectorTiming& t) {
        ++detected_frames;
        preprocess += t.preprocess_ms;
        inference += t.inference_ms;
        postprocess += t.postprocess_ms;
    }

    // Returns the FPS of the finished window, or a negative value while the window is open.
    double maybe_report(int total_frames, std::size_t tracks) {
        const double elapsed_ms = window.elapsed();
        if (elapsed_ms < 1000.0) return -1.0;
        const double fps = frames * 1000.0 / elapsed_ms;
        const double d = std::max(1, detected_frames);
        const double n = std::max(1, frames);
        std::cout << std::fixed << std::setprecision(1) << "frames=" << total_frames
                  << " tracks=" << tracks << " fps=" << fps << " | ms: read=" << read / n << " pre=" << preprocess / d
                  << " npu=" << inference / d << " post=" << postprocess / d
                  << " track=" << tracker / n << " display=" << display / n << std::endl;
        *this = Stats{};
        return fps;
    }
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

        DetectorConfig detector_config;
        detector_config.confidence_threshold = options.confidence_threshold;
        detector_config.nms_threshold = options.nms_threshold;
        detector_config.class_filter = options.class_filter;
        detector_config.use_rga = options.use_rga;
        Yolo26Detector detector(options.yolo_model, detector_config);
        DeepSort tracker(options.reid_model, options.detect_interval);
        const auto labels = load_labels(options.labels);

        std::unique_ptr<DrmDisplay> display;
        if (options.display) display = std::make_unique<DrmDisplay>();
        std::unique_ptr<MotWriter> track_writer, detection_writer;
        if (!options.mot_output.empty()) track_writer = std::make_unique<MotWriter>(options.mot_output);
        if (!options.det_output.empty()) detection_writer = std::make_unique<MotWriter>(options.det_output);

        cv::Mat frame;
        Stats stats;
        double fps = 0.0;
        int frame_index = 0;
        while (g_running) {
            Stopwatch read_watch;
            if (!source->read(frame)) {
                if (source->finished()) break;
                continue;  // camera wait interrupted by a signal
            }
            stats.read += read_watch.lap();

            std::vector<DetectBox> tracks;
            Stopwatch watch;
            if (frame_index % options.detect_interval == 0) {
                tracks = detector.detect(frame);
                stats.add_detection(detector.last_timing());
                if (detection_writer) detection_writer->write(frame_index + 1, tracks);
                watch.lap();
                tracker.sort(frame, tracks);
            } else {
                tracker.sort_interval(tracks);
            }
            stats.tracker += watch.lap();

            ++frame_index;
            ++stats.frames;
            if (track_writer) track_writer->write(frame_index, tracks);
            const double reported = stats.maybe_report(frame_index, tracks.size());
            if (reported >= 0.0) fps = reported;

            if (display) {
                draw_tracks(frame, tracks, labels, fps);
                display->show(frame);
                stats.display += watch.lap();
            }
            if (options.max_frames > 0 && frame_index >= options.max_frames) break;
        }
        std::cout << "finished after " << frame_index << " frames" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << std::endl;
        return 1;
    }
}
