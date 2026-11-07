#include "options.h"

#include <charconv>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {

int to_int(const std::string& value, const std::string& option) {
    int result = 0;
    const auto [end, ec] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (ec != std::errc() || end != value.data() + value.size()) {
        throw std::invalid_argument("invalid value for " + option + ": " + value);
    }
    return result;
}

float to_float(const std::string& value, const std::string& option) {
    try {
        std::size_t used = 0;
        const float result = std::stof(value, &used);
        if (used == value.size()) return result;
    } catch (const std::exception&) {
    }
    throw std::invalid_argument("invalid value for " + option + ": " + value);
}

}  // namespace

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " [options]\n"
        << "  --input <index|device|dir> V4L2 camera index/device, or a directory of images\n"
        << "                             (default: /dev/video-camera0)\n"
        << "  --model <file>             YOLO26 RKNN model (default: model/yolo26n_person_rk3568.rknn)\n"
        << "  --reid-model <file>        OSNet ReID RKNN model\n"
        << "  --labels <file>            class name list, one per line\n"
        << "  --class <id>               class to track, -1 = all (default: 0 person)\n"
        << "  --tracker <name>           bytetrack (default), ocsort or deepsort\n"
        << "  --conf-threshold <0..1>    score needed to start/keep a track (default: bytetrack 0.5,\n"
        << "                             ocsort 0.3, deepsort 0.25; bytetrack also uses boxes >= 0.1)\n"
        << "  --nms-threshold <0..1>     duplicate-box IoU (default: 0.7)\n"
        << "  --detect-interval <n>      run the detector every n frames (default: 1)\n"
        << "  --camera-width <px>        requested capture width (default: 1280)\n"
        << "  --camera-height <px>       requested capture height (default: 720)\n"
        << "  --max-frames <n>           stop after n frames, 0 = unlimited\n"
        << "  --max-age <frames>         keep lost tracks this long (default: 30)\n"
        << "  --n-init <n>               deepsort: detections before a new track is shown (default: 2)\n"
        << "  --no-pipeline              process frames one after another (for timing comparisons)\n"
        << "  --mot-output <file>        write tracks in MOTChallenge format\n"
        << "  --det-output <file>        write raw detections in MOTChallenge format\n"
        << "  --no-rga                   resize on the CPU instead of RGA\n"
        << "  --display                  draw results on the DRM screen\n"
        << "  --help                     show this help\n";
}

Options parse_options(int argc, char** argv) {
    namespace fs = std::filesystem;
    Options o;
    const fs::path model_dir = fs::absolute(argv[0]).parent_path() / "model";
    o.yolo_model = (model_dir / "yolo26n_person_rk3568.rknn").string();
    o.reid_model = (model_dir / "osnet_x0_25_market_rk3568.rknn").string();
    o.labels = (model_dir / "person_labels.txt").string();

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (++i >= argc) throw std::invalid_argument("missing value after " + arg);
            return argv[i];
        };
        if (arg == "--input") o.input = value();
        else if (arg == "--model") o.yolo_model = value();
        else if (arg == "--reid-model") o.reid_model = value();
        else if (arg == "--labels") o.labels = value();
        else if (arg == "--class") o.class_filter = to_int(value(), arg);
        else if (arg == "--conf-threshold") o.confidence_threshold = to_float(value(), arg);
        else if (arg == "--nms-threshold") o.nms_threshold = to_float(value(), arg);
        else if (arg == "--detect-interval") o.detect_interval = to_int(value(), arg);
        else if (arg == "--camera-width") o.camera_width = to_int(value(), arg);
        else if (arg == "--camera-height") o.camera_height = to_int(value(), arg);
        else if (arg == "--max-frames") o.max_frames = to_int(value(), arg);
        else if (arg == "--tracker") o.tracker = value();
        else if (arg == "--no-pipeline") o.pipeline = false;
        else if (arg == "--max-age") o.max_age = to_int(value(), arg);
        else if (arg == "--n-init") o.n_init = to_int(value(), arg);
        else if (arg == "--mot-output") o.mot_output = value();
        else if (arg == "--det-output") o.det_output = value();
        else if (arg == "--no-rga") o.use_rga = false;
        else if (arg == "--display") o.display = true;
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }

    if (o.class_filter < -1) throw std::invalid_argument("--class must be >= -1");
    if (o.detect_interval < 1) throw std::invalid_argument("--detect-interval must be >= 1");
    if (o.max_frames < 0) throw std::invalid_argument("--max-frames must be >= 0");
    if (o.max_age < 1 || o.n_init < 1) throw std::invalid_argument("--max-age and --n-init must be >= 1");
    if (o.camera_width < 1 || o.camera_height < 1) throw std::invalid_argument("bad camera size");
    if (o.confidence_threshold != -1.0f && (o.confidence_threshold <= 0.0f || o.confidence_threshold >= 1.0f)) {
        throw std::invalid_argument("--conf-threshold must be in (0, 1)");
    }
    if (o.tracker != "bytetrack" && o.tracker != "ocsort" && o.tracker != "deepsort") {
        throw std::invalid_argument("--tracker must be bytetrack, ocsort or deepsort");
    }
    if (o.nms_threshold <= 0.0f || o.nms_threshold > 1.0f) {
        throw std::invalid_argument("--nms-threshold must be in (0, 1]");
    }
    return o;
}
