// Run ByteTrack or OC-SORT on saved detections (MOTChallenge format) without images.
// Used to compare trackers and tune thresholds on the VM with detections recorded on the board
// (pedestrian_tracker --det-output, or tools/benchmark/eval_detectors_board.py).
//
// Build (host):  tools/benchmark/build_track_offline.sh
// Usage:  track_offline --tracker ocsort|bytetrack --dets DET.txt --frames N --out TRACKS.txt
//                       [--det-thresh 0.4] [--high 0.5] [--low 0.1] [--new 0.6] [--iou 0.3]
//                       [--max-age 30] [--min-hits 3] [--use-byte] [--min-score 0.0]
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "bytetrack.h"
#include "ocsort.h"

int main(int argc, char** argv) {
    std::string tracker = "ocsort", dets_path, out_path;
    int frames = 0;
    mot::OcSortConfig oc;
    mot::ByteTrackConfig bt;
    float min_score = 0.0f;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto v = [&]() { return std::string(argv[++i]); };
        if (a == "--tracker") tracker = v();
        else if (a == "--dets") dets_path = v();
        else if (a == "--frames") frames = std::stoi(v());
        else if (a == "--out") out_path = v();
        else if (a == "--det-thresh") oc.det_thresh = std::stof(v());
        else if (a == "--iou") oc.iou_thresh = std::stof(v());
        else if (a == "--max-age") oc.max_age = bt.track_buffer = std::stoi(v());
        else if (a == "--min-hits") oc.min_hits = std::stoi(v());
        else if (a == "--use-byte") oc.use_byte = true;
        else if (a == "--high") bt.high_thresh = std::stof(v());
        else if (a == "--low") bt.low_thresh = oc.low_thresh = std::stof(v());
        else if (a == "--new") bt.new_track_thresh = std::stof(v());
        else if (a == "--match") bt.match_thresh = std::stof(v());
        else if (a == "--min-score") min_score = std::stof(v());
        else { std::cerr << "unknown option " << a << "\n"; return 2; }
    }
    std::map<int, std::vector<DetectBox>> by_frame;
    std::ifstream in(dets_path);
    if (!in) { std::cerr << "cannot read " << dets_path << "\n"; return 1; }
    for (std::string line; std::getline(in, line);) {
        std::stringstream ss(line);
        std::string f;
        std::vector<float> v;
        while (std::getline(ss, f, ',')) v.push_back(std::stof(f));
        if (v.size() < 7 || v[6] < min_score) continue;
        by_frame[static_cast<int>(v[0])].emplace_back(v[2], v[3], v[2] + v[4], v[3] + v[5], v[6], 0.0f);
        frames = std::max(frames, static_cast<int>(v[0]));
    }
    mot::OcSort ocsort(oc);
    mot::ByteTrack bytetrack(bt);
    std::ofstream out(out_path);
    double total_ms = 0;
    for (int f = 1; f <= frames; ++f) {
        const auto& d = by_frame[f];
        const auto t0 = std::chrono::steady_clock::now();
        const auto tracks = tracker == "bytetrack" ? bytetrack.update(d) : ocsort.update(d);
        total_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        for (const auto& t : tracks)
            out << f << ',' << static_cast<int>(t.trackID) << ',' << t.x1 << ',' << t.y1 << ',' << t.width() << ','
                << t.height() << ',' << t.confidence << ",-1,-1,-1\n";
    }
    std::cerr << tracker << ": " << frames << " frames, " << total_ms / std::max(1, frames) << " ms/frame\n";
    return 0;
}
