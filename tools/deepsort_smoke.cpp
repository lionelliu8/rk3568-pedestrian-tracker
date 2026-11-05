#include <iostream>

#include "tracker.h"

int main() {
    tracker state(0.2f, 100, 0.7f, 30, 2);

    DETECTION_ROW detection;
    detection.tlwh << 100.0f, 100.0f, 40.0f, 80.0f;
    detection.confidence = 0.9f;
    detection.feature.setZero();

    for (int frame = 0; frame < 3; ++frame) {
        state.predict();
        state.update(DETECTIONS{detection});
    }

    if (state.tracks.size() != 1 || !state.tracks.front().is_confirmed() ||
        state.tracks.front().track_id != 1 || !state.tracks.front().mean.allFinite()) {
        std::cerr << "track was not confirmed with a finite state and stable ID\n";
        return 1;
    }

    for (int frame = 0; frame < 31; ++frame) {
        state.predict();
        state.update(DETECTIONS{});
    }

    if (!state.tracks.empty()) {
        std::cerr << "expired track was not deleted\n";
        return 1;
    }

    std::cout << "DeepSORT smoke test passed\n";
    return 0;
}
