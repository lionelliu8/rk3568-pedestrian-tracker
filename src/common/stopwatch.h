#ifndef PEDESTRIAN_TRACKER_STOPWATCH_H
#define PEDESTRIAN_TRACKER_STOPWATCH_H

#include <chrono>

// Milliseconds elapsed since construction or the last lap().
class Stopwatch {
public:
    Stopwatch() : start_(Clock::now()) {}

    double lap() {
        const auto now = Clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - start_).count();
        start_ = now;
        return ms;
    }

    double elapsed() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
    }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point start_;
};

#endif
