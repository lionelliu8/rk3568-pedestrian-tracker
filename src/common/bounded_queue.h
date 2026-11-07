#ifndef PEDESTRIAN_TRACKER_BOUNDED_QUEUE_H
#define PEDESTRIAN_TRACKER_BOUNDED_QUEUE_H

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>

// Small blocking queue between pipeline stages.
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {}

    // Blocks while full. Returns false if the queue was closed.
    bool push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [&] { return closed_ || items_.size() < capacity_; });
        if (closed_) return false;
        items_.push_back(std::move(item));
        not_empty_.notify_one();
        return true;
    }

    // Never blocks: drops the oldest item when full (live camera: prefer fresh frames).
    // Returns true if an item was dropped.
    bool push_drop_oldest(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) return false;
        bool dropped = false;
        if (items_.size() >= capacity_) {
            items_.pop_front();
            dropped = true;
        }
        items_.push_back(std::move(item));
        not_empty_.notify_one();
        return dropped;
    }

    // Blocks until an item is available; empty result once closed and drained.
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [&] { return closed_ || !items_.empty(); });
        if (items_.empty()) return std::nullopt;
        T item = std::move(items_.front());
        items_.pop_front();
        not_full_.notify_one();
        return item;
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

private:
    std::size_t capacity_;
    std::deque<T> items_;
    bool closed_ = false;
    std::mutex mutex_;
    std::condition_variable not_empty_, not_full_;
};

#endif
