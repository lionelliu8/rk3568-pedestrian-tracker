#ifndef PEDESTRIAN_TRACKER_POSIX_UTIL_H
#define PEDESTRIAN_TRACKER_POSIX_UTIL_H

#include <cerrno>

#include <sys/ioctl.h>

// ioctl() that retries when interrupted by a signal.
inline int xioctl(int fd, unsigned long request, void* argument) {
    int result;
    do {
        result = ioctl(fd, request, argument);
    } while (result == -1 && errno == EINTR);
    return result;
}

#endif
