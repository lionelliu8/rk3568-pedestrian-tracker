#ifndef RK3568_IMAGE_TYPES_H
#define RK3568_IMAGE_TYPES_H

struct image_rect_t {
    int left;
    int top;
    int right;
    int bottom;
};

struct letterbox_t {
    int x_pad;
    int y_pad;
    float scale;
};

#endif
