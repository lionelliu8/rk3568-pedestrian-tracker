#ifndef RK3568_YOLOV5_POSTPROCESS_H
#define RK3568_YOLOV5_POSTPROCESS_H

#include "common.h"

#ifndef OBJ_NAME_MAX_SIZE
#define OBJ_NAME_MAX_SIZE 64
#endif
#ifndef OBJ_NUMB_MAX_SIZE
#define OBJ_NUMB_MAX_SIZE 128
#endif
#define OBJ_CLASS_NUM 80
#define NMS_THRESH 0.45f
#define BOX_THRESH 0.25f
#define PROP_BOX_SIZE (5 + OBJ_CLASS_NUM)

struct rknn_app_context_t;

struct object_detect_result {
    image_rect_t box;
    float prop;
    int cls_id;
};

struct object_detect_result_list {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
};

int init_post_process();
void deinit_post_process();
const char* coco_cls_to_name(int cls_id);
int post_process(rknn_app_context_t* app_ctx, void* outputs, letterbox_t* letter_box,
                 float conf_threshold, float nms_threshold, object_detect_result_list* results);

#endif
