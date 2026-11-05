cmake_minimum_required(VERSION 3.16)
project(pedestrian_tracker VERSION 2.0.0 LANGUAGES C CXX)

# RK3568 pedestrian tracker: YOLO26 detection + DeepSORT/OSNet tracking on the RKNPU.
# Cross-build with scripts/build_rk3568.sh (see README.md).

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(RKNN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknn" CACHE PATH "RKNN runtime (headers + librknnrt.so)")
set(RGA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/rga" CACHE PATH "RGA SDK (headers + librga.so)")
set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3" CACHE PATH "Eigen3 headers (header-only)")

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Threads REQUIRED)

add_library(rknnrt SHARED IMPORTED)
set_target_properties(rknnrt PROPERTIES
    IMPORTED_LOCATION "${RKNN_ROOT}/lib/librknnrt.so"
    INTERFACE_INCLUDE_DIRECTORIES "${RKNN_ROOT}/include")

add_library(rga SHARED IMPORTED)
set_target_properties(rga PROPERTIES
    IMPORTED_LOCATION "${RGA_ROOT}/lib/librga.so"
    INTERFACE_INCLUDE_DIRECTORIES "${RGA_ROOT}/include")

# DeepSORT (third-party derived code, kept warning-free only at default level)
add_library(deepsort STATIC
    src/tracker/deepsort.cpp
    src/tracker/featuretensor.cpp
    src/tracker/hungarianoper.cpp
    src/tracker/kalmanfilter.cpp
    src/tracker/linear_assignment.cpp
    src/tracker/nn_matching.cpp
    src/tracker/rknn_fp.cpp
    src/tracker/track.cpp
    src/tracker/tracker.cpp
)
target_include_directories(deepsort PUBLIC src/tracker src/common ${EIGEN3_INCLUDE_DIR})
target_link_libraries(deepsort PUBLIC ${OpenCV_LIBS} rknnrt Threads::Threads)

add_executable(pedestrian_tracker
    src/main.cpp
    src/app/options.cpp
    src/app/mot_writer.cpp
    src/app/overlay.cpp
    src/detector/yolo26_detector.cpp
    src/io/drm_display.cpp
    src/io/frame_source.cpp
    src/io/image_sequence.cpp
    src/io/v4l2_camera.cpp
)
target_include_directories(pedestrian_tracker PRIVATE src/app src/common src/detector src/io)
target_compile_options(pedestrian_tracker PRIVATE -Wall -Wextra)
target_link_libraries(pedestrian_tracker PRIVATE deepsort rga ${OpenCV_LIBS} rknnrt Threads::Threads dl)
# The board's libstdc++ is older than the cross compiler's; link the C++ runtime statically.
target_link_options(pedestrian_tracker PRIVATE -static-libstdc++ -static-libgcc)
set_target_properties(pedestrian_tracker PROPERTIES INSTALL_RPATH "$ORIGIN/lib")

# Camera-free detector check (installed as tests/detector_selftest)
add_executable(detector_selftest tests/detector_selftest.cpp src/detector/yolo26_detector.cpp)
target_include_directories(detector_selftest PRIVATE src/common src/detector)
target_compile_options(detector_selftest PRIVATE -Wall -Wextra)
target_link_libraries(detector_selftest PRIVATE rga ${OpenCV_LIBS} rknnrt Threads::Threads dl)
target_link_options(detector_selftest PRIVATE -static-libstdc++ -static-libgcc)
set_target_properties(detector_selftest PROPERTIES INSTALL_RPATH "$ORIGIN/../lib")

# Board package layout: <prefix>/{pedestrian_tracker, run.sh, lib/, model/}
install(TARGETS pedestrian_tracker RUNTIME DESTINATION .)
install(TARGETS detector_selftest RUNTIME DESTINATION tests)
install(PROGRAMS scripts/run.sh DESTINATION .)
install(FILES README.md DESTINATION .)
install(FILES "${RKNN_ROOT}/lib/librknnrt.so" "${RGA_ROOT}/lib/librga.so" DESTINATION lib)
install(FILES
    model/yolo26n_rk3568.rknn
    model/osnet_x0_25_market_rk3568.rknn
    model/coco_80_labels_list.txt
    DESTINATION model)
