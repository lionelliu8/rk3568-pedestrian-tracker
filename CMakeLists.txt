cmake_minimum_required(VERSION 3.16)
project(yolov5_deepsort_rk3568 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_INSTALL_RPATH "$ORIGIN/lib")

set(RKNN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknn" CACHE PATH "RKNN runtime directory")
set(EIGEN3_INCLUDE_DIR "/usr/include/eigen3" CACHE PATH "Eigen3 include directory")
set(RGA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/rga" CACHE PATH "RGA SDK directory")

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc)
find_package(Threads REQUIRED)

set(DEEPSORT_SOURCES
    deepsort/src/deepsort.cpp
    deepsort/src/featuretensor.cpp
    deepsort/src/hungarianoper.cpp
    deepsort/src/kalmanfilter.cpp
    deepsort/src/linear_assignment.cpp
    deepsort/src/nn_matching.cpp
    deepsort/src/track.cpp
    deepsort/src/tracker.cpp
    deepsort/src/rknn_fp.cpp
)

add_library(deepsort STATIC ${DEEPSORT_SOURCES})
target_include_directories(deepsort PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/deepsort/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${RKNN_ROOT}/include
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(deepsort PUBLIC ${OpenCV_LIBS} ${RKNN_ROOT}/lib/librknnrt.so Threads::Threads)

add_executable(yolov5_deepsort
    yolov5_deepsort.cpp
    rk3568/detector.cpp
    rk3568/postprocess.cc
)
target_include_directories(yolov5_deepsort PRIVATE
    ${RGA_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/rk3568
    ${CMAKE_CURRENT_SOURCE_DIR}/deepsort/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${RKNN_ROOT}/include
    ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(yolov5_deepsort PRIVATE deepsort ${OpenCV_LIBS} ${RKNN_ROOT}/lib/librknnrt.so Threads::Threads dl)
target_link_libraries(yolov5_deepsort PRIVATE ${RGA_ROOT}/lib/librga.so)

install(TARGETS yolov5_deepsort RUNTIME DESTINATION .)
install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/run.sh DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/README_RK3568.md DESTINATION .)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/RGA_TEST.md DESTINATION .)
install(FILES ${RKNN_ROOT}/lib/librknnrt.so DESTINATION lib)
install(FILES ${RGA_ROOT}/lib/librga.so DESTINATION lib)
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/model/yolov5_rk3568.rknn
    ${CMAKE_CURRENT_SOURCE_DIR}/model/osnet_x0_25_market_rk3568.rknn
    ${CMAKE_CURRENT_SOURCE_DIR}/model/coco_80_labels_list.txt
    DESTINATION model
)
