cmake_minimum_required(VERSION 3.4.1)

project(rknn_yolov8_demo)

include_directories(${CMAKE_SOURCE_DIR}/include)

set(RKNN_RT_LIB ${CMAKE_SOURCE_DIR}/lib/librknnrt.so)

set(OpenCV_DIR /root/lib/opencv454_install/lib/cmake/opencv4)
find_package(OpenCV REQUIRED)

find_package(Threads REQUIRED) # find and include pthreads

add_executable(${PROJECT_NAME}_picture
        src/picture_demo.cc
        src/rknn_net.cc
        src/yolov8_post.cc
)

add_executable(${PROJECT_NAME}_video
        src/video_demo.cc
        src/rknn_net.cc
        src/yolov8_post.cc
)

target_link_libraries(${PROJECT_NAME}_picture
  ${RKNN_RT_LIB}
  ${OpenCV_LIBS}
)

target_link_libraries(${PROJECT_NAME}_video
  ${RKNN_RT_LIB}
  ${OpenCV_LIBS}
  Threads::Threads
)