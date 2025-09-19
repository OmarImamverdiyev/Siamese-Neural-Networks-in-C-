cmake_minimum_required(VERSION 3.10)
project(SiamFCTracker)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

# Path to ONNX Runtime (relative to project folder)
set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-1.19.0)
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(tracker tracking.cpp)
target_link_libraries(tracker ${OpenCV_LIBS} onnxruntime)

