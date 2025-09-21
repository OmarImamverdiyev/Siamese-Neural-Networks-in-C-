cmake_minimum_required(VERSION 3.15)
project(tracking_siamtpn)

set(CMAKE_CXX_STANDARD 17)

# OpenCV
set(OpenCV_DIR "C:/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ONNX Runtime
set(ONNXRUNTIME_DIR "C:/onnxruntime")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# Executable
add_executable(tracker tracking.cpp)

# Libraries
target_link_libraries(tracker
    ${OpenCV_LIBS}
    onnxruntime
)
