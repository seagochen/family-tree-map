cmake_minimum_required(VERSION 3.18)
project(fire_detection_trt LANGUAGES CXX CUDA)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find CUDA
find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Find OpenCV
find_package(OpenCV REQUIRED)

# TensorRT paths (adjust as needed)
set(TENSORRT_ROOT "/opt/tensorrt/tensorrt" CACHE PATH "TensorRT installation root")

# Find TensorRT libraries
find_library(NVINFER_LIB nvinfer
    HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib64 /usr/lib/x86_64-linux-gnu)
find_library(NVONNXPARSER_LIB nvonnxparser
    HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib64 /usr/lib/x86_64-linux-gnu)

if(NOT NVINFER_LIB)
    message(FATAL_ERROR "Could not find nvinfer library. Please set TENSORRT_ROOT.")
endif()

if(NOT NVONNXPARSER_LIB)
    message(FATAL_ERROR "Could not find nvonnxparser library. Please set TENSORRT_ROOT.")
endif()

# CUDATT paths
set(SCT_ROOT "/opt/cudatt" CACHE PATH "CUDATT installation root")
set(SCT_INCLUDE_DIR "${SCT_ROOT}/include")
set(SCT_LIB_DIR "${SCT_ROOT}/lib")

find_library(SCT_LIB cuda_tensors
    HINTS ${SCT_LIB_DIR})

find_library(TRT_ENGINE_LIB cuda_tensors_trt
    HINTS ${SCT_LIB_DIR})

if(NOT SCT_LIB)
    message(FATAL_ERROR "Could not find cuda_tensors library at ${SCT_LIB_DIR}")
endif()

if(NOT TRT_ENGINE_LIB)
    message(FATAL_ERROR "Could not find cuda_tensors_trt library at ${SCT_LIB_DIR}")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_ROOT}/include
    ${SCT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# Source files
set(TRT_ENGINE_SOURCES
    src/yolov8_postprocess.cpp
    src/fire_pipeline.cpp
    src/roi_extractor.cpp
    src/convlstm_classifier.cpp
)

# Create static library for trt_engine
add_library(trt_engine STATIC ${TRT_ENGINE_SOURCES})

target_link_libraries(trt_engine
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}
    ${SCT_LIB}
    ${TRT_ENGINE_LIB}
    ${OpenCV_LIBS}
    CUDA::cudart
)

# Main executable
add_executable(fire_detection_test src/main.cpp)

target_link_libraries(fire_detection_test
    trt_engine
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}
    ${SCT_LIB}
    ${TRT_ENGINE_LIB}
    ${OpenCV_LIBS}
    CUDA::cudart
    pthread
)

# Install targets
install(TARGETS fire_detection_test trt_engine
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# Print configuration summary
message(STATUS "")
message(STATUS "========================================")
message(STATUS "  Fire Detection TRT Configuration")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA: ${CUDA_VERSION}")
message(STATUS "TensorRT: ${TENSORRT_ROOT}")
message(STATUS "  nvinfer: ${NVINFER_LIB}")
message(STATUS "  nvonnxparser: ${NVONNXPARSER_LIB}")
message(STATUS "SimpleCudaToolkits: ${SCT_ROOT}")
message(STATUS "  cuda_tensors: ${SCT_LIB}")
message(STATUS "  cuda_tensors_trt: ${TRT_ENGINE_LIB}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "========================================")
message(STATUS "")
