cmake_minimum_required(VERSION 3.18)
project(fire_detection LANGUAGES CXX CUDA)

# 版本信息
set(FIRE_DETECTION_VERSION_MAJOR 1)
set(FIRE_DETECTION_VERSION_MINOR 0)
set(FIRE_DETECTION_VERSION_PATCH 0)
set(FIRE_DETECTION_VERSION "${FIRE_DETECTION_VERSION_MAJOR}.${FIRE_DETECTION_VERSION_MINOR}.${FIRE_DETECTION_VERSION_PATCH}")

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

# 符号可见性（用于共享库）
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

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

#=============================================================================
# 库源文件
#=============================================================================
set(FIRE_DETECTION_SOURCES
    src/api/fire_detection_api.cpp
    src/api/fire_detection_api_c.cpp
    src/detector/segment_detector.cpp
    src/detector/yolov8_postprocess.cpp
    src/roi/roi_extractor.cpp
    src/temporal/convlstm_classifier.cpp
)

# 依赖库列表
set(FIRE_DETECTION_LIBS
    ${NVINFER_LIB}
    ${NVONNXPARSER_LIB}
    ${SCT_LIB}
    ${TRT_ENGINE_LIB}
    ${OpenCV_LIBS}
    CUDA::cudart
)

#=============================================================================
# 静态库
#=============================================================================
add_library(fire_detection_static STATIC ${FIRE_DETECTION_SOURCES})

set_target_properties(fire_detection_static PROPERTIES
    OUTPUT_NAME fire_detection
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(fire_detection_static
    ${FIRE_DETECTION_LIBS}
)

#=============================================================================
# 共享库
#=============================================================================
add_library(fire_detection_shared SHARED ${FIRE_DETECTION_SOURCES})

set_target_properties(fire_detection_shared PROPERTIES
    OUTPUT_NAME fire_detection
    VERSION ${FIRE_DETECTION_VERSION}
    SOVERSION ${FIRE_DETECTION_VERSION_MAJOR}
)

target_compile_definitions(fire_detection_shared PRIVATE FIRE_DETECTION_EXPORTS)

target_link_libraries(fire_detection_shared
    ${FIRE_DETECTION_LIBS}
)

#=============================================================================
# 示例程序
#=============================================================================
add_executable(fire_detection_demo src/demo/main.cpp)

target_link_libraries(fire_detection_demo
    fire_detection_static
    ${FIRE_DETECTION_LIBS}
    pthread
)

#=============================================================================
# 安装规则
#=============================================================================

# 安装目标
install(TARGETS fire_detection_static fire_detection_shared fire_detection_demo
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# 安装公共头文件
install(FILES include/fire_detection_api.h
    DESTINATION include
)

# 安装内部头文件（可选，供开发者使用）
install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

#=============================================================================
# 打印配置摘要
#=============================================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "  Fire Detection Library v${FIRE_DETECTION_VERSION}")
message(STATUS "========================================")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA: ${CUDA_VERSION}")
message(STATUS "TensorRT: ${TENSORRT_ROOT}")
message(STATUS "  nvinfer: ${NVINFER_LIB}")
message(STATUS "  nvonnxparser: ${NVONNXPARSER_LIB}")
message(STATUS "CUDATT: ${SCT_ROOT}")
message(STATUS "  cuda_tensors: ${SCT_LIB}")
message(STATUS "  cuda_tensors_trt: ${TRT_ENGINE_LIB}")
message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "")
message(STATUS "Build targets:")
message(STATUS "  - fire_detection_static (libfire_detection.a)")
message(STATUS "  - fire_detection_shared (libfire_detection.so)")
message(STATUS "  - fire_detection_demo")
message(STATUS "========================================")
message(STATUS "")
