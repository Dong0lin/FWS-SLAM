cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

rosbuild_init()

# 可执行文件输出到 /home/dl/FWS-SLAM/Examples/FWS-SLAM-ROS/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY /home/dl/FWS-SLAM/Examples/FWS-SLAM-ROS/bin)

IF(NOT ROS_BUILD_TYPE)
  SET(ROS_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${ROS_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3")


# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   set(CMAKE_CXX_STANDARD 14)
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../cmake_modules)

set(cv_bridge_DIR /usr/local/share/cv_bridge/cmake)

find_package(OpenCV 4.2 EXACT REQUIRED)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

# ── 语义 SLAM 额外路径：TensorRT + CUDA ──
set(TENSORRT_ROOT /home/dl/TensorRT/TensorRT-8.6.0.12)
set(CUDA_ROOT /usr/local/cuda-11.6/targets/x86_64-linux)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/../../
${PROJECT_SOURCE_DIR}/../../include
${PROJECT_SOURCE_DIR}/../../include/CameraModels
${Pangolin_INCLUDE_DIRS}
# TensorRT + CUDA (语义 SLAM 依赖)
${TENSORRT_ROOT}/include
${CUDA_ROOT}/include
)

link_directories(
${TENSORRT_ROOT}/lib
${CUDA_ROOT}/lib
)

set(LIBS 
${OpenCV_LIBS} 
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/../../Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/../../Thirdparty/g2o/lib/libg2o.so
${PROJECT_SOURCE_DIR}/../../lib/libFWS-SLAM.so
-lboost_system
# TensorRT + CUDA (语义 SLAM 依赖)
-lnvinfer
-lnvinfer_plugin
-lnvonnxparser
-lnvparsers
-lcudart
)

# ═══════════════════════════════════════════════════════════════
# slam_runner_ros — Qt 共享内存 + ROS 相机订阅（语义 SLAM 版）
# ═══════════════════════════════════════════════════════════════
rosbuild_add_executable(slam_runner_ros
src/slam_runner_ros.cc
../../src/SlamInterface.cc
)

target_link_libraries(slam_runner_ros
${LIBS}
rt
)
# 强制 RPATH 优先于 LD_LIBRARY_PATH（防止 Qt 的 LD_LIBRARY_PATH 覆盖）
set_target_properties(slam_runner_ros PROPERTIES
    LINK_FLAGS "-Wl,--disable-new-dtags -Wl,-rpath,${PROJECT_SOURCE_DIR}/../../lib:/usr/local/lib:${PROJECT_SOURCE_DIR}/../../Thirdparty/DBoW2/lib:${PROJECT_SOURCE_DIR}/../../Thirdparty/g2o/lib:${TENSORRT_ROOT}/lib:${CUDA_ROOT}/lib")

# ═══════════════════════════════════════════════════════════════
# Mono_Stereo_Left — Pangolin 版单目 SLAM 节点（左半图裁剪）
# ═══════════════════════════════════════════════════════════════
rosbuild_add_executable(Mono_Stereo_Left
src/ros_mono_stereo_left.cc
)

target_link_libraries(Mono_Stereo_Left
${LIBS}
)
set_target_properties(Mono_Stereo_Left PROPERTIES
    LINK_FLAGS "-Wl,--disable-new-dtags -Wl,-rpath,${PROJECT_SOURCE_DIR}/../../lib:/usr/local/lib:${PROJECT_SOURCE_DIR}/../../Thirdparty/DBoW2/lib:${PROJECT_SOURCE_DIR}/../../Thirdparty/g2o/lib:${TENSORRT_ROOT}/lib:${CUDA_ROOT}/lib")

# Node for monocular camera
rosbuild_add_executable(Mono
src/ros_mono.cc
)

target_link_libraries(Mono
${LIBS}
)

# Node for monocular camera (Augmented Reality Demo)
rosbuild_add_executable(MonoAR
src/AR/ros_mono_ar.cc
src/AR/ViewerAR.h
src/AR/ViewerAR.cc
)

target_link_libraries(MonoAR
${LIBS}
)

# Node for stereo camera
rosbuild_add_executable(Stereo
src/ros_stereo.cc
)

target_link_libraries(Stereo
${LIBS}
)

# Node for RGB-D camera
rosbuild_add_executable(RGBD
src/ros_rgbd.cc
)

target_link_libraries(RGBD
${LIBS}
)

# Node for monocular-inertial camera
rosbuild_add_executable(Mono_Inertial
src/ros_mono_inertial.cc
)

target_link_libraries(Mono_Inertial
${LIBS}
)

# Node for stereo-inertial camera
rosbuild_add_executable(Stereo_Inertial
src/ros_stereo_inertial.cc
)

target_link_libraries(Stereo_Inertial
${LIBS}
)
