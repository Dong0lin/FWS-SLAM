cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

rosbuild_init()

# 可执行文件输出到 <本包目录>/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

IF(NOT ROS_BUILD_TYPE)
  SET(ROS_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${ROS_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3")
# 与主库保持一致（主库 -march=native，AVX 下 Eigen 固定尺寸类型要求 64 字节对齐；
# ROS 节点若不匹配会在运行时触发 Eigen unaligned assert）
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")


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

# cv_bridge 安装路径：仅在显式指定或默认位置存在时设置，否则交给 ROS 自动查找
if(DEFINED ENV{cv_bridge_DIR})
        set(cv_bridge_DIR "$ENV{cv_bridge_DIR}")
elseif(EXISTS "/usr/local/share/cv_bridge/cmake")
        set(cv_bridge_DIR "/usr/local/share/cv_bridge/cmake")
endif()

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
# 优先取环境变量，否则用默认值；可用 `cmake -DTENSORRT_ROOT=... -DCUDA_ROOT=...` 覆盖
if(DEFINED ENV{TENSORRT_ROOT})
        set(_TENSORRT_DEFAULT "$ENV{TENSORRT_ROOT}")
else()
        set(_TENSORRT_DEFAULT "/home/dl/TensorRT/TensorRT-8.6.0.12")
endif()
set(TENSORRT_ROOT "${_TENSORRT_DEFAULT}" CACHE PATH "TensorRT root directory")

if(DEFINED ENV{CUDA_ROOT})
        set(CUDA_ROOT "$ENV{CUDA_ROOT}")
elseif(EXISTS "/usr/local/cuda/targets/x86_64-linux")
        set(CUDA_ROOT "/usr/local/cuda/targets/x86_64-linux")
elseif(EXISTS "/usr/local/cuda-11.6/targets/x86_64-linux")
        set(CUDA_ROOT "/usr/local/cuda-11.6/targets/x86_64-linux")
endif()

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
# slam_runner_ros_stereo — Qt 共享内存 + ROS 相机订阅（长短焦双目语义 SLAM 版）
# ═══════════════════════════════════════════════════════════════
rosbuild_add_executable(slam_runner_ros_stereo
src/slam_runner_ros_stereo.cc
../../src/SlamInterface.cc
)

target_link_libraries(slam_runner_ros_stereo
${LIBS}
rt
)
set_target_properties(slam_runner_ros_stereo PROPERTIES
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

# ═══════════════════════════════════════════════════════════════
# MultiFocal_Stereo — Pangolin 版长短焦双目 SLAM 节点（拼接图左右半）
# ═══════════════════════════════════════════════════════════════
rosbuild_add_executable(MultiFocal_Stereo
src/ros_multifocal_stereo.cc
)

target_link_libraries(MultiFocal_Stereo
${LIBS}
)
set_target_properties(MultiFocal_Stereo PROPERTIES
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
