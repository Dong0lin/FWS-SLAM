cmake_minimum_required(VERSION 3.12)
project(FWS-SLAM)

# 启用CUDA语言支持（关键修复）
enable_language(CUDA)

# Suppress CMake developer warnings about removed/changed Find modules
if(POLICY CMP0146)
        cmake_policy(SET CMP0146 NEW)
endif()
if(POLICY CMP0057)
        cmake_policy(SET CMP0057 NEW)
endif()
if(POLICY CMP0167)
        cmake_policy(SET CMP0167 NEW)
endif()

# Put all executables into a single `bin` folder for easier management
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)


IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
# -march=native 放基础选项（而不是只放 Release）：
# 否则 Debug 配置下会丢失该优化，单目等重计算路径速度下降约 10%（29fps→26fps）
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

# 设置CUDA编译选项（参考yolov11.txt）
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_80,code=sm_80")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -std=c++14 --use_fast_math")

# Check C++14 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++14" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++14.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++14 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 4.2 EXACT REQUIRED)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 4.2 not found.")
   endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Boost COMPONENTS thread REQUIRED)

# find_package(realsense2)
# Prefer modern CMake CUDA package when available, fall back to legacy FindCUDA

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
        message(STATUS "Found CUDAToolkit")
else()
        # Add common CUDA install prefixes so CMake can locate CUDAConfig.cmake
        list(APPEND CMAKE_PREFIX_PATH /usr/local/cuda /usr/local/cuda-11.6)

        # If user didn't set CUDA_DIR, try common locations and cache the value
        if(NOT DEFINED CUDA_DIR)
                if(EXISTS "/usr/local/cuda")
                        set(CUDA_DIR "/usr/local/cuda" CACHE PATH "CUDA root dir")
                elseif(EXISTS "/usr/local/cuda-11.6")
                        set(CUDA_DIR "/usr/local/cuda-11.6" CACHE PATH "CUDA root dir")
                endif()
        endif()

        find_package(CUDA QUIET)
        if(CUDA_FOUND)
                message(STATUS "Found legacy CUDA (FindCUDA)")
        else()
                message(WARNING "CUDA not found by CUDAToolkit or FindCUDA. You can set CUDA_DIR or CMAKE_PREFIX_PATH to /usr/local/cuda or /usr/local/cuda-11.6")
        endif()
endif()

# Locate libcudart (CUDA runtime). Try common CUDA library locations as hints.
find_library(CUDART_LIBRARY
                                                 cudart
                                                 HINTS
                                                 /usr/local/cuda/lib64
                                                 /usr/local/cuda-11.6/lib64
                                                 /usr/local/cuda-11.6/targets/x86_64-linux/lib
                                                 /usr/lib/x86_64-linux-gnu)
if(CUDART_LIBRARY)
        message(STATUS "Found libcudart: ${CUDART_LIBRARY}")
else()
        message(WARNING "libcudart not found by find_library; linking may fail. You can set CUDART_LIBRARY manually.")
endif()

# CUDA 根目录（TensorRT 插件头文件/库搜索用）
# 优先取环境变量 CUDA_ROOT → CUDAToolkit → 常见安装位置
if(DEFINED ENV{CUDA_ROOT})
        set(CUDA_ROOT "$ENV{CUDA_ROOT}")
elseif(CUDAToolkit_FOUND)
        set(CUDA_ROOT "${CUDAToolkit_TARGET_DIR}")
elseif(EXISTS "/usr/local/cuda")
        set(CUDA_ROOT "/usr/local/cuda")
endif()


# TensorRT 安装路径：优先取环境变量 TENSORRT_ROOT，否则用下面的默认值
# 可用 `cmake -DTENSORRT_ROOT=/path/to/TensorRT ..` 覆盖
if(DEFINED ENV{TENSORRT_ROOT})
        set(_TENSORRT_DEFAULT "$ENV{TENSORRT_ROOT}")
else()
        set(_TENSORRT_DEFAULT "/home/dl/TensorRT/TensorRT-8.6.0.12")
endif()
set(TENSORRT_ROOT "${_TENSORRT_DEFAULT}" CACHE PATH "TensorRT root directory")

# 添加TensorRT头文件目录（确保包含NvInferRuntimeCommon.h）
include_directories(${TENSORRT_ROOT}/include)

# 添加TensorRT库文件目录（用于链接时查找库）
link_directories(${TENSORRT_ROOT}/lib)



include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/include/CameraModels
${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
${OpenCV_INCLUDE_DIRS}
${Boost_INCLUDE_DIR}
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# 定义CUDA源文件（关键修复）
set(CUDA_SOURCES src/preprocess.cu)

add_library(${PROJECT_NAME} SHARED
src/Detector.cc
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Atlas.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Viewer.cc
src/ImuTypes.cc
src/G2oTypes.cc
src/CameraModels/Pinhole.cpp
src/CameraModels/KannalaBrandt8.cpp
src/OptimizableTypes.cpp
src/MLPnPsolver.cpp
src/GeometricTools.cc
src/TwoViewReconstruction.cc
src/Config.cc
src/Settings.cc
src/SlamInterface.cc
${CUDA_SOURCES}  # 添加CUDA源文件到库目标（关键修复）
include/System.h
include/Detector.h
include/logging.h
include/Tracking.h
include/LocalMapping.h
include/LoopClosing.h
include/ORBextractor.h
include/ORBmatcher.h
include/FrameDrawer.h
include/Converter.h
include/MapPoint.h
include/KeyFrame.h
include/Atlas.h
include/Map.h
include/MapDrawer.h
include/Optimizer.h
include/Frame.h
include/KeyFrameDatabase.h
include/Sim3Solver.h
include/Viewer.h
include/ImuTypes.h
include/G2oTypes.h
include/CameraModels/GeometricCamera.h
include/CameraModels/Pinhole.h
include/CameraModels/KannalaBrandt8.h
include/OptimizableTypes.h
include/MLPnPsolver.h
include/GeometricTools.h
include/TwoViewReconstruction.h
include/SerializationUtils.h
include/Config.h
include/Settings.h

include/common_coco.h
include/macros.h
include/cuda_utils.h
include/preprocess.h
)

add_subdirectory(Thirdparty/g2o)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${Boost_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
-lboost_serialization
-lcrypto
-lnvinfer
-lnvinfer_plugin
-lnvonnxparser
-lnvparsers
${CUDART_LIBRARY}
-lpthread
)

# YOLOX---------------------------------------------------------------------------------------------------------------------------------
# cuda
if(CUDA_ROOT)
        include_directories(${CUDA_ROOT}/include)
        link_directories(${CUDA_ROOT}/lib)
endif()

#------------------------------------------------------------------------------------------------------------------------------------------

# 打印TensorRT头文件路径（用于调试）
message("TensorRT include dir: ${TENSORRT_ROOT}/include")
# 确认路径是否正确
include_directories(${TENSORRT_ROOT}/include)


set(CUDA_SOURCES src/preprocess.cu)



# Build examples

# RGB-D examples

# add_executable(rgbd_tum
#         Examples/RGB-D/rgbd_tum.cc)
# target_link_libraries(rgbd_tum ${PROJECT_NAME})

# if(realsense2_FOUND)
#     add_executable(rgbd_realsense_D435i
#             Examples/RGB-D/rgbd_realsense_D435i.cc)
#     target_link_libraries(rgbd_realsense_D435i ${PROJECT_NAME})
# endif()


# # RGB-D inertial examples

# if(realsense2_FOUND)
#     add_executable(rgbd_inertial_realsense_D435i
#             Examples/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc)
#     target_link_libraries(rgbd_inertial_realsense_D435i ${PROJECT_NAME})
# endif()

#Stereo examples

add_executable(stereo_kitti
        Examples/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(stereo_multifocal
        Examples/Stereo/stereo_multifocal.cc)
target_link_libraries(stereo_multifocal ${PROJECT_NAME})

add_executable(stereo_mf_verify
        Examples/Stereo/mf_verify.cc)
target_link_libraries(stereo_mf_verify ${PROJECT_NAME})

add_executable(stereo_euroc
        Examples/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc ${PROJECT_NAME})

add_executable(stereo_tum_vi
        Examples/Stereo/stereo_tum_vi.cc)
target_link_libraries(stereo_tum_vi ${PROJECT_NAME})

# if(realsense2_FOUND)
#     add_executable(stereo_realsense_t265
#             Examples/Stereo/stereo_realsense_t265.cc)
#     target_link_libraries(stereo_realsense_t265 ${PROJECT_NAME})

#     add_executable(stereo_realsense_D435i
#             Examples/Stereo/stereo_realsense_D435i.cc)
#     target_link_libraries(stereo_realsense_D435i ${PROJECT_NAME})
# endif()



#单独提取特征点
add_executable(extractor_orb
        Examples/Monocular/extractor_orb.cc)
target_link_libraries(extractor_orb ${PROJECT_NAME})

#Monocular examples
add_executable(mono_tum
        Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti
        Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

add_executable(mono_euroc
        Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})

add_executable(mono_euroc_mine
        Examples/Monocular/mono_euroc_mine.cc)
target_link_libraries(mono_euroc_mine ${PROJECT_NAME})

add_executable(mono_euroc_mine_png
        Examples/Monocular/mono_euroc_mine_png.cc)
target_link_libraries(mono_euroc_mine_png ${PROJECT_NAME})

# SLAM Runner — 独立进程，写共享内存供 Qt 界面读取
add_executable(slam_runner
        Examples/Monocular/slam_runner.cc)
target_link_libraries(slam_runner ${PROJECT_NAME} rt)

# 双目/长短焦 Qt 界面 runner（独立于单目 slam_runner）
add_executable(slam_runner_stereo
        Examples/Stereo/slam_runner_stereo.cc)
target_link_libraries(slam_runner_stereo ${PROJECT_NAME} rt)

add_executable(mono_tum_vi
        Examples/Monocular/mono_tum_vi.cc)
target_link_libraries(mono_tum_vi ${PROJECT_NAME})

# add_executable(myvideo2
#         Examples/Monocular/myvideo2.cc)
# target_link_libraries(myvideo2 ${PROJECT_NAME})

# if(realsense2_FOUND)
#     add_executable(mono_realsense_t265
#             Examples/Monocular/mono_realsense_t265.cc)
#     target_link_libraries(mono_realsense_t265 ${PROJECT_NAME})

#     add_executable(mono_realsense_D435i
#             Examples/Monocular/mono_realsense_D435i.cc)
#     target_link_libraries(mono_realsense_D435i ${PROJECT_NAME})
# endif()

# #Monocular inertial examples

# add_executable(mono_inertial_euroc
#         Examples/Monocular-Inertial/mono_inertial_euroc.cc)
# target_link_libraries(mono_inertial_euroc ${PROJECT_NAME})

# add_executable(mono_inertial_tum_vi
#         Examples/Monocular-Inertial/mono_inertial_tum_vi.cc)
# target_link_libraries(mono_inertial_tum_vi ${PROJECT_NAME})

# # add_executable(mono_inertial_kitti
# #         Examples/Monocular-Inertial/mono_inertial_kitti.cc)
# # target_link_libraries(mono_inertial_kitti ${PROJECT_NAME})

# if(realsense2_FOUND)
#     add_executable(mono_inertial_realsense_t265
#             Examples/Monocular-Inertial/mono_inertial_realsense_t265.cc)
#     target_link_libraries(mono_inertial_realsense_t265 ${PROJECT_NAME})

#     add_executable(mono_inertial_realsense_D435i
#             Examples/Monocular-Inertial/mono_inertial_realsense_D435i.cc)
#     target_link_libraries(mono_inertial_realsense_D435i ${PROJECT_NAME})
# endif()

# #Stereo Inertial examples

# add_executable(stereo_inertial_euroc
#         Examples/Stereo-Inertial/stereo_inertial_euroc.cc)
# target_link_libraries(stereo_inertial_euroc ${PROJECT_NAME})

# add_executable(stereo_inertial_tum_vi
#         Examples/Stereo-Inertial/stereo_inertial_tum_vi.cc)
# target_link_libraries(stereo_inertial_tum_vi ${PROJECT_NAME})

# # add_executable(stereo_inertial_kitti
# #         Examples/Stereo-Inertial/stereo_inertial_kitti.cc)
# # target_link_libraries(stereo_inertial_kitti ${PROJECT_NAME})

# if(realsense2_FOUND)
#     add_executable(stereo_inertial_realsense_t265
#             Examples/Stereo-Inertial/stereo_inertial_realsense_t265.cc)
#     target_link_libraries(stereo_inertial_realsense_t265 ${PROJECT_NAME})

#     add_executable(stereo_inertial_realsense_D435i
#             Examples/Stereo-Inertial/stereo_inertial_realsense_D435i.cc)
#     target_link_libraries(stereo_inertial_realsense_D435i ${PROJECT_NAME})
# endif()

# # Calibration examples (use global bin)
# if(realsense2_FOUND)
#     add_executable(recorder_realsense_D435i
#             Examples/Calibration/recorder_realsense_D435i.cc)
#     target_link_libraries(recorder_realsense_D435i ${PROJECT_NAME})

#     add_executable(recorder_realsense_T265
#             Examples/Calibration/recorder_realsense_T265.cc)
#     target_link_libraries(recorder_realsense_T265 ${PROJECT_NAME})
# endif()

# #Old examples

# # RGB-D examples (old) - use global bin

# add_executable(rgbd_tum_old
#         Examples_old/RGB-D/rgbd_tum.cc)
# target_link_libraries(rgbd_tum_old ${PROJECT_NAME})

# if(realsense2_FOUND)
#     add_executable(rgbd_realsense_D435i_old
#             Examples_old/RGB-D/rgbd_realsense_D435i.cc)
#     target_link_libraries(rgbd_realsense_D435i_old ${PROJECT_NAME})
# endif()


# # RGB-D inertial examples (old) - use global bin

# if(realsense2_FOUND)
#     add_executable(rgbd_inertial_realsense_D435i_old
#             Examples_old/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc)
#     target_link_libraries(rgbd_inertial_realsense_D435i_old ${PROJECT_NAME})
# endif()

# #Stereo examples (old) - use global bin

# add_executable(stereo_kitti_old
#         Examples_old/Stereo/stereo_kitti.cc)
# target_link_libraries(stereo_kitti_old ${PROJECT_NAME})

# add_executable(stereo_euroc_old
#         Examples_old/Stereo/stereo_euroc.cc)
# target_link_libraries(stereo_euroc_old ${PROJECT_NAME})

# add_executable(stereo_tum_vi_old
#         Examples_old/Stereo/stereo_tum_vi.cc)
# target_link_libraries(stereo_tum_vi_old ${PROJECT_NAME})

# if(realsense2_FOUND)
#     add_executable(stereo_realsense_t265_old
#             Examples_old/Stereo/stereo_realsense_t265.cc)
#     target_link_libraries(stereo_realsense_t265_old ${PROJECT_NAME})

#     add_executable(stereo_realsense_D435i_old
#             Examples_old/Stereo/stereo_realsense_D435i.cc)
#     target_link_libraries(stereo_realsense_D435i_old ${PROJECT_NAME})
# endif()

# #Monocular examples (old) - use global bin

# add_executable(mono_tum_old
#         Examples_old/Monocular/mono_tum.cc)
# target_link_libraries(mono_tum_old ${PROJECT_NAME})

# add_executable(mono_kitti_old
#         Examples_old/Monocular/mono_kitti.cc)
# target_link_libraries(mono_kitti_old ${PROJECT_NAME})

# add_executable(mono_euroc_old
#         Examples_old/Monocular/mono_euroc.cc)
# target_link_libraries(mono_euroc_old ${PROJECT_NAME})

# add_executable(mono_tum_vi_old
#         Examples_old/Monocular/mono_tum_vi.cc)
# target_link_libraries(mono_tum_vi_old ${PROJECT_NAME})

# if(realsense2_FOUND)
#     add_executable(mono_realsense_t265_old
#             Examples_old/Monocular/mono_realsense_t265.cc)
#     target_link_libraries(mono_realsense_t265_old ${PROJECT_NAME})

#     add_executable(mono_realsense_D435i_old
#             Examples_old/Monocular/mono_realsense_D435i.cc)
#     target_link_libraries(mono_realsense_D435i_old ${PROJECT_NAME})
# endif()

# #Monocular inertial examples (old) - use global bin

# add_executable(mono_inertial_euroc_old
#         Examples_old/Monocular-Inertial/mono_inertial_euroc.cc)
# target_link_libraries(mono_inertial_euroc_old ${PROJECT_NAME})

# add_executable(mono_inertial_tum_vi_old
#         Examples_old/Monocular-Inertial/mono_inertial_tum_vi.cc)
# target_link_libraries(mono_inertial_tum_vi_old ${PROJECT_NAME})

# if(realsense2_FOUND)
#     add_executable(mono_inertial_realsense_t265_old
#             Examples_old/Monocular-Inertial/mono_inertial_realsense_t265.cc)
#     target_link_libraries(mono_inertial_realsense_t265_old ${PROJECT_NAME})

#     add_executable(mono_inertial_realsense_D435i_old
#             Examples_old/Monocular-Inertial/mono_inertial_realsense_D435i.cc)
#     target_link_libraries(mono_inertial_realsense_D435i_old ${PROJECT_NAME})
# endif()

# #Stereo Inertial examples (old) - use global bin

# add_executable(stereo_inertial_euroc_old
#         Examples_old/Stereo-Inertial/stereo_inertial_euroc.cc)
# target_link_libraries(stereo_inertial_euroc_old ${PROJECT_NAME})

# add_executable(stereo_inertial_tum_vi_old
#         Examples_old/Stereo-Inertial/stereo_inertial_tum_vi.cc)
# target_link_libraries(stereo_inertial_tum_vi_old ${PROJECT_NAME})

# if(realsense2_FOUND)
#     add_executable(stereo_inertial_realsense_t265_old
#             Examples_old/Stereo-Inertial/stereo_inertial_realsense_t265.cc)
#     target_link_libraries(stereo_inertial_realsense_t265_old ${PROJECT_NAME})

#     add_executable(stereo_inertial_realsense_D435i_old
#             Examples_old/Stereo-Inertial/stereo_inertial_realsense_D435i.cc)
#     target_link_libraries(stereo_inertial_realsense_D435i_old ${PROJECT_NAME})
# endif()
