# FWS-SLAM

面向扑翼飞行机器人户外航拍场景的语义 SLAM 系统，**基于 [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) 修改而来**。

## 项目来源

本项目在 ORB-SLAM3（作者 Carlos Campos 等，University of Zaragoza）的基础上进行了二次开发，在原有单目/双目定位与建图流程之上增加了：

- YOLOv11 + TensorRT 语义感知（独立检测线程、2D→3D 检测框、持久化语义地标）
- 扑翼振动自适应跟踪
- 基于极线约束的动态目标剔除
- 地面平面约束与尺度锚定
- 语义增强回环检测
- 长短焦双目模式
- 供 Qt 前端读取的共享内存接口（`include/ShmData.h`）

除新增/修改的文件外，其余代码与模块（`System`、`Tracking`、`LocalMapping`、`LoopClosing`、`ORBextractor`、`Map`、`Atlas`、`Optimizer`、`CameraModels` 等）均沿用 ORB-SLAM3。DBoW2、g2o、Sophus 等第三方库位于 `Thirdparty/`。

> 本项目为 ORB-SLAM3 的衍生作品，遵循 GPLv3 分发。原 ORB-SLAM3 版权归 Carlos Campos 等作者所有。

## 开发与测试环境

| 项目 | 版本 / 说明 |
|------|-------------|
| 操作系统 | Ubuntu 20.04.6 LTS |
| 内核 | 5.15.0-139-generic |
| GPU | NVIDIA GeForce RTX 4060 Laptop（驱动 580.105.08） |
| GCC / G++ | 9.4.0 |
| CMake | 3.30.5 |
| CUDA | 11.6（`/usr/local/cuda-11.6`，nvcc V11.6.55） |
| cuDNN | 8.9.7 |
| TensorRT | 8.6.0.12（`/home/dl/TensorRT/TensorRT-8.6.0.12`） |
| OpenCV | 4.2.0 |
| Eigen | 3.3.7 |
| Pangolin | 0.6 |
| Boost | 1.71.0 |
| ROS | Noetic |
| C++ 标准 | C++14 |

## 依赖库

**系统 / 第三方库**

- OpenCV（`find_package(OpenCV)`）
- Eigen3（`find_package(Eigen3)`）
- Pangolin（`find_package(Pangolin)`，用于可视化）
- Boost（`thread` 组件，链接 `boost_serialization`）
- OpenSSL（`libcrypto`）
- pthread、librt

**CUDA / TensorRT（语义检测必需）**

- CUDA Toolkit 11.6 + cuDNN 8.9.7
- TensorRT 8.6.0.12：`nvinfer`、`nvinfer_plugin`、`nvonnxparser`、`nvparsers`、`cudart`

**仓库内第三方源码（`Thirdparty/`，随工程一起编译）**

- DBoW2（词袋）
- g2o（图优化）
- Sophus（李群李代数）

**ROS 版本额外依赖**（`Examples/FWS-SLAM-ROS/manifest.xml`）

- `roscpp`、`tf`、`sensor_msgs`、`image_transport`、`cv_bridge`

**外部数据文件（不打进仓库，需自行准备）**

- `Vocabulary/ORBvoc.txt`：ORB 词袋，取自 ORB-SLAM3
- `engine/*.engine`：TensorRT 检测引擎（YOLOv11，VisDrone 数据集训练）

## 编译配置

### 1. 安装基础依赖

```bash
sudo apt update
sudo apt install -y build-essential cmake git libgtk2.0-dev pkg-config \
    libavcodec-dev libavformat-dev libswscale-dev libssl-dev \
    libboost-thread-dev libboost-serialization-dev
```

OpenCV、Eigen、Pangolin 请按上表版本自行安装（Pangolin 0.6 需源码编译）。

### 2. 编译 TensorRT 与 CUDA

安装 CUDA 11.6、cuDNN 8.9.7，并将 TensorRT 8.6.0.12 解压到任意目录。

### 3. 配置外部依赖路径

`TensorRT`、`CUDA` 的安装位置通过环境变量或 CMake 变量指定，工程内不含任何硬编码的绝对路径，输出目录均相对工程根目录。

```bash
# 方式一：环境变量
export TENSORRT_ROOT=/path/to/TensorRT-8.6.0.12
export CUDA_ROOT=/usr/local/cuda-11.6/targets/x86_64-linux

# 方式二：cmake 命令行（推荐）
cmake .. -DTENSORRT_ROOT=/path/to/TensorRT-8.6.0.12 \
         -DCUDA_ROOT=/usr/local/cuda-11.6/targets/x86_64-linux
```

若 `CMakeLists.txt` 中 `find_package(OpenCV 4.2 EXACT REQUIRED)` 未自动找到 4.2，可显式指定：

```bash
cmake .. -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4
```

（若本机只用 4.2 版本可忽略；ROS 与本项目需使用同一 OpenCV 版本。）

### 4. 编译

```bash
cd FWS-SLAM
# 编译第三方库
cd Thirdparty/DBoW2 && mkdir -p build && cd build && cmake .. && make -j$(nproc) && cd ../../..
cd Thirdparty/g2o   && mkdir -p build && cd build && cmake .. && make -j$(nproc) && cd ../../..
# 编译主库与示例
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

产物输出到 `bin/`（库输出到 `lib/`），主要可执行文件：

| 可执行文件 | 说明 |
|------------|------|
| `bin/mono_euroc_mine` | 扑翼单目（自定义图像序列） |
| `bin/mono_euroc` | EuRoC 单目 |
| `bin/stereo_multifocal` | 长短焦双目 |
| `bin/stereo_euroc` | EuRoC 双目 |
| `bin/slam_runner` | 单目 + 共享内存（供 Qt 前端调用） |
| `bin/slam_runner_stereo` | 长短焦双目 + 共享内存 |

ROS 版本单独编译：

```bash
cd Examples/FWS-SLAM-ROS
mkdir -p build && cd build && cmake .. && make -j$(nproc)
# 产物在 Examples/FWS-SLAM-ROS/bin/
```

### 5. 准备词袋与检测模型

- 将 ORB-SLAM3 的 `ORBvoc.txt` 放入 `Vocabulary/`；
- 将 YOLOv11 TensorRT 引擎放入 `engine/`，默认文件名 `visdrone_11m.engine`（单目）与 `visdrone_11s.engine`（长短焦右目）。

引擎路径按以下顺序自动解析，无需修改源码：

1. 环境变量 `FWS_ENGINE_MONO` / `FWS_ENGINE_STEREO`（优先级最高）
2. 可执行文件同级的 `engine/`（如 `bin/slam_runner` → `<工程根>/engine/`）
3. 当前工作目录下的 `engine/`

```bash
export FWS_ENGINE_MONO=/path/to/visdrone_11m.engine
export FWS_ENGINE_STEREO=/path/to/visdrone_11s.engine
```

### 6. 配置文件（YAML）

相机内参、ORB 参数、检测参数等通过 YAML 指定，示例见 `Examples/`：

- 单目（扑翼 D435i）：`Examples/Monocular/FWAF-VID_D435i_mono.yaml`
- 长短焦双目：`Examples/Stereo/...`（如 `2026_4mm6mm.yaml`）

运行时由命令行传入，例如：

```bash
./bin/mono_euroc_mine Vocabulary/ORBvoc.txt \
    <配置文件.yaml> <图像目录> <时间戳文件>
```

### 7. ROS 运行

```bash
roscore
rosrun FWS-SLAM-ROS Mono_Stereo_Left \
    /path/to/Vocabulary/ORBvoc.txt \
    /path/to/left4mm720p.yaml \
    /camera/image_raw:=/usb_cam/image_raw
```

## 许可

本项目以 **GPLv3** 分发（由于基于 GPLv3 的 ORB-SLAM3 修改而来）。完整许可证见 `LICENSE`，来源与第三方库版权声明见 `NOTICE`。
