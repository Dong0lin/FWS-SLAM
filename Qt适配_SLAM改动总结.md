# Qt 适配 — ORB-SLAM3 SLAM 端改动总结

本文档列出了为了支持 Qt 界面，在 `/home/dl/Universal` 中对 ORB-SLAM3 源码所做的所有新增和修改。可参照此文档同步到另一个 ORB-SLAM3 仓库。

---

## 一、新增文件

### 1. SlamInterface — Qt 封装接口层

| 文件 | 用途 |
|------|------|
| `include/SlamInterface.h` | 对外 C++ 接口头文件，不暴露任何 ORB-SLAM3 内部依赖 |
| `src/SlamInterface.cc` | PIMPL 实现，封装所有 SLAM 调用 |

**核心功能：**
- `Init()` — 以 `VIEWER_QT` 模式初始化 System，跳过 Pangolin 窗口
- `TrackMonocular()` — 处理单目帧，返回位姿
- `GetCurrentFrame()` — 获取当前绘制好的帧图像（RGB 格式，给 Qt 显示）
- `GetAllMapPoints(outPositions, outColors, maxCount)` — 地图点 + 语义颜色
- `GetAllKeyFramePoses(outPoses, outStatus, maxCount)` — 关键帧位姿
- `GetPersistentBoxes(outBoxes, outPlaneNormal, outPlaneOffset, maxCount)` — 3D 框 + 平面参数
- `SetEnable3DBoxDetection()` / `SetVisualizationMode()` / `SetStepByStep()` — 运行时控制
- `GetCameraPose()` / `GetTrackingState()` 等状态查询

**基础依赖**（在 `.cc` 中 include，不暴露给调用者）：
```cpp
#include "System.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "common.h"
```

### 2. ShmData — 共享内存数据结构

| 文件 | 用途 |
|------|------|
| `include/ShmData.h` | 定义 SLAM 进程与 Qt 界面之间的共享内存协议 |

**三大结构体：**

- `ShmCtrl` — 控制命令和基本状态（相机位姿、帧计数、跟踪状态、步进/重置/3D框开关等命令字）
- `ShmFrame` — 图像帧数据（RGB 像素缓冲 + 帧 ID）
- `ShmMap` — 3D 地图数据（轨迹环形缓冲、关键帧位姿、地图点坐标+颜色、平面参数、3D 框数据）

**关键宏：**
```cpp
#define SHM_NAME_CTRL   "/slam_ctrl"
#define SHM_NAME_FRAME  "/slam_frame"
#define SHM_NAME_MAP    "/slam_map"
#define SEM_NAME_FRAME  "/slam_frame_sem"
#define SHM_MAX_FRAME_BYTES  (1920 * 1080 * 3)
#define MAX_TRAJ_POINTS  100000
#define MAX_KF_POSES     5000
#define MAX_MAP_POINTS   200000
#define MAX_3D_BOXES     100
```

### 3. slam_runner — 独立 SLAM 进程入口

| 文件 | 用途 |
|------|------|
| `Examples/Monocular/slam_runner.cc` | 独立进程：读取图像 → 运行 SLAM → 写共享内存供 Qt 读取 |

**用法：**
```
./slam_runner <vocab> <settings> <image_folder> <timestamp_file> [trajectory_name]
```

**核心逻辑：**
- 主循环：读图 → `TrackMonocular` → `WriteShm`（写控制帧、图像帧、3D 地图数据到共享内存）
- `ProcessCommands()` 轮询 Qt 发来的命令（定位模式、步进模式、可视化切换、3D 框开关）
- 步进模式：等待 `cmd_step` 信号再处理下一帧
- 共享内存生命周期由 slam_runner 创建，退出时清理

---

## 二、现有文件的修改

### 4. `include/System.h` — 新增 enum 和访问器

```cpp
// eViewer 枚举新增 VIEWER_QT = 2
enum eViewerType { VIEWER_NONE = 0, VIEWER_PANGOLIN = 1, VIEWER_QT = 2 };
```

**新增公有访问器：**
```cpp
float GetImageScale();
FrameDrawer* GetFrameDrawer() { return mpFrameDrawer; }
MapDrawer*   GetMapDrawer()   { return mpMapDrawer; }
Tracking*    GetTracker()     { return mpTracker; }
```

### 5. `src/System.cc` — VIEWER_QT 模式跳过 Pangolin

```cpp
// 构造函数中：仅 VIEWER_PANGOLIN 创建 mpViewer
if(viewerType == VIEWER_PANGOLIN) {
    mpViewer = new Viewer(...);
}
// VIEWER_QT 时 mpViewer 保持 nullptr，不启动 Pangolin 线程
```

**新增 `GetImageScale()`：**
```cpp
float System::GetImageScale() { return mpTracker->GetImageScale(); }
```

### 6. `include/FrameDrawer.h` — Qt 模式标记

```cpp
void SetQtMode(bool qt) { mbQtMode = qt; }
bool mbQtMode = false;
```

### 7. `src/FrameDrawer.cc` — Qt 模式跳过文字叠加

在 `DrawFrame()` 末尾，Qt 模式下跳过 Pangolin 风格的状态信息栏拼接：
```cpp
if(mbQtMode) {
    imText = im;  // 直接输出图像，不拼接底部状态栏
}
```

### 8. `include/MapDrawer.h` — 相机位姿查询接口

```cpp
Eigen::Matrix4f GetCurrentCameraPose();
```

### 9. `src/MapDrawer.cc` — 实现 GetCurrentCameraPose

```cpp
Eigen::Matrix4f MapDrawer::GetCurrentCameraPose() {
    unique_lock<mutex> lock(mMutexCamera);
    return mCameraPose.matrix();
}
```

### 10. `include/Tracking.h` — 新增控制接口

```cpp
void SetStepByStep(bool bSet);
bool GetStepByStep();

void SetEnable3DBoxDetection(bool flag) { mbEnable3DBoxDetection = flag; }
bool Is3DBoxDetectionEnabled() { return mbEnable3DBoxDetection; }

void SetShowDynamicVis(bool flag) { mbShowDynamicVis = flag; }
bool IsShowDynamicVis() { return mbShowDynamicVis; }

float GetImageScale();
```

**新增成员变量：**
```cpp
bool mbEnable3DBoxDetection = false;
bool mbShowDynamicVis = false;
```

### 11. `src/Tracking.cc` — 实现新增接口

```cpp
void Tracking::SetStepByStep(bool bSet) { bStepByStep = bSet; }

float Tracking::GetImageScale() { return mImageScale; }
```

### 12. `include/Map.h` — 平面和 3D 框访问接口

```cpp
const Eigen::Vector3f& GetPlaneNormal() const;
const std::vector<float>& GetPlaneOffsets() const;
bool IsPlaneEstimated();
const std::vector<Detection3D>& GetPersistentBoxes() const;
```

### 13. `src/Map.cc` — 实现 IsPlaneEstimated

```cpp
bool Map::IsPlaneEstimated() {
    unique_lock<mutex> lock(mMutexMap);
    return mbPlaneEstimated;
}
```

### 14. `include/Detector.h` — 暴露检测结果图像

```cpp
cv::Mat mImg;  // 目标检测可视化图像（含检测框）
```

### 15. `include/common.h` — 语义颜色表

共享的类别名称和颜色定义表，`SlamInterface.cc` 中的地图点着色依赖此文件：
```cpp
const std::vector<std::string> CLASS_NAMES = {
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor" };

const std::vector<std::vector<unsigned int>> COLORS = {
    {0, 114, 189},  {217, 83, 25},  {237, 177, 32}, {126, 47, 142},
    {119, 172, 48}, {77, 190, 238}, {162, 20, 47},  {76, 76, 76},
    {153, 153, 153},{255, 0, 0},    {255, 128, 0},  {191, 191, 0} };
```

### 16. `CMakeLists.txt` — 新增编译目标

```cmake
# slam_runner — 独立进程，写入共享内存供 Qt 界面读取
add_executable(slam_runner
        Examples/Monocular/slam_runner.cc)
target_link_libraries(slam_runner ${PROJECT_NAME} rt)
```

---

## 三、关键数据流

```
slam_runner 进程                    Qt 界面进程
     │                                   │
     ├─ SlamInterface::TrackMonocular()  │
     ├─ SlamInterface::GetCurrentFrame() ──→ ShmFrame ──→ Image_Label
     ├─ SlamInterface::GetCameraPose()   ──→ ShmCtrl  ──→ CloudGLWidget (相机)
     ├─ SlamInterface::GetAllMapPoints() ──→ ShmMap   ──→ CloudGLWidget (点云)
     ├─ SlamInterface::GetAllKeyFramePoses()→ ShmMap  ──→ CloudGLWidget (关键帧)
     ├─ SlamInterface::GetPersistentBoxes()→ ShmMap  ──→ CloudGLWidget (3D框+平面)
     │                                   │
     │        ShmCtrl 命令字  ←────────── Qt 界面控件交互
     │   (cmd_step, cmd_localization,
     │    draw_3dbox, cmd_vis_mode, ...)
```

---

## 四、同步清单（按依赖顺序）

1. 复制 `include/ShmData.h`
2. 复制 `include/SlamInterface.h` + `src/SlamInterface.cc`
3. 复制 `Examples/Monocular/slam_runner.cc`
4. 确保 `include/common.h` 存在
5. 修改 `include/System.h` — 加 `VIEWER_QT`、访问器
6. 修改 `src/System.cc` — 加 `VIEWER_QT` 分支、`GetImageScale()`
7. 修改 `include/FrameDrawer.h` — 加 `SetQtMode`、`mbQtMode`
8. 修改 `src/FrameDrawer.cc` — `mbQtMode` 跳过状态栏
9. 修改 `include/MapDrawer.h` — 加 `GetCurrentCameraPose()`
10. 修改 `src/MapDrawer.cc` — 实现 `GetCurrentCameraPose()`
11. 修改 `include/Tracking.h` — 加控制接口 + 成员变量
12. 修改 `src/Tracking.cc` — 实现 `SetStepByStep`、`GetImageScale`
13. 修改 `include/Map.h` — 加平面/3D框访问器（已有则跳过）
14. 修改 `src/Map.cc` — 加 `IsPlaneEstimated()`（已有则跳过）
15. 修改 `CMakeLists.txt` — 加 `slam_runner` 目标
