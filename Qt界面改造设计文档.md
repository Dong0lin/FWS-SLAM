# SLAM Qt 界面改造设计文档

> **环境**：Qt 6.5.3 | OpenCV 4.x | CUDA/TensorRT | Ubuntu 20.04+

## 1. 目标

将当前基于 **Pangolin + OpenCV HighGUI** 的双窗口可视化，整合为一个统一的 **Qt6 应用程序**，包含三个面板：

```
+------------------------------------------------------------------+
|                         QMainWindow                               |
|  +---------------------+------------------+--------------------+  |
|  |                     |                  |                    |  |
|  |  QOpenGLWidget      |  QLabel          |  Control Panel     |  |
|  |  (3D 地图面板)      |  (当前帧面板)    |  (交互面板)        |  |
|  |                     |                  |                    |  |
|  |  - 轨迹 / 关键帧    |  - 当前帧图像    |  词汇文件: [___]   |  |
|  |  - 点云             |  - ORB 特征点    |  配置文件: [___]   |  |
|  |  - 地面平面         |  - 动态点标注    |  时间戳:   [___]   |  |
|  |  - 3D 检测框        |  - 跟踪状态      |  图像目录: [___]   |  |
|  |  - 共视图           |                  |  [Browse] 按钮x4   |  |
|  |                     |                  |                    |  |
|  |  (鼠标拖拽旋转/缩放)|                  |  传感器类型: [...] |  |
|  |                     |                  |  显示点云:   [√]   |  |
|  |                     |                  |  显示关键帧: [√]   |  |
|  |                     |                  |  显示共视图: [ ]   |  |
|  |                     |                  |  跟随相机:   [√]   |  |
|  |                     |                  |                    |  |
|  |                     |                  |  [开始SLAM]        |  |
|  |                     |                  |  [停止SLAM]        |  |
|  |                     |                  |  [保存轨迹]        |  |
|  |                     |                  |                    |  |
|  |                     |                  |  状态: 运行中      |  |
|  |                     |                  |  FPS: 30.5         |  |
|  |                     |                  |  地图点: 1523      |  |
|  |                     |                  |  关键帧: 47        |  |
|  +---------------------+------------------+--------------------+  |
+------------------------------------------------------------------+
```

---

## 2. 核心挑战与解决方案

### 2.0 Qt 的角色：纯显示，不参与计算

**Qt 的职责仅仅是**：接收 SLAM 输出的数据，渲染到屏幕上。不参与任何 SLAM 计算（跟踪、建图、回环检测等都在原有线程中运行）。

### 2.1 可视化效果：与原始 ORB-SLAM3 完全一致

**可以做到效果完全相同**，原因：

- `MapDrawer` 中所有 OpenGL 绘制代码（`glBegin/glEnd`、`glColor3f`、`glPointSize`、`glLineWidth` 等）是**标准 OpenGL 指令**，与窗口框架无关
- Pangolin 窗口和 `QOpenGLWidget` 都是 OpenGL 的"画布"，底层 GPU 渲染管线完全一致
- 点大小、线宽、颜色、透明度、深度测试、混合模式 —— 全部由相同的 OpenGL 状态机控制
- 唯一变化的是"谁创建 OpenGL 上下文"（从 Pangolin 变为 Qt），但上下文中的绘制行为一致

| 渲染元素 | 原 Pangolin 效果 | Qt 效果 |
|---------|-----------------|---------|
| 地图点 | 黑色 POINTS + 参考点红色 | 完全一致 |
| 关键帧相机 | 蓝色线框锥体 | 完全一致 |
| 当前相机 | 绿色线框锥体 | 完全一致 |
| 共视图连线 | 绿色半透明线段 | 完全一致 |
| 地面平面 | 绿色半透明网格 + 法向量箭头 | 完全一致 |
| 3D 检测框 | 彩色半透明长方体 | 完全一致 |
| 当前帧图像 | cv::imshow 窗口 | QLabel 嵌入布局（内容一致） |

### 2.2 Pangolin 与 Qt 的事件循环冲突

**问题**：Pangolin 内部使用 GLFW 或 X11，拥有自己的事件循环。Qt 也拥有自己的事件循环（`QApplication::exec()`）。两者无法共存。

**解决方案**：完全移除 Pangolin 依赖，将 3D 渲染移植到 `QOpenGLWidget`。

- `QOpenGLWidget` 提供合法的 OpenGL 上下文，在 Qt 事件循环中运行
- `MapDrawer` 中的所有绘制代码使用 Legacy OpenGL（`glBegin/glEnd`），这与 `QOpenGLWidget` 完全兼容
- 鼠标交互（旋转/缩放/平移）通过 `QOpenGLWidget` 的 `mousePressEvent`/`mouseMoveEvent`/`wheelEvent` 实现

### 2.3 OpenCV HighGUI 的替换

**问题**：`cv::imshow` 创建独立窗口，无法嵌入 Qt 布局。

**解决方案**：将 `cv::Mat` 转为 `QImage` 再转为 `QPixmap`，显示在 `QLabel` 中。

### 2.4 线程安全

**问题**：SLAM 的 Viewer 线程不能直接操作 Qt Widget（只能在主线程操作 UI）。

**解决方案**：使用 Qt 的 **信号槽（Signal/Slot）** 机制跨线程传递数据：

```
Viewer Thread (SLAM)              Qt Main Thread (UI)
==================                ===================
每帧循环:
  读取 FrameDrawer → cv::Mat
  读取 MapDrawer   → 相机位姿
        │
        ├─ emit FrameUpdated(cv::Mat) ──→ slot: 更新 QLabel
        │
        └─ emit MapUpdated() ──→ slot: mGLWidget->update()
                                       └→ paintGL() 读取 MapDrawer 数据
```

---

## 3. 新增文件清单

| 文件 | 作用 |
|------|------|
| `include/QtViewer.h` | Qt 主窗口类声明 + 3D Widget 声明 |
| `src/QtViewer.cc` | Qt 主窗口类实现（布局、信号槽、控制逻辑） |
| `src/QtGLWidget.cc` | QOpenGLWidget 子类实现（3D 渲染、鼠标交互） |

### 3.1 `QtGLWidget` — 3D 渲染面板

继承 `QOpenGLWidget`，重写三个关键虚函数：

```cpp
// include/QtViewer.h

#pragma once

// Qt6: QOpenGLWidget 在 QtOpenGLWidgets 模块中
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QMainWindow>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QCheckBox>
#include <QComboBox>
#include <QTimer>
#include <QStatusBar>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QSplitter>
#include <QFileDialog>
#include <QMouseEvent>
#include <QWheelEvent>
#include <QImage>
#include <QPixmap>

#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <mutex>

#include "MapDrawer.h"
#include "FrameDrawer.h"
#include "System.h"

namespace ORB_SLAM3 {

// ============ 3D 渲染 Widget ============
class QtGLWidget : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT

public:
    explicit QtGLWidget(MapDrawer* pMapDrawer, QWidget* parent = nullptr);
    ~QtGLWidget();

    // 相机控制
    void SetFollowCamera(bool follow) { mbFollowCamera = follow; }
    void SetCameraView(bool camView);
    void SetTopView();

protected:
    // QOpenGLWidget 接口
    void initializeGL() override;
    void paintGL() override;
    void resizeGL(int w, int h) override;

    // 鼠标交互
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    void UpdateCameraMatrix();

    MapDrawer* mpMapDrawer;

    // 视图参数 (从 Pangolin 对应迁移)
    float mViewpointF;
    // 模型视图矩阵（相机位姿 + 视角偏移）
    float mModelView[16];
    float mProjection[16];

    // 鼠标旋转/缩放状态
    bool mbFollowCamera = true;
    bool mbCameraView = true;
    QPointF mLastMousePos;
    float mRotationX = 0, mRotationY = 0;
    float mZoom = 1.0f;
};

// ============ 主窗口 ============
class QtViewer : public QMainWindow
{
    Q_OBJECT

public:
    explicit QtViewer(System* pSystem, FrameDrawer* pFrameDrawer,
                      MapDrawer* pMapDrawer, Tracking* pTracking,
                      const std::string& strSettingsPath,
                      QWidget* parent = nullptr);
    ~QtViewer();

    // 由 SLAM 线程调用的更新接口
    void UpdateFromSLAM();  // 替代原 Viewer::Run() 的循环体

signals:
    // 跨线程信号：SLAM 线程 → UI 线程
    void FrameReady(const QImage& image);
    void MapReady();
    void StatsUpdated(int fps, int nMapPoints, int nKeyFrames, int nMatches);
    void SlamFinished();

public slots:
    // UI 线程槽函数
    void onFrameReady(const QImage& image);
    void onMapReady();
    void onStatsUpdated(int fps, int nMapPoints, int nKeyFrames, int nMatches);

    // 按钮槽函数
    void onStartClicked();
    void onStopClicked();
    void onSaveTrajectory();
    void onBrowseVocab();
    void onBrowseSettings();
    void onBrowseTimestamps();
    void onBrowseImages();

private:
    void SetupUI();              // 初始化布局
    void SetupConnections();     // 连接信号槽
    void UpdateStatusLabel();

    // Widget 成员
    QtGLWidget* mpGLWidget;
    QLabel* mpImageLabel;
    QLabel* mpStatusLabel;

    // 控制面板控件
    QLineEdit* mpVocabPath;
    QLineEdit* mpSettingsPath;
    QLineEdit* mpTimestampsPath;
    QLineEdit* mpImagesPath;
    QComboBox* mpSensorType;
    QCheckBox* mpShowPoints;
    QCheckBox* mpShowKeyFrames;
    QCheckBox* mpShowGraph;
    QCheckBox* mpShowPlane;
    QCheckBox* mpShowDetection3D;
    QCheckBox* mpFollowCamera;
    QPushButton* mpBtnStart;
    QPushButton* mpBtnStop;
    QPushButton* mpBtnSaveTraj;

    // SLAM 组件指针
    System* mpSystem;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;
    Tracking* mpTracker;
    std::string mSettingsPath;

    // 状态
    bool mbRunning = false;
    float mImageViewerScale = 1.0f;
    int mImageWidth, mImageHeight;
};

} // namespace ORB_SLAM3
```

---

## 4. 线程模型详述

### 4.1 改造前（当前）

```
Main Thread:   图像读取 → SLAM.TrackMonocular(im, t) → 循环
Thread 1:      LocalMapping::Run()
Thread 2:      LoopClosing::Run()
Thread 3:      Detector::Run()
Thread 4:      Viewer::Run()  ← 内部 while(1) 循环
                   ├─ pangolin::FinishFrame()
                   ├─ cv::imshow()
                   └─ cv::waitKey()
```

### 4.2 改造后（Qt）

```
Qt Main Thread:   QApplication::exec() → 事件循环 → 响应用户交互
                      ↑
                      │ 信号槽连接 (QueuedConnection)
                      │
SLAM Thread:     while(有图像) { SLAM.TrackMonocular(im,t); qtViewer->UpdateFromSLAM(); }
                      │
                      ├─ emit FrameReady(qImage)  →  QLabel::setPixmap()
                      ├─ emit MapReady()           →  QOpenGLWidget::update()
                      └─ emit StatsUpdated(...)    →  QStatusBar::showMessage()

Thread 1:        LocalMapping::Run()
Thread 2:        LoopClosing::Run()
Thread 3:        Detector::Run()
```

**关键变化**：
- 原 `Viewer::Run()` 线程被**移除**。
- 图像循环线程（原来的 main thread）改为 **SLAM worker 线程**。
- **Qt 主线程只负责 UI 渲染和事件响应**，不在主线程中跑 SLAM 计算。
- `QTimer` 驱动 3D 视图的定时刷新（替代 `pangolin::FinishFrame` 的循环）。

---

## 5. 关键代码片段

### 5.1 QOpenGLWidget 的 paintGL() — 替代 pangolin 渲染

```cpp
void QtGLWidget::paintGL()
{
    QOpenGLFunctions* f = QOpenGLContext::currentContext()->functions();
    f->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    f->glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    // 设置投影和模型视图矩阵 (Legacy OpenGL)
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    // 使用 gluPerspective 的等价实现或手动构建
    glMultMatrixf(mProjection);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    if (mbFollowCamera) {
        UpdateCameraMatrix();  // 从 MapDrawer 读取当前相机位姿
    }
    glMultMatrixf(mModelView);

    // ===== 绘制代码：直接从 MapDrawer 移植，无需修改 =====
    mpMapDrawer->DrawCurrentCamera(TwcMapped);  // 需适配参数类型
    mpMapDrawer->DrawKeyFrames(bKF, bGraph, bInertial, bOptLba);
    mpMapDrawer->DrawPlane();
    mpMapDrawer->DrawDetection3Ds();
    mpMapDrawer->DrawMapPoints();
}
```

**MapDrawer 接口适配**：

`MapDrawer` 的绘制方法目前使用 Pangolin 的 `pangolin::OpenGlMatrix`。需要将其改为 `Eigen::Matrix4f` 或直接使用 bare `GLfloat*`：

```cpp
// 改造前：void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);
// 改造后：void DrawCurrentCamera(const Eigen::Matrix4f &Twc);
```

修改量很小——`MapDrawer.cc` 中 `GetCurrentOpenGLCameraMatrix` 本来就已经在构造 `Eigen::Matrix4f`，只需要把 `DrawCurrentCamera` 的参数从 `pangolin::OpenGlMatrix` 换成 `Eigen::Matrix4f`，然后在 `glMultMatrixf` 处改用 `.data()` 指针。

### 5.2 cv::Mat → QImage 转换

```cpp
QImage QtViewer::MatToQImage(const cv::Mat& mat)
{
    // 假设输入为 BGR (OpenCV 默认)
    cv::Mat rgb;
    cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    return QImage(rgb.data, rgb.cols, rgb.rows,
                  rgb.step, QImage::Format_RGB888).copy();
}

// 在 SLAM 线程中调用
void QtViewer::UpdateFromSLAM()
{
    cv::Mat im = mpFrameDrawer->DrawFrame(mImageViewerScale);
    QImage qimg = MatToQImage(im);
    emit FrameReady(qimg);
    emit MapReady();

    // 统计信息
    // (需要从 System/Map 读取实际值)
    emit StatsUpdated(currentFps, nMapPoints, nKeyFrames, nMatches);
}
```

### 5.3 按钮槽函数 — 启动/停止

```cpp
void QtViewer::onStartClicked()
{
    if (mbRunning) return;

    // 1. 校验输入
    std::string vocab   = mpVocabPath->text().toStdString();
    std::string settings = mpSettingsPath->text().toStdString();
    std::string timestamps = mpTimestampsPath->text().toStdString();
    std::string images  = mpImagesPath->text().toStdString();

    if (vocab.empty() || settings.empty() || timestamps.empty() || images.empty()) {
        mpStatusLabel->setText("请填写所有路径");
        return;
    }

    // 2. 创建 SLAM System（在 SLAM 线程中或提前创建）
    //    选项 A：构造函数中已创建 mpSystem
    //    选项 B：在此处创建新 System

    // 3. 启动 SLAM 工作线程
    mbRunning = true;
    mpBtnStart->setEnabled(false);
    mpBtnStop->setEnabled(true);

    std::thread slamThread([this, vocab, settings, timestamps, images]() {
        // 加载图像列表 & 时间戳
        // 循环调用 mpSystem->TrackMonocular(im, tframe)
        // 每帧调用 mQtViewer->UpdateFromSLAM()
        // 循环结束后 emit SlamFinished()
    });
    slamThread.detach();
}

void QtViewer::onStopClicked()
{
    if (!mbRunning) return;
    mpSystem->Shutdown();
    mbRunning = false;
    mpBtnStart->setEnabled(true);
    mpBtnStop->setEnabled(false);
}
```

### 5.4 main() 函数

```cpp
#include <QApplication>
#include "QtViewer.h"

int main(int argc, char** argv)
{
    QApplication app(argc, argv);

    ORB_SLAM3::QtViewer window(/* parameters */);
    window.resize(1600, 900);
    window.show();

    return app.exec();
}
```

---

## 6. CMakeLists.txt 修改

需要在 `CMakeLists.txt` 中添加 Qt6 支持（Qt6 与 Qt5 的 CMake 接口有重要差异）：

```cmake
# ========== Qt6 依赖 (6.5.3) ==========
# 关键：QOpenGLWidget 在 Qt6 中属于独立模块 OpenGLWidgets
find_package(Qt6 REQUIRED COMPONENTS Widgets OpenGLWidgets)
set(CMAKE_AUTOMOC ON)   # 自动处理 Q_OBJECT 宏
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# ========== 新增 Qt 相关源文件 ==========
set(QT_SOURCES
    src/QtViewer.cc
    src/QtGLWidget.cc
)

set(QT_HEADERS
    include/QtViewer.h
)

# ========== 编译目标 ==========
add_executable(slam_qt
    ${QT_SOURCES}
    ${QT_HEADERS}
    Examples/Monocular/mono_euroc_mine.cc   # 或新建 Qt 入口
)

target_link_libraries(slam_qt
    ${PROJECT_NAME}           # ORB-SLAM3 核心库
    Qt6::Widgets
    Qt6::OpenGLWidgets        # Qt6 中 QOpenGLWidget 的独立模块
    ${OpenCV_LIBS}
    # ... 其余依赖
)
```

### 6.1 Qt5 vs Qt6 CMake 差异

| 项目 | Qt5 | Qt6 |
|------|-----|-----|
| `find_package` | `Qt5 COMPONENTS Widgets OpenGL` | `Qt6 COMPONENTS Widgets OpenGLWidgets` |
| QOpenGLWidget 链接目标 | `Qt5::OpenGL` | `Qt6::OpenGLWidgets` |
| OpenGL 功能 | 包含在 `Qt5::OpenGL` 中 | `QOpenGLFunctions` 仍在 `Qt6::Widgets` 中可用 |

### 6.2 Pangolin 依赖处理（条件编译）

- **方案 A（推荐）**：通过 CMake 选项 `-DWITH_QT_VIEWER=ON` 条件编译。Qt 模式下不链接 Pangolin，原 Pangolin Viewer 仍可独立编译。
- **方案 B**：完全替换，移除 Pangolin 依赖（如果确定不再需要 Pangolin 窗口）。

推荐方案 A，用编译宏控制：

```cmake
option(WITH_QT_VIEWER "Use Qt6-based viewer instead of Pangolin" OFF)

if(WITH_QT_VIEWER)
    find_package(Qt6 REQUIRED COMPONENTS Widgets OpenGLWidgets)
    add_definitions(-DUSE_QT_VIEWER)
    # Qt 模式下不链接 Pangolin
else()
    find_package(Pangolin REQUIRED)
    target_link_libraries(${PROJECT_NAME} ${Pangolin_LIBRARIES})
endif()
```

---

## 7. MapDrawer 接口需要的小幅修改

`MapDrawer` 类当前依赖 Pangolin 的唯一位置：

| 方法 | 当前签名 | 修改后签名 |
|------|---------|-----------|
| `DrawCurrentCamera` | `(pangolin::OpenGlMatrix &Twc)` | `(const Eigen::Matrix4f &Twc)` 或 `(const Sophus::SE3f &Twc)` |
| `GetCurrentOpenGLCameraMatrix` | 返回两个 `pangolin::OpenGlMatrix` | 改为返回 `Eigen::Matrix4f`（更简单） |

修改量极小——`MapDrawer.cc` 中 `GetCurrentOpenGLCameraMatrix` 函数体内已经在构造 `Eigen::Matrix4f`，只是最后拷贝到了 `pangolin::OpenGlMatrix`。改为直接返回 `Eigen::Matrix4f` 即可。

`MapDrawer.h` 中移除 `#include <pangolin/pangolin.h>`，绘图代码（`glBegin/glEnd`）不变。

---

## 8. 实施步骤

| 步骤 | 内容 | 预估修改量 |
|------|------|-----------|
| **Step 1** | 修改 `MapDrawer.h/cc`：移除 Pangolin 依赖，`DrawCurrentCamera` 参数改为 `Eigen::Matrix4f` | ~20 行 |
| **Step 2** | 新增 `include/QtViewer.h`：`QtGLWidget` 和 `QtViewer` 类声明 | ~150 行 |
| **Step 3** | 新增 `src/QtGLWidget.cc`：QOpenGLWidget 的 `paintGL` + 鼠标交互 | ~250 行 |
| **Step 4** | 新增 `src/QtViewer.cc`：主窗口布局、信号槽、按钮逻辑 | ~350 行 |
| **Step 5** | 修改 `CMakeLists.txt`：添加 Qt5 查找 + 条件编译 | ~30 行 |
| **Step 6** | 修改 `System.cc`：条件编译下不创建 Pangolin Viewer | ~10 行 |
| **Step 7** | 新建 Qt 入口文件（或修改现有 `mono_euroc_mine.cc`） | ~80 行 |
| **Step 8** | 编译测试、调试 | — |

---

## 9. 风险与注意事项

1. **Legacy OpenGL 兼容性**：`MapDrawer` 使用 `glBegin/glEnd`（Legacy OpenGL / Compatibility Profile）。在 Qt6 中，`QOpenGLWidget` 需要在创建时显式设置 `QSurfaceFormat::setProfile(QSurfaceFormat::CompatibilityProfile)`，确保支持固定管线。大多数 Linux 桌面环境都支持 Compatibility Profile。

   ```cpp
   // 在 main() 中设置（QApplication 创建之前）
   QSurfaceFormat fmt;
   fmt.setProfile(QSurfaceFormat::CompatibilityProfile);
   fmt.setDepthBufferSize(24);
   QSurfaceFormat::setDefaultFormat(fmt);
   ```

2. **QImage 深拷贝开销**：`FrameReady(QImage)` 信号传递的 QImage 会被拷贝一次。对于 640x480 图像，开销可忽略（~1ms）。如需极致优化，可用 `QSharedPointer<QImage>` 传递。

3. **3D 渲染帧率控制**：不再有 `cv::waitKey(mT)` 控制帧率。改用 `QTimer` 以固定间隔触发 `update()`（例如 33ms ≈ 30fps）。

4. **`Viewer.h/cc` 保留**：原 `Viewer` 类保留，通过 `#ifdef USE_QT_VIEWER` 条件编译切换。非 Qt 模式下功能不受影响。

5. **`FrameDrawer` 不变**：`FrameDrawer::DrawFrame()` 返回 `cv::Mat`，对上层透明，无需修改。

6. **Qt6.5.3 注意事项**：
   - `QOpenGLWidget` 模块名从 Qt5 的 `Qt5::OpenGL` 变为 Qt6 的 `Qt6::OpenGLWidgets`
   - `QOpenGLFunctions` 可直接通过 `QOpenGLContext::currentContext()->functions()` 获取（无需继承）
   - `QWheelEvent::angleDelta()` 替代 Qt5 已废弃的 `QWheelEvent::delta()`
