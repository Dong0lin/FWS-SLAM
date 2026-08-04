# ORB-SLAM3 与语义 SLAM 改进对比分析文档

> **项目背景**：扑翼飞行机器人户外航拍数据集，飞行高度约 50m，飞行速度 10m/s，摄像头视角斜向下。
> **基线版本**：ORB_SLAM3（路径 `/home/dl/ORB_SLAM3`）
> **改进版本**：本语义 SLAM 系统（路径 `/home/dl/Universal`）

---

## 目录

1. [新增模块概览](#1-新增模块概览)
2. [改进一：目标检测模块（YOLOv11 TensorRT）](#2-改进一目标检测模块yolov11-tensorrt)
3. [改进二：动态一致性模块（极线约束）](#3-改进二动态一致性模块极线约束)
4. [改进三：初始化平面系统](#4-改进三初始化平面系统)
5. [改进四：扑翼飞行机器人初始化优化](#5-改进四扑翼飞行机器人初始化优化)
6. [改进五：抖动强度计算与运用](#6-改进五抖动强度计算与运用)
7. [其他改进](#7-其他改进)
8. [完整文件改动清单](#8-完整文件改动清单)

---

## 1. 新增模块概览

相比 ORB_SLAM3，本系统新增了以下模块和文件：

| 模块 | 新增文件 | 功能 |
|------|---------|------|
| TensorRT YOLOv11 检测 | `src/tensorrt/YOLOv11.h`, `YOLOv11.cpp`, `main.cpp` | GPU 加速目标检测推理 |
| 检测器主模块 | `include/Detector.h`, `src/Detector.cc` | 目标检测线程，动态类别管理 |
| 通用定义 | `include/common.h`, `common_coco.h` | 类别名/颜色/运动概率定义 |
| GPU 预处理 | `include/preprocess.h`, `src/preprocess.cu` | CUDA 图像预处理（resize/BGR→RGB/归一化） |
| 图像质量评估 | `include/ImageQuality.h` | 锐度/对比度/特征分布度量 |
| 工具头文件 | `include/cuda_utils.h`, `logging.h`, `macros.h` | CUDA 错误检查/TensorRT 日志/跨平台宏 |
| 图像增强工具 | `src/extractor_orb/fw_enhance.cpp` | 局部自适应对比度增强 |
| 图像融合工具 | `src/extractor_orb/fusion.cpp` | 双焦距图像融合 |

---

## 2. 改进一：目标检测模块（YOLOv11 TensorRT）

### 2.1 架构概述

本系统新增了完整的 **TensorRT 加速目标检测管线**，在 ORB-SLAM3 的基础上添加了一条并行的语义感知通道。

**关键文件**：
- [Detector.h](file:///home/dl/Universal/include/Detector.h) / [Detector.cc](file:///home/dl/Universal/src/Detector.cc) — 检测器主类
- [YOLOv11.h](file:///home/dl/Universal/src/tensorrt/YOLOv11.h) / [YOLOv11.cpp](file:///home/dl/Universal/src/tensorrt/YOLOv11.cpp) — TensorRT 推理引擎
- [System.cc](file:///home/dl/Universal/src/System.cc) — 在构造函数中启动检测器线程

### 2.2 检测器初始化（System 层改动）

在 System 构造函数中，新增了检测器线程的创建与绑定：

```cpp
// 原 ORB_SLAM3：仅三个线程（LocalMapping, LoopClosing, Viewer）
// 新增：检测器线程
mpDetector = new Detector();
mptDetector = new thread(&ORB_SLAM3::Detector::Run, mpDetector);
mpDetector->SetTracker(mpTracker);
mpTracker->SetDetector(mpDetector);
```

在每次 `TrackStereo/TrackRGBD/TrackMonocular` 调用时，新增图像投喂：

```cpp
mpTracker->GetImgForDetector(imToFeed);  // 将当前帧交给检测器线程
```

### 2.3 TensorRT 推理管线

**模型**：YOLOv11，使用 VisDrone 数据集训练的 8 类检测模型（`best_fp16.engine`）
- 类别：pedestrian(0), people(1), bicycle(2), car(3), van(4), truck(5), tricycle(6), awning-tricycle(7)
- 精度：FP16
- 置信度阈值：0.5，NMS IoU 阈值：0.5

**预处理流水线（GPU 端到端）**：
1. Letterbox resize（保持宽高比的缩放）
2. BGR → RGB 色彩空间转换
3. 归一化至 [0, 1]
4. Interleaved → Planar 通道重排（HWC → CHW）

**后处理流水线**：
1. GPU → CPU 异步拷贝结果
2. 逐检测列提取最大类别得分
3. 置信度过滤 + NMS 去重
4. 输出 `vector<Detection>`（包含置信度、类别ID、边界框）

### 2.4 类别特定的后处理

检测器对每类目标进行针对性处理（[Detector.cc](file:///home/dl/Universal/src/Detector.cc) 中 `ProcessPedestrian/People/Bicycle/Car/Unknown` 函数）：

- **行人（pedestrian）**：基于宽高比判断站立/坐姿，调整运动概率
- **人群（people）**：更大的检测框，更高的运动概率预设
- **自行车（bicycle）**：检测骑手关联
- **汽车（car）**：检测近处行人以调整动态概率
- **未知类别（unknown）**：通用几何启发式处理

每类均维护独立的 `moving_prob`，支持随时间演化的贝叶斯动态概率更新。

### 2.5 帧间同步机制

检测器在独立线程中异步运行。Tracking 线程通过以下机制同步等待检测结果：

```cpp
// Tracking.cc 中 GrabImageMonocular
while(!isNewDetectedImgArrived()) {
    usleep(1);  // 微秒级轮询
}
mCurrentFrame.SetBoxes(mpDetector->objects);  // 将检测框绑定到当前帧
```

### 2.6 与 ORB_SLAM3 的对比

| 特性 | ORB_SLAM3 | 本系统 |
|------|-----------|--------|
| 目标检测 | 无 | YOLOv11 TensorRT FP16 |
| GPU 推理 | 无 | CUDA 端到端预处理+推理 |
| 语义类别 | 无 | 8 类交通场景目标 |
| 动态概率 | 无 | 类别特定的运动概率与贝叶斯更新 |
| 检测线程 | 无 | 独立线程异步运行 |

---

## 3. 改进二：动态一致性模块（极线约束）

### 3.1 架构概述

本系统构建了 **三层级动态点检测与过滤管道**：

**层级 1 — 检测框预过滤**：[Tracking.cc](file:///home/dl/Universal/src/Tracking.cc) `FilterFeaturePointsByDetection()`
**层级 2 — 极线约束语义点分类**：[Tracking.cc](file:///home/dl/Universal/src/Tracking.cc) `ClassifySemanticPoints()`
**层级 3 — 贝叶斯动态概率演化**：[Tracking.cc](file:///home/dl/Universal/src/Tracking.cc) `RefineDynamicStatusFromSemanticRatio()`

### 3.2 Frame 层改动

在 [Frame.h](file:///home/dl/Universal/include/Frame.h) 中新增了特征点匹配分类队列：

| 成员 | 类型 | 用途 |
|------|------|------|
| `mvFrameMatches` | `vector<pair<int,int>>` | 帧间特征匹配对 |
| `mvStaticMatches` | `vector<pair<int,int>>` | 静态区域匹配队列 |
| `mvSemanticMatches` | `vector<pair<int,int>>` | 语义框内匹配队列 |
| `mvPeopleMatches` | `vector<pair<int,int>>` | 人体区域匹配队列 |
| `mRelativePose` | `Sophus::SE3f` | 相对上一帧的位姿 |

### 3.3 MapPoint 层改动

在 [MapPoint.h](file:///home/dl/Universal/include/MapPoint.h) 中新增了语义/动态状态标记：

| 成员 | 类型 | 用途 |
|------|------|------|
| `mFeatureStatus` | `enum FeatureStatus` | STATIC/DYNAMIC/SEMANTIC/UNKNOWN |
| `mMovingProbability` | `float` | 运动概率（随时间演化） |
| `mStaticProbability` | `float` | 静态概率（= 1 - mMovingProbability） |
| `mnObservedDynamic` | `int` | 被观测为动态的次数 |
| `mnSemanticClass` | `int` | 语义类别 ID |

`mbBad` 标记升级为 `std::atomic<bool>`，实现无锁线程安全读写。

### 3.4 极线约束分类算法

`ClassifySemanticPoints()` 的核心流程：

1. **计算基础矩阵**：利用帧间相对位姿计算本质矩阵 E，进而求基础矩阵 F = K⁻ᵀ · E · K⁻¹
2. **逐语义框处理**：对每个检测框内的特征点，计算其到对应极线的距离
3. **自适应阈值**：
   - 基线自适应：`threshold = 0.8 × median + 1.5 × stddev`（下限 1.0 像素）
   - **抖动拓宽**：当旋转角或垂向位移超过阈值时，乘以 `1.0 + maxExceedRatio × 0.6` 的调整因子
4. **分类判定**：超过阈值的点为动态候选，结合贝叶斯框架更新概率

### 3.5 贝叶斯动态概率框架

`UpdateDynamicProbabilities()` 每 N 帧更新一次类别级动态概率：

- 统计每类检测框中动态/静态特征点的比例
- 使用贝叶斯规则更新 `mvDynamicProbabilities[class_id]`
- 概率超过 0.5 阈值的 MapPoint 被标记为 `DYNAMIC`

### 3.6 可视化

`DrawDynamicSemanticPoints()` 在图像上以不同颜色绘制：
- 静态语义点：绿色
- 动态语义点：红色

同时将带检测框的图像写入 `mImColor` 供 FrameDrawer 显示。

### 3.7 与 ORB_SLAM3 的对比

| 特性 | ORB_SLAM3 | 本系统 |
|------|-----------|--------|
| 动态点检测 | 无 | 检测框 + 极线约束 + 贝叶斯概率三层管线 |
| 极线几何 | 仅用于位姿估计 | 同时用于语义点动态/静态分类 |
| 自适应阈值 | 无 | 基线自适应 + 抖动拓宽 |
| 概率演化 | 无 | 类别级贝叶斯概率更新 |
| MapPoint 状态 | 仅 bad/good | STATIC/DYNAMIC/SEMANTIC/UNKNOWN 四态 |

---

## 4. 改进三：初始化平面系统

### 4.1 架构概述

本系统在单目模式下构建了完整的 **地面平面估计与约束系统**，为核心 SLAM 运算（三角化、BA）提供额外的几何先验。

**关键文件**：
- [Map.h](file:///home/dl/Universal/include/Map.h) — 平面模型存储
- [Tracking.cc](file:///home/dl/Universal/src/Tracking.cc) — 平面拟合与更新方法
- [LocalMapping.cc](file:///home/dl/Universal/src/LocalMapping.cc) — 平面引导三角化
- [Optimizer.cc](file:///home/dl/Universal/src/Optimizer.cc) — BA 平面约束
- [G2oTypes.h](file:///home/dl/Universal/include/G2oTypes.h) — 平面 g2o 边类型

### 4.2 Map 层平面模型存储

[Map.h](file:///home/dl/Universal/include/Map.h) 中新增：

| 成员 | 类型 | 用途 |
|------|------|------|
| `mPlaneNormal` | `Eigen::Vector3f` | 地面平面法向量 |
| `mvpPlaneOffsets` | `vector<float>` | 平面偏移量历史 |
| `mPlaneRefHeight` | `float` | 拟合时的相机高度 |
| `mPlaneRefOffset` | `float` | 初始冻结偏移（FitGroundPlane 时固定） |
| `mPlaneDynamicOffset` | `float` | 动态偏移（每 30 帧由 PlaneRemark 更新） |
| `mPlaneScaleLambda` | `float` | 尺度漂移比率 |
| `mvTriangRatios` | `vector<float>` | 三角化高度比缓冲 |
| `mbPlaneEstimated` | `bool` | 平面是否已估计 |

Map 层提供 15 个新公共接口：`SetPlaneModel()`, `IsPlaneEstimated()`, `GetPlaneNormal()`, `GetPlaneRefOffset()`, `SetPlaneDynamicOffset()`, `SetPlaneScaleLambda()`, `PushTriangRatio()`, 等。

### 4.3 地面平面拟合（Tracking 层）

**`CollectGroundPlaneData()`**：
- 在单目初始化完成后启动
- 从语义地面类检测框（class_id 3=car, 4=van, 5=truck, 8=bus）中收集 MapPoint 指针
- 每新关键帧收集一次，持续 5 秒，然后触发拟合

**`FitGroundPlane()` — 双遍 SVD 拟合**：
1. **第一遍**：对所有非动态 MapPoint 做 SVD，得到初始法向量 `initNormal`
2. **第二遍**：取 `initNormal` 投影最低的 60% 点做第二次 SVD，得到精细化法向量 `planeNormal`（偏离超过 25° 时回退到 `initNormal`）
3. 计算平面偏移：`offset = planeNormal.dot(lowCentroid)`
4. 写入 Map：`SetPlaneModel(normal, offset)`, `SetPlaneRefHeight`, `SetPlaneRefOffset`
5. 标记平面内点：距离在场景尺度 8% 以内的点设 `mnPlaneID=0`, `mfPlaneInfo=1.0`

**`PlaneRemark()` — 动态平面更新**：
- 每 30 帧调用一次
- 以平面标记点的 `n·P` 中位数更新 `d_dynamic`（动态偏移）
- 重新分类 MapPoint：三角化高度比 `t_triang/t_plane < 0.35`（排除建筑物立面）且射线方向向下（`ndir < 0`）的点标记为平面点
- 通过 `Map::SetPlaneDynamicOffset()` 和 `Map::SetPlaneScaleLambda()` 更新

### 4.4 平面引导三角化（LocalMapping 层）

在 `LocalMapping::CreateNewMapPoints()` 中，当平面已估计时，对低视差三角化进行深度修正：

| cos(视差角) | alpha（融合权重） | 含义 |
|------------|-------------------|------|
| > 0.9998 | 0.60 | 近纯旋转，强依赖平面 |
| > 0.9995 | 0.45 | 弱视差 |
| > 0.9990 | 0.30 | 中等视差 |
| > 0.9980 | 0.15 | 过渡阶段 |
| 其他 | 0 | 正常三角化，不干预 |

修正公式：
```
x3D_corrected = Ow1 + dir × (t_triang + alpha × (t_target - t_triang))
```

此外还有 **尺度修正分支**（`alpha_scale`）：当三角化深度与冻结平面深度比值 `scaleRatio < 0.8` 时，`alpha_scale = (1.0 - scaleRatio) × 0.85`，补偿转弯旋转导致的三角化深度系统性偏小。

取 `max(alpha, alpha_scale)` 作为最终融合权重。当 `lambda`（退化指标）严重时，进一步放宽视差要求。

修正后的 MapPoint 会被预标记为 `mnPlaneID=0`, `mfPlaneInfo=3.0`（在 BA 中获得更强的平面约束）。

### 4.5 BA 平面约束（Optimizer 层）

在 [Optimizer.cc](file:///home/dl/Universal/src/Optimizer.cc) 的 6 个 BA 入口函数中，当 `pMap->IsPlaneEstimated()` 为 true 时，为所有平面标记的 MapPoint 添加软约束边：

**MapPoint 级** — `EdgePlaneConstraint`：
```
error = planeNormal · Point_world - refOffset
```
信息矩阵权重：`INFO × mfPlaneInfo`（INFO = 5.0f）

**关键帧级** — `EdgeKFCamCenterToPlaneSE3`：
```
error = planeNormal · CameraCenter_world - refOffset - refHeight
```
信息矩阵权重：0.05（弱约束，仅防止漂移）

受影响的 BA 函数：
- `BundleAdjustment()`
- `FullInertialBA()`
- `LocalBundleAdjustment(KeyFrame*)`（两个重载）
- `LocalInertialBA()`
- `MergeInertialBA()`

### 4.6 2D 检测框升维 3D

`Lift2DBoxesTo3D()` 通过射线-地面平面求交将语义 2D 检测框提升为 3D 边界框：
- 使用类别特定的尺寸先验（来自 `common.h` 中的 `THRESHOLD_RATIOS`）
- 通过投影方程求解高度
- 结果存入 Map 的 `mvPersistentBoxes`（带移动平均去重）

### 4.7 与 ORB_SLAM3 的对比

| 特性 | ORB_SLAM3 | 本系统 |
|------|-----------|--------|
| 地面平面估计 | 无 | 双遍 SVD 拟合 |
| 动态平面更新 | 无 | 每 30 帧 PlaneRemark |
| 平面引导三角化 | 无 | 低视差时 alpha 融合修正 |
| BA 平面约束 | 无 | 6 个 BA 入口的点和帧级平面约束 |
| 尺度漂移检测 | 无 | lambda 退化指标 |
| 2D→3D 框提升 | 无 | 射线-平面求交 + 类别先验 |
| 3D 持久化检测 | 无 | Detection3D 移动平均去重 |

---

## 5. 改进四：扑翼飞行机器人初始化优化

### 5.1 问题背景

扑翼飞行机器人的特点：
- **高频抖动**：翅膀扑动导致相机姿态周期性微小变化
- **近纯旋转场景**：飞行中转弯时视差极小
- **低纹理场景**：50m 高空航拍，地面特征相对稀疏
- **快速运动**：10m/s 飞行速度使帧间运动较大

以上特点导致 ORB_SLAM3 的标准单目初始化经常失败或质量不佳。

### 5.2 自适应匹配数阈值

**ORB_SLAM3**：固定阈值 `nmatches < 100` 触发初始化失败

**本系统**（[Tracking.cc](file:///home/dl/Universal/src/Tracking.cc) `MonocularInitialization()`）：
```cpp
int minMatches = min(200, static_cast<int>(mCurrentFrame.mvKeys.size() * 0.2));
minMatches = max(minMatches, 80);  // 绝对下限
```
- 特征丰富时（500 特征点）：阈值为 `min(200, 100) = 100`
- 特征稀疏时（200 特征点）：阈值为 `min(200, 40) = 40` → 保护为 80
- 自适应匹配帧的实际特征数量

### 5.3 初始化三重质量门控

在 `MonocularInitialization()` 中新增三个预初始化检查（ORB_SLAM3 中不存在）：

**门控 1 — 有效三角化点数**：
```cpp
int minValid3D = min(100, nmatches * 0.5);
minValid3D = max(minValid3D, 40);  // 绝对下限
```
确保足够的有效三维点。

**门控 2 — 10×10 网格覆盖率**：
- 将图像划分为 10×10 空间网格
- 最小覆盖率阈值：`MIN_GRID_COVERAGE = 0.08f`（8%）
- 确保三角化点空间分布均匀，非聚集在单一区域

**门控 3 — SVD 地面平面验证**：
- 对有效 3D 点做 SVD 拟合平面
- 验证质心 Z > 0.2（点在相机前方）
- 验证法向量方向：`ny >= 0.2`（大致指向上方），`nz >= -0.3`
- 处理 SVD 退化情况（奇异值比率检查）

### 5.4 初始化健康检查

`CheckInitializationHealth()` 执行 6 维度初始化质量评估（当前注释掉，按需启用）：

1. 有效 MapPoint ≥ 50
2. 重投影误差中位数 < 2.0，标准差 < 0.8
3. 深度比中位数 < 2.0
4. 基线/深度比 ∈ [0.025, 2.0]
5. 旋转角 < 8.0°（扑翼场景放宽至 25°，注释中注明 "2.5x margin"）
6. 视差中位数 ≥ 0.3°
7. 深度展宽 P80/P20 < 2.0

### 5.5 放宽的关键帧检查

**ORB_SLAM3**：`pKFcur->TrackedMapPoints(1) < 100` 触发重置

**本系统**：放宽至 `< 50`，适应扑翼飞行中可能出现的较低跟踪质量。

### 5.6 图像预处理增强

`fw_enhance.cpp` 实现了论文 "FW-ORB-SLAM: A Monocular Visual SLAM Algorithm for Flapping-Wing Flying Robots"（IEEE RAL 2025）中的局部自适应对比度增强算法：

- **局部纹理增益**：`alpha = 1.0 + k × sigma/mu`（在均质区域保持增益 = 1.0）
- **色彩显著性增益**：像素色彩偏离局部均值的 L2 距离作为额外增强
- **导向滤波**：自导向模式，边缘保持平滑增益图
- **优化**：单通道灰度计算增益（BGR 三通道复用），单次导向滤波，3 倍加速

### 5.7 与 ORB_SLAM3 的对比

| 特性 | ORB_SLAM3 | 本系统 |
|------|-----------|--------|
| 匹配数阈值 | 固定 100 | 自适应 min(200, 20%特征数)，下限 80 |
| 初始化质量门控 | 无 | 3 重门控（有效点数/网格覆盖/平面验证） |
| 关键帧检查阈值 | 100 | 50（放宽） |
| 初始化健康检查 | 无 | 6 维度评估（可配） |
| 图像预处理 | 无 | FW-ORB-SLAM 局部自适应增强 |
| 旋转阈值 | 固定 | 针对扑翼放宽至 25° |

---

## 6. 改进五：抖动强度计算与运用

### 6.1 问题背景

扑翼飞行机器人在飞行中产生周期性高频抖动，表现为：
- 帧间旋转角的快速微小变化
- 垂直方向像素级位移
- 抖动导致特征点检测框内产生"伪动态点"

传统 SLAM 系统将抖动误判为动态物体，或在抖动帧插入低质量关键帧。

### 6.2 抖动指标计算

`CalculateVibrationMetrics()`（[Tracking.cc](file:///home/dl/Universal/src/Tracking.cc)）在每帧计算：

- **旋转抖动强度** `mRotationAngle`：从帧间相对位姿提取旋转角（`AngleAxisd`）
- **垂向抖动强度** `mVerticalDisplacement`：归一化平移向量与 (0,0,1) 的点积绝对值

```cpp
Eigen::AngleAxisd aa(mCurrentFrame.mRelativePose.rotationMatrix());
mCurrentFrame.mRotationAngle = fabs(aa.angle());
mCurrentFrame.mVerticalDisplacement =
    fabs(mCurrentFrame.mRelativePose.translation().normalized().dot(Eigen::Vector3d::UnitZ()));
```

### 6.3 自适应阈值学习

`InitializeVibrationThresholds()`：
- 收集前 **200 帧**（`INITIALIZATION_PHASE = 200`）的旋转角和垂向位移
- 使用 **中位数**（而非百分位数）作为优化阈值，对离群抖动帧具有鲁棒性
- 产出 `mOptimizedRotationThreshold` 和 `mOptimizedVerticalThreshold`

### 6.4 关键帧插入抑制

在 `NeedNewKeyFrame()` 中应用抖动阈值：

```cpp
double rotExceed = mCurrentFrame.mRotationAngle / mOptimizedRotationThreshold;
double vertExceed = mCurrentFrame.mVerticalDisplacement / mOptimizedVerticalThreshold;

if (rotExceed > 2.0 || vertExceed > 2.0) {
    return false;  // 抑制关键帧插入
}
```
当抖动强度超过学习阈值的 **3 倍**（rotExceed > 2.0 即比值 > 3.0），拒绝插入关键帧，避免抖动帧污染地图。

### 6.5 极线约束阈值抖动拓宽

在 `ClassifySemanticPoints()` 的极线检查中，根据当前帧抖动强度动态调整阈值：

```cpp
float adjustmentFactor = 1.0 + maxExceedRatio * 0.6;
threshold *= adjustmentFactor;
```

当旋转或垂向位移超过学习阈值时，极线距离容忍度相应增大，防止抖动导致的静态点被误分类为动态点。

### 6.6 Frame 层存储

[Frame.h](file:///home/dl/Universal/include/Frame.h) 中新增：
- `float mRotationAngle` — 帧间旋转角（弧度）
- `float mVerticalDisplacement` — 帧间垂向位移（像素单位）
- `SetVibrationMetrics()` / `GetRotationAngle()` / `GetVerticalDisplacement()` — 读写接口

### 6.7 与 ORB_SLAM3 的对比

| 特性 | ORB_SLAM3 | 本系统 |
|------|-----------|--------|
| 抖动感知 | 无 | 每帧计算旋转角 + 垂向位移 |
| 自适应阈值 | 无 | 200 帧中位数学习 |
| 关键帧抖动抑制 | 无 | 超过 3× 阈值拒绝插入 |
| 极线阈值抖动补偿 | 无 | 自适应拓宽调整因子 |
| Frame 振动数据 | 无 | mRotationAngle + mVerticalDisplacement |

---

## 7. 其他改进

### 7.1 图像质量评估

[ImageQuality.h](file:///home/dl/Universal/include/ImageQuality.h) 提供全面的帧级图像质量度量：

| 指标 | 函数 | 用途 |
|------|------|------|
| 锐度 | `ComputeSharpness()` | 拉普拉斯方差 |
| 响应分布 | `ComputeResponseHistogram()` | 6 区间 Harris 响应直方图 |
| 金字塔分布 | `ComputeOctaveDistribution()` | 每层关键点数量 |
| 空间均匀性 | `ComputeDistribution()` | 10×10 网格覆盖率 |
| 匹配质量 | `MatchingMetrics` 结构体 | 匹配率/内点率/内点响应统计 |

Tracking 层在每帧调用 `LogImageQuality()` 和 `LogMatchingQuality()`，便于离线分析哪些帧导致跟踪退化。

### 7.2 分阶段计时统计

[Tracking.h](file:///home/dl/Universal/include/Tracking.h) 新增细粒度计时变量：

| 变量 | 记录内容 |
|------|---------|
| `mdStageOrbExtract` | ORB 特征提取耗时 |
| `mdStageWaitDetect` | 等待检测器结果耗时 |
| `mdStageTrack` | Track() 函数总耗时 |
| `mdCurOrbExtractMs` | 当前帧 ORB 提取耗时 |
| `mdCurWaitDetectMs` | 当前帧等待检测耗时 |

通过全局标志 `gEnableTimingStats` 开关，支持详细的性能剖析。

### 7.3 CUDA 图像预处理

[preprocess.cu](file:///home/dl/Universal/src/preprocess.cu) 实现 GPU 端图像预处理：
- Letterbox 缩放 + 反向仿射映射
- 双线性插值（边界外填灰 128）
- BGR → RGB + 归一化至 [0,1]
- Interleaved → Planar 通道重排
- 异步 CUDA 流执行

256 线程/块配置，最大化 GPU 占用率。使用页锁定内存加速 PCIe 传输。

### 7.4 类别统计与可视化

新增的语义相关能力：
- `CategoryStatistics` 结构体：跟踪每类检测目标的动态/静态/语义出现次数
- `UpdateCategoryStatistics()`：每帧更新统计
- `SaveMapPointsPLY()`：导出带语义着色的稀疏点云（STATIC=蓝色, DYNAMIC=红色, 语义类=类别色, 未知=灰色）
- `SavePersistentBoxesPLY()`：导出持久化 3D 检测框的线框 PLY 文件

### 7.5 线程安全改进

- `MapPoint::mbBad` 从普通 `bool` 升级为 `std::atomic<bool>`，`isBad()` 不再需要 mutex 锁
- `cv::destroyAllWindows()` 在 Shutdown() 中由主线程统一执行，避免多线程 X11 BadWindow 崩溃

### 7.6 未改动部分

以下 ORB_SLAM3 核心模块在本系统中**完全未修改**：
- `Converter.h/cc` — 坐标转换
- `Atlas.h/cc` — 地图集管理
- `Config.h/cc` — 配置解析
- `GeometricTools.h/cc` — 基础几何工具
- `KeyFrame.h/cc` — 关键帧管理
- `ORBmatcher.h/cc` — 特征匹配（仅新增中文注释）
- `ORBextractor.h/cc` — ORB 特征提取
- `TwoViewReconstruction.h/cc` — 双视图重建
- `Sim3Solver.cc` — Sim3 求解
- `MLPnPsolver.cpp` — MLPnP 求解
- `Viewer.cc` / `FrameDrawer.cc` / `MapDrawer.cc` — 可视化模块

---

## 8. 完整文件改动清单

### 新增文件（16 个）

| 文件路径 | 功能 |
|---------|------|
| `include/Detector.h` | 目标检测器头文件 |
| `src/Detector.cc` | 检测器实现：TensorRT 引擎管理、推理、类别处理 |
| `src/tensorrt/YOLOv11.h` | YOLOv11 TensorRT 类声明 |
| `src/tensorrt/YOLOv11.cpp` | YOLOv11 TensorRT 推理实现 |
| `src/tensorrt/main.cpp` | 独立推理入口程序 |
| `include/common.h` | 自定义 8 类语义定义 |
| `include/common_coco.h` | COCO 80 类语义定义 |
| `include/macros.h` | 跨平台 API 宏 / TensorRT 版本兼容 |
| `include/logging.h` | TensorRT 日志框架（NVIDIA） |
| `include/cuda_utils.h` | CUDA 错误检查宏 |
| `include/preprocess.h` | CUDA 预处理头文件 |
| `src/preprocess.cu` | CUDA 图像预处理内核 |
| `include/ImageQuality.h` | 图像质量评估工具 |
| `src/extractor_orb/fw_enhance.cpp` | FW-ORB-SLAM 对比度增强 |
| `src/extractor_orb/enhance_contrast.cpp` | CLAHE 对比度增强 |
| `src/extractor_orb/fusion.cpp` | 双焦距图像融合 |

### 修改文件（8 个）

| 文件 | 改动量（估算） | 改动内容 |
|------|---------------|---------|
| `include/System.h` | +13 行 | Detector 指针/线程、PLY 导出方法 |
| `src/System.cc` | +305 行 | 检测器线程管理、图像投喂、Shutdown 清理、PLY 导出 |
| `include/Tracking.h` | +191 行 | 检测器接口、动态检测方法、平面拟合、振动阈值、类别统计 |
| `src/Tracking.cc` | +1084 行 | 所有新增方法的实现（最核心的改动文件） |
| `include/Frame.h` | +50 行 | 匹配队列、振动指标、检测框 |
| `src/Frame.cc` | +61 行 | 新增方法的实现 |
| `include/MapPoint.h` | +45 行 | FeatureStatus 枚举、平面成员、动态概率、序列化 |
| `src/MapPoint.cc` | +95 行 | 新增方法的实现、原子操作、线程安全 |
| `include/Map.h` | +47 行 | Detection3D 结构体、平面模型 API、持久化检测框 |
| `src/Map.cc` | +135 行 | 平面模型实现、持久化检测框去重 |
| `include/G2oTypes.h` | +95 行 | 3 个平面约束 g2o 边类型 |
| `src/Optimizer.cc` | +28 处 | 6 个 BA 函数的平面约束边注入 |
| `src/LocalMapping.cc` | +73 行 | 平面引导三角化融合 |

### 未修改文件（按模块分类）

- **基础模块**：`Config.h/cc`, `Converter.h/cc`, `Atlas.h/cc`
- **几何模块**：`GeometricTools.h/cc`, `Sim3Solver.cc`, `TwoViewReconstruction.h/cc`, `MLPnPsolver.cpp`, `OptimizableTypes.cpp`
- **特征模块**：`ORBextractor.h/cc`, `ORBmatcher.h/cc`, `KeyFrameDatabase.h/cc`
- **框架模块**：`KeyFrame.h/cc`, `MapDrawer.h/cc`, `FrameDrawer.h/cc`, `Viewer.cc`
- **其他**：`LoopClosing.cc`, `ImuTypes.cc`, `G2oTypes.cc`, `Settings.cc`, `CameraModels/`

---

## 附录：改进路线图总结

```
ORB_SLAM3（原始）
    │
    ├── [新增] TensorRT YOLOv11 目标检测模块
    │   ├── GPU 端到端预处理 + FP16 推理
    │   ├── 8 类 VisDrone 语义标签
    │   └── 类别特定后处理 + 贝叶斯动态概率
    │
    ├── [新增] 动态一致性模块
    │   ├── 检测框预过滤 + 极线约束分类
    │   ├── 贝叶斯动态概率演化框架
    │   ├── 抖动自适应阈值补偿
    │   └── MapPoint 四态分类（STATIC/DYNAMIC/SEMANTIC/UNKNOWN）
    │
    ├── [新增] 地面平面系统
    │   ├── 双遍 SVD 平面拟合 + PlaneRemark 动态更新
    │   ├── 平面引导三角化（低视差融合 + 尺度修正）
    │   ├── 6 个 BA 入口的软约束边注入
    │   ├── lambda 退化检测与门控
    │   └── 2D→3D 检测框提升 + 持久化
    │
    ├── [改进] 扑翼初始化优化
    │   ├── 自适应匹配阈值
    │   ├── 三重初始化质量门控
    │   ├── 放宽关键帧检查
    │   └── FW-ORB-SLAM 对比度增强
    │
    ├── [新增] 抖动强度系统
    │   ├── 帧间旋转角 + 垂向位移计算
    │   ├── 200 帧中位数自适应阈值学习
    │   ├── 关键帧插入抖动抑制
    │   └── 极线阈值抖动补偿
    │
    └── [辅助] 图像质量 / 计时 / 可视化 / 线程安全
```

---

> **文档生成日期**：2026-07-14
> **代码路径**：Universal（本系统）`/home/dl/Universal` | ORB_SLAM3（基线）`/home/dl/ORB_SLAM3`
