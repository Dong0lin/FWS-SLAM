/**
* 对外接口头文件 — Qt 项目只需 include 这一个头文件
* 不暴露任何 ORB-SLAM3 内部依赖（OpenCV, CUDA, Pangolin, TensorRT 等）
*/

#ifndef SLAM_INTERFACE_H
#define SLAM_INTERFACE_H

#include <string>
#include <Eigen/Dense>

namespace ORB_SLAM3 {

// 前向声明，隐藏内部实现
class SlamInterfaceImpl;

class SlamInterface
{
public:
    // 传感器类型
    enum SensorType { MONOCULAR = 0, STEREO = 1, RGBD = 2 };

    SlamInterface();
    ~SlamInterface();

    // 初始化 SLAM 系统（bUseViewer 固定为 false）
    bool Init(const std::string& vocabPath, const std::string& settingsPath,
              SensorType sensor);

    // 处理一帧图像，返回相机位姿（如果跟踪失败则返回单位阵）
    Eigen::Matrix4f TrackMonocular(const unsigned char* rgbData,
                                   int width, int height, double timestamp);

    // 获取当前帧图像（带特征点绘制），返回 RGB 数据
    // width/height 为输出参数
    const unsigned char* GetCurrentFrame(int& width, int& height);

    // 获取相机世界位姿矩阵（用于 3D 视图）
    Eigen::Matrix4f GetCameraPose();

    // 获取跟踪状态：0=未初始化, 1=正在初始化, 2=正常, 3=短暂丢失, 4=丢失
    int GetTrackingState();

    // 获取统计信息
    int GetMapPointsCount();
    int GetKeyFramesCount();

    // 获取地图点位置和颜色 (最多 maxCount 个，返回实际数量)
    // outPositions: float[3*N] 存储 x,y,z 坐标
    // outColors:    uint8_t[3*N] 存储 R,G,B (可为nullptr表示不需要颜色)
    int GetAllMapPoints(float* outPositions, uint8_t* outColors, int maxCount);

    // 获取关键帧位姿和状态 (最多 maxCount 个，返回实际数量)
    // outPoses:  float[16*N] 存储 4x4 列主序 T_wc
    // outStatus: uint8_t[N]  状态标记: 0=普通, 1=首帧, 2=固定, 3=LBA (可为nullptr)
    int GetAllKeyFramePoses(float* outPoses, uint8_t* outStatus, int maxCount);

    // 控制 3D 检测框计算
    void SetEnable3DBoxDetection(bool enable);

    // 获取持久化 3D 框 (最多 maxCount 个) 和平面参数
    // outBoxes: float[8*N], 每框: center(3)+width+depth+height+class_id+nObs
    // outPlaneNormal: float[3] 地面平面法向量 (可为nullptr)
    // outPlaneOffset: float   平面方程偏移 d (可为nullptr)
    int GetPersistentBoxes(float* outBoxes, float* outPlaneNormal, float* outPlaneOffset, int maxCount);

    // 保存轨迹
    void SaveTrajectory(const std::string& path);

    // 激活/关闭 仅定位模式
    void SetLocalizationMode(bool enabled);

    // 步进模式
    void SetStepByStep(bool enabled);

    // 可视化模式：false=原图+检测框, true=动态一致性语义点
    void SetVisualizationMode(bool showDynamic);

    // 重置地图
    void ResetActiveMap();

    // 停止并清理
    void Shutdown();

    // 图像缩放因子
    float GetImageScale();

private:
    SlamInterfaceImpl* mImpl;
};

} // namespace ORB_SLAM3

#endif // SLAM_INTERFACE_H
