/**
* SlamInterface 实现 — 在这里包含所有 ORB-SLAM3 内部头文件
* Qt 项目不会直接 include 这个文件
*/

#include "SlamInterface.h"

// 所有内部依赖都在 .cc 中，不暴露给调用者
#include "System.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "common.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>   // cv::imwrite
#include <sys/stat.h>   // mkdir
#include <cstdio>       // snprintf

namespace ORB_SLAM3 {

class SlamInterfaceImpl
{
public:
    System*              pSystem = nullptr;
    FrameDrawer*         pFrameDrawer = nullptr;
    MapDrawer*           pMapDrawer = nullptr;
    float                imageScale = 1.0f;

    // 缓存的最后一帧（RGB 格式）
    std::vector<unsigned char> cachedFrame;
    int                 cachedWidth = 0;
    int                 cachedHeight = 0;

    // 可视化模式：0=原图(短焦), 1=动态一致性, 2=长短焦(右目长焦)
    int                 visMode = 0;

    // 动态一致性图像保存
    bool                saveDynamicVis = false;   // 是否保存
    int                 visSaveCounter = 0;       // 已保存帧计数

    // 帧刷新节流：每 N 帧才调用一次 DrawFrame（减少 Qt 模式下的同步绘制开销）
    int                 frameSkipInterval = 2;   // 每 2 帧刷新一次 → 约 9-10fps 刷新率
    int                 frameSkipCounter = 0;
};

namespace {
// 保存动态一致性图像：visMode==1 时缓存帧即为动态一致性语义点图像
void SaveDynamicVisImage(ORB_SLAM3::SlamInterfaceImpl* impl, const cv::Mat& frameBGR)
{
    static bool dirCreated = false;
    const char* dir = "dynamic_vis";
    if (!dirCreated) {
        if (mkdir(dir, 0755) != 0 && errno != EEXIST) {
            std::cerr << "[SlamInterface] 无法创建动态一致性保存目录 " << dir << std::endl;
            return;
        }
        dirCreated = true;
    }
    char name[128];
    snprintf(name, sizeof(name), "%s/frame_%06d.png", dir, impl->visSaveCounter++);
    if (!cv::imwrite(name, frameBGR))
        std::cerr << "[SlamInterface] 保存动态一致性图像失败: " << name << std::endl;
}
// 按当前可视化模式缓存一帧图像（0=原图/短焦, 1=动态一致性, 2=长短焦/右目长焦）
void CacheFrame(ORB_SLAM3::SlamInterfaceImpl* impl, float imageScale)
{
    ORB_SLAM3::FrameDrawer* drawer = impl->pFrameDrawer;
    if (!drawer) return;

    cv::Mat frameBGR;
    if (impl->visMode == 2)
        frameBGR = drawer->DrawRightFrame(imageScale);   // 右目长焦 + 覆盖
    else
        frameBGR = drawer->DrawFrame(imageScale);        // 左目短焦（原图/动态一致性）
    if (frameBGR.empty()) return;

    // 动态一致性模式：保存当前帧图像（含语义点可视化）
    if (impl->saveDynamicVis && impl->visMode == 1)
        SaveDynamicVisImage(impl, frameBGR);

    cv::Mat frameRGB;
    cv::cvtColor(frameBGR, frameRGB, cv::COLOR_BGR2RGB);
    // 确保连续存储，避免 QImage 读取时出现行对齐问题
    if (!frameRGB.isContinuous())
        frameRGB = frameRGB.clone();

    impl->cachedWidth  = frameRGB.cols;
    impl->cachedHeight = frameRGB.rows;
    impl->cachedFrame.assign(frameRGB.data, frameRGB.data + frameRGB.total() * 3);
}
} // namespace

SlamInterface::SlamInterface()
    : mImpl(new SlamInterfaceImpl())
{}

SlamInterface::~SlamInterface()
{
    Shutdown();
    delete mImpl;
}

bool SlamInterface::Init(const std::string& vocabPath,
                         const std::string& settingsPath,
                         SensorType sensor)
{
    if (mImpl->pSystem) {
        Shutdown();  // 先清理旧的
    }

    System::eSensor s;
    switch (sensor) {
        case MONOCULAR: s = System::MONOCULAR; break;
        case STEREO:    s = System::STEREO;    break;
        case RGBD:      s = System::RGBD;      break;
        default: return false;
    }

    mImpl->pSystem = new System(vocabPath, settingsPath, s, System::VIEWER_QT);
    mImpl->pFrameDrawer = mImpl->pSystem->GetFrameDrawer();
    mImpl->pFrameDrawer->SetQtMode(true);
    mImpl->pMapDrawer   = mImpl->pSystem->GetMapDrawer();
    mImpl->imageScale   = mImpl->pSystem->GetImageScale();

    // 读取动态一致性图像保存开关（可选 YAML 键 SaveDynamicVis: 1，缺省关闭）
    try {
        cv::FileStorage fSettings(settingsPath, cv::FileStorage::READ);
        if (fSettings.isOpened()) {
            int saveVis = (int)fSettings["SaveDynamicVis"];
            mImpl->saveDynamicVis = (saveVis != 0);
            fSettings.release();
            if (mImpl->saveDynamicVis)
                std::cout << "[SlamInterface] 动态一致性图像保存已开启 → ./dynamic_vis/" << std::endl;
        }
    } catch (...) {
        mImpl->saveDynamicVis = false;
    }

    return true;
}

void SlamInterface::SetSaveDynamicVis(bool enable)
{
    mImpl->saveDynamicVis = enable;
    if (enable)
        std::cout << "[SlamInterface] 动态一致性图像保存已开启 → ./dynamic_vis/" << std::endl;
    else
        std::cout << "[SlamInterface] 动态一致性图像保存已关闭" << std::endl;
}

bool SlamInterface::IsSaveDynamicVis() const
{
    return mImpl->saveDynamicVis;
}

Eigen::Matrix4f SlamInterface::TrackMonocular(const unsigned char* rgbData,
                                              int width, int height,
                                              double timestamp)
{
    if (!mImpl->pSystem) return Eigen::Matrix4f::Identity();

    // 输入数据保持原始格式（cv::imread 读的是 BGR），
    // System 内部根据 Camera.RGB 配置自行处理颜色转换
    cv::Mat im(height, width, CV_8UC3, (void*)rgbData);

    // 缩放
    if (mImpl->imageScale != 1.f) {
        int w = im.cols * mImpl->imageScale;
        int h = im.rows * mImpl->imageScale;
        cv::resize(im, im, cv::Size(w, h));
    }

    Sophus::SE3f pose = mImpl->pSystem->TrackMonocular(im, timestamp);

    // 帧刷新节流：每 N 帧才同步调用 DrawFrame（减少 Qt 模式下的同步绘制开销）
    mImpl->frameSkipCounter++;
    if (mImpl->frameSkipCounter % mImpl->frameSkipInterval == 0)
        CacheFrame(mImpl, mImpl->imageScale);

    return pose.matrix();
}

Eigen::Matrix4f SlamInterface::TrackStereo(const unsigned char* leftRgb,
                                           const unsigned char* rightRgb,
                                           int width, int height,
                                           double timestamp)
{
    if (!mImpl->pSystem) return Eigen::Matrix4f::Identity();

    cv::Mat imL(height, width, CV_8UC3, (void*)leftRgb);
    cv::Mat imR(height, width, CV_8UC3, (void*)rightRgb);

    if (mImpl->imageScale != 1.f) {
        int w = imL.cols * mImpl->imageScale;
        int h = imL.rows * mImpl->imageScale;
        cv::resize(imL, imL, cv::Size(w, h));
        cv::resize(imR, imR, cv::Size(w, h));
    }

    Sophus::SE3f pose = mImpl->pSystem->TrackStereo(imL, imR, timestamp);

    // 帧刷新节流：每 N 帧才同步调用 DrawFrame（减少 Qt 模式下的同步绘制开销）
    mImpl->frameSkipCounter++;
    if (mImpl->frameSkipCounter % mImpl->frameSkipInterval == 0)
        CacheFrame(mImpl, mImpl->imageScale);

    return pose.matrix();
}

const unsigned char* SlamInterface::GetCurrentFrame(int& width, int& height)
{
    width  = mImpl->cachedWidth;
    height = mImpl->cachedHeight;
    return mImpl->cachedFrame.empty() ? nullptr : mImpl->cachedFrame.data();
}

Eigen::Matrix4f SlamInterface::GetCameraPose()
{
    if (!mImpl->pMapDrawer) return Eigen::Matrix4f::Identity();
    return mImpl->pMapDrawer->GetCurrentCameraPose();
}

int SlamInterface::GetTrackingState()
{
    if (!mImpl->pSystem) return 0;
    return mImpl->pSystem->GetTrackingState();
}

int SlamInterface::GetMapPointsCount()
{
    if (!mImpl->pMapDrawer || !mImpl->pMapDrawer->mpAtlas) return 0;
    auto* pMap = mImpl->pMapDrawer->mpAtlas->GetCurrentMap();
    return pMap ? (int)pMap->GetAllMapPoints().size() : 0;
}

int SlamInterface::GetKeyFramesCount()
{
    if (!mImpl->pMapDrawer || !mImpl->pMapDrawer->mpAtlas) return 0;
    auto* pMap = mImpl->pMapDrawer->mpAtlas->GetCurrentMap();
    return pMap ? (int)pMap->GetAllKeyFrames().size() : 0;
}

int SlamInterface::GetAllMapPoints(float* outPositions, uint8_t* outColors, int maxCount)
{
    if (!mImpl->pMapDrawer || !mImpl->pMapDrawer->mpAtlas) return 0;
    auto* pMap = mImpl->pMapDrawer->mpAtlas->GetCurrentMap();
    if (!pMap) return 0;

    auto vpMPs = pMap->GetAllMapPoints();
    int count = 0;
    for (auto* pMP : vpMPs) {
        if (!pMP || pMP->isBad()) continue;
        if (count >= maxCount) break;
        Eigen::Vector3f pos = pMP->GetWorldPos();
        outPositions[count*3 + 0] = pos.x();
        outPositions[count*3 + 1] = pos.y();
        outPositions[count*3 + 2] = pos.z();
        if (outColors) {
            // 点云着色规则：
            //   - 非语义点 (mnSemanticClass == -1)  → 蓝色
            //   - 动态语义点 (IsDynamicMapPoint)     → 红色
            //   - 静态语义点                          → 按 common.h COLORS 类别颜色
            int semClass = pMP->mnSemanticClass;
            if (semClass == -1) {
                // 非语义点：蓝色
                outColors[count*3 + 0] = 0;
                outColors[count*3 + 1] = 0;
                outColors[count*3 + 2] = 255;
            } else if (pMP->IsDynamicMapPoint()) {
                // 动态语义点：红色
                outColors[count*3 + 0] = 255;
                outColors[count*3 + 1] = 0;
                outColors[count*3 + 2] = 0;
            } else {
                // 静态语义点：类别颜色
                const auto& c = COLORS[semClass % COLORS.size()];
                outColors[count*3 + 0] = (uint8_t)c[0];
                outColors[count*3 + 1] = (uint8_t)c[1];
                outColors[count*3 + 2] = (uint8_t)c[2];
            }
        }
        count++;
    }
    return count;
}

int SlamInterface::GetAllKeyFramePoses(float* outPoses, uint8_t* outStatus, int maxCount)
{
    if (!mImpl->pMapDrawer || !mImpl->pMapDrawer->mpAtlas) return 0;
    auto* pMap = mImpl->pMapDrawer->mpAtlas->GetCurrentMap();
    if (!pMap) return 0;

    auto vpKFs = pMap->GetAllKeyFrames();
    int count = 0;
    for (auto* pKF : vpKFs) {
        if (!pKF || pKF->isBad()) continue;
        if (count >= maxCount) break;
        Eigen::Matrix4f Twc = pKF->GetPoseInverse().matrix();
        for (int r = 0; r < 4; r++)
            for (int c = 0; c < 4; c++)
                outPoses[count*16 + c*4 + r] = Twc(r, c);
        if (outStatus) {
            // 状态: 0=普通, 1=首帧, 2=固定, 3=LBA优化
            if (!pKF->GetParent())
                outStatus[count] = 1;  // 首帧
            else if (pMap->msFixedKFs.count(pKF->mnId))
                outStatus[count] = 2;  // 固定
            else if (pMap->msOptKFs.count(pKF->mnId))
                outStatus[count] = 3;  // LBA优化
            else
                outStatus[count] = 0;  // 普通
        }
        count++;
    }
    return count;
}

void SlamInterface::SetEnable3DBoxDetection(bool enable)
{
    if (!mImpl->pSystem) return;
    mImpl->pSystem->GetTracker()->SetEnable3DBoxDetection(enable);
}

int SlamInterface::GetPersistentBoxes(float* outBoxes, float* outPlaneNormal, float* outPlaneOffset, int maxCount)
{
    if (!mImpl->pMapDrawer || !mImpl->pMapDrawer->mpAtlas) return 0;
    auto* pMap = mImpl->pMapDrawer->mpAtlas->GetCurrentMap();
    if (!pMap) return 0;

    const auto& boxes = pMap->GetPersistentBoxes();
    int count = 0;
    for (const auto& box : boxes) {
        if (!box.bValid) continue;
        if (count >= maxCount) break;
        outBoxes[count*11 + 0] = box.center.x();
        outBoxes[count*11 + 1] = box.center.y();
        outBoxes[count*11 + 2] = box.center.z();
        outBoxes[count*11 + 3] = box.width;
        outBoxes[count*11 + 4] = box.depth;
        outBoxes[count*11 + 5] = box.height;
        outBoxes[count*11 + 6] = (float)box.class_id;
        outBoxes[count*11 + 7] = (float)box.nObservations;
        outBoxes[count*11 + 8] = box.heading.x();
        outBoxes[count*11 + 9] = box.heading.y();
        outBoxes[count*11 + 10] = box.heading.z();
        count++;
    }

    if (pMap->IsPlaneEstimated()) {
        if (outPlaneNormal) {
            Eigen::Vector3f N = pMap->GetPlaneNormal();
            outPlaneNormal[0] = N.x();
            outPlaneNormal[1] = N.y();
            outPlaneNormal[2] = N.z();
        }
        if (outPlaneOffset) {
            *outPlaneOffset = pMap->GetPlaneOffsets()[0];
        }
    }

    return count;
}

void SlamInterface::SaveTrajectory(const std::string& dir)
{
    if (mImpl->pSystem) {
        // TUM 格式（时间戳为秒），与 stereo_multifocal 及 evo_ape tum 工作流一致；
        // EuRoC 版输出纳秒，与 GT(time.tum) 对不上
        mImpl->pSystem->SaveTrajectoryTUM(dir + "/CameraTrajectory.txt");
        mImpl->pSystem->SaveKeyFrameTrajectoryTUM(dir + "/KeyFrameTrajectory.txt");
    }
}

void SlamInterface::SetLocalizationMode(bool enabled)
{
    if (!mImpl->pSystem) return;
    if (enabled)
        mImpl->pSystem->ActivateLocalizationMode();
    else
        mImpl->pSystem->DeactivateLocalizationMode();
}

void SlamInterface::SetStepByStep(bool enabled)
{
    if (!mImpl->pSystem) return;
    mImpl->pSystem->GetTracker()->SetStepByStep(enabled);
}

void SlamInterface::SetVisualizationMode(int mode)
{
    mImpl->visMode = mode;
    if (!mImpl->pSystem) return;
    mImpl->pSystem->GetTracker()->SetShowDynamicVis(mode == 1);
}

void SlamInterface::ResetActiveMap()
{
    if (mImpl->pSystem)
        mImpl->pSystem->ResetActiveMap();
}

void SlamInterface::Shutdown()
{
    if (mImpl->pSystem) {
        mImpl->pSystem->Shutdown();
        delete mImpl->pSystem;
        mImpl->pSystem = nullptr;
        mImpl->pFrameDrawer = nullptr;
        mImpl->pMapDrawer = nullptr;
    }
}

float SlamInterface::GetImageScale()
{
    return mImpl->imageScale;
}

} // namespace ORB_SLAM3
