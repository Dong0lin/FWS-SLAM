/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Viewer.h"
#include "FrameDrawer.h"
#include "Atlas.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "Frame.h"
#include "ORBVocabulary.h"
#include "KeyFrameDatabase.h"
#include "ORBextractor.h"
#include "MapDrawer.h"
#include "System.h"
#include "ImuTypes.h"
#include "Settings.h"

#include "GeometricCamera.h"

#include "Detector.h"
#include "ImageQuality.h"

#include <mutex>
#include <unordered_set>

// #include "pointcloudmapping.h"

class PointCloudMapping;

namespace ORB_SLAM3
{

// 全局耗时统计开关：true 时统计并在结束时打印各阶段耗时；false 时零开销。
// 可在程序入口（如 main）设置，或直接改此默认值。
extern bool gEnableTimingStats;

class Viewer;
class FrameDrawer;
class Atlas;
class LocalMapping;
class LoopClosing;
class System;
class Settings;
class Detector;

struct CategoryStatistics
{
    int class_id;
    std::string class_name;
    int total_count;
    int dynamic_count;
};

class Tracking
{  

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Atlas* pAtlas,
             KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, Settings* settings, const string &_nameSeq=std::string());

    ~Tracking();

    // Parse the config file
    bool ParseCamParamFile(cv::FileStorage &fSettings);
    bool ParseORBParamFile(cv::FileStorage &fSettings);
    bool ParseIMUParamFile(cv::FileStorage &fSettings);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    Sophus::SE3f GrabImageStereo(const cv::Mat &imRectLeft,const cv::Mat &imRectRight, const double &timestamp, string filename);
    Sophus::SE3f GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, string filename);
    Sophus::SE3f GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename);

    void GrabImuData(const IMU::Point &imuMeasurement);

    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);
    void SetDetector(Detector* pDetector);
    void SetStepByStep(bool bSet);

    // 3D 检测框开关
    void SetEnable3DBoxDetection(bool flag) { mbEnable3DBoxDetection = flag; }
    bool Is3DBoxDetectionEnabled()          { return mbEnable3DBoxDetection; }
    // 图像/匹配质量日志：默认关闭；开启时打开 image_quality.txt 并按30帧采样写入
    void SetSaveQuality(bool flag);

    // 动态一致性可视化开关（替代检测框绘制，用于 Qt 界面）
    void SetShowDynamicVis(bool flag)       { mbShowDynamicVis = flag; }
    bool IsShowDynamicVis()                 { return mbShowDynamicVis; }

    // Load new settings
    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag);

    void UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame);
    
    KeyFrame* GetLastKeyFrame()
    {
        return mpLastKeyFrame;
    }

    void CreateMapInAtlas();
    //std::mutex mMutexTracks;

    //--
    void NewDataset();
    int GetMatchesInliers();

    //DEBUG
    void SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, string strFolder="");
    void SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, Map* pMap);

    float GetImageScale();

#ifdef REGISTER_LOOP
    void RequestStop();
    bool isStopped();
    void Release();
    bool stopRequested();
#endif

public:

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        RECENTLY_LOST=3,
        LOST=4,
        OK_KLT=5
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Input sensor
    int mSensor;

    // Current Frame
    Frame mCurrentFrame;
    Frame mLastFrame;
    cv::Mat mImRGB;
    cv::Mat mImGray;
    cv::Mat imDepth; // adding mImDepth member to realize pointcloud view

    cv::Mat mImColor;//保存来自Detector的带有检测框的图像，供FrameDrawer使用
    cv::Mat mImColorRight;//右目检测框绘制后的图像，供FrameDrawer右窗口显示

    // Initialization Variables (Monocular)
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<Sophus::SE3f> mlRelativeFramePoses;
    list<KeyFrame*> mlpReferences;
    list<double> mlFrameTimes;
    list<bool> mlbLost;

    // frames with estimated pose
    int mTrackedFr;
    bool mbStep;

    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking;

    void Reset(bool bLocMap = false);
    void ResetActiveMap(bool bLocMap = false);

    float mMeanTrack;
    bool mbInitWith3KFs;
    double t0; // time-stamp of first read frame
    double t0vis; // time-stamp of first inserted keyframe
    double t0IMU; // time-stamp of IMU initialization
    bool mFastInit = false;


    vector<MapPoint*> GetLocalMapMPS();

    bool mbWriteStats;
    
        //detection functions/variables
	void GetImgForDetector(const cv::Mat& img);
	void GetImgForDetector(const cv::Mat& imgLeft, const cv::Mat& imgRight);  // 双目：左+右一起送入检测线程
	void SetDetectorRight(Detector* pDetector) { mpDetectorRight = pDetector; }  // 右目独立检测线程（长短焦模式）
	bool mbNewDetImgFlag;
	bool mbNewDetImgFlagRight;

	// 检测完成等待：用条件变量替代 usleep 忙等待
	void WaitForDetection();

#ifdef REGISTER_TIMES
    void LocalMapStats2File();
    void TrackStats2File();
    void PrintTimeStats();

    vector<double> vdRectStereo_ms;
    vector<double> vdResizeImage_ms;
    vector<double> vdORBExtract_ms;
    vector<double> vdStereoMatch_ms;
    vector<double> vdIMUInteg_ms;
    vector<double> vdPosePred_ms;
    vector<double> vdLMTrack_ms;
    vector<double> vdNewKF_ms;
    vector<double> vdTrackTotal_ms;
#endif

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();

    // Map initialization for stereo and RGB-D
    void StereoInitialization();
    // 双目初始化质量评估（针对扑翼航拍：有效深度点数量 + 空间网格覆盖率）
    bool CheckInitializationQuality(int &nValidDepthPoints, int &nOccupiedCells);

    // Map initialization for monocular
    void MonocularInitialization();
    //void CreateNewMapPoints();
    void CreateInitialMapMonocular();
    bool CheckInitializationHealth(KeyFrame* pKF1, KeyFrame* pKF2);
    void CollectGroundPlaneData();       // 收集语义地面点数据(初始化后100帧)
    void FitGroundPlane();               // 拟合地面平面0(100帧后执行)
    void PlaneRemark();                  // 每30帧刷新平面标记 + 动态偏移量 + λ + BA约束强度
    void ProjectPlanePoints();           // 将平面归属点硬投影到平面上（锚定尺度）
    void Lift2DBoxesTo3D();              // 2D检测框 → 3D框：按模式分发（单目/长短焦）
    void Lift2DBoxesTo3D_Mono();         // 单目模式实现（地面反投影 + 高度/先验融合）
    void Lift2DBoxesTo3D_Stereo();       // 长短焦模式实现（左右目匹配+前后帧匹配特征点）
    // 单目：2D框+位姿+地面平面 → 地面足迹（位置/朝向/宽深）。
    // 不依赖稀疏地图点/立体视差，远距离航拍下更稳。失败(无平面/射线不相交)返回 false。
    bool EstimateGroundFootprintMono(const Detection& det, Eigen::Vector3f& P_ctr,
                                     Eigen::Vector3f& heading, float& widthSlam, float& depthSlam);
    void AlignBoxRows();                 // 持续维护：同类成排目标微调位置/朝向（中心近+共线）

    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();
    bool TrackWithMotionModel();
    bool PredictStateIMU();

    bool Relocalization();

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap();
    void SearchLocalPoints();

    // 特征点筛选和动态点检测函数
    void CollectFrameMatches();              // O(N)帧间匹配收集(替换原O(N²)嵌套循环)
    void CalculateVibrationMetrics();        // 每帧计算振动强度参数（旋转角、垂直位移）
    bool FilterFeaturePointsByDetection();  // 根据目标检测结果筛选特征点
    void DetectDynamicPoints();             // 基于相对位姿和特征点匹配结果检测动态点
    Sophus::SE3f ComputePoseWithStaticPoints();  // 使用静态点计算位姿
    void ClassifySemanticPoints(const Sophus::SE3f& staticPose); // 分类语义点（静态/动态）- 极线约束方法
    void VisualizeSemanticPoints(const std::vector<std::pair<int, int>>& semanticMatches,
                                double rotationAngle, double verticalDisplacement,
                                double dynamicThreshold); // 可视化语义点

    void RefineDynamicStatusFromSemanticRatio();  // 基于语义点动态占比精化检测框动态状态
    void UpdateCategoryStatistics();              // 统计各类别目标的数量和动态数量
    void UpdateDynamicProbabilities();            // 基于统计结果动态调整各类别动态概率
    void MarkDynamicMapPointsAsOutliers();        // 将动态MapPoint预设为外点，存入可视化容器
    void DrawDynamicSemanticPoints();             // 在Detector图像上绘制动态语义点（红色）
    
    

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    // 用当前帧检测框填充关键帧的语义概要（类别直方图），供回环/重定位语义校验使用
    void UpdateKeyFrameSemanticSummary(KeyFrame* pKF);

    // Perform preintegration from last frame
    void PreintegrateIMU();

    // Reset IMU biases and compute frame velocity
    void ResetFrameIMU();

    bool mbMapUpdated;

    // Imu preintegration from last frame
    IMU::Preintegrated *mpImuPreintegratedFromLastKF;

    // Queue of IMU measurements between frames
    std::list<IMU::Point> mlQueueImuData;

    // Vector of IMU measurements from previous to current frame (to be filled by PreintegrateIMU)
    std::vector<IMU::Point> mvImuFromLastFrame;
    std::mutex mMutexImuQueue;

    // Imu calibration parameters
    IMU::Calib *mpImuCalib;

    // Last Bias Estimation (at keyframe creation)
    IMU::Bias mLastBias;

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO;

    //Other Thread Pointers
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;
    Detector* mpDetector;
    Detector* mpDetectorRight = nullptr;   // 右目（长焦）独立检测线程，仅长短焦模式创建
    	

    //ORB
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;

    //BoW
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization (only for monocular)
    bool mbReadyToInitializate;
    bool mbSetInit;

    //Local Map
    KeyFrame* mpReferenceKF;
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    std::vector<MapPoint*> mvpLocalMapPoints;
    
    // System
    System* mpSystem;
    
    //Drawers
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;
    bool bStepByStep;

    //Atlas
    Atlas* mpAtlas;

    //Calibration matrix
    cv::Mat mK;
    Eigen::Matrix3f mK_;
    cv::Mat mDistCoef;
    float mbf;
    float mImageScale;

    float mImuFreq;
    double mImuPer;
    bool mInsertKFsLost;

    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    int mnFirstImuFrameId;
    int mnFramesToResetIMU;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;
    // 立体深度合理性门控（长短焦模式）：太近/太远的深度点不建图
    float mfMinDepth = 0.f;
    float mfMaxDepth = 1e9f;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;

    //Current matches in frame
    int mnMatchesInliers;

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;
    double mTimeStampLost;
    double time_recently_lost;

    unsigned int mnFirstFrameId;
    unsigned int mnInitialFrameId;
    unsigned int mnLastInitFrameId;

    bool mbCreatedMap;

    //Motion Model
    bool mbVelocity{false};
    Sophus::SE3f mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;
    bool mbEnable3DBoxDetection{true};   // 3D检测框开关（默认开启：语义分层静态物体建图）
    bool mbShowDynamicVis{false};         // 动态一致性可视化开关（默认关闭）

    list<MapPoint*> mlpTemporalPoints;
    
    //ORB parameters
   int nFeatures;
   float fScaleFactor;
   int nLevels;
   int fIniThFAST;
   int fMinThFAST;

    int nMapChangeIndex;

    int mnNumDataset;

    ofstream f_track_stats;

    ofstream f_track_times;

    // 图像质量评估
    std::ofstream mQualityLog;
    bool mbSaveQuality;
    ImageQualityMetrics mPendingQualityMetrics;  // 暂存 Track() 前的质量指标
    bool mPendingQualityValid;                    // 是否有待写入的质量数据
    void LogImageQuality(const cv::Mat& img, const std::vector<cv::KeyPoint>& vKeys,
                         int nLevels, long unsigned int frameId);
    void LogMatchingQuality(long unsigned int frameId);  // Track() 后调用，收集匹配指标并写入完整日志

    // 语义类别统计
    std::vector<CategoryStatistics> mvCategoryStats;

    // 动态点检测去重（同帧只检测一次）
    long unsigned int mnLastDynamicCheckFrameId = 0;

    // 动态语义点可视化容器（存储当前帧特征点索引）
    std::vector<int> mvDynamicSemanticPointIndices;

    // 动态概率序列（每20帧根据统计结果使用贝叶斯方法更新一次）
    std::vector<float> mvDynamicProbabilities;
    int PROB_UPDATE_INTERVAL = 20;
    int mnProbUpdateCounter = 0;

    // 振动检测阈值管理（新增）- 包含垂直位移
    int INITIALIZATION_PHASE = 200;  // 初始化阶段：200帧
    bool mbVibrationThresholdsInitialized;                    // 阈值是否已初始化
    double mOptimizedRotationThreshold;                       // 优化的旋转阈值
    double mOptimizedVerticalThreshold;                       // 优化的垂直位移阈值
    float mfPrevVibrationLevel = 0.0f;                        // 前一帧的振动强度（归一化，0=正常，>0=超出阈值）
    float mfCurrentVibrationLevel = 0.0f;                     // 当前帧的振动强度（匹配后可用）
    std::vector<double> mRotationHistory;                     // 旋转角历史数据
    std::vector<double> mVerticalHistory;                     // 垂直位移历史数据
    // 阈值初始化函数
    void InitializeVibrationThresholds();
    // 计算振动等级：取旋转和垂直位移超出阈值百分比的最大值
    float ComputeVibrationLevel(float rotAngle, float vertDisp);

    // 地面平面拟合（初始化后100帧执行）
    long unsigned int mMonoInitFrameId;     // 单目初始化完成时的帧ID
    bool  mbPlaneFitted;                    // 平面是否已拟合
    long unsigned int mnLastPlaneCollectKFId;  // 上次收集数据时的关键帧ID（仅关键帧更新时收集）
    long unsigned int mnPlaneFitFrameId = 0;   // FitGroundPlane 完成时的帧ID（平面约束权重平滑过渡用）
    long unsigned int mnPlaneFitNextRetry = 0; // 拟合失败后的下次重试帧ID（避免每帧都跑RANSAC）
    int mnPlaneLowLambdaStreak = 0;            // λ<0.7 的连续轮数（每30帧一轮），用于“连续确认”后才增强平面约束
    std::unordered_set<MapPoint*> mspGroundPoints;  // 收集的语义地面点指针（去重，拟合时实时读取坐标）
    std::unordered_set<MapPoint*> mspAllPoints;     // 收集的所有点指针（去重，拟合时实时读取坐标）

    // 语义车辆高度统计（尺度健康时收集，冻结后用于BA约束）
    std::vector<float> mvVehicleHeightSamples;   // 收集的车辆语义点到平面距离样本
    float mfVehicleRefHeight;                    // 冻结的车辆参考高度（中位数）
    bool  mbVehicleHeightFrozen;                 // 是否已冻结（lambda>0.85时触发）

    std::vector<Detection3D> mvDetection3Ds;        // 3D检测框：每帧Lift2DBoxesTo3D的结果
    long unsigned int mnLastLift3DFrameId = 0;      // 上次执行Lift2DBoxesTo3D的帧ID（频率控制）
    long unsigned int mnLastAlignRowsFrameId = 0;   // 上次执行AlignBoxRows的帧ID（频率控制）

    double mTime_PreIntIMU;
    double mTime_PosePred;
    double mTime_LocalMapTrack;
    double mTime_NewKF_Dec;

    GeometricCamera* mpCamera, *mpCamera2;

    int initID, lastID;

    Sophus::SE3f mTlr;

    void newParameterLoader(Settings* settings);
    
        // for point cloud viewing
    shared_ptr<PointCloudMapping> mpPointCloudMapping;

#ifdef REGISTER_LOOP
    bool Stop();

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;
#endif

public:
    cv::Mat mImRight;

    // 帧处理时间统计
    std::chrono::steady_clock::time_point mTrackingStartTime;
    std::chrono::steady_clock::time_point mFrameStartTime;
    long long mnTotalFrames;
    double mdTotalProcessingTime;
    double mdMaxFrameTime;
    double mdMinFrameTime;
    std::vector<double> mvFrameTimes;
    double mdLastFpsUpdateTime;
    int mnFramesSinceLastFpsUpdate;
    double mdCurrentFps;
    bool mbFrameStatsStarted;   // 初始化完成后才开始统计（初始化阶段耗时不计入平均）
    void UpdateFrameStatistics();
    void PrintFrameStatistics();
    void SaveFrameStatisticsToFile();

    // ===== 分阶段精准耗时统计（受 gEnableTimingStats 控制）=====
    // 单帧全流程被拆成三段：ORB特征提取 / 等待目标检测 / Track()核心跟踪
    double mdStageOrbExtract   = 0.0;  ///< 累计 ORB 特征提取耗时(ms)
    double mdStageWaitDetect   = 0.0;  ///< 累计等待目标检测耗时(ms)
    double mdStageTrack        = 0.0;  ///< 累计 Track() 核心耗时(ms)
    double mdStageSemantic     = 0.0;  ///< 累计 振动+动态点+地面+3D框 语义管线耗时(ms)
    double mdStageSemVibration = 0.0;  ///< 累计 振动指标计算 耗时(ms)
    double mdStageSemDynamic   = 0.0;  ///< 累计 动态点检测/分类/离群标记 耗时(ms)
    double mdStageSemGround    = 0.0;  ///< 累计 地面点收集+平面标记 耗时(ms)
    double mdStageSemLift      = 0.0;  ///< 累计 3D检测框提升 耗时(ms)
    double mdStageTotalPipeline= 0.0;  ///< 累计单帧全流程耗时(ms)
    double mdMaxWaitDetect     = 0.0;  ///< 单帧最大等待检测耗时(ms)
    // 当前帧的临时打点（由 GrabImage* 写入，UpdateFrameStatistics 读取）
    double mdCurOrbExtractMs   = 0.0;
    double mdCurWaitDetectMs   = 0.0;
    double mdCurSemanticMs     = 0.0;
    double mdCurSemVibrationMs = 0.0;
    double mdCurSemDynamicMs   = 0.0;
    double mdCurSemGroundMs    = 0.0;
    double mdCurSemLiftMs      = 0.0;
};

} //namespace ORB_SLAM

#endif // TRACKING_H
