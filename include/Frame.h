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


#ifndef FRAME_H
#define FRAME_H

#include<vector>

#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include "Thirdparty/Sophus/sophus/geometry.hpp"

#include "ImuTypes.h"
#include "ORBVocabulary.h"

#include "Converter.h"
#include "Settings.h"

#include <mutex>
#include <opencv2/opencv.hpp>

#include "Eigen/Core"
#include "sophus/se3.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace ORB_SLAM3
{
struct Detection;
}

namespace ORB_SLAM3
{
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class MapPoint;
class KeyFrame;
class ConstraintPoseImu;
class GeometricCamera;
class ORBextractor;

class Frame
{
public:
    Frame();

    // Copy constructor.
    Frame(const Frame &frame);

    // Constructor for stereo cameras.
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera,Frame* pPrevF = static_cast<Frame*>(NULL), const IMU::Calib &ImuCalib = IMU::Calib());

    // Constructor for RGB-D cameras.
    Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera,Frame* pPrevF = static_cast<Frame*>(NULL), const IMU::Calib &ImuCalib = IMU::Calib());

    // Constructor for Monocular cameras.
    Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, GeometricCamera* pCamera, cv::Mat &distCoef, const float &bf, const float &thDepth, Frame* pPrevF = static_cast<Frame*>(NULL), const IMU::Calib &ImuCalib = IMU::Calib());

    // Destructor
    // ~Frame();

    // Extract ORB on the image. 0 for left image and 1 for right image.
    void ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1);

    // Compute Bag of Words representation.
    void ComputeBoW();

    // Set the camera pose. (Imu pose is not modified!)
    void SetPose(const Sophus::SE3<float> &Tcw);

    // Set IMU velocity
    void SetVelocity(Eigen::Vector3f Vw);

    Eigen::Vector3f GetVelocity() const;

    // Set IMU pose and velocity (implicitly changes camera pose)
    void SetImuPoseVelocity(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const Eigen::Vector3f &Vwb);

    Eigen::Matrix<float,3,1> GetImuPosition() const;
    Eigen::Matrix<float,3,3> GetImuRotation();
    Sophus::SE3<float> GetImuPose();

    Sophus::SE3f GetRelativePoseTrl();
    Sophus::SE3f GetRelativePoseTlr();

    void SetNewBias(const IMU::Bias &b);

    // Check if a MapPoint is in the frustum of the camera
    // and fill variables of the MapPoint to be used by the tracking
    bool isInFrustum(MapPoint* pMP, float viewingCosLimit);

    bool ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v);

    // Compute the cell of a keypoint (return false if outside the grid)
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel=-1, const int maxLevel=-1, const bool bRight = false) const;

    // Search a match for each keypoint in the left image to a keypoint in the right image.
    // If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.
    void ComputeStereoMatches();

    // ------------------------------------------------------------------
    // 长短焦（Multi-focal）双目：特征点校正 + 焦距比补偿匹配
    // 移植自 MF-SLAM（undistComputeStereoMatches）
    // ------------------------------------------------------------------
    void UndistortLeftKeyPoints();
    void UndistortRightKeyPoints();
    void undistComputeStereoMatches();

    // 长短焦标定（由 Tracking::newParameterLoader 从 Settings 设置一次）
    static bool mbMultiFocal;
    static cv::Mat mLeftK, mLeftD, mLeftR, mLeftP;
    static cv::Mat mRightK, mRightD, mRightR, mRightP;
    static float mFscale;                  // 焦距比 Rfx/fx
    static float mInvFscale;
    static cv::Point2f mROILeftUp, mROIRightBottom;          // 左图（原始图像坐标）重叠视场ROI（提取/显示用）
    static cv::Point2f mROIRectLeftUp, mROIRectRightBottom;  // 校正坐标系下的重叠视场ROI（匹配门控用）
    static float mMinDepth, mMaxDepth;       // 立体深度合理性门控（假近点/假远点过滤）
    static bool mbGateAboveCamera;           // 扑翼航拍场景：特征方向不可能指向相机上方（校正坐标 v<cy 即上方）
    static float mAboveCameraGate;           // 相机上方超视界门控（米）：与 MaxDepth 解耦，固定 ~100m（见 SetMultiFocalCalib）
    static float mBelowPrincipalGate;        // 主点下方超远门控（米）：45°俯视下主点下方(v>cy)地面最远~50m，>60m 必为假匹配

    // 长焦对应区域自适应精化（前 ROIFINALIZE_FRAMES 帧扩大匹配范围，
    // 用实际匹配特征点的分布重新划分重叠区域，替代纯标定初始区域）
    // 注意：当前弃用（SetMultiFocalCalib 中 mbROIAdapting 置 false），
    // 使用标定固定区域；自适应代码保留，需要时置 true 重新启用。
    static bool mbROIAdapting;               // 自适应期间：匹配门控放宽到全图，同时累积匹配点
    static int mnROIAdaptFrameCount;         // 已累积帧数
    static std::vector<cv::Point2f> mvsROIMatchPts;   // 累积的匹配特征点（校正坐标）
    static const int ROIFINALIZE_FRAMES = 100;
    static void RefineROIFromMatches();      // 用累积匹配点分位数+边距重划 ROI（校正+原始坐标）
    static void SetMultiFocalCalib(const cv::Mat& leftK, const cv::Mat& leftD, const cv::Mat& leftR, const cv::Mat& leftP,
                                   const cv::Mat& rightK, const cv::Mat& rightD, const cv::Mat& rightR, const cv::Mat& rightP,
                                   float fScale, const cv::Point2f& roiLU, const cv::Point2f& roiRB,
                                   const int imW, const int imH, const float minDepth, const float maxDepth);
    static void ResetMultiFocalCalib();

    // 把检测框从原始左图坐标变换到校正坐标（用于语义/动态点管线与校正后特征点对齐）
    static void RectifyDetectionBoxes(std::vector<Detection>& vBoxes);
    // 把检测框从原始右图坐标变换到校正坐标（左右目检测框匹配用）
    static void RectifyDetectionBoxesRight(std::vector<Detection>& vBoxes);

    // Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
    void ComputeStereoFromRGBD(const cv::Mat &imDepth);

    // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
    bool UnprojectStereo(const int &i, Eigen::Vector3f &x3D);
    
    ConstraintPoseImu* mpcpi;

    bool imuIsPreintegrated();
    void setIntegrated();

    bool isSet() const;
    
    // Computes rotation, translation and camera center matrices from the camera pose.
    void UpdatePoseMatrices();

    // Returns the camera center.
    inline Eigen::Vector3f GetCameraCenter(){
        return mOw;
    }

    // Returns inverse of rotation
    inline Eigen::Matrix3f GetRotationInverse(){
        return mRwc;
    }

    inline Sophus::SE3<float> GetPose() const {
        //TODO: can the Frame pose be accsessed from several threads? should this be protected somehow?
        return mTcw;
    }

    inline Eigen::Matrix3f GetRwc() const {
        return mRwc;
    }

    inline Eigen::Vector3f GetOw() const {
        return mOw;
    }

    inline bool HasPose() const {
        return mbHasPose;
    }

    inline bool HasVelocity() const {
        return mbHasVelocity;
    }

private:
    //Sophus/Eigen migration
    Sophus::SE3<float> mTcw;
    Eigen::Matrix<float,3,3> mRwc;
    Eigen::Matrix<float,3,1> mOw;
    Eigen::Matrix<float,3,3> mRcw;
    Eigen::Matrix<float,3,1> mtcw;
    bool mbHasPose;

    //Rcw_ not necessary as Sophus has a method for extracting the rotation matrix: Tcw_.rotationMatrix()
    //tcw_ not necessary as Sophus has a method for extracting the translation vector: Tcw_.translation()
    //Twc_ not necessary as Sophus has a method for easily computing the inverse pose: Tcw_.inverse()

    Sophus::SE3<float> mTlr, mTrl;
    Eigen::Matrix<float,3,3> mRlr;
    Eigen::Vector3f mtlr;


    // IMU linear velocity
    Eigen::Vector3f mVw;
    bool mbHasVelocity;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Vocabulary used for relocalization.
    ORBVocabulary* mpORBvocabulary;

    // Feature extractor. The right is used only in the stereo case.
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;

    // Frame timestamp.
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat mK;
    Eigen::Matrix3f mK_;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat mDistCoef;

    // Stereo baseline multiplied by fx.
    float mbf;

    // Stereo baseline in meters.
    float mb;

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
    float mThDepth;

    // Number of KeyPoints.
    int N;

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    std::vector<cv::KeyPoint> mvKeys, mvKeysRight;
    std::vector<cv::KeyPoint> mvKeysUn;
    std::vector<cv::KeyPoint> mvKeysRightUn;
    std::vector<cv::KeyPoint> refermvKeys, refermvKeysRight;  // 校正前的原始关键点（SAD精匹配/可视化用）

    // Corresponding stereo coordinate and depth for each keypoint.
    std::vector<MapPoint*> mvpMapPoints;
    // "Monocular" keypoints have a negative value.
    std::vector<float> mvuRight;
    std::vector<float> mvuLeft;
    std::vector<float> mvDepth;
    std::vector<float> LeftIdtoRightId;   // 左特征点对应右特征点ID
    std::vector<float> RightIdToLeftId;   // 右特征点对应左特征点ID

    // Bag of Words Vector structures.
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // ORB descriptor, each row associated to a keypoint.
    cv::Mat mDescriptors, mDescriptorsRight;

    // MapPoints associated to keypoints, NULL pointer if no association.
    // Flag to identify outlier associations.
    std::vector<bool> mvbOutlier;
    int mnCloseMPs;

    // 特征点匹配结果保存 - 用于存储与上一帧的匹配关系
    std::vector<std::pair<int, int>> mvFrameMatches;  // 存储帧间特征点匹配对 (上一帧索引, 当前帧索引)
    Sophus::SE3f mRelativePose;                      // 当前帧相对于上一帧的位姿变换

    
    // 特征点分类队列 - 用于动态点检测
    std::vector<std::pair<int, int>> mvStaticMatches;     // 静态特征点匹配队列
    std::vector<std::pair<int, int>> mvSemanticMatches;    // 语义特征点匹配队列（含行人/车辆等所有检测框内目标）

    // 振动强度参数（扑翼飞行场景专用）
    float mRotationAngle;              // 帧间旋转角度（弧度），反映扑翼振动强度
    float mVerticalDisplacement;       // 帧间垂直方向位移量（像素单位），反映上下振动幅度

    // 特征点匹配结果接口函数
    void SetFrameMatches(const std::vector<std::pair<int, int>>& matches);  // 设置帧间匹配结果
    void SetRelativePose(const Sophus::SE3f& relativePose);                 // 设置相对位姿
    const std::vector<std::pair<int, int>>& GetFrameMatches() const;        // 获取帧间匹配结果
    Sophus::SE3f GetRelativePose() const;                                   // 获取相对位姿
    // 特征点分类队列接口函数
    void SetStaticMatches(const std::vector<std::pair<int, int>>& matches);    // 设置静态匹配队列
    void SetSemanticMatches(const std::vector<std::pair<int, int>>& matches);  // 设置语义匹配队列
    const std::vector<std::pair<int, int>>& GetStaticMatches() const;         // 获取静态匹配队列
    const std::vector<std::pair<int, int>>& GetSemanticMatches() const;       // 获取语义匹配队列
    void ClearSemanticData();                                                 // 清空语义数据

    // 振动强度参数接口
    void SetVibrationMetrics(float rotationAngle, float verticalDisplacement);   // 设置振动强度参数
    float GetRotationAngle() const;                                              // 获取帧间旋转角度
    float GetVerticalDisplacement() const;                                       // 获取帧间垂直位移量

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    IMU::Bias mPredBias;

    // IMU bias
    IMU::Bias mImuBias;

    // Imu calibration
    IMU::Calib mImuCalib;

    // Imu preintegration from last keyframe
    IMU::Preintegrated* mpImuPreintegrated;
    KeyFrame* mpLastKeyFrame;

    // Pointer to previous frame
    Frame* mpPrevFrame;
    IMU::Preintegrated* mpImuPreintegratedFrame;

    // Current and Next Frame id.
    static long unsigned int nNextId;
    long unsigned int mnId;

    // Reference Keyframe.
    KeyFrame* mpReferenceKF;

    // Scale pyramid info.
    int mnScaleLevels;
    float mfScaleFactor;
    float mfLogScaleFactor;
    vector<float> mvScaleFactors;
    vector<float> mvInvScaleFactors;
    vector<float> mvLevelSigma2;
    vector<float> mvInvLevelSigma2;

    // Undistorted Image Bounds (computed once).
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    static bool mbInitialComputations;

    map<long unsigned int, cv::Point2f> mmProjectPoints;
    map<long unsigned int, cv::Point2f> mmMatchedInImage;

    string mNameFile;

    int mnDataset; 

     //Detector variable and functions
//    std::vector<cv::Rect> detectedBoxes;
    std::vector<Detection> detectedBoxes_dynamic;
    std::vector<Detection> detectedBoxes;
    // 左右目检测框匹配（长短焦模式，仅非人目标）：detectedBoxes[i]（左目校正坐标）
    // 匹配到右目检测框 objectsRight[mvMatchedRightBoxIdx[i]]（右目原始坐标），-1=未匹配。
    // 深度取左目框内已立体匹配特征点的中位深度（50m 高空下框中心视差常不足 1px，不可靠）。
    std::vector<int> mvMatchedRightBoxIdx;
    std::vector<float> mvDetectionDepth;
    // std::vector<cv::Rect> detectedBoxes_half;
//    std::vector<cv::Rect> detectedBoxes_stay;
    void SetBoxes(const std::vector<Detection>& newBoxes);
    void MatchRightDetections(const std::vector<Detection>& vRightBoxes);

    // 语义分类：每帧在 SetBoxes() 时预计算，O(1) 查询
    std::vector<int> mvKeypointSemanticClass;   // 每个关键点的语义类别（-1=非语义）
    std::vector<int> mvKeypointBoxIndex;        // 关键点所属检测框索引（-1=非语义）
    void ComputeSemanticClassForKeys();         // 根据检测框分类所有关键点
    inline int GetSemanticClassAt(int idx) const {
        return (idx >= 0 && idx < (int)mvKeypointSemanticClass.size()) 
               ? mvKeypointSemanticClass[idx] : -1;
    }
    inline int GetBoxIndexAt(int idx) const {
        return (idx >= 0 && idx < (int)mvKeypointBoxIndex.size()) 
               ? mvKeypointBoxIndex[idx] : -1;
    }

#ifdef REGISTER_TIMES
    double mTimeORB_Ext;
    double mTimeStereoMatch;
#endif

private:

    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).
    void UndistortKeyPoints();

    // Computes image bounds for the undistorted image (called in the constructor).
    void ComputeImageBounds(const cv::Mat &imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void AssignFeaturesToGrid();

    bool mbIsSet;

    bool mbImuPreintegrated;

    std::mutex *mpMutexImu;

public:
    GeometricCamera* mpCamera, *mpCamera2;

    //Number of KeyPoints extracted in the left and right images
    int Nleft, Nright;
    //Number of Non Lapping Keypoints
    int monoLeft, monoRight;

    //For stereo matching
    std::vector<int> mvLeftToRightMatch, mvRightToLeftMatch;

    //For stereo fisheye matching
    static cv::BFMatcher BFmatcher;

    //Triangulated stereo observations using as reference the left camera. These are
    //computed during ComputeStereoFishEyeMatches
    std::vector<Eigen::Vector3f> mvStereo3Dpoints;

    //Grid for the right image
    std::vector<std::size_t> mGridRight[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, GeometricCamera* pCamera2, Sophus::SE3f& Tlr,Frame* pPrevF = static_cast<Frame*>(NULL), const IMU::Calib &ImuCalib = IMU::Calib());

    //Stereo fisheye
    void ComputeStereoFishEyeMatches();

    bool isInFrustumChecks(MapPoint* pMP, float viewingCosLimit, bool bRight = false);

    Eigen::Vector3f UnprojectStereoFishEye(const int &i);

    cv::Mat imgLeft, imgRight;

};

}// namespace ORB_SLAM

#endif // FRAME_H
