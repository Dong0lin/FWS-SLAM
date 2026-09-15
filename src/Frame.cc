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

#include "Frame.h"

#include "G2oTypes.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "ORBextractor.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include "GeometricCamera.h"
#include "Detector.h" 
#include "common.h"

#include <thread>
#include <include/CameraModels/Pinhole.h>
#include <include/CameraModels/KannalaBrandt8.h>
#include <algorithm>
#include <cmath>
#include <limits>
namespace ORB_SLAM3
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

// 长短焦（Multi-focal）双目静态标定
bool Frame::mbMultiFocal = false;
cv::Mat Frame::mLeftK, Frame::mLeftD, Frame::mLeftR, Frame::mLeftP;
cv::Mat Frame::mRightK, Frame::mRightD, Frame::mRightR, Frame::mRightP;
float Frame::mFscale = 1.0f;
float Frame::mInvFscale = 1.0f;
cv::Point2f Frame::mROILeftUp, Frame::mROIRightBottom;
cv::Point2f Frame::mROIRectLeftUp, Frame::mROIRectRightBottom;
float Frame::mMinDepth = 0.f;
float Frame::mMaxDepth = 1e9f;
bool Frame::mbGateAboveCamera = false;
float Frame::mAboveCameraGate = 1e9f;    // 默认关闭（单目/普通双目不受影响）
float Frame::mBelowPrincipalGate = 1e9f; // 默认关闭（单目/普通双目不受影响）
bool Frame::mbROIAdapting = false;
int Frame::mnROIAdaptFrameCount = 0;
std::vector<cv::Point2f> Frame::mvsROIMatchPts;

//For stereo fisheye matching
cv::BFMatcher Frame::BFmatcher = cv::BFMatcher(cv::NORM_HAMMING);

Frame::Frame(): mpcpi(NULL), mpImuPreintegrated(NULL), mpPrevFrame(NULL), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mbHasPose(false), mbHasVelocity(false)
{
#ifdef REGISTER_TIMES
    mTimeStereoMatch = 0;
    mTimeORB_Ext = 0;
#endif
    
    // 初始化特征点匹配相关成员变量
    mvFrameMatches.clear();           // 清空匹配结果
    mRelativePose = Sophus::SE3f();   // 初始化相对位姿为单位矩阵
    
    // 初始化特征点分类队列成员变量
    mvStaticMatches.clear();          // 清空静态匹配队列
    mvSemanticMatches.clear();        // 清空语义匹配队列
    mRotationAngle = 0.0f;             // 初始化旋转角度
    mVerticalDisplacement = 0.0f;      // 初始化垂直位移
}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpcpi(frame.mpcpi),mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mK_(Converter::toMatrix3f(frame.mK)), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvKeysRightUn(frame.mvKeysRightUn), mvuRight(frame.mvuRight),
     refermvKeys(frame.refermvKeys), refermvKeysRight(frame.refermvKeysRight),
     mvuLeft(frame.mvuLeft), LeftIdtoRightId(frame.LeftIdtoRightId), RightIdToLeftId(frame.RightIdToLeftId),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mImuCalib(frame.mImuCalib), mnCloseMPs(frame.mnCloseMPs),
     mpImuPreintegrated(frame.mpImuPreintegrated), mpImuPreintegratedFrame(frame.mpImuPreintegratedFrame), mImuBias(frame.mImuBias),
     mnId(frame.mnId), mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors), mNameFile(frame.mNameFile), mnDataset(frame.mnDataset),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2), mpPrevFrame(frame.mpPrevFrame), mpLastKeyFrame(frame.mpLastKeyFrame),
     mbIsSet(frame.mbIsSet), mbImuPreintegrated(frame.mbImuPreintegrated), mpMutexImu(frame.mpMutexImu),
     mpCamera(frame.mpCamera), mpCamera2(frame.mpCamera2), Nleft(frame.Nleft), Nright(frame.Nright),
     monoLeft(frame.monoLeft), monoRight(frame.monoRight), mvLeftToRightMatch(frame.mvLeftToRightMatch),
     mvRightToLeftMatch(frame.mvRightToLeftMatch), mvStereo3Dpoints(frame.mvStereo3Dpoints),
     mTlr(frame.mTlr), mRlr(frame.mRlr), mtlr(frame.mtlr), mTrl(frame.mTrl),
     mTcw(frame.mTcw), mbHasPose(false), mbHasVelocity(false),
     mvFrameMatches(frame.mvFrameMatches), mRelativePose(frame.mRelativePose),
     mvStaticMatches(frame.mvStaticMatches), mvSemanticMatches(frame.mvSemanticMatches),
     mRotationAngle(frame.mRotationAngle), mVerticalDisplacement(frame.mVerticalDisplacement)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++){
            mGrid[i][j]=frame.mGrid[i][j];
            if(frame.Nleft > 0){
                mGridRight[i][j] = frame.mGridRight[i][j];
            }
        }

    if(frame.mbHasPose)
        SetPose(frame.GetPose());

    if(frame.HasVelocity())
    {
        SetVelocity(frame.GetVelocity());
    }

    mmProjectPoints = frame.mmProjectPoints;
    mmMatchedInImage = frame.mmMatchedInImage;

#ifdef REGISTER_TIMES
    mTimeStereoMatch = frame.mTimeStereoMatch;
    mTimeORB_Ext = frame.mTimeORB_Ext;
#endif
}

//双目的初始化
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL), mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
     mpCamera(pCamera) ,mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft,0,0);
    thread threadRight(&Frame::ExtractORB,this,1,imRight,0,0);
    threadLeft.join();
    threadRight.join();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    
    N = mvKeys.size();
    if(mvKeys.empty())
        return;

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
    if(mbMultiFocal)
    {
        // 长短焦双目：先校正特征点到公共坐标系，再做焦距比补偿匹配
        UndistortLeftKeyPoints();
        UndistortRightKeyPoints();
        undistComputeStereoMatches();
    }
    else
    {
        UndistortKeyPoints();
        ComputeStereoMatches();
    }
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);
    mmProjectPoints.clear();
    mmMatchedInImage.clear();
    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    if(pPrevF)
    {
        if(pPrevF->HasVelocity())
            SetVelocity(pPrevF->GetVelocity());
    }
    else
    {
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();
}
//RGB-D初始化
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera,Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
     mpCamera(pCamera),mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    ExtractORB(0,imGray,0,0);
#ifdef REGISTER_TIMES
std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif
    
    
    
    N = mvKeys.size();

    if(mvKeys.empty())
        return;


    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    if(pPrevF){
        if(pPrevF->HasVelocity())
            SetVelocity(pPrevF->GetVelocity());
    }
    else{
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();
}

/**
 * Frame类构造函数 - 单目图像帧初始化
 * 
 * @param imGray 输入灰度图像
 * @param timeStamp 时间戳
 * @param extractor ORB特征提取器指针
 * @param voc ORB词典指针
 * @param pCamera 相机模型指针
 * @param distCoef 畸变系数矩阵
 * @param bf 基线长度（双目相机参数，单目时为0）
 * @param thDepth 深度阈值
 * @param pPrevF 前一帧指针（用于IMU预积分）
 * @param ImuCalib IMU标定参数
 */
Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, GeometricCamera* pCamera, cv::Mat &distCoef, const float &bf, const float &thDepth, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(static_cast<Pinhole*>(pCamera)->toK()), mK_(static_cast<Pinhole*>(pCamera)->toK_()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL),mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mpCamera(pCamera),
     mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // std::cout << "[DEBUG] Frame构造函数: 开始创建帧" << std::endl;
    // Frame ID - 分配唯一的帧ID
    mnId=nNextId++;

    // Scale Level Info - 获取ORB特征提取器的尺度信息
    mnScaleLevels = mpORBextractorLeft->GetLevels();        // 金字塔层数
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();   // 尺度因子
    mfLogScaleFactor = log(mfScaleFactor);                  // 尺度因子的对数
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors(); // 各层尺度因子
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors(); // 各层尺度因子的倒数
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();       // 各层尺度方差
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares(); // 各层尺度方差倒数

    // std::cout << "[DEBUG] Frame构造函数: 尺度信息已初始化" << std::endl;

    // ORB extraction - 提取ORB特征点
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    ExtractORB(0,imGray,0,1000); // 提取左目图像特征点
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();
    // 记录特征提取耗时（如果启用了时间记录）
    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif
    
    // 检查特征点数量
    N = mvKeys.size(); // 获取特征点数量
    if(mvKeys.empty()) // 如果没有提取到特征点
        return;        // 直接返回

    // std::cout << "[DEBUG] Frame构造函数: 特征点数量为 " << N << " 构造函数结束 " <<std::endl;
    UndistortKeyPoints();

    mvuRight = vector<float>(N,-1);  // 右目图像对应点横坐标
    mvDepth = vector<float>(N,-1);   // 特征点深度
    mnCloseMPs = 0;                  // 近距离地图点数量

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N,false);

    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray); // 计算图像边界

        // 计算网格单元宽度和高度倒数
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        // 提取相机内参
        fx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,0); // 焦距x
        fy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,1); // 焦距y
        cx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,2); // 主点x
        cy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,2); // 主点y
        invfx = 1.0f/fx; // 焦距x倒数
        invfy = 1.0f/fy; // 焦距y倒数

        mbInitialComputations=false; // 标记初始计算已完成
    }

    mb = mbf/fx;

    Nleft = -1;  // 左目特征点数量
    Nright = -1; // 右目特征点数量
    mvLeftToRightMatch = vector<int>(0);  // 左到右匹配
    mvRightToLeftMatch = vector<int>(0);  // 右到左匹配
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0); // 立体3D点
    monoLeft = -1;  // 左目单目标志
    monoRight = -1; // 右目单目标志

    AssignFeaturesToGrid();

    if(pPrevF) // 如果存在前一帧
    {
        if(pPrevF->HasVelocity()) // 如果前一帧有速度信息
        {
            SetVelocity(pPrevF->GetVelocity()); // 设置当前帧速度
        }
    }
    else // 如果是首帧
    {
        mVw.setZero(); // 速度置零
    }

    mpMutexImu = new std::mutex();

}
void Frame::AssignFeaturesToGrid()
{
    // Fill matrix with points
    const int nCells = FRAME_GRID_COLS*FRAME_GRID_ROWS;

    int nReserve = 0.5f*N/(nCells);

    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++){
            mGrid[i][j].reserve(nReserve);
            if(Nleft != -1){
                mGridRight[i][j].reserve(nReserve);
            }
        }

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = (Nleft == -1) ? mvKeysUn[i]
                                                 : (i < Nleft) ? mvKeys[i]
                                                                 : mvKeysRight[i - Nleft];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY)){
            if(Nleft == -1 || i < Nleft)
                mGrid[nGridPosX][nGridPosY].push_back(i);
            else
                mGridRight[nGridPosX][nGridPosY].push_back(i - Nleft);
        }
    }
}
void Frame::ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1)
{
    vector<int> vLapping = {x0,x1};
    if(flag==0)
        monoLeft = (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors,vLapping,0);
    else
        monoRight = (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight,vLapping,1);
}

// }

bool Frame::isSet() const {
    return mbIsSet;
}

void Frame::SetPose(const Sophus::SE3<float> &Tcw) {
    mTcw = Tcw;

    UpdatePoseMatrices();
    mbIsSet = true;
    mbHasPose = true;
}

void Frame::SetNewBias(const IMU::Bias &b)
{
    mImuBias = b;
    if(mpImuPreintegrated)
        mpImuPreintegrated->SetNewBias(b);
}

void Frame::SetVelocity(Eigen::Vector3f Vwb)
{
    mVw = Vwb;
    mbHasVelocity = true;
}

Eigen::Vector3f Frame::GetVelocity() const
{
    return mVw;
}

void Frame::SetImuPoseVelocity(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const Eigen::Vector3f &Vwb)
{
    mVw = Vwb;
    mbHasVelocity = true;

    Sophus::SE3f Twb(Rwb, twb);
    Sophus::SE3f Tbw = Twb.inverse();

    mTcw = mImuCalib.mTcb * Tbw;

    UpdatePoseMatrices();
    mbIsSet = true;
    mbHasPose = true;
}

void Frame::UpdatePoseMatrices()
{
    Sophus::SE3<float> Twc = mTcw.inverse();
    mRwc = Twc.rotationMatrix();
    mOw = Twc.translation();
    mRcw = mTcw.rotationMatrix();
    mtcw = mTcw.translation();
}

Eigen::Matrix<float,3,1> Frame::GetImuPosition() const {
    return mRwc * mImuCalib.mTcb.translation() + mOw;
}

Eigen::Matrix<float,3,3> Frame::GetImuRotation() {
    return mRwc * mImuCalib.mTcb.rotationMatrix();
}

Sophus::SE3<float> Frame::GetImuPose() {
    return mTcw.inverse() * mImuCalib.mTcb;
}

Sophus::SE3f Frame::GetRelativePoseTrl()
{
    return mTrl;
}

Sophus::SE3f Frame::GetRelativePoseTlr()
{
    return mTlr;
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    if(Nleft == -1){
        pMP->mbTrackInView = false;
        pMP->mTrackProjX = -1;
        pMP->mTrackProjY = -1;

        // 3D in absolute coordinates
        Eigen::Matrix<float,3,1> P = pMP->GetWorldPos();

        // 3D in camera coordinates
        const Eigen::Matrix<float,3,1> Pc = mRcw * P + mtcw;
        const float Pc_dist = Pc.norm();

        // Check positive depth
        const float &PcZ = Pc(2);
        const float invz = 1.0f/PcZ;
        if(PcZ<0.0f)
            return false;

        const Eigen::Vector2f uv = mpCamera->project(Pc);

        if(uv(0)<mnMinX || uv(0)>mnMaxX)
            return false;
        if(uv(1)<mnMinY || uv(1)>mnMaxY)
            return false;

        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjY = uv(1);

        // Check distance is in the scale invariance region of the MapPoint
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const Eigen::Vector3f PO = P - mOw;
        const float dist = PO.norm();

        if(dist<minDistance || dist>maxDistance)
            return false;

        // Check viewing angle
        Eigen::Vector3f Pn = pMP->GetNormal();

        const float viewCos = PO.dot(Pn)/dist;

        if(viewCos<viewingCosLimit)
            return false;

        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist,this);

        // Data used by the tracking
        pMP->mbTrackInView = true;
        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjXR = uv(0) - mbf*invz;

        pMP->mTrackDepth = Pc_dist;

        pMP->mTrackProjY = uv(1);
        pMP->mnTrackScaleLevel= nPredictedLevel;
        pMP->mTrackViewCos = viewCos;

        return true;
    }
    else{
        pMP->mbTrackInView = false;
        pMP->mbTrackInViewR = false;
        pMP -> mnTrackScaleLevel = -1;
        pMP -> mnTrackScaleLevelR = -1;

        pMP->mbTrackInView = isInFrustumChecks(pMP,viewingCosLimit);
        pMP->mbTrackInViewR = isInFrustumChecks(pMP,viewingCosLimit,true);

        return pMP->mbTrackInView || pMP->mbTrackInViewR;
    }
}

bool Frame::ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    // 3D in camera coordinates
    const Eigen::Vector3f Pc = mRcw * P + mtcw;
    const float &PcX = Pc(0);
    const float &PcY= Pc(1);
    const float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
    {
        cout << "Negative depth: " << PcZ << endl;
        return false;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    u=fx*PcX*invz+cx;
    v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    float u_distort, v_distort;

    float x = (u - cx) * invfx;
    float y = (v - cy) * invfy;
    float r2 = x * x + y * y;
    float k1 = mDistCoef.at<float>(0);
    float k2 = mDistCoef.at<float>(1);
    float p1 = mDistCoef.at<float>(2);
    float p2 = mDistCoef.at<float>(3);
    float k3 = 0;
    if(mDistCoef.total() == 5)
    {
        k3 = mDistCoef.at<float>(4);
    }

    // Radial distorsion
    float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    u_distort = x_distort * fx + cx;
    v_distort = y_distort * fy + cy;
    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel, const bool bRight) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    float factorX = r;
    float factorY = r;

    const int nMinCellX = max(0,(int)floor((x-mnMinX-factorX)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
    {
        return vIndices;
    }

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+factorX)*mfGridElementWidthInv));
    if(nMaxCellX<0)
    {
        return vIndices;
    }

    const int nMinCellY = max(0,(int)floor((y-mnMinY-factorY)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
    {
        return vIndices;
    }

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+factorY)*mfGridElementHeightInv));
    if(nMaxCellY<0)
    {
        return vIndices;
    }

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = (Nleft == -1) ? mvKeysUn[vCell[j]]
                                                         : (!bRight) ? mvKeys[vCell[j]]
                                                                     : mvKeysRight[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<factorX && fabs(disty)<factorY)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}
void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}
// 以下为修改点
void Frame::SetBoxes(const std::vector<Detection>& newBoxes)
{
	detectedBoxes = newBoxes;
    ComputeSemanticClassForKeys();
}

void Frame::MatchRightDetections(const std::vector<Detection>& vRightBoxes)
{
    const size_t nL = detectedBoxes.size();
    mvMatchedRightBoxIdx.assign(nL, -1);
    mvDetectionDepth.assign(nL, -1.f);

    // 仅长短焦模式启用；左目检测框需为校正坐标（GrabImageStereo 已校正）
    if(!mbMultiFocal || nL == 0 || vRightBoxes.empty())
        return;

    // 右目框校正到公共坐标系（用副本，不改动原始 objectsRight，绘制仍用原始坐标）
    std::vector<Detection> vRight = vRightBoxes;
    RectifyDetectionBoxesRight(vRight);
    const size_t nR = vRight.size();

    // 候选对：仅非人目标（class_id>=2，排除 pedestrian/people），
    // 类别一致，且校正坐标系下 IoU>0.3（同一目标在公共坐标系下几乎重合）
    struct DetCand { float fIoU; int iL; int iR; };
    std::vector<DetCand> vCand;
    vCand.reserve(nL);
    for(size_t i = 0; i < nL; i++)
    {
        const Detection& dL = detectedBoxes[i];
        if(dL.class_id < 2)
            continue;
        const cv::Rect& rL = dL.bbox;
        if(rL.width <= 0 || rL.height <= 0)
            continue;

        for(size_t j = 0; j < nR; j++)
        {
            const Detection& dR = vRight[j];
            if(dR.class_id != dL.class_id)
                continue;
            const cv::Rect& rR = dR.bbox;
            if(rR.width <= 0 || rR.height <= 0)
                continue;

            const cv::Rect inter = rL & rR;
            if(inter.width <= 0 || inter.height <= 0)
                continue;
            const float fIoU = (float)inter.area() /
                               (float)(rL.area() + rR.area() - inter.area());
            if(fIoU > 0.3f)
                vCand.push_back({fIoU, (int)i, (int)j});
        }
    }

    // 按 IoU 降序贪心一对一分配（每个左框最多匹配一个右框）
    std::sort(vCand.begin(), vCand.end(),
              [](const DetCand& a, const DetCand& b){ return a.fIoU > b.fIoU; });
    std::vector<char> vbUsedR(nR, 0);
    for(const DetCand& c : vCand)
    {
        if(mvMatchedRightBoxIdx[c.iL] >= 0 || vbUsedR[c.iR])
            continue;

        mvMatchedRightBoxIdx[c.iL] = c.iR;
        vbUsedR[c.iR] = 1;

        // 深度：取左目框内已立体匹配特征点的中位深度。
        // 50m 高空下框中心视差仅 0.2~1.4px，直接用视差算深度误差极大，
        // 框内特征点深度（mvDepth，已通过 Min/MaxDepth 门控）可靠得多。
        float depth = -1.f;
        const cv::Rect& rL = detectedBoxes[c.iL].bbox;
        std::vector<float> vD;
        vD.reserve(32);
        for(int k = 0; k < N; k++)
        {
            if(mvDepth[k] <= 0.f)
                continue;
            if(rL.contains(mvKeys[k].pt))
                vD.push_back(mvDepth[k]);
        }
        if(!vD.empty())
        {
            std::sort(vD.begin(), vD.end());
            depth = vD[vD.size() / 2];
        }
        else
        {
            // 框内无立体特征点时退回框中心视差（要求视差>=1px且深度在合理范围）
            const cv::Rect& rR = vRight[c.iR].bbox;
            const float disp = (rL.x + rL.width * 0.5f) - (rR.x + rR.width * 0.5f);
            if(disp >= 1.f && mbf > 0.f)
            {
                const float z = mbf / disp;
                if(z >= mMinDepth && z <= mMaxDepth)
                    depth = z;
            }
        }
        mvDetectionDepth[c.iL] = depth;
    }
}

// 根据检测框为每个关键点分配语义类别（在匹配前预计算，O(1)查询）
void Frame::ComputeSemanticClassForKeys()
{
    mvKeypointSemanticClass.assign(N, -1);
    mvKeypointBoxIndex.assign(N, -1);

    if(detectedBoxes.empty() || N == 0)
        return;

    for(int i = 0; i < N; i++)
    {
        const float kx = mvKeys[i].pt.x;
        const float ky = mvKeys[i].pt.y;

        for(size_t b = 0; b < detectedBoxes.size(); b++)
        {
            const Detection& det = detectedBoxes[b];
            int margin = 3;
            if(det.class_id >= 0 && det.class_id < (int)SEMANTIC_BOX_MARGINS.size())
                margin = SEMANTIC_BOX_MARGINS[det.class_id];

            const float x1 = det.bbox.x - margin;
            const float y1 = det.bbox.y - margin;
            const float x2 = det.bbox.x + det.bbox.width + margin;
            const float y2 = det.bbox.y + det.bbox.height + margin;

            if(kx >= x1 && kx <= x2 && ky >= y1 && ky <= y2)
            {
                mvKeypointSemanticClass[i] = det.class_id;
                mvKeypointBoxIndex[i] = (int)b;
                break;
            }
        }
    }
}
void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);

    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat, static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);
    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }

}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        // Undistort corners
        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));
    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}

// --------------------------------------------------------------------------
// 长短焦（Multi-focal）双目实现，移植自 MF-SLAM
// --------------------------------------------------------------------------
void Frame::SetMultiFocalCalib(const cv::Mat& leftK, const cv::Mat& leftD, const cv::Mat& leftR, const cv::Mat& leftP,
                               const cv::Mat& rightK, const cv::Mat& rightD, const cv::Mat& rightR, const cv::Mat& rightP,
                               float fScale, const cv::Point2f& roiLU, const cv::Point2f& roiRB,
                               const int imW, const int imH, const float minDepth, const float maxDepth)
{
    mLeftK = leftK.clone(); mLeftD = leftD.clone(); mLeftR = leftR.clone(); mLeftP = leftP.clone();
    mRightK = rightK.clone(); mRightD = rightD.clone(); mRightR = rightR.clone(); mRightP = rightP.clone();
    mFscale = fScale;
    mInvFscale = (fScale > 1e-6f) ? 1.0f / fScale : 1.0f;
    mROILeftUp = roiLU;
    mROIRightBottom = roiRB;
    mbMultiFocal = true;

    // 同步给 ORB 提取器：ROI 内加密提取、跳过无法匹配的低层金字塔
    ORBextractor::SetFocalScale(fScale);
    ORBextractor::SetROI(roiLU, roiRB);

    // 匹配门控ROI：把右图（长焦）四个角映射到校正坐标系，取包围盒。
    // 左图中落在该区域内的特征点才可能与右图匹配（比近似公式更准）
    cv::Mat pts(4, 2, CV_32F);
    pts.at<float>(0, 0) = 0.f;              pts.at<float>(0, 1) = 0.f;
    pts.at<float>(1, 0) = (float)imW;       pts.at<float>(1, 1) = 0.f;
    pts.at<float>(2, 0) = 0.f;              pts.at<float>(2, 1) = (float)imH;
    pts.at<float>(3, 0) = (float)imW;       pts.at<float>(3, 1) = (float)imH;
    pts = pts.reshape(2);
    cv::undistortPoints(pts, pts, mRightK, mRightD, mRightR, mRightP);
    pts = pts.reshape(1);
    float minx = std::numeric_limits<float>::max(), miny = std::numeric_limits<float>::max();
    float maxx = -std::numeric_limits<float>::max(), maxy = -std::numeric_limits<float>::max();
    for(int j = 0; j < 4; j++)
    {
        minx = std::min(minx, pts.at<float>(j, 0));
        miny = std::min(miny, pts.at<float>(j, 1));
        maxx = std::max(maxx, pts.at<float>(j, 0));
        maxy = std::max(maxy, pts.at<float>(j, 1));
    }
    mROIRectLeftUp = cv::Point2f(minx, miny);
    mROIRectRightBottom = cv::Point2f(maxx, maxy);
    ORBextractor::SetRectifiedROI(mROIRectLeftUp, mROIRectRightBottom);

    mMinDepth = minDepth;
    mMaxDepth = maxDepth;
    mbGateAboveCamera = true;   // 扑翼航拍：相机恒朝下，特征方向不可能指向相机上方

    // 相机上方超视界门控（与 MaxDepth 解耦）：
    // 45° 俯视 50m 航拍、校正后 fy≈1110/cy≈250：图像顶边对应俯角约 32°，
    // 可见地面距离约 20~79m；主点上方(v<cy)的真实地面点范围为 50~79m。
    // 这里固定 ~100m 而不是 0.5×MaxDepth——收紧 MaxDepth（如 120~150）时
    // 不会误杀真实远地面点；主点上方且 >100m 的基本都是近零视差假匹配。
    mAboveCameraGate = 100.0f;

    // 主点下方超远门控（对称补充）：
    // 主点下方(v>cy)的射线俯角 >45°，真实地面距离 <~50m（h=50m 时）；
    // z>60m 且 v>cy 的点物理上不可能，是近零视差假匹配。
    // 这类点被反投影到相机“下方”，会把地面平面拟合整体拉向相机、
    // 法向量带歪（实测单帧拟合高度 5.6m vs 真实 50m），必须剔除。
    // 60m 阈值给振动/俯角变化留了 ~10m 余量（俯角低至 ~40° 仍安全）。
    mBelowPrincipalGate = 60.0f;

    // 长焦对应区域自适应：当前弃用。实测首 100 帧左右目匹配特征点过少，
    // 收敛出的区域偏差较大；先用标定推导的固定区域（mROIRect*/mROI*）。
    // 代码保留，后续需要时把 mbROIAdapting 改为 true 即可重新启用。
    mbROIAdapting = false;
    mnROIAdaptFrameCount = 0;
    mvsROIMatchPts.clear();
    cout << "[Frame] 长焦对应区域自适应已禁用，使用标定固定区域 ROI=["
         << mROILeftUp << " -> " << mROIRightBottom << "]" << endl;
}

void Frame::ResetMultiFocalCalib()
{
    mbMultiFocal = false;
    mLeftK.release(); mLeftD.release(); mLeftR.release(); mLeftP.release();
    mRightK.release(); mRightD.release(); mRightR.release(); mRightP.release();
    mFscale = 1.0f;
    mInvFscale = 1.0f;
    mROILeftUp = cv::Point2f(0.f, 0.f);
    mROIRightBottom = cv::Point2f(0.f, 0.f);
    mROIRectLeftUp = cv::Point2f(0.f, 0.f);
    mROIRectRightBottom = cv::Point2f(0.f, 0.f);
    mMinDepth = 0.f;
    mMaxDepth = 1e9f;
    mbGateAboveCamera = false;
    mAboveCameraGate = 1e9f;
    mBelowPrincipalGate = 1e9f;
    mbROIAdapting = false;
    mnROIAdaptFrameCount = 0;
    mvsROIMatchPts.clear();
    ORBextractor::SetFocalScale(1.0f);
    ORBextractor::SetROI(cv::Point2f(0.f, 0.f), cv::Point2f(0.f, 0.f));
    ORBextractor::SetRectifiedROI(cv::Point2f(0.f, 0.f), cv::Point2f(0.f, 0.f));
}

void Frame::RectifyDetectionBoxes(std::vector<Detection>& vBoxes)
{
    if(vBoxes.empty() || mLeftK.empty() || mLeftP.empty())
        return;

    for(size_t i = 0; i < vBoxes.size(); i++)
    {
        cv::Rect& r = vBoxes[i].bbox;
        if(r.width <= 0 || r.height <= 0)
            continue;

        cv::Mat pts(4, 2, CV_32F);
        pts.at<float>(0, 0) = (float)r.x;       pts.at<float>(0, 1) = (float)r.y;
        pts.at<float>(1, 0) = (float)(r.x + r.width);  pts.at<float>(1, 1) = (float)r.y;
        pts.at<float>(2, 0) = (float)r.x;       pts.at<float>(2, 1) = (float)(r.y + r.height);
        pts.at<float>(3, 0) = (float)(r.x + r.width);  pts.at<float>(3, 1) = (float)(r.y + r.height);

        pts = pts.reshape(2);
        cv::undistortPoints(pts, pts, mLeftK, mLeftD, mLeftR, mLeftP);
        pts = pts.reshape(1);

        float minx = std::numeric_limits<float>::max(), miny = std::numeric_limits<float>::max();
        float maxx = -std::numeric_limits<float>::max(), maxy = -std::numeric_limits<float>::max();
        for(int j = 0; j < 4; j++)
        {
            const float px = pts.at<float>(j, 0);
            const float py = pts.at<float>(j, 1);
            minx = std::min(minx, px); miny = std::min(miny, py);
            maxx = std::max(maxx, px); maxy = std::max(maxy, py);
        }

        minx = std::max(0.f, minx);
        miny = std::max(0.f, miny);
        r.x = (int)minx;
        r.y = (int)miny;
        r.width = (int)(maxx - minx + 0.5f);
        r.height = (int)(maxy - miny + 0.5f);
    }
}

void Frame::RectifyDetectionBoxesRight(std::vector<Detection>& vBoxes)
{
    if(vBoxes.empty() || mRightK.empty() || mRightP.empty())
        return;

    for(size_t i = 0; i < vBoxes.size(); i++)
    {
        cv::Rect& r = vBoxes[i].bbox;
        if(r.width <= 0 || r.height <= 0)
            continue;

        cv::Mat pts(4, 2, CV_32F);
        pts.at<float>(0, 0) = (float)r.x;       pts.at<float>(0, 1) = (float)r.y;
        pts.at<float>(1, 0) = (float)(r.x + r.width);  pts.at<float>(1, 1) = (float)r.y;
        pts.at<float>(2, 0) = (float)r.x;       pts.at<float>(2, 1) = (float)(r.y + r.height);
        pts.at<float>(3, 0) = (float)(r.x + r.width);  pts.at<float>(3, 1) = (float)(r.y + r.height);

        pts = pts.reshape(2);
        cv::undistortPoints(pts, pts, mRightK, mRightD, mRightR, mRightP);
        pts = pts.reshape(1);

        float minx = std::numeric_limits<float>::max(), miny = std::numeric_limits<float>::max();
        float maxx = -std::numeric_limits<float>::max(), maxy = -std::numeric_limits<float>::max();
        for(int j = 0; j < 4; j++)
        {
            const float px = pts.at<float>(j, 0);
            const float py = pts.at<float>(j, 1);
            minx = std::min(minx, px); miny = std::min(miny, py);
            maxx = std::max(maxx, px); maxy = std::max(maxy, py);
        }

        minx = std::max(0.f, minx);
        miny = std::max(0.f, miny);
        r.x = (int)minx;
        r.y = (int)miny;
        r.width = (int)(maxx - minx + 0.5f);
        r.height = (int)(maxy - miny + 0.5f);
    }
}

void Frame::UndistortLeftKeyPoints()
{
    const float widthF = (float)mpORBextractorLeft->mvImagePyramid[0].cols;
    const float heightF = (float)mpORBextractorLeft->mvImagePyramid[0].rows;

    if(mLeftD.at<double>(0) == 0.0 && fabs(mFscale - 1.0f) < 1e-6f)
    {
        mvKeysUn = mvKeys;
        refermvKeys = mvKeys;
        return;
    }

    cv::Mat mat(N, 2, CV_32F);
    for(int i = 0; i < N; i++)
    {
        mat.at<float>(i, 0) = mvKeys[i].pt.x;
        mat.at<float>(i, 1) = mvKeys[i].pt.y;
    }

    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, mLeftK, mLeftD, mLeftR, mLeftP);
    mat = mat.reshape(1);

    mvKeysUn.resize(N);
    refermvKeys.clear();
    refermvKeys.resize(N);
    refermvKeys = mvKeys;   // 原始（校正前）坐标，供 SAD 精匹配使用

    for(int i = 0; i < N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        const float px = mat.at<float>(i, 0);
        const float py = mat.at<float>(i, 1);
        if(px < 0.f || px > widthF - 1.f || py < 0.f || py > heightF - 1.f)
        {
            // 校正后越界点保留原坐标（行带/ROI 门控会将其排除）
            mvKeysUn[i] = kp;
            continue;
        }
        kp.pt.x = px;
        kp.pt.y = py;
        mvKeys[i] = kp;
        mvKeysUn[i] = kp;
    }
}

void Frame::UndistortRightKeyPoints()
{
    const float widthF = (float)mpORBextractorLeft->mvImagePyramid[0].cols;
    const float heightF = (float)mpORBextractorLeft->mvImagePyramid[0].rows;

    if(mRightD.at<double>(0) == 0.0 && fabs(mFscale - 1.0f) < 1e-6f)
    {
        mvKeysRightUn = mvKeysRight;
        refermvKeysRight = mvKeysRight;
        return;
    }

    const int Nr = (int)mvKeysRight.size();
    if(Nr == 0)
        return;

    cv::Mat mat(Nr, 2, CV_32F);
    for(int i = 0; i < Nr; i++)
    {
        mat.at<float>(i, 0) = mvKeysRight[i].pt.x;
        mat.at<float>(i, 1) = mvKeysRight[i].pt.y;
    }

    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, mRightK, mRightD, mRightR, mRightP);
    mat = mat.reshape(1);

    mvKeysRightUn.resize(Nr);
    refermvKeysRight.clear();
    refermvKeysRight.resize(Nr);
    refermvKeysRight = mvKeysRight;

    for(int i = 0; i < Nr; i++)
    {
        cv::KeyPoint kp = mvKeysRight[i];
        const float px = mat.at<float>(i, 0);
        const float py = mat.at<float>(i, 1);
        if(px < 0.f || px > widthF - 1.f || py < 0.f || py > heightF - 1.f)
        {
            mvKeysRightUn[i] = kp;
            continue;
        }
        kp.pt.x = px;
        kp.pt.y = py;
        mvKeysRight[i] = kp;
        mvKeysRightUn[i] = kp;
    }
}

void Frame::undistComputeStereoMatches()
{
    mvuRight = vector<float>(N, -1.0f);
    mvuLeft = vector<float>(N, -1.0f);
    mvDepth = vector<float>(N, -1.0f);
    LeftIdtoRightId = vector<float>(N, -1.0f);
    RightIdToLeftId = vector<float>(mvKeysRight.size(), -1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH + ORBmatcher::TH_LOW) / 2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;
    const int nCols = mpORBextractorLeft->mvImagePyramid[0].cols;
    const float widthF = (float)nCols;
    const float heightF = (float)nRows;

    // 诊断（MF_MATCH_DEBUG=1）：左右目特征数/ROI/匹配各阶段计数，
    // 用于排查"ROS 实机双目匹配为 0"的问题（如右目无特征/标定不匹配）
    const bool bMatchDebug = (getenv("MF_MATCH_DEBUG") != nullptr);
    if(bMatchDebug)
        cout << "[MFMatch] N=" << N << " Nr=" << (int)mvKeysRight.size()
             << " roi=[" << mROIRectLeftUp.x << "," << mROIRectLeftUp.y
             << "]-[" << mROIRectRightBottom.x << "," << mROIRectRightBottom.y << "]" << endl;

    // 右图特征点行索引：CSR 扁平结构（先计数→前缀和→填充），
    // 替代 vector<vector>(nRows) + reserve(400)——旧写法每帧预分配 ~2.3MB
    // （720 行 × 400 × 8B）再释放，分配抖动明显。
    const int Nr = (int)mvKeysRight.size();
    vector<int> vRowCount(nRows + 1, 0);
    for(int iR = 0; iR < Nr; iR++)
    {
        const cv::KeyPoint& kp = mvKeysRight[iR];
        const float kpY = kp.pt.y;
        if(kpY < 0.f || kpY >= heightF)
            continue;

        // 校正坐标系下极线水平对齐，但原始金字塔行坐标仍受畸变/焦距补偿影响，
        // 行带保持 2.0×尺度（收紧会丢边缘匹配，得不偿失）
        const float r = 2.0f * mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = std::min((int)ceil(kpY + r), nRows - 1);
        const int minr = std::max((int)floor(kpY - r), 0);

        for(int yi = minr; yi <= maxr; yi++)
            vRowCount[yi + 1]++;
    }
    for(int i = 0; i < nRows; i++)
        vRowCount[i + 1] += vRowCount[i];   // 前缀和：vRowCount[y] = 第 y 行起点

    vector<int> vRowIdx(vRowCount[nRows]);
    {
        vector<int> vRowFill = vRowCount;   // 填充游标
        for(int iR = 0; iR < Nr; iR++)
        {
            const cv::KeyPoint& kp = mvKeysRight[iR];
            const float kpY = kp.pt.y;
            if(kpY < 0.f || kpY >= heightF)
                continue;

            const float r = 2.0f * mvScaleFactors[mvKeysRight[iR].octave];
            const int maxr = std::min((int)ceil(kpY + r), nRows - 1);
            const int minr = std::max((int)floor(kpY - r), 0);

            for(int yi = minr; yi <= maxr; yi++)
                vRowIdx[vRowFill[yi]++] = iR;
        }
    }

    // 视差搜索范围：maxD = mbf / minZ。
    // minZ 用深度门控下限（默认10m），而不是基线长度——
    // 否则最大搜索视差接近整幅图像宽，假匹配会大量混入
    const float minZ = (mMinDepth > 0.f) ? mMinDepth : ((mb > 1e-6f) ? mb : 0.1f);
    const float minD = 0;
    const float maxD = mbf / minZ;

    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    // 焦距比对应的金字塔层差：长焦右图特征点层数 ≈ 左图 + Leyermis
    int Leyermis = 0;
    if(mFscale > 0.f && fabs(mfLogScaleFactor) > 1e-6f)
        Leyermis = cvRound(log(mFscale) / mfLogScaleFactor);

    // ================= 阶段1：描述子粗匹配（记录左右互指最优） =================
    vector<int> vBestIdxL(N, -1);
    int nCoarse = 0;   // 诊断：描述子粗匹配通过数

    for(int iL = 0; iL < N; iL++)
    {
        const cv::KeyPoint& kpL = mvKeys[iL];        // 校正后坐标
        const int levelL = kpL.octave;
        const float vL = kpL.pt.y;
        const float uL = kpL.pt.x;

        if(vL < 0.f || vL >= heightF)
            continue;
        // ROI 门控：只匹配两目重叠视场内的左特征点（校正坐标系）。
        // 自适应精化期间（前 ROIFINALIZE_FRAMES 帧）放宽到全图，让边缘特征
        // 参与匹配并累积实测分布，之后用实测分布重划的 ROI 限制。
        if(!mbROIAdapting)
        {
            if(uL < mROIRectLeftUp.x || uL > mROIRectRightBottom.x)
                continue;
            if(vL < mROIRectLeftUp.y || vL > mROIRectRightBottom.y)
                continue;
        }

        const int iRow = (int)vL;
        const int iCandStart = vRowCount[iRow];
        const int iCandEnd = vRowCount[iRow + 1];
        if(iCandStart == iCandEnd)
            continue;

        float minU = uL - maxD;
        minU = std::max(minU, 0.f);
        float maxU = uL - minD;
        maxU = std::min(maxU, widthF - 1.f);

        int bestDist = ORBmatcher::TH_HIGH;
        int bestIdxR = -1;
        const cv::Mat& dL = mDescriptors.row(iL);

        for(int iC = iCandStart; iC < iCandEnd; iC++)
        {
            const int iR = vRowIdx[iC];
            const cv::KeyPoint& kpR = mvKeysRight[iR];

            // 焦距比金字塔层补偿
            if(kpR.octave - Leyermis < levelL - 1 || kpR.octave - Leyermis > levelL + 1)
                continue;

            const float uR = kpR.pt.x;
            if(uR >= minU && uR <= maxU)
            {
                const cv::Mat& dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL, dR);
                if(dist < bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        if(bestIdxR < 0 || bestDist >= thOrbDist)
            continue;

        vBestIdxL[iL] = bestIdxR;
        nCoarse++;
    }

    // ================= 阶段2：互指校验 + SAD 亚像素精匹配 + 深度门控 =================
    int nGated = 0;   // 诊断：通过全部门控的有效匹配数

    // SAD 精匹配结果暂存（右目原始坐标），全部算完后批量 undistortPoints
    // （旧写法对每个匹配点单独构造 1×2 Mat 调一次 undistortPoints，每次都有
    //  Mat 分配+函数调用开销；批量一次调用即可，行为完全一致）
    struct SADResult { int iL, bestIdxR; float bestuR, bestvR; int bestDistSAD; };
    vector<SADResult> vSAD;
    vSAD.reserve(N);

    for(int iL = 0; iL < N; iL++)
    {
        const int bestIdxR = vBestIdxL[iL];
        if(bestIdxR < 0)
            continue;

        // 说明：50m 高空长焦图像中，右目特征比左目稀疏，
        // 一个右点对应多个左点是正常现象；左右互指校验会把大量真实匹配拒掉，
        // 导致地图过稀、跟踪易断。质量由 SAD 亚像素 + 深度门控 + 中位数剔除把关。

        const cv::KeyPoint& kpL = mvKeys[iL];
        const cv::KeyPoint& refkpL = refermvKeys[iL];

        // ---- SAD 亚像素精匹配（各目自己的原始金字塔与 octave，抵消焦距缩放） ----
        const float uR0 = refermvKeysRight[bestIdxR].pt.x;
        const float vR0 = refermvKeysRight[bestIdxR].pt.y;
        const float scaleFactor = mvInvScaleFactors[kpL.octave];
        const cv::KeyPoint& kpr = mvKeysRight[bestIdxR];
        const float scaleFactorright = mvInvScaleFactors[kpr.octave];

        const float scaleduL = round(refkpL.pt.x * scaleFactor);
        const float scaledvL = round(refkpL.pt.y * scaleFactor);
        const float scaleduR0 = round(uR0 * scaleFactorright);
        const float scaledvR0 = round(vR0 * scaleFactorright);

        // SAD 亚像素：窗口保持 5（精度），搜索半径 5→3。
        // 描述子粗匹配已给出亚像素级起点 uR0，±3px 足够覆盖残差；
        // 谷值门槛+抛物线插值仍在，匹配质量基本不变，SAD 计算量约 -36%。
        const int w = 5;
        const int L = 3;

        const float leftsadrowmax = mpORBextractorLeft->mvImagePyramid[kpL.octave].rows;
        const float leftsadcolmax = mpORBextractorLeft->mvImagePyramid[kpL.octave].cols;
        if(scaledvL - w < 0 || scaledvL + w + 1 > leftsadrowmax || scaleduL - w < 0 || scaleduL + w + 1 > leftsadcolmax)
            continue;

        // 左窗口去中心均值只算一次（旧写法对 7 个 incR 每次都 convertTo+减法+norm，
        // 每次都有 Mat 分配；这里直接用指针算，等价且零分配）
        const cv::Mat& pyrL = mpORBextractorLeft->mvImagePyramid[kpL.octave];
        const int iLeftStep = (int)pyrL.step;
        const uchar* pILBase = pyrL.ptr<uchar>((int)scaledvL - w) + (int)scaleduL - w;
        const int cL = pILBase[w * iLeftStep + w];
        int ILw[25];
        for(int y = 0; y < 5; y++)
            for(int x = 0; x < 5; x++)
                ILw[y * 5 + x] = (int)pILBase[y * iLeftStep + x] - cL;

        int bestDistSAD = std::numeric_limits<int>::max();
        int bestincR = 0;
        float vDists[7];   // 2*L+1 = 7，固定长度避免逐点 vector 分配

        const float iniu = scaleduR0 + L - w;
        const float endu = scaleduR0 + L + w + 1;
        const cv::Mat& pyrR = mpORBextractorRight->mvImagePyramid[kpr.octave];
        if(iniu < 0 || endu >= pyrR.cols)
            continue;

        const float sadrowmax = pyrR.rows;
        const float sadcolmax = pyrR.cols;
        const int iRightStep = (int)pyrR.step;
        for(int incR = -L; incR <= +L; incR++)
        {
            if(scaledvR0 - w < 0 || scaledvR0 + w + 1 > sadrowmax ||
               scaleduR0 + incR - w < 0 || scaleduR0 + incR + w + 1 > sadcolmax)
            {
                vDists[L + incR] = -1.f;   // 越界标记：原逻辑中 norm 不会执行、dist 保持默认构造 0
                continue;
            }

            const uchar* pIRBase = pyrR.ptr<uchar>((int)scaledvR0 - w) + (int)scaleduR0 + incR - w;
            const int cR = pIRBase[w * iRightStep + w];
            int sad = 0;
            for(int k = 0; k < 25; k++)
            {
                const int d = ILw[k] - ((int)pIRBase[(k / 5) * iRightStep + (k % 5)] - cR);
                sad += (d < 0) ? -d : d;
            }
            const float dist = (float)sad;
            if(dist < bestDistSAD)
            {
                bestDistSAD = (int)dist;
                bestincR = incR;
            }
            vDists[L + incR] = dist;
        }

        if(bestincR == -L || bestincR == L)
            continue;

        // 抛物线亚像素插值
        const float dist1 = vDists[L + bestincR - 1];
        const float dist2 = vDists[L + bestincR];
        const float dist3 = vDists[L + bestincR + 1];
        // SAD 谷值质量门槛：中心必须是明显更深的谷（否则亚像素峰不可靠，
        // 会在重复纹理上锁到错误位置，造成同一目标上深度差异极大）。
        // dist<=0 表示该窗口位置越界未计算，直接放弃。
        if(dist1 <= 0.f || dist2 <= 0.f || dist3 <= 0.f)
            continue;
        if(dist2 > 0.99f * std::min(dist1, dist3))
            continue;
        const float deltaR = (dist1 - dist3) / (2.0f * (dist1 + dist3 - 2.0f * dist2));
        if(deltaR < -1.f || deltaR > 1.f)
            continue;

        // 回到右目全分辨率原始坐标，稍后批量校正到公共坐标系
        const float bestuR = mvScaleFactors[kpr.octave] * ((float)scaleduR0 + (float)bestincR + deltaR);
        vSAD.push_back({iL, bestIdxR, bestuR, vR0, bestDistSAD});
    }

    // 批量校正：一次 undistortPoints 处理所有 SAD 通过的候选（等价于逐点调用）
    if(!vSAD.empty())
    {
        cv::Mat mat((int)vSAD.size(), 2, CV_32F);
        for(size_t k = 0; k < vSAD.size(); k++)
        {
            mat.at<float>((int)k, 0) = vSAD[k].bestuR;
            mat.at<float>((int)k, 1) = vSAD[k].bestvR;
        }
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mRightK, mRightD, mRightR, mRightP);
        mat = mat.reshape(1);

        for(size_t k = 0; k < vSAD.size(); k++)
        {
            const SADResult& r = vSAD[k];
            const int iL = r.iL;
            const int bestIdxR = r.bestIdxR;
            const float bestuR = mat.at<float>((int)k, 0);
            const cv::KeyPoint& kpL = mvKeys[iL];
            const float uL = kpL.pt.x;

            const float disparity = uL - bestuR;
            if(disparity >= 0 && disparity < maxD)
            {
                float d = disparity;
                float uR_final = bestuR;
                if(d <= 0)
                {
                    d = 0.01f;
                    uR_final = uL - 0.01f;
                }
                const float depth = mbf / d;

                // 深度合理性门控：50m 航拍下，太近(<MinDepth)或太远(>MaxDepth)
                // 的立体匹配基本都是假匹配（大视差假近点 / 近零视差假远点）
                if(depth < mMinDepth || depth > mMaxDepth)
                    continue;

                // 相机上方 + 超视界门控：45° 俯视 50m 航拍中，主点上方(v<cy)且
                // 深度超过 mAboveCameraGate(~100m) 的射线已超出可见地面范围
                // （图像顶边最远约 79m），只可能是误匹配。注意：主点上方但
                // 深度合理的点（远处地面）是真实特征，不能一刀切剔除。
                if(mbGateAboveCamera && kpL.pt.y < cy && depth > mAboveCameraGate)
                    continue;

                // 主点下方超远门控：v>cy 且深度超出主点下方地面可达范围（~50m）的
                // 点只可能是近零视差假匹配，反投影后落在相机“下方”，会污染地面
                // 平面拟合（把拟合平面从真实 50m 拉近到几米处）。阈值见
                // SetMultiFocalCalib 的 mBelowPrincipalGate 注释。
                if(mbGateAboveCamera && kpL.pt.y > cy && depth > mBelowPrincipalGate)
                    continue;

                mvDepth[iL] = depth;
                nGated++;
                mvuRight[iL] = uR_final;
                vDistIdx.push_back(pair<int, int>(r.bestDistSAD, iL));
                mvuLeft[iL] = (float)iL;
                LeftIdtoRightId[iL] = (float)bestIdxR;
                RightIdToLeftId[bestIdxR] = (float)iL;
                // 累积匹配特征点（校正坐标）：用于长焦对应区域自适应重划
                if(mbROIAdapting && mvsROIMatchPts.size() < 40000)
                    mvsROIMatchPts.push_back(kpL.pt);
            }
        }
    }

    // SAD 中位数离群剔除
    if(!vDistIdx.empty())
    {
        sort(vDistIdx.begin(), vDistIdx.end());
        const float median = vDistIdx[vDistIdx.size() / 2].first;
        const float thDist = 1.5f * 1.4f * median;

        for(int i = (int)vDistIdx.size() - 1; i >= 0; i--)
        {
            if(vDistIdx[i].first < thDist)
                break;
            else
            {
                const int iL = vDistIdx[i].second;
                mvuRight[iL] = -1;
                mvDepth[iL] = -1;
                mvuLeft[iL] = -1;
                if(LeftIdtoRightId[iL] >= 0)
                    RightIdToLeftId[(int)LeftIdtoRightId[iL]] = -1;
                LeftIdtoRightId[iL] = -1;
            }
        }
    }

    // 邻域深度一致性过滤：修复"同一目标上特征点深度差异极大"的问题。
    // 对每个匹配点，取图像邻域内其它匹配点的中位深度，
    // 若自身深度在 log 域偏离中位超过阈值，判为错误深度并剔除
    // （同一物体/同一局部区域的特征点深度应当相近）。
    {
        const float R2 = 80.f * 80.f;      // 邻域半径 80px（校正坐标系）
        const float fMaxLogDev = 1.6f;     // exp(±1.6) ≈ ×0.20 ~ ×5：只剔极端离谱点

        vector<int> vMatchedIdx;
        vMatchedIdx.reserve(N);
        for(int i = 0; i < N; i++)
            if(mvDepth[i] > 0)
                vMatchedIdx.push_back(i);

        if(vMatchedIdx.size() > 1)
        {
            vector<float> vLogD(N, 0.f);
            for(int i : vMatchedIdx)
                vLogD[i] = log(mvDepth[i]);

            vector<bool> vbKeep(N, false);
            // 单缓冲复用：旧写法每个点都 new 一个 vector<float>(reserve 16) 再排序，
            // M≈200 时每帧上百次小分配；结果完全相同
            vector<float> vNeighLog;
            vNeighLog.reserve(vMatchedIdx.size());
            for(size_t a = 0; a < vMatchedIdx.size(); a++)
            {
                const int iA = vMatchedIdx[a];
                const cv::Point2f& pA = mvKeys[iA].pt;

                vNeighLog.clear();
                for(size_t b = 0; b < vMatchedIdx.size(); b++)
                {
                    if(a == b)
                        continue;
                    const int iB = vMatchedIdx[b];
                    const cv::Point2f& pB = mvKeys[iB].pt;
                    const float dx = pA.x - pB.x;
                    const float dy = pA.y - pB.y;
                    if(dx * dx + dy * dy <= R2)
                        vNeighLog.push_back(vLogD[iB]);
                }

                if(vNeighLog.empty())
                {
                    vbKeep[iA] = true;   // 邻域无其它匹配点：保留
                    continue;
                }
                sort(vNeighLog.begin(), vNeighLog.end());
                const float med = vNeighLog[vNeighLog.size() / 2];
                if(fabs(vLogD[iA] - med) <= fMaxLogDev)
                    vbKeep[iA] = true;
            }

            for(int i : vMatchedIdx)
            {
                if(!vbKeep[i])
                {
                    mvuRight[i] = -1;
                    mvDepth[i] = -1;
                    mvuLeft[i] = -1;
                    if(LeftIdtoRightId[i] >= 0)
                        RightIdToLeftId[(int)LeftIdtoRightId[i]] = -1;
                    LeftIdtoRightId[i] = -1;
                }
            }
        }
    }

    // 长焦对应区域自适应：累积满 ROIFINALIZE_FRAMES 帧后，用实测匹配分布重划区域
    if(mbROIAdapting && ++mnROIAdaptFrameCount >= ROIFINALIZE_FRAMES)
        RefineROIFromMatches();

    if(bMatchDebug)
    {
        int nValid = 0;
        for(float d : mvDepth)
            if(d > 0.f) nValid++;
        cout << "[MFMatch] coarse=" << nCoarse << " gated=" << nGated
             << " valid=" << nValid << endl;
    }
}

void Frame::RefineROIFromMatches()
{
    // 匹配数据不足（首100帧匹配太少时）保持标定初始区域，不再自适应
    if(mvsROIMatchPts.size() < 50)
    {
        mbROIAdapting = false;
        return;
    }

    std::vector<float> vX, vY;
    vX.reserve(mvsROIMatchPts.size());
    vY.reserve(mvsROIMatchPts.size());
    for(const cv::Point2f& p : mvsROIMatchPts)
    {
        vX.push_back(p.x);
        vY.push_back(p.y);
    }
    std::sort(vX.begin(), vX.end());
    std::sort(vY.begin(), vY.end());

    const size_t n = vX.size();
    const float p5x = vX[n / 20];            // 5% 分位
    const float p95x = vX[n - 1 - n / 20];   // 95% 分位
    const float p5y = vY[n / 20];
    const float p95y = vY[n - 1 - n / 20];

    // 边距：给分布边缘留余量，避免把真实匹配区域边缘裁掉
    const float PAD = 15.f;
    const float minX = std::max(0.f, p5x - PAD);
    const float maxX = std::min(mnMaxX, p95x + PAD);
    const float minY = std::max(0.f, p5y - PAD);
    const float maxY = std::min(mnMaxY, p95y + PAD);
    if(maxX <= minX || maxY <= minY)
    {
        mbROIAdapting = false;
        return;
    }

    mROIRectLeftUp = cv::Point2f(minX, minY);
    mROIRectRightBottom = cv::Point2f(maxX, maxY);

    // 校正 ROI 四角 -> 左目原始图像坐标（绘制/提取用）
    {
        float minox = std::numeric_limits<float>::max(), minoy = std::numeric_limits<float>::max();
        float maxox = -std::numeric_limits<float>::max(), maxoy = -std::numeric_limits<float>::max();
        cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
        cv::Point2f corners[4] = { mROIRectLeftUp,
                                   cv::Point2f(maxX, minY),
                                   cv::Point2f(minX, maxY),
                                   mROIRectRightBottom };
        for(int k = 0; k < 4; k++)
        {
            cv::Mat xn(3, 1, CV_64F);
            xn.at<double>(0, 0) = (corners[k].x - cx) / fx;
            xn.at<double>(1, 0) = (corners[k].y - cy) / fy;
            xn.at<double>(2, 0) = 1.0;
            cv::Mat xnL = mLeftR.t() * xn;   // LEFT.P 平移为 0
            // 注意：本机 OpenCV 构建的 projectPoints 不接受 CV_64FC3 输入，统一用 CV_32FC3
            cv::Mat objPts(1, 1, CV_32FC3);
            objPts.at<cv::Vec3f>(0, 0) = cv::Vec3f((float)xnL.at<double>(0, 0),
                                                   (float)xnL.at<double>(1, 0),
                                                   (float)xnL.at<double>(2, 0));
            std::vector<cv::Point2f> vOut;
            cv::projectPoints(objPts, rvec, tvec, mLeftK, mLeftD, vOut);
            minox = std::min(minox, vOut[0].x); minoy = std::min(minoy, vOut[0].y);
            maxox = std::max(maxox, vOut[0].x); maxoy = std::max(maxoy, vOut[0].y);
        }
        mROILeftUp = cv::Point2f(std::max(minox, 0.f), std::max(minoy, 0.f));
        mROIRightBottom = cv::Point2f(std::min(maxox, mnMaxX), std::min(maxoy, mnMaxY));
    }

    ORBextractor::SetROI(mROILeftUp, mROIRightBottom);
    ORBextractor::SetRectifiedROI(mROIRectLeftUp, mROIRectRightBottom);

    cout << "[Frame] 长焦对应区域自适应完成（" << mvsROIMatchPts.size() << " 个匹配点）: "
         << "rect=[" << mROIRectLeftUp << " -> " << mROIRectRightBottom
         << "] orig=[" << mROILeftUp << " -> " << mROIRightBottom << "]" << endl;

    mbROIAdapting = false;
    mvsROIMatchPts.clear();
}

void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

bool Frame::UnprojectStereo(const int &i, Eigen::Vector3f &x3D)
{
    const float z = mvDepth[i];
    if(z>0) {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        Eigen::Vector3f x3Dc(x, y, z);
        x3D = mRwc * x3Dc + mOw;
        return true;
    } else
        return false;
}

bool Frame::imuIsPreintegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    return mbImuPreintegrated;
}

void Frame::setIntegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    mbImuPreintegrated = true;
}
//这个也是双目？但是左右的特征点一开始就分开处理？
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, GeometricCamera* pCamera2, Sophus::SE3f& Tlr,Frame* pPrevF, const IMU::Calib &ImuCalib)
        :mpcpi(NULL), mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),  mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
         mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false), mpCamera(pCamera), mpCamera2(pCamera2),
         mbHasPose(false), mbHasVelocity(false)

{
    imgLeft = imLeft.clone();
    imgRight = imRight.clone();

    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft,static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1]);
    thread threadRight(&Frame::ExtractORB,this,1,imRight,static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1]);
    threadLeft.join();
    threadRight.join();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    Nleft = mvKeys.size();
    Nright = mvKeysRight.size();
    N = Nleft + Nright;

    if(N == 0)
        return;

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf / fx;

    // Sophus/Eigen
    mTlr = Tlr;
    mTrl = mTlr.inverse();
    mRlr = mTlr.rotationMatrix();
    mtlr = mTlr.translation();

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
    ComputeStereoFishEyeMatches();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

    //Put all descriptors in the same matrix
    cv::vconcat(mDescriptors,mDescriptorsRight,mDescriptors);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(nullptr));
    mvbOutlier = vector<bool>(N,false);

    AssignFeaturesToGrid();

    mpMutexImu = new std::mutex();

    UndistortKeyPoints();

}
void Frame::ComputeStereoFishEyeMatches() {
    vector<cv::KeyPoint> stereoLeft(mvKeys.begin() + monoLeft, mvKeys.end());
    vector<cv::KeyPoint> stereoRight(mvKeysRight.begin() + monoRight, mvKeysRight.end());

    cv::Mat stereoDescLeft = mDescriptors.rowRange(monoLeft, mDescriptors.rows);
    cv::Mat stereoDescRight = mDescriptorsRight.rowRange(monoRight, mDescriptorsRight.rows);

    mvLeftToRightMatch = vector<int>(Nleft,-1);
    mvRightToLeftMatch = vector<int>(Nright,-1);
    mvDepth = vector<float>(Nleft,-1.0f);
    mvuRight = vector<float>(Nleft,-1);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(Nleft);
    mnCloseMPs = 0;

    vector<vector<cv::DMatch>> matches;
    BFmatcher.knnMatch(stereoDescLeft,stereoDescRight,matches,2);

    int nMatches = 0;
    int descMatches = 0;

    for(vector<vector<cv::DMatch>>::iterator it = matches.begin(); it != matches.end(); ++it){
        if((*it).size() >= 2 && (*it)[0].distance < (*it)[1].distance * 0.7){
            Eigen::Vector3f p3D;
            mvuRight[monoLeft + (it - matches.begin())] = stereoRight[(*it)[0].trainIdx].pt.x;
            nMatches++;
        }
    }
}

void Frame::SetFrameMatches(const std::vector<std::pair<int, int>>& matches)
{
    mvFrameMatches = matches;
}

void Frame::SetRelativePose(const Sophus::SE3f& relativePose)
{
    mRelativePose = relativePose;
}

const std::vector<std::pair<int, int>>& Frame::GetFrameMatches() const
{
    return mvFrameMatches;
}

Sophus::SE3f Frame::GetRelativePose() const
{
    return mRelativePose;
}

void Frame::SetStaticMatches(const std::vector<std::pair<int, int>>& matches)
{
    mvStaticMatches = matches;
}

void Frame::SetSemanticMatches(const std::vector<std::pair<int, int>>& matches)
{
    mvSemanticMatches = matches;
}

const std::vector<std::pair<int, int>>& Frame::GetStaticMatches() const
{
    return mvStaticMatches;
}

const std::vector<std::pair<int, int>>& Frame::GetSemanticMatches() const
{
    return mvSemanticMatches;
}

void Frame::ClearSemanticData()
{
    mvStaticMatches.clear();
    mvSemanticMatches.clear();
}

void Frame::SetVibrationMetrics(float rotationAngle, float verticalDisplacement)
{
    mRotationAngle = rotationAngle;
    mVerticalDisplacement = verticalDisplacement;
}

float Frame::GetRotationAngle() const
{
    return mRotationAngle;
}

float Frame::GetVerticalDisplacement() const
{
    return mVerticalDisplacement;
}

bool Frame::isInFrustumChecks(MapPoint* pMP, float viewingCosLimit, bool bRight)
{
    Eigen::Matrix<float,3,1> P = pMP->GetWorldPos();

    Eigen::Matrix<float,3,1> Pc;
    if(bRight)
    {
        Eigen::Matrix3f Rrl = mTrl.rotationMatrix();
        Eigen::Vector3f trl = mTrl.translation();
        Pc = Rrl * (mRcw * P + mtcw) + trl;
    }
    else
        Pc = mRcw * P + mtcw;

    if(Pc(2) <= 0.0f)
        return false;

    Eigen::Vector2f uv;
    if(bRight)
        uv = mpCamera2->project(Pc);
    else
        uv = mpCamera->project(Pc);

    if(uv(0) < mnMinX || uv(0) > mnMaxX)
        return false;
    if(uv(1) < mnMinY || uv(1) > mnMaxY)
        return false;

    float maxDistance = pMP->GetMaxDistanceInvariance();
    float minDistance = pMP->GetMinDistanceInvariance();
    Eigen::Vector3f PO = P - mOw;
    float dist = PO.norm();

    if(dist < minDistance || dist > maxDistance)
        return false;

    Eigen::Vector3f Pn = pMP->GetNormal();
    float viewCos = PO.dot(Pn) / dist;

    if(viewCos < viewingCosLimit)
        return false;

    return true;
}

Eigen::Vector3f Frame::UnprojectStereoFishEye(const int &i)
{
    Eigen::Vector3f x3D;
    x3D << mvStereo3Dpoints[i];
    return x3D;
}

} //namespace ORB_SLAM3
