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
#include "Tracking.h"

#include <unistd.h>   // usleep：全局尺度回拉时等待LocalMapping停止

#include "Settings.h"
#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "G2oTypes.h"
#include "Optimizer.h"
#include "Pinhole.h"
#include "KannalaBrandt8.h"
#include "MLPnPsolver.h"
#include "GeometricTools.h"
#include "Detector.h"
#include "common.h"

#include <iostream>
#include <limits>
#include <unordered_map>

#include <mutex>
#include <chrono>
#include <cstdlib>

using namespace std;

namespace ORB_SLAM3
{

// 全局耗时统计开关定义，默认关闭（零开销）。可在 main 中置 true 开启。
bool gEnableTimingStats = false;

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Atlas *pAtlas, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, Settings* settings, const string &_nameSeq):
    mState(NO_IMAGES_YET), mSensor(sensor), mTrackedFr(0), mbStep(false),
    mbOnlyTracking(false), mbMapUpdated(false), mbVO(false), mpORBVocabulary(pVoc), mpKeyFrameDB(pKFDB),
    mbReadyToInitializate(false), mpSystem(pSys), mpViewer(NULL), bStepByStep(false),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpAtlas(pAtlas), mnLastRelocFrameId(0), time_recently_lost(5.0),
    mnInitialFrameId(0), mbCreatedMap(false), mnFirstFrameId(0), mpCamera2(nullptr), mpLastKeyFrame(static_cast<KeyFrame*>(NULL)),
    mbVibrationThresholdsInitialized(false), mOptimizedRotationThreshold(0.018), mOptimizedVerticalThreshold(0.5),
    mMonoInitFrameId(0), mbPlaneFitted(false), mnLastPlaneCollectKFId(0),
        mfVehicleRefHeight(0.0f), mbVehicleHeightFrozen(false),
        mfPendingFrameScale(1.0f)
{
    // Load camera parameters from settings file
    if(settings){
        newParameterLoader(settings);
    }
    else{
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        bool b_parse_cam = ParseCamParamFile(fSettings);
        if(!b_parse_cam)
        {
            std::cout << "*Error with the camera parameters in the config file*" << std::endl;
        }

        // Load ORB parameters
        bool b_parse_orb = ParseORBParamFile(fSettings);
        if(!b_parse_orb)
        {
            std::cout << "*Error with the ORB parameters in the config file*" << std::endl;
        }

        bool b_parse_imu = true;
        if(sensor==System::IMU_MONOCULAR || sensor==System::IMU_STEREO || sensor==System::IMU_RGBD)
        {
            b_parse_imu = ParseIMUParamFile(fSettings);
            if(!b_parse_imu)
            {
                std::cout << "*Error with the IMU parameters in the config file*" << std::endl;
            }

            mnFramesToResetIMU = mMaxFrames;
        }

        if(!b_parse_cam || !b_parse_orb || !b_parse_imu)
        {
            std::cerr << "**ERROR in the config file, the format is not correct**" << std::endl;
            try
            {
                throw -1;
            }
            catch(exception &e)
            {

            }
        }
    }

    mvDynamicProbabilities = MOTION_PROBABILITIES;

    initID = 0; lastID = 0;
    mbInitWith3KFs = false;
    mnNumDataset = 0;

    vector<GeometricCamera*> vpCams = mpAtlas->GetAllCameras();
    std::cout << "There are " << vpCams.size() << " cameras in the atlas" << std::endl;
    for(GeometricCamera* pCam : vpCams)
    {
        std::cout << "Camera " << pCam->GetId();
        if(pCam->GetType() == GeometricCamera::CAM_PINHOLE)
        {
            std::cout << " is pinhole" << std::endl;
        }
        else if(pCam->GetType() == GeometricCamera::CAM_FISHEYE)
        {
            std::cout << " is fisheye" << std::endl;
        }
        else
        {
            std::cout << " is unknown" << std::endl;
        }
    }

#ifdef REGISTER_TIMES
    vdRectStereo_ms.clear();
    vdResizeImage_ms.clear();
    vdORBExtract_ms.clear();
    vdStereoMatch_ms.clear();
    vdIMUInteg_ms.clear();
    vdPosePred_ms.clear();
    vdLMTrack_ms.clear();
    vdNewKF_ms.clear();
    vdTrackTotal_ms.clear();
#endif

    // 初始化图像质量日志
    mbSaveQuality = true;
    mPendingQualityValid = false;
    mQualityLog.open("image_quality.txt", std::ios::out);
    if (mQualityLog.is_open()) {
        mQualityLog << FormatMetricsHeader() << std::endl;
    }

    // 初始化帧处理时间统计
    mTrackingStartTime = std::chrono::steady_clock::now();
    mnTotalFrames = 0;
    mdTotalProcessingTime = 0.0;
    mdMaxFrameTime = 0.0;
    mdMinFrameTime = std::numeric_limits<double>::max();
    mdLastFpsUpdateTime = 0.0;
    mnFramesSinceLastFpsUpdate = 0;
    mdCurrentFps = 0.0;
    mbFrameStatsStarted = false;
}

#ifdef REGISTER_TIMES
double calcAverage(vector<double> v_times)
{
    double accum = 0;
    for(double value : v_times)
    {
        accum += value;
    }

    return accum / v_times.size();
}

double calcDeviation(vector<double> v_times, double average)
{
    double accum = 0;
    for(double value : v_times)
    {
        accum += pow(value - average, 2);
    }
    return sqrt(accum / v_times.size());
}

double calcAverage(vector<int> v_values)
{
    double accum = 0;
    int total = 0;
    for(double value : v_values)
    {
        if(value == 0)
            continue;
        accum += value;
        total++;
    }

    return accum / total;
}

double calcDeviation(vector<int> v_values, double average)
{
    double accum = 0;
    int total = 0;
    for(double value : v_values)
    {
        if(value == 0)
            continue;
        accum += pow(value - average, 2);
        total++;
    }
    return sqrt(accum / total);
}

void Tracking::LocalMapStats2File()
{
    ofstream f;
    f.open("LocalMapTimeStats.txt");
    f << fixed << setprecision(6);
    f << "#Stereo rect[ms], MP culling[ms], MP creation[ms], LBA[ms], KF culling[ms], Total[ms]" << endl;
    for(int i=0; i<mpLocalMapper->vdLMTotal_ms.size(); ++i)
    {
        f << mpLocalMapper->vdKFInsert_ms[i] << "," << mpLocalMapper->vdMPCulling_ms[i] << ","
          << mpLocalMapper->vdMPCreation_ms[i] << "," << mpLocalMapper->vdLBASync_ms[i] << ","
          << mpLocalMapper->vdKFCullingSync_ms[i] <<  "," << mpLocalMapper->vdLMTotal_ms[i] << endl;
    }

    f.close();

    f.open("LBA_Stats.txt");
    f << fixed << setprecision(6);
    f << "#LBA time[ms], KF opt[#], KF fixed[#], MP[#], Edges[#]" << endl;
    for(int i=0; i<mpLocalMapper->vdLBASync_ms.size(); ++i)
    {
        f << mpLocalMapper->vdLBASync_ms[i] << "," << mpLocalMapper->vnLBA_KFopt[i] << ","
          << mpLocalMapper->vnLBA_KFfixed[i] << "," << mpLocalMapper->vnLBA_MPs[i] << ","
          << mpLocalMapper->vnLBA_edges[i] << endl;
    }
    f.close();
}

void Tracking::TrackStats2File()
{
    ofstream f;
    f.open("SessionInfo.txt");
    f << fixed;
    f << "Number of KFs: " << mpAtlas->GetAllKeyFrames().size() << endl;
    f << "Number of MPs: " << mpAtlas->GetAllMapPoints().size() << endl;

    f << "OpenCV version: " << CV_VERSION << endl;

    f.close();

    f.open("TrackingTimeStats.txt");
    f << fixed << setprecision(6);

    f << "#Image Rect[ms], Image Resize[ms], ORB ext[ms], Stereo match[ms], IMU preint[ms], Pose pred[ms], LM track[ms], KF dec[ms], Total[ms]" << endl;

    for(int i=0; i<vdTrackTotal_ms.size(); ++i)
    {
        double stereo_rect = 0.0;
        if(!vdRectStereo_ms.empty())
        {
            stereo_rect = vdRectStereo_ms[i];
        }

        double resize_image = 0.0;
        if(!vdResizeImage_ms.empty())
        {
            resize_image = vdResizeImage_ms[i];
        }

        double stereo_match = 0.0;
        if(!vdStereoMatch_ms.empty())
        {
            stereo_match = vdStereoMatch_ms[i];
        }

        double imu_preint = 0.0;
        if(!vdIMUInteg_ms.empty())
        {
            imu_preint = vdIMUInteg_ms[i];
        }

        f << stereo_rect << "," << resize_image << "," << vdORBExtract_ms[i] << "," << stereo_match << "," << imu_preint << ","
          << vdPosePred_ms[i] <<  "," << vdLMTrack_ms[i] << "," << vdNewKF_ms[i] << "," << vdTrackTotal_ms[i] << endl;
    }

    f.close();
}

void Tracking::PrintTimeStats()
{
    // Save data in files
    TrackStats2File();
    LocalMapStats2File();
    ofstream f;
    f.open("ExecMean.txt");
    f << fixed;
    //Report the mean and std of each one
    std::cout << std::endl << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
    f << " TIME STATS in ms (mean$\\pm$std)" << std::endl;
    cout << "OpenCV version: " << CV_VERSION << endl;
    f << "OpenCV version: " << CV_VERSION << endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << "Tracking" << std::setprecision(5) << std::endl << std::endl;
    f << "---------------------------" << std::endl;
    f << "Tracking" << std::setprecision(5) << std::endl << std::endl;
    double average, deviation;
    if(!vdRectStereo_ms.empty())
    {
        average = calcAverage(vdRectStereo_ms);
        deviation = calcDeviation(vdRectStereo_ms, average);
        std::cout << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
        f << "Stereo Rectification: " << average << "$\\pm$" << deviation << std::endl;
    }

    if(!vdResizeImage_ms.empty())
    {
        average = calcAverage(vdResizeImage_ms);
        deviation = calcDeviation(vdResizeImage_ms, average);
        std::cout << "Image Resize: " << average << "$\\pm$" << deviation << std::endl;
        f << "Image Resize: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdORBExtract_ms);
    deviation = calcDeviation(vdORBExtract_ms, average);
    std::cout << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;
    f << "ORB Extraction: " << average << "$\\pm$" << deviation << std::endl;

    if(!vdStereoMatch_ms.empty())
    {
        average = calcAverage(vdStereoMatch_ms);
        deviation = calcDeviation(vdStereoMatch_ms, average);
        std::cout << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
        f << "Stereo Matching: " << average << "$\\pm$" << deviation << std::endl;
    }

    if(!vdIMUInteg_ms.empty())
    {
        average = calcAverage(vdIMUInteg_ms);
        deviation = calcDeviation(vdIMUInteg_ms, average);
        std::cout << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
        f << "IMU Preintegration: " << average << "$\\pm$" << deviation << std::endl;
    }

    average = calcAverage(vdPosePred_ms);
    deviation = calcDeviation(vdPosePred_ms, average);
    std::cout << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;
    f << "Pose Prediction: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdLMTrack_ms);
    deviation = calcDeviation(vdLMTrack_ms, average);
    std::cout << "LM Track: " << average << "$\\pm$" << deviation << std::endl;
    f << "LM Track: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdNewKF_ms);
    deviation = calcDeviation(vdNewKF_ms, average);
    std::cout << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;
    f << "New KF decision: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(vdTrackTotal_ms);
    deviation = calcDeviation(vdTrackTotal_ms, average);
    std::cout << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;
    f << "Total Tracking: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping time stats
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "Local Mapping" << std::endl << std::endl;
    f << std::endl << "Local Mapping" << std::endl << std::endl;

    average = calcAverage(mpLocalMapper->vdKFInsert_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFInsert_ms, average);
    std::cout << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;
    f << "KF Insertion: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCulling_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCulling_ms, average);
    std::cout << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;
    f << "MP Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdMPCreation_ms);
    deviation = calcDeviation(mpLocalMapper->vdMPCreation_ms, average);
    std::cout << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;
    f << "MP Creation: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLBA_ms);
    deviation = calcDeviation(mpLocalMapper->vdLBA_ms, average);
    std::cout << "LBA: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdKFCulling_ms);
    deviation = calcDeviation(mpLocalMapper->vdKFCulling_ms, average);
    std::cout << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;
    f << "KF Culling: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vdLMTotal_ms);
    deviation = calcDeviation(mpLocalMapper->vdLMTotal_ms, average);
    std::cout << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;
    f << "Total Local Mapping: " << average << "$\\pm$" << deviation << std::endl;

    // Local Mapping LBA complexity
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "LBA complexity (mean$\\pm$std)" << std::endl;
    f << "---------------------------" << std::endl;
    f << std::endl << "LBA complexity (mean$\\pm$std)" << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_edges);
    deviation = calcDeviation(mpLocalMapper->vnLBA_edges, average);
    std::cout << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA Edges: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFopt);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFopt, average);
    std::cout << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA KF optimized: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_KFfixed);
    deviation = calcDeviation(mpLocalMapper->vnLBA_KFfixed, average);
    std::cout << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;
    f << "LBA KF fixed: " << average << "$\\pm$" << deviation << std::endl;

    average = calcAverage(mpLocalMapper->vnLBA_MPs);
    deviation = calcDeviation(mpLocalMapper->vnLBA_MPs, average);
    std::cout << "LBA MP: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    f << "LBA MP: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    std::cout << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
    std::cout << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;
    f << "LBA executions: " << mpLocalMapper->nLBA_exec << std::endl;
    f << "LBA aborts: " << mpLocalMapper->nLBA_abort << std::endl;

    // Map complexity
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "Map complexity" << std::endl;
    std::cout << "KFs in map: " << mpAtlas->GetAllKeyFrames().size() << std::endl;
    std::cout << "MPs in map: " << mpAtlas->GetAllMapPoints().size() << std::endl;
    f << "---------------------------" << std::endl;
    f << std::endl << "Map complexity" << std::endl;
    vector<Map*> vpMaps = mpAtlas->GetAllMaps();
    Map* pBestMap = vpMaps[0];
    for(int i=1; i<vpMaps.size(); ++i)
    {
        if(pBestMap->GetAllKeyFrames().size() < vpMaps[i]->GetAllKeyFrames().size())
        {
            pBestMap = vpMaps[i];
        }
    }

    f << "KFs in map: " << pBestMap->GetAllKeyFrames().size() << std::endl;
    f << "MPs in map: " << pBestMap->GetAllMapPoints().size() << std::endl;

    f << "---------------------------" << std::endl;
    f << std::endl << "Place Recognition (mean$\\pm$std)" << std::endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << std::endl << "Place Recognition (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdDataQuery_ms);
    deviation = calcDeviation(mpLoopClosing->vdDataQuery_ms, average);
    f << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Database Query: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdEstSim3_ms);
    deviation = calcDeviation(mpLoopClosing->vdEstSim3_ms, average);
    f << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "SE3 estimation: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdPRTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdPRTotal_ms, average);
    f << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Place Recognition: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << std::endl << "Loop Closing (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Loop Closing (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopFusion_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopFusion_ms, average);
    f << "Loop Fusion: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Loop Fusion: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopOptEss_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopOptEss_ms, average);
    f << "Essential Graph: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Essential Graph: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdLoopTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdLoopTotal_ms, average);
    f << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Loop Closing: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nLoop << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nLoop << std::endl;
    average = calcAverage(mpLoopClosing->vnLoopKFs);
    deviation = calcDeviation(mpLoopClosing->vnLoopKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;

    f << std::endl << "Map Merging (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Map Merging (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeMaps_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeMaps_ms, average);
    f << "Merge Maps: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Merge Maps: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdWeldingBA_ms);
    deviation = calcDeviation(mpLoopClosing->vdWeldingBA_ms, average);
    f << "Welding BA: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Welding BA: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeOptEss_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeOptEss_ms, average);
    f << "Optimization Ess.: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Optimization Ess.: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdMergeTotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdMergeTotal_ms, average);
    f << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Map Merging: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nMerges << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nMerges << std::endl;
    average = calcAverage(mpLoopClosing->vnMergeKFs);
    deviation = calcDeviation(mpLoopClosing->vnMergeKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vnMergeMPs);
    deviation = calcDeviation(mpLoopClosing->vnMergeMPs, average);
    f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;

    f << std::endl << "Full GBA (mean$\\pm$std)" << std::endl;
    std::cout << std::endl << "Full GBA (mean$\\pm$std)" << std::endl;
    average = calcAverage(mpLoopClosing->vdGBA_ms);
    deviation = calcDeviation(mpLoopClosing->vdGBA_ms, average);
    f << "GBA: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "GBA: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdUpdateMap_ms);
    deviation = calcDeviation(mpLoopClosing->vdUpdateMap_ms, average);
    f << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Map Update: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vdFGBATotal_ms);
    deviation = calcDeviation(mpLoopClosing->vdFGBATotal_ms, average);
    f << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl << std::endl;
    std::cout << "Total Full GBA: " << average << "$\\pm$" << deviation << std::endl << std::endl;

    f << "Numb exec: " << mpLoopClosing->nFGBA_exec << std::endl;
    std::cout << "Num exec: " << mpLoopClosing->nFGBA_exec << std::endl;
    f << "Numb abort: " << mpLoopClosing->nFGBA_abort << std::endl;
    std::cout << "Num abort: " << mpLoopClosing->nFGBA_abort << std::endl;
    average = calcAverage(mpLoopClosing->vnGBAKFs);
    deviation = calcDeviation(mpLoopClosing->vnGBAKFs, average);
    f << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of KFs: " << average << "$\\pm$" << deviation << std::endl;
    average = calcAverage(mpLoopClosing->vnGBAMPs);
    deviation = calcDeviation(mpLoopClosing->vnGBAMPs, average);
    f << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;
    std::cout << "Number of MPs: " << average << "$\\pm$" << deviation << std::endl;

    f.close();

}

#endif

Tracking::~Tracking()
{
    //f_track_stats.close();

}

void Tracking::newParameterLoader(Settings *settings) {
    mpCamera = settings->camera1();
    mpCamera = mpAtlas->AddCamera(mpCamera);

    if(settings->needToUndistort()){
        mDistCoef = settings->camera1DistortionCoef();
    }
    else{
        mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    }

    //TODO: missing image scaling and rectification
    mImageScale = 1.0f;

    mK = cv::Mat::eye(3,3,CV_32F);
    mK.at<float>(0,0) = mpCamera->getParameter(0);
    mK.at<float>(1,1) = mpCamera->getParameter(1);
    mK.at<float>(0,2) = mpCamera->getParameter(2);
    mK.at<float>(1,2) = mpCamera->getParameter(3);

    mK_.setIdentity();
    mK_(0,0) = mpCamera->getParameter(0);
    mK_(1,1) = mpCamera->getParameter(1);
    mK_(0,2) = mpCamera->getParameter(2);
    mK_(1,2) = mpCamera->getParameter(3);

    if((mSensor==System::STEREO || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD) &&
        settings->cameraType() == Settings::KannalaBrandt){
        mpCamera2 = settings->camera2();
        mpCamera2 = mpAtlas->AddCamera(mpCamera2);

        mTlr = settings->Tlr();

        mpFrameDrawer->both = true;
    }

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD ){
        mbf = settings->bf();
        mThDepth = settings->b() * settings->thDepth();
    }

    if(mSensor==System::RGBD || mSensor==System::IMU_RGBD){
        mDepthMapFactor = settings->depthMapFactor();
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

    mMinFrames = 0;
    mMaxFrames = settings->fps();
    mbRGB = settings->rgb();

    //ORB parameters
    nFeatures = settings->nFeatures();
    nLevels = settings->nLevels();
    fIniThFAST = settings->initThFAST();
    fMinThFAST = settings->minThFAST();
    fScaleFactor = settings->scaleFactor();

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::STEREO || mSensor==System::IMU_STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR)
        mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    //IMU parameters
    Sophus::SE3f Tbc = settings->Tbc();
    mInsertKFsLost = settings->insertKFsWhenLost();
    mImuFreq = settings->imuFrequency();
    mImuPer = 0.001; //1.0 / (double) mImuFreq;     //TODO: ESTO ESTA BIEN?
    float Ng = settings->noiseGyro();
    float Na = settings->noiseAcc();
    float Ngw = settings->gyroWalk();
    float Naw = settings->accWalk();

    const float sf = sqrt(mImuFreq);
    mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
}

bool Tracking::ParseCamParamFile(cv::FileStorage &fSettings)
{
    mDistCoef = cv::Mat::zeros(4,1,CV_32F);
    cout << endl << "Camera Parameters: " << endl;
    bool b_miss_params = false;

    string sCameraName = fSettings["Camera.type"];
    if(sCameraName == "PinHole")
    {
        float fx, fy, cx, cy;
        mImageScale = 1.f;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if(!node.empty() && node.isReal())
        {
            fx = node.real();
        }
        else
        {
            std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.fy"];
        if(!node.empty() && node.isReal())
        {
            fy = node.real();
        }
        else
        {
            std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if(!node.empty() && node.isReal())
        {
            cx = node.real();
        }
        else
        {
            std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if(!node.empty() && node.isReal())
        {
            cy = node.real();
        }
        else
        {
            std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        // Distortion parameters
        node = fSettings["Camera.k1"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(0) = node.real();
        }
        else
        {
            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k2"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(1) = node.real();
        }
        else
        {
            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p1"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(2) = node.real();
        }
        else
        {
            std::cerr << "*Camera.p1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.p2"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.at<float>(3) = node.real();
        }
        else
        {
            std::cerr << "*Camera.p2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if(!node.empty() && node.isReal())
        {
            mDistCoef.resize(5);
            mDistCoef.at<float>(4) = node.real();
        }

        node = fSettings["Camera.imageScale"];
        if(!node.empty() && node.isReal())
        {
            mImageScale = node.real();
        }

        if(b_miss_params)
        {
            return false;
        }

        if(mImageScale != 1.f)
        {
            // K matrix parameters must be scaled.
            fx = fx * mImageScale;
            fy = fy * mImageScale;
            cx = cx * mImageScale;
            cy = cy * mImageScale;
        }

        vector<float> vCamCalib{fx,fy,cx,cy};

        mpCamera = new Pinhole(vCamCalib);

        mpCamera = mpAtlas->AddCamera(mpCamera);

        std::cout << "- Camera: Pinhole" << std::endl;
        std::cout << "- Image scale: " << mImageScale << std::endl;
        std::cout << "- fx: " << fx << std::endl;
        std::cout << "- fy: " << fy << std::endl;
        std::cout << "- cx: " << cx << std::endl;
        std::cout << "- cy: " << cy << std::endl;
        std::cout << "- k1: " << mDistCoef.at<float>(0) << std::endl;
        std::cout << "- k2: " << mDistCoef.at<float>(1) << std::endl;
        std::cout << "- p1: " << mDistCoef.at<float>(2) << std::endl;
        std::cout << "- p2: " << mDistCoef.at<float>(3) << std::endl;

        if(mDistCoef.rows==5)
            std::cout << "- k3: " << mDistCoef.at<float>(4) << std::endl;

        mK = cv::Mat::eye(3,3,CV_32F);
        mK.at<float>(0,0) = fx;
        mK.at<float>(1,1) = fy;
        mK.at<float>(0,2) = cx;
        mK.at<float>(1,2) = cy;

        mK_.setIdentity();
        mK_(0,0) = fx;
        mK_(1,1) = fy;
        mK_(0,2) = cx;
        mK_(1,2) = cy;
    }
    else if(sCameraName == "KannalaBrandt8")
    {
        float fx, fy, cx, cy;
        float k1, k2, k3, k4;
        mImageScale = 1.f;

        // Camera calibration parameters
        cv::FileNode node = fSettings["Camera.fx"];
        if(!node.empty() && node.isReal())
        {
            fx = node.real();
        }
        else
        {
            std::cerr << "*Camera.fx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
        node = fSettings["Camera.fy"];
        if(!node.empty() && node.isReal())
        {
            fy = node.real();
        }
        else
        {
            std::cerr << "*Camera.fy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cx"];
        if(!node.empty() && node.isReal())
        {
            cx = node.real();
        }
        else
        {
            std::cerr << "*Camera.cx parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.cy"];
        if(!node.empty() && node.isReal())
        {
            cy = node.real();
        }
        else
        {
            std::cerr << "*Camera.cy parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        // Distortion parameters
        node = fSettings["Camera.k1"];
        if(!node.empty() && node.isReal())
        {
            k1 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k1 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
        node = fSettings["Camera.k2"];
        if(!node.empty() && node.isReal())
        {
            k2 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k2 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k3"];
        if(!node.empty() && node.isReal())
        {
            k3 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k3 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.k4"];
        if(!node.empty() && node.isReal())
        {
            k4 = node.real();
        }
        else
        {
            std::cerr << "*Camera.k4 parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

        node = fSettings["Camera.imageScale"];
        if(!node.empty() && node.isReal())
        {
            mImageScale = node.real();
        }

        if(!b_miss_params)
        {
            if(mImageScale != 1.f)
            {
                // K matrix parameters must be scaled.
                fx = fx * mImageScale;
                fy = fy * mImageScale;
                cx = cx * mImageScale;
                cy = cy * mImageScale;
            }

            vector<float> vCamCalib{fx,fy,cx,cy,k1,k2,k3,k4};
            mpCamera = new KannalaBrandt8(vCamCalib);
            mpCamera = mpAtlas->AddCamera(mpCamera);
            std::cout << "- Camera: Fisheye" << std::endl;
            std::cout << "- Image scale: " << mImageScale << std::endl;
            std::cout << "- fx: " << fx << std::endl;
            std::cout << "- fy: " << fy << std::endl;
            std::cout << "- cx: " << cx << std::endl;
            std::cout << "- cy: " << cy << std::endl;
            std::cout << "- k1: " << k1 << std::endl;
            std::cout << "- k2: " << k2 << std::endl;
            std::cout << "- k3: " << k3 << std::endl;
            std::cout << "- k4: " << k4 << std::endl;

            mK = cv::Mat::eye(3,3,CV_32F);
            mK.at<float>(0,0) = fx;
            mK.at<float>(1,1) = fy;
            mK.at<float>(0,2) = cx;
            mK.at<float>(1,2) = cy;

            mK_.setIdentity();
            mK_(0,0) = fx;
            mK_(1,1) = fy;
            mK_(0,2) = cx;
            mK_(1,2) = cy;
        }

        if(mSensor==System::STEREO || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD){
            // Right camera
            // Camera calibration parameters
            cv::FileNode node = fSettings["Camera2.fx"];
            if(!node.empty() && node.isReal())
            {
                fx = node.real();
            }
            else
            {
                std::cerr << "*Camera2.fx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera2.fy"];
            if(!node.empty() && node.isReal())
            {
                fy = node.real();
            }
            else
            {
                std::cerr << "*Camera2.fy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.cx"];
            if(!node.empty() && node.isReal())
            {
                cx = node.real();
            }
            else
            {
                std::cerr << "*Camera2.cx parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.cy"];
            if(!node.empty() && node.isReal())
            {
                cy = node.real();
            }
            else
            {
                std::cerr << "*Camera2.cy parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            // Distortion parameters
            node = fSettings["Camera2.k1"];
            if(!node.empty() && node.isReal())
            {
                k1 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k1 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            node = fSettings["Camera2.k2"];
            if(!node.empty() && node.isReal())
            {
                k2 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k2 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.k3"];
            if(!node.empty() && node.isReal())
            {
                k3 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k3 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }

            node = fSettings["Camera2.k4"];
            if(!node.empty() && node.isReal())
            {
                k4 = node.real();
            }
            else
            {
                std::cerr << "*Camera2.k4 parameter doesn't exist or is not a real number*" << std::endl;
                b_miss_params = true;
            }
            int leftLappingBegin = -1;
            int leftLappingEnd = -1;

            int rightLappingBegin = -1;
            int rightLappingEnd = -1;

            node = fSettings["Camera.lappingBegin"];
            if(!node.empty() && node.isInt())
            {
                leftLappingBegin = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera.lappingBegin not correctly defined" << std::endl;
            }
            node = fSettings["Camera.lappingEnd"];
            if(!node.empty() && node.isInt())
            {
                leftLappingEnd = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera.lappingEnd not correctly defined" << std::endl;
            }
            node = fSettings["Camera2.lappingBegin"];
            if(!node.empty() && node.isInt())
            {
                rightLappingBegin = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera2.lappingBegin not correctly defined" << std::endl;
            }
            node = fSettings["Camera2.lappingEnd"];
            if(!node.empty() && node.isInt())
            {
                rightLappingEnd = node.operator int();
            }
            else
            {
                std::cout << "WARNING: Camera2.lappingEnd not correctly defined" << std::endl;
            }

            node = fSettings["Tlr"];
            cv::Mat cvTlr;
            if(!node.empty())
            {
                cvTlr = node.mat();
                if(cvTlr.rows != 3 || cvTlr.cols != 4)
                {
                    std::cerr << "*Tlr matrix have to be a 3x4 transformation matrix*" << std::endl;
                    b_miss_params = true;
                }
            }
            else
            {
                std::cerr << "*Tlr matrix doesn't exist*" << std::endl;
                b_miss_params = true;
            }

            if(!b_miss_params)
            {
                if(mImageScale != 1.f)
                {
                    // K matrix parameters must be scaled.
                    fx = fx * mImageScale;
                    fy = fy * mImageScale;
                    cx = cx * mImageScale;
                    cy = cy * mImageScale;

                    leftLappingBegin = leftLappingBegin * mImageScale;
                    leftLappingEnd = leftLappingEnd * mImageScale;
                    rightLappingBegin = rightLappingBegin * mImageScale;
                    rightLappingEnd = rightLappingEnd * mImageScale;
                }

                static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0] = leftLappingBegin;
                static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1] = leftLappingEnd;

                mpFrameDrawer->both = true;

                vector<float> vCamCalib2{fx,fy,cx,cy,k1,k2,k3,k4};
                mpCamera2 = new KannalaBrandt8(vCamCalib2);
                mpCamera2 = mpAtlas->AddCamera(mpCamera2);

                mTlr = Converter::toSophus(cvTlr);

                static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0] = rightLappingBegin;
                static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1] = rightLappingEnd;

                std::cout << "- Camera1 Lapping: " << leftLappingBegin << ", " << leftLappingEnd << std::endl;

                std::cout << std::endl << "Camera2 Parameters:" << std::endl;
                std::cout << "- Camera: Fisheye" << std::endl;
                std::cout << "- Image scale: " << mImageScale << std::endl;
                std::cout << "- fx: " << fx << std::endl;
                std::cout << "- fy: " << fy << std::endl;
                std::cout << "- cx: " << cx << std::endl;
                std::cout << "- cy: " << cy << std::endl;
                std::cout << "- k1: " << k1 << std::endl;
                std::cout << "- k2: " << k2 << std::endl;
                std::cout << "- k3: " << k3 << std::endl;
                std::cout << "- k4: " << k4 << std::endl;

                std::cout << "- mTlr: \n" << cvTlr << std::endl;

                std::cout << "- Camera2 Lapping: " << rightLappingBegin << ", " << rightLappingEnd << std::endl;
            }
        }

        if(b_miss_params)
        {
            return false;
        }

    }
    else
    {
        std::cerr << "*Not Supported Camera Sensor*" << std::endl;
        std::cerr << "Check an example configuration file with the desired sensor" << std::endl;
    }

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD )
    {
        cv::FileNode node = fSettings["Camera.bf"];
        if(!node.empty() && node.isReal())
        {
            mbf = node.real();
            if(mImageScale != 1.f)
            {
                mbf *= mImageScale;
            }
        }
        else
        {
            std::cerr << "*Camera.bf parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

    }

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << "- fps: " << fps << endl;
    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD)
    {
        float fx = mpCamera->getParameter(0);
        cv::FileNode node = fSettings["ThDepth"];
        if(!node.empty()  && node.isReal())
        {
            mThDepth = node.real();
            mThDepth = mbf*mThDepth/fx;
            cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
        }
        else
        {
            std::cerr << "*ThDepth parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }
    }

    if(mSensor==System::RGBD || mSensor==System::IMU_RGBD)
    {
        cv::FileNode node = fSettings["DepthMapFactor"];
        if(!node.empty() && node.isReal())
        {
            mDepthMapFactor = node.real();
            if(fabs(mDepthMapFactor)<1e-5)
                mDepthMapFactor=1;
            else
                mDepthMapFactor = 1.0f/mDepthMapFactor;
        }
        else
        {
            std::cerr << "*DepthMapFactor parameter doesn't exist or is not a real number*" << std::endl;
            b_miss_params = true;
        }

    }

    if(b_miss_params)
    {
        return false;
    }

    return true;
}

bool Tracking::ParseORBParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;

    cv::FileNode node = fSettings["ORBextractor.nFeatures"];
    if(!node.empty() && node.isInt())
    {
        nFeatures = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.nFeatures parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.scaleFactor"];
    if(!node.empty() && node.isReal())
    {
        fScaleFactor = node.real();
    }
    else
    {
        std::cerr << "*ORBextractor.scaleFactor parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.nLevels"];
    if(!node.empty() && node.isInt())
    {
        nLevels = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.nLevels parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.iniThFAST"];
    if(!node.empty() && node.isInt())
    {
        fIniThFAST = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.iniThFAST parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["ORBextractor.minThFAST"];
    if(!node.empty() && node.isInt())
    {
        fMinThFAST = node.operator int();
    }
    else
    {
        std::cerr << "*ORBextractor.minThFAST parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    if(b_miss_params)
    {
        return false;
    }

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::STEREO || mSensor==System::IMU_STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR)
        mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    return true;
}

bool Tracking::ParseIMUParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;

    cv::Mat cvTbc;
    cv::FileNode node = fSettings["Tbc"];
    if(!node.empty())
    {
        cvTbc = node.mat();
        if(cvTbc.rows != 4 || cvTbc.cols != 4)
        {
            std::cerr << "*Tbc matrix have to be a 4x4 transformation matrix*" << std::endl;
            b_miss_params = true;
        }
    }
    else
    {
        std::cerr << "*Tbc matrix doesn't exist*" << std::endl;
        b_miss_params = true;
    }
    cout << endl;
    cout << "Left camera to Imu Transform (Tbc): " << endl << cvTbc << endl;
    Eigen::Matrix<float,4,4,Eigen::RowMajor> eigTbc(cvTbc.ptr<float>(0));
    Sophus::SE3f Tbc(eigTbc);

    node = fSettings["InsertKFsWhenLost"];
    mInsertKFsLost = true;
    if(!node.empty() && node.isInt())
    {
        mInsertKFsLost = (bool) node.operator int();
    }

    if(!mInsertKFsLost)
        cout << "Do not insert keyframes when lost visual tracking " << endl;

    float Ng, Na, Ngw, Naw;

    node = fSettings["IMU.Frequency"];
    if(!node.empty() && node.isInt())
    {
        mImuFreq = node.operator int();
        mImuPer = 0.001; //1.0 / (double) mImuFreq;
    }
    else
    {
        std::cerr << "*IMU.Frequency parameter doesn't exist or is not an integer*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseGyro"];
    if(!node.empty() && node.isReal())
    {
        Ng = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseGyro parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.NoiseAcc"];
    if(!node.empty() && node.isReal())
    {
        Na = node.real();
    }
    else
    {
        std::cerr << "*IMU.NoiseAcc parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.GyroWalk"];
    if(!node.empty() && node.isReal())
    {
        Ngw = node.real();
    }
    else
    {
        std::cerr << "*IMU.GyroWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.AccWalk"];
    if(!node.empty() && node.isReal())
    {
        Naw = node.real();
    }
    else
    {
        std::cerr << "*IMU.AccWalk parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["IMU.fastInit"];
    mFastInit = false;
    if(!node.empty())
    {
        mFastInit = static_cast<int>(fSettings["IMU.fastInit"]) != 0;
    }

    if(mFastInit)
        cout << "Fast IMU initialization. Acceleration is not checked \n";

    if(b_miss_params)
    {
        return false;
    }

    const float sf = sqrt(mImuFreq);
    cout << endl;
    cout << "IMU frequency: " << mImuFreq << " Hz" << endl;
    cout << "IMU gyro noise: " << Ng << " rad/s/sqrt(Hz)" << endl;
    cout << "IMU gyro walk: " << Ngw << " rad/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer noise: " << Na << " m/s^2/sqrt(Hz)" << endl;
    cout << "IMU accelerometer walk: " << Naw << " m/s^3/sqrt(Hz)" << endl;

    mpImuCalib = new IMU::Calib(Tbc,Ng*sf,Na*sf,Ngw/sf,Naw/sf);

    mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
    return true;
}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}


void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
    
}

void Tracking::SetDetector(Detector *pDetector)
{
    mpDetector=pDetector;
}

void Tracking::SetStepByStep(bool bSet)
{
    bStepByStep = bSet;
}

bool Tracking::GetStepByStep()
{
    return bStepByStep;
}

Sophus::SE3f Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, string filename)
{
    // cout << "GrabImageStereo" << endl;

        //------------------------------------------------------------------------------------------------

	mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST); 
	mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    //------------------------------------------------------------------------------------------------
    
    mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;
    mImRight = imRectRight;

    if(mImGray.channels()==3)
    {
        //cout << "Image with 3 channels" << endl;
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_BGR2GRAY);
        }
    }
    else if(mImGray.channels()==4)
    {
        //cout << "Image with 4 channels" << endl;
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,cv::COLOR_BGRA2GRAY);
        }
    }

    // cout << "Incoming frame creation" << endl;

    if (mSensor == System::STEREO && !mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera);
       // cout << "GrabImageStereo1" << endl;}
    else if(mSensor == System::STEREO && mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr);
    else if(mSensor == System::IMU_STEREO && !mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,&mLastFrame,*mpImuCalib);
    else if(mSensor == System::IMU_STEREO && mpCamera2)
        mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,mpCamera2,mTlr,&mLastFrame,*mpImuCalib);

    // 记录图像质量指标（每10帧采样一次，降低CPU开销）
    {
        static int qualityCounter = 0;
        if (++qualityCounter % 10 == 0)
            LogImageQuality(mImGray, mCurrentFrame.mvKeysUn, nLevels, mCurrentFrame.mnId);
    }
    
     // cout << "GrabImageStereo2" << endl;
    //-------------------------------------------------------------------------------
	std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
	// cout << "GrabImageStereo3" << endl;
	// struct timespec ts, ts1;
	// ts.tv_nsec = 500000000;    // 1500ms
        // ts.tv_sec = 1;
	// cout << !isNewDetectedImgArrived() <<"New"<< endl;
     while(!isNewDetectedImgArrived()) 
    {
        // cout << !isNewDetectedImgArrived() <<"New"<< endl;
        //cout << "GrabImageStereo4" << endl;
        usleep(1);
        // cout << "GrabImageStereo5" << endl;
    }
    // cout << "GrabImageStereo4" << endl;
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();
    // cout << "time waiting for detection: " << ttrack*1000 << endl;

	mCurrentFrame.SetBoxes(mpDetector->objects);

	//------------------------------------------------------------------------------------

    
    // cout << "Incoming frame ended" << endl;

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;
    

#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
    vdStereoMatch_ms.push_back(mCurrentFrame.mTimeStereoMatch);
#endif

    //cout << "Tracking start" << endl;
    Track();
    //cout << "Tracking end" << endl;

    // 记录匹配质量指标
    LogMatchingQuality(mCurrentFrame.mnId);

// FEATURE NUMBER UPDATE
    //------------------------------------------------------------------------------------------------

	//cout << "FACTOR: " << mpDetector->fig_factor << endl;

	delete mpORBextractorLeft;
	delete mpORBextractorRight;
		
    //------------------------------------------------------------------------------------------------
    return mCurrentFrame.GetPose();
}

Sophus::SE3f Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp, string filename)
{
    // 保持和Detector相同的输入图片和目标检测框
    // FEATURE NUMBER UPDATE
    //------------------------------------------------------------------------------------------------
	
	mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST); 

    //------------------------------------------------------------------------------------------------

    // mImGray = imRGB;
    // cv::Mat imDepth = imD;
    
    mImRGB = imRGB;
    mImGray = imRGB;
    imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    if (mSensor == System::RGBD)
        mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera);
    else if(mSensor == System::IMU_RGBD)
        mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth,mpCamera,&mLastFrame,*mpImuCalib);

 //-------------------------------------------------------------------------------------------------
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    while(!isNewDetectedImgArrived()) 
    {
        usleep(1);
    }
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();
    //cout << "time waiting for detection: " << ttrack*1000 << endl;

	// send detected boxes to Frame
	mCurrentFrame.SetBoxes(mpDetector->objects);

    mCurrentFrame.mNameFile = filename;
    mCurrentFrame.mnDataset = mnNumDataset;
   
#ifdef REGISTER_TIMES
    vdORBExtract_ms.push_back(mCurrentFrame.mTimeORB_Ext);
#endif

    Track();
    // 记录匹配质量指标
    LogMatchingQuality(mCurrentFrame.mnId);

	delete mpORBextractorLeft;
		
    //------------------------------------------------------------------------------------------------

    return mCurrentFrame.GetPose();
}


void Tracking::LogImageQuality(const cv::Mat& img, const std::vector<cv::KeyPoint>& vKeys,
                               int nLevels, long unsigned int frameId)
{
    if (!mbSaveQuality || !mQualityLog.is_open())
        return;

    // 计算并暂存图像质量指标（Track() 后与匹配指标合并写入）
    mPendingQualityMetrics = ComputeAllMetrics(vKeys, img, nLevels);
    mPendingQualityValid = true;
}

void Tracking::LogMatchingQuality(long unsigned int frameId)
{
    if (!mbSaveQuality || !mQualityLog.is_open() || !mPendingQualityValid)
        return;

    Frame& f = mCurrentFrame;

    // 计算匹配指标
    MatchingMetrics mm;
    mm.totalKeypoints = static_cast<int>(f.mvKeysUn.size());

    mm.matchedPoints = 0;
    mm.outlierPoints = 0;
    std::vector<float> matchedResps;
    matchedResps.reserve(mm.totalKeypoints);

    for (int i = 0; i < mm.totalKeypoints; i++) {
        if (f.mvpMapPoints[i]) {
            mm.matchedPoints++;
            if (f.mvbOutlier[i])
                mm.outlierPoints++;
            matchedResps.push_back(f.mvKeysUn[i].response);
        }
    }

    mm.inlierPoints = mm.matchedPoints - mm.outlierPoints;
    mm.matchRatio = (mm.totalKeypoints > 0) ?
        (float)mm.matchedPoints / mm.totalKeypoints : 0.0f;
    mm.inlierRatio = (mm.matchedPoints > 0) ?
        (float)mm.inlierPoints / mm.matchedPoints : 0.0f;

    if (!matchedResps.empty()) {
        std::sort(matchedResps.begin(), matchedResps.end());
        size_t n = matchedResps.size();
        float sum = std::accumulate(matchedResps.begin(), matchedResps.end(), 0.0f);
        mm.matchedRespMean = sum / n;
        mm.matchedRespMedian = (n % 2 == 0) ?
            (matchedResps[n/2 - 1] + matchedResps[n/2]) / 2.0f : matchedResps[n/2];
        float sq_sum = 0;
        for (float r : matchedResps)
            sq_sum += (r - mm.matchedRespMean) * (r - mm.matchedRespMean);
        mm.matchedRespStddev = std::sqrt(sq_sum / n);
    } else {
        mm.matchedRespMean = mm.matchedRespMedian = mm.matchedRespStddev = 0;
    }

    mQualityLog << FormatMetricsLine(frameId, mPendingQualityMetrics, mm) << std::endl;
    mQualityLog.flush();
    mPendingQualityValid = false;
}

void Tracking::GetImgForDetector(const cv::Mat& img)
{
    {
        unique_lock<mutex> lock(mpDetector->mMutexGetNewImg);
        mpDetector->mbNewImgFlag=true;
        // 浅拷贝：仅增加引用计数，避免每帧深拷贝整张图像的开销
        // 安全性保证：Tracking 线程会等待检测完成后才继续处理图像
        mpDetector->mImg = img;
    }
    // 通知检测线程有新图像到达（零延迟唤醒）
    mpDetector->mCvNewImg.notify_one();
}

bool Tracking::isNewDetectedImgArrived()
{
    std::unique_lock <std::mutex> lock(mpDetector->mMutexNewImgDetection);
    if(mbNewDetImgFlag)
    {
        mbNewDetImgFlag=false;
        return true;
    }
    else 
    {
	    return false;
    }
}

// 使用条件变量等待检测完成（替代 usleep 忙等待，零CPU开销）
void Tracking::WaitForDetection()
{
    std::unique_lock<std::mutex> lock(mpDetector->mMutexCvDetDone);
    mpDetector->mCvDetDone.wait(lock, [this]{ return mbNewDetImgFlag; });
    mbNewDetImgFlag = false;
}

/**
 * 单目图像跟踪处理函数
 * @param im 输入的单目图像
 * @param timestamp 时间戳
 * @param filename 文件名
 * @return 相机位姿 (SE3变换矩阵)
 */
Sophus::SE3f Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp, string filename)
{
    // 特征点提取器初始化
    // ------------------------------------------------------------------------------------------------
    // 根据系统状态选择不同的特征点提取器
    // 未初始化或没有图像时使用5倍特征点的初始化提取器
    // 正常运行时使用标准特征点提取器

	// if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
    // {
    //     mpIniORBextractor = new ORBextractor(5*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);
    // }
    // else
    // {
    //     mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST); 
    // }

    
    // 图像预处理：转换为灰度图
    mImGray = im;
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGB2GRAY);  // RGB转灰度
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGR2GRAY);  // BGR转灰度
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,cv::COLOR_RGBA2GRAY); // RGBA转灰度
        else
            cvtColor(mImGray,mImGray,cv::COLOR_BGRA2GRAY); // BGRA转灰度
    }

    // 创建当前帧对象
    std::chrono::steady_clock::time_point tOrbStart;
    if (gEnableTimingStats) tOrbStart = std::chrono::steady_clock::now();
    if (mSensor == System::MONOCULAR)  // 单目相机模式
    {
        if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET ||(lastID - initID) < mMaxFrames)
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
        else
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth);
    }

    else if(mSensor == System::IMU_MONOCULAR)  // 单目+IMU模式
    {
        if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        {
            mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib);
        }
        else
            mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mpCamera,mDistCoef,mbf,mThDepth,&mLastFrame,*mpImuCalib);
    }
    // 记录 ORB 特征提取耗时（Frame 构造内部完成特征提取）
    if (gEnableTimingStats)
    {
        auto tOrbEnd = std::chrono::steady_clock::now();
        mdCurOrbExtractMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(tOrbEnd - tOrbStart).count();
    }

    // 记录图像质量指标（每10帧采样一次，降低CPU开销）
    {
        static int qualityCounter = 0;
        if (++qualityCounter % 10 == 0)
            LogImageQuality(mImGray, mCurrentFrame.mvKeysUn, nLevels, mCurrentFrame.mnId);
    }
    
    //-------------------------------------------------------------------------
    // 等待目标检测完成（使用条件变量，零CPU开销，唤醒延迟仅几微秒）
    std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
    WaitForDetection();
    std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
    // 记录等待目标检测耗时（单目流水线的关键阻塞点）
    if (gEnableTimingStats)
        mdCurWaitDetectMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t4 - t3).count();

	// 将检测结果传递给当前帧
    mCurrentFrame.SetBoxes(mpDetector->objects);   // 设置静态物体检测框
	
//--------------------------------------------------------------------------
    // 系统初始化相关设置
    if (mState==NO_IMAGES_YET)
    {
        t0=timestamp;  // 记录第一帧的时间戳
    }

    // 设置帧的附加信息
    mCurrentFrame.mNameFile = filename;    // 文件名
    mCurrentFrame.mnDataset = mnNumDataset; // 数据集编号

    lastID = mCurrentFrame.mnId;  // 更新最后处理的帧ID

    Track();  // 执行主要的跟踪算法

    // 记录匹配质量指标
    LogMatchingQuality(mCurrentFrame.mnId);

    return mCurrentFrame.GetPose();

}
void Tracking::GrabImuData(const IMU::Point &imuMeasurement)
{
    unique_lock<mutex> lock(mMutexImuQueue);
    mlQueueImuData.push_back(imuMeasurement);
}

void Tracking::PreintegrateIMU()
{

    if(!mCurrentFrame.mpPrevFrame)
    {
        Verbose::PrintMess("non prev frame ", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated();
        return;
    }

    mvImuFromLastFrame.clear();
    mvImuFromLastFrame.reserve(mlQueueImuData.size());
    if(mlQueueImuData.size() == 0)
    {
        Verbose::PrintMess("Not IMU data in mlQueueImuData!!", Verbose::VERBOSITY_NORMAL);
        mCurrentFrame.setIntegrated();
        return;
    }

    while(true)
    {
        bool bSleep = false;
        {
            unique_lock<mutex> lock(mMutexImuQueue);
            if(!mlQueueImuData.empty())
            {
                IMU::Point* m = &mlQueueImuData.front();
                cout.precision(17);
                if(m->t<mCurrentFrame.mpPrevFrame->mTimeStamp-mImuPer)
                {
                    mlQueueImuData.pop_front();
                }
                else if(m->t<mCurrentFrame.mTimeStamp-mImuPer)
                {
                    mvImuFromLastFrame.push_back(*m);
                    mlQueueImuData.pop_front();
                }
                else
                {
                    mvImuFromLastFrame.push_back(*m);
                    break;
                }
            }
            else
            {
                break;
                bSleep = true;
            }
        }
        if(bSleep)
            usleep(500);
    }

    const int n = mvImuFromLastFrame.size()-1;
    if(n==0){
        cout << "Empty IMU measurements vector!!!\n";
        return;
    }

    IMU::Preintegrated* pImuPreintegratedFromLastFrame = new IMU::Preintegrated(mLastFrame.mImuBias,mCurrentFrame.mImuCalib);

    for(int i=0; i<n; i++)
    {
        float tstep;
        Eigen::Vector3f acc, angVel;
        if((i==0) && (i<(n-1)))
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tini = mvImuFromLastFrame[i].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tini/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tini/tab))*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mCurrentFrame.mpPrevFrame->mTimeStamp;
        }
        else if(i<(n-1))
        {
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a)*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w)*0.5f;
            tstep = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
        }
        else if((i>0) && (i==(n-1)))
        {
            float tab = mvImuFromLastFrame[i+1].t-mvImuFromLastFrame[i].t;
            float tend = mvImuFromLastFrame[i+1].t-mCurrentFrame.mTimeStamp;
            acc = (mvImuFromLastFrame[i].a+mvImuFromLastFrame[i+1].a-
                    (mvImuFromLastFrame[i+1].a-mvImuFromLastFrame[i].a)*(tend/tab))*0.5f;
            angVel = (mvImuFromLastFrame[i].w+mvImuFromLastFrame[i+1].w-
                    (mvImuFromLastFrame[i+1].w-mvImuFromLastFrame[i].w)*(tend/tab))*0.5f;
            tstep = mCurrentFrame.mTimeStamp-mvImuFromLastFrame[i].t;
        }
        else if((i==0) && (i==(n-1)))
        {
            acc = mvImuFromLastFrame[i].a;
            angVel = mvImuFromLastFrame[i].w;
            tstep = mCurrentFrame.mTimeStamp-mCurrentFrame.mpPrevFrame->mTimeStamp;
        }

        if (!mpImuPreintegratedFromLastKF)
            cout << "mpImuPreintegratedFromLastKF does not exist" << endl;
        mpImuPreintegratedFromLastKF->IntegrateNewMeasurement(acc,angVel,tstep);
        pImuPreintegratedFromLastFrame->IntegrateNewMeasurement(acc,angVel,tstep);
    }

    mCurrentFrame.mpImuPreintegratedFrame = pImuPreintegratedFromLastFrame;
    mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
    mCurrentFrame.mpLastKeyFrame = mpLastKeyFrame;

    mCurrentFrame.setIntegrated();

    //Verbose::PrintMess("Preintegration is finished!! ", Verbose::VERBOSITY_DEBUG);
}

bool Tracking::PredictStateIMU()
{
    if(!mCurrentFrame.mpPrevFrame)
    {
        Verbose::PrintMess("No last frame", Verbose::VERBOSITY_NORMAL);
        return false;
    }

    if(mbMapUpdated && mpLastKeyFrame)
    {
        const Eigen::Vector3f twb1 = mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mpLastKeyFrame->GetVelocity();

        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float t12 = mpImuPreintegratedFromLastKF->dT;

        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaRotation(mpLastKeyFrame->GetImuBias()));
        Eigen::Vector3f twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mpImuPreintegratedFromLastKF->GetDeltaPosition(mpLastKeyFrame->GetImuBias());
        Eigen::Vector3f Vwb2 = Vwb1 + t12*Gz + Rwb1 * mpImuPreintegratedFromLastKF->GetDeltaVelocity(mpLastKeyFrame->GetImuBias());
        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2);

        mCurrentFrame.mImuBias = mpLastKeyFrame->GetImuBias();
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    }
    else if(!mbMapUpdated)
    {
        const Eigen::Vector3f twb1 = mLastFrame.GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame.GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.GetVelocity();
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const float t12 = mCurrentFrame.mpImuPreintegratedFrame->dT;

        Eigen::Matrix3f Rwb2 = IMU::NormalizeRotation(Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaRotation(mLastFrame.mImuBias));
        Eigen::Vector3f twb2 = twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaPosition(mLastFrame.mImuBias);
        Eigen::Vector3f Vwb2 = Vwb1 + t12*Gz + Rwb1 * mCurrentFrame.mpImuPreintegratedFrame->GetDeltaVelocity(mLastFrame.mImuBias);

        mCurrentFrame.SetImuPoseVelocity(Rwb2,twb2,Vwb2);

        mCurrentFrame.mImuBias = mLastFrame.mImuBias;
        mCurrentFrame.mPredBias = mCurrentFrame.mImuBias;
        return true;
    }
    else
        cout << "not IMU prediction!!" << endl;

    return false;
}

void Tracking::ResetFrameIMU()
{
    // TODO To implement...
}

void Tracking::UpdateFrameStatistics()
{
    auto frameEndTime = std::chrono::steady_clock::now();
    double frameTimeMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(frameEndTime - mFrameStartTime).count();

    // 初始化阶段（首次进入 OK 之前）耗时明显偏慢，不计入平均统计。
    // 待跟踪首次达到 OK 后，从下一帧开始正式计时。
    if (!mbFrameStatsStarted)
    {
        if (mState == OK)
        {
            mbFrameStatsStarted = true;
            // 以初始化完成时刻为统计起点，使平均FPS/总时长不含初始化阶段
            mTrackingStartTime = frameEndTime;
            mdLastFpsUpdateTime = 0.0;
            mnFramesSinceLastFpsUpdate = 0;
        }
        return;   // 初始化阶段（含完成初始化的那一帧）不计入统计
    }

    mnTotalFrames++;
    mdTotalProcessingTime += frameTimeMs;
    mvFrameTimes.push_back(frameTimeMs);

    if (frameTimeMs > mdMaxFrameTime)
        mdMaxFrameTime = frameTimeMs;
    if (frameTimeMs < mdMinFrameTime)
        mdMinFrameTime = frameTimeMs;

    // 分阶段耗时汇总：frameTimeMs 即 Track() 核心耗时（mFrameStartTime 在 Track() 开头打点）
    if (gEnableTimingStats)
    {
        mdStageTrack         += frameTimeMs;
        mdStageOrbExtract    += mdCurOrbExtractMs;
        mdStageWaitDetect    += mdCurWaitDetectMs;
        mdStageTotalPipeline += (mdCurOrbExtractMs + mdCurWaitDetectMs + frameTimeMs);
        if (mdCurWaitDetectMs > mdMaxWaitDetect)
            mdMaxWaitDetect = mdCurWaitDetectMs;
    }

    mnFramesSinceLastFpsUpdate++;
    double currentTime = std::chrono::duration_cast<std::chrono::duration<double>>(frameEndTime - mTrackingStartTime).count();
    
    // if (currentTime - mdLastFpsUpdateTime >= 1.0)
    // {
    //     mdCurrentFps = mnFramesSinceLastFpsUpdate / (currentTime - mdLastFpsUpdateTime);
    //     mdLastFpsUpdateTime = currentTime;
    //     mnFramesSinceLastFpsUpdate = 0;

    //     std::cout << std::fixed << std::setprecision(2);
    //     std::cout << "[FPS] Current: " << mdCurrentFps << " fps, ";
    //     std::cout << "Avg: " << (mnTotalFrames / currentTime) << " fps, ";
    //     std::cout << "Frame Time: " << frameTimeMs << " ms" << std::endl;
    // }
}

// 打印帧处理统计信息
void Tracking::PrintFrameStatistics()
{
    if (mnTotalFrames == 0)
    {
        std::cout << "[FPS] No frames processed yet." << std::endl;
        return;
    }

    auto currentTime = std::chrono::steady_clock::now();
    double totalTimeSec = std::chrono::duration_cast<std::chrono::duration<double>>(currentTime - mTrackingStartTime).count();
    double avgFrameTime = mdTotalProcessingTime / mnTotalFrames;

    std::cout << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "          FRAME PROCESSING STATS       " << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Total frames processed: " << mnTotalFrames << std::endl;
    std::cout << "Total processing time:  " << totalTimeSec << " s" << std::endl;
    std::cout << "Average FPS:           " << (mnTotalFrames / totalTimeSec) << " fps" << std::endl;
    std::cout << "Average frame time:     " << avgFrameTime << " ms" << std::endl;
    std::cout << "Max frame time:         " << mdMaxFrameTime << " ms" << std::endl;
    std::cout << "Min frame time:         " << mdMinFrameTime << " ms" << std::endl;

    if (mvFrameTimes.size() > 0)
    {
        double variance = 0.0;
        for (double t : mvFrameTimes)
            variance += std::pow(t - avgFrameTime, 2);
        variance /= mvFrameTimes.size();
        double stdDev = std::sqrt(variance);
        std::cout << "Frame time std dev:     " << stdDev << " ms" << std::endl;
    }
    std::cout << "======================================" << std::endl;

    // 分阶段精准耗时（仅在开启统计时输出），用于定位真实瓶颈
    if (gEnableTimingStats)
    {
        double avgOrb   = mdStageOrbExtract    / mnTotalFrames;
        double avgWait  = mdStageWaitDetect    / mnTotalFrames;
        double avgTrack = mdStageTrack         / mnTotalFrames;
        double avgTotal = mdStageTotalPipeline / mnTotalFrames;

        std::cout << std::endl;
        std::cout << "======================================" << std::endl;
        std::cout << "     PER-STAGE TIMING (per frame)     " << std::endl;
        std::cout << "======================================" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "ORB extract:      " << avgOrb   << " ms  (" << (avgTotal>0? 100.0*avgOrb/avgTotal:0.0)   << "%)" << std::endl;
        std::cout << "Wait detection:   " << avgWait  << " ms  (" << (avgTotal>0? 100.0*avgWait/avgTotal:0.0)  << "%)   max=" << mdMaxWaitDetect << " ms" << std::endl;
        std::cout << "Track() core:     " << avgTrack << " ms  (" << (avgTotal>0? 100.0*avgTrack/avgTotal:0.0) << "%)" << std::endl;
        std::cout << "--------------------------------------" << std::endl;
        std::cout << "Full pipeline:    " << avgTotal << " ms  ->  " << (avgTotal>0? 1000.0/avgTotal:0.0) << " fps (compute limit)" << std::endl;
        std::cout << "======================================" << std::endl;
    }
}

void Tracking::SaveFrameStatisticsToFile()
{
    if (mnTotalFrames == 0)
        return;

    std::ofstream f("FrameStats.txt");
    if (!f.is_open())
        return;

    auto currentTime = std::chrono::steady_clock::now();
    double totalTimeSec = std::chrono::duration_cast<std::chrono::duration<double>>(currentTime - mTrackingStartTime).count();
    double avgFrameTime = mdTotalProcessingTime / mnTotalFrames;

    f << std::fixed << std::setprecision(6);
    f << "# Frame Processing Statistics" << std::endl;
    f << "# Total frames: " << mnTotalFrames << std::endl;
    f << "# Total time (s): " << totalTimeSec << std::endl;
    f << "# Average FPS: " << (mnTotalFrames / totalTimeSec) << std::endl;
    f << "# Average frame time (ms): " << avgFrameTime << std::endl;
    f << "# Max frame time (ms): " << mdMaxFrameTime << std::endl;
    f << "# Min frame time (ms): " << mdMinFrameTime << std::endl;

    if (mvFrameTimes.size() > 0)
    {
        double variance = 0.0;
        for (double t : mvFrameTimes)
            variance += std::pow(t - avgFrameTime, 2);
        variance /= mvFrameTimes.size();
        double stdDev = std::sqrt(variance);
        f << "# Frame time std dev (ms): " << stdDev << std::endl;
    }

    f << std::endl;
    f << "# Frame times (ms)" << std::endl;
    for (size_t i = 0; i < mvFrameTimes.size(); i++)
    {
        f << i << "," << mvFrameTimes[i] << std::endl;
    }

    f.close();
}

void Tracking::Track()
{
    mFrameStartTime = std::chrono::steady_clock::now();

    // 若LocalMapping已执行尺度回拉，同步当前/上一帧位姿（保证与地图尺度一致）
    SyncCurrentFrameScale();

    // 步进模式处理：如果启用步进模式，等待用户触发下一步
    if (bStepByStep)
    {
        std::cout << "Tracking: Waiting to the next step" << std::endl;
        // 等待步进信号，避免CPU过度占用
        while(!mbStep && bStepByStep)
            usleep(500);
        mbStep = false; // 重置步进标志
    }

    // 检查IMU数据质量：如果局部建图器检测到IMU数据异常，重置地图
    if(mpLocalMapper->mbBadImu)
    {
        cout << "[ERROR] Tracking: 局部建图器检测到IMU数据异常，重置地图" << endl;
        mpSystem->ResetActiveMap(); // 重置当前活动地图
        return; // 直接返回，不进行后续跟踪
    }

    // 获取当前活动地图并检查有效性
    Map* pCurrentMap = mpAtlas->GetCurrentMap();
    if(!pCurrentMap)
    {
        // cout << "[ERROR] Tracking: 无活动地图，无法跟踪" << endl;
        // 注意：这里没有返回，因为可能需要在无地图情况下继续处理
    }

    // 时间戳一致性检查：确保帧序列的时间戳是递增的
    if(mState!=NO_IMAGES_YET)
    {
        // 检查时间戳异常：当前帧时间戳小于上一帧（时间倒流）
        if(mLastFrame.mTimeStamp>mCurrentFrame.mTimeStamp)
        {
            cout << "[ERROR] Tracking: 检测到时间戳异常，当前帧时间戳早于上一帧" << endl;
            unique_lock<mutex> lock(mMutexImuQueue); // 锁定IMU队列
            mlQueueImuData.clear(); // 清空IMU数据队列
            CreateMapInAtlas(); // 创建新地图
            return; // 返回，重新开始跟踪
        }
        // 检查时间戳跳跃：当前帧与上一帧时间差超过1秒
        else if(mCurrentFrame.mTimeStamp>mLastFrame.mTimeStamp+1.0)
        {
            // 调试信息（已注释）
            // cout << mCurrentFrame.mTimeStamp << ", " << mLastFrame.mTimeStamp << endl;
            // cout << "id last: " << mLastFrame.mnId << "    id curr: " << mCurrentFrame.mnId << endl;
            
            // 仅对惯性系统处理时间戳跳跃
            if(mpAtlas->isInertial())
            {
                // IMU已初始化情况下的处理
                if(mpAtlas->isImuInitialized())
                {
                    cout << "Timestamp jump detected. State set to LOST. Reseting IMU integration..." << endl;
                    // 根据是否完成惯性BA2来决定重置策略
                    if(!pCurrentMap->GetIniertialBA2())
                    {
                        mpSystem->ResetActiveMap(); // 重置活动地图
                    }
                    else
                    {
                        CreateMapInAtlas(); // 创建新地图
                    }
                }
                // IMU未初始化情况下的处理
                else
                {
                    cout << "Timestamp jump detected, before IMU initialization. Reseting..." << endl;
                    mpSystem->ResetActiveMap(); // 直接重置活动地图
                }
                return; // 返回，重新开始跟踪
            }

        }
    }

    // 惯性传感器处理：如果使用IMU传感器且有上一关键帧，设置当前帧的IMU偏置
    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpLastKeyFrame)
        mCurrentFrame.SetNewBias(mpLastKeyFrame->GetImuBias());

    // 状态转换：如果尚未接收到图像，将状态从未接收图像转为未初始化
    if(mState==NO_IMAGES_YET)
    {
        mState = NOT_INITIALIZED;
    }

    // 记录上一次处理的状态，用于状态变化检测
    mLastProcessedState=mState;

    // IMU预积分：对于IMU传感器且地图未创建的情况，进行IMU数据预积分
    if ((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mbCreatedMap)
    {
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartPreIMU = std::chrono::steady_clock::now();
#endif
        PreintegrateIMU(); // 执行IMU预积分计算
#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_EndPreIMU = std::chrono::steady_clock::now();

        double timePreImu = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPreIMU - time_StartPreIMU).count();
        vdIMUInteg_ms.push_back(timePreImu); // 记录IMU预积分耗时
#endif

    }
    mbCreatedMap = false; // 重置地图创建标志

    // 获取地图互斥锁 -> 防止地图在跟踪过程中被修改
    unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);

    mbMapUpdated = false; // 初始化地图更新标志

    // 检查地图是否发生变化：比较当前地图变化索引和上次记录的变化索引
    int nCurMapChangeIndex = pCurrentMap->GetMapChangeIndex();
    int nMapChangeIndex = pCurrentMap->GetLastMapChange();
    
    if(nCurMapChangeIndex>nMapChangeIndex)
    {
        pCurrentMap->SetLastMapChange(nCurMapChangeIndex); // 更新最后地图变化索引
        mbMapUpdated = true; // 设置地图更新标志
    }
    // 系统初始化阶段处理
    // 调用条件：系统状态为未初始化，且未创建地图
    if(mState==NOT_INITIALIZED)
    {
        // 根据传感器类型选择不同的初始化方法
        if(mSensor==System::STEREO || mSensor==System::RGBD || mSensor==System::IMU_STEREO || mSensor==System::IMU_RGBD)
        {
            StereoInitialization(); // 双目或RGBD传感器初始化
        }
        else
        {
            MonocularInitialization(); // 单目传感器初始化
        }

        // 帧绘制器更新（已注释）
        mpFrameDrawer->Update(this);

        // 检查初始化是否成功：成功初始化后状态应为OK
        if(mState!=OK) // If rightly initialized, mState=OK
        {
            // std::cout << "[ERROR] Tracking: 初始化失败，状态: " << mState << std::endl;
            mLastFrame = Frame(mCurrentFrame); // 保存当前帧到上一帧
            return; // 初始化失败，直接返回
        }

        // 如果是第一个地图，记录第一帧的ID
        if(mpAtlas->GetAllMaps().size() == 1)
        {
            mnFirstFrameId = mCurrentFrame.mnId;
        }
    }
    else
    {
        // 系统已初始化，开始帧跟踪
        bool bOK; // 跟踪结果标志

#ifdef REGISTER_TIMES
        std::chrono::steady_clock::time_point time_StartPosePred = std::chrono::steady_clock::now();
#endif

        // 初始相机位姿估计：使用运动模型或重定位（如果跟踪丢失）
        // 正常SLAM模式：同时进行跟踪和建图
        // 局部建图激活：会创建新的关键帧和地图点
        // 地图更新：地图会随着时间推移不断扩展和优化
        // 应用场景：需要构建完整地图的应用
        if(!mbOnlyTracking) // 非纯跟踪模式（正常模式，包含局部建图）
        {
            // 状态OK：正常跟踪状态
            // 局部建图已激活。这是正常行为，除非显式激活"仅跟踪"模式
            if(mState==OK)
            {
                // 检查上一帧中被替换的地图点：局部建图可能修改了上一帧跟踪的地图点
                CheckReplacedInLastFrame();

                // 跟踪策略选择：根据运动模型可用性和重定位历史选择跟踪方法
                if((!mbVelocity && !pCurrentMap->isImuInitialized()) || mCurrentFrame.mnId<mnLastRelocFrameId+2)
                {
                    // 条件：无速度信息且IMU未初始化，或刚完成重定位（2帧内）
                    Verbose::PrintMess("TRACK: Track with respect to the reference KF ", Verbose::VERBOSITY_DEBUG);
                    bOK = TrackReferenceKeyFrame(); // 基于参考关键帧的跟踪
                }
                else
                {
                    // 正常情况：使用运动模型进行跟踪
                    Verbose::PrintMess("TRACK: Track with motion model", Verbose::VERBOSITY_DEBUG);
                    bOK = TrackWithMotionModel(); // 基于运动模型的跟踪
                    if(!bOK)
                        bOK = TrackReferenceKeyFrame(); // 运动模型失败时回退到参考关键帧跟踪
                }
                // 跟踪失败处理：根据失败情况设置不同的丢失状态
                if (!bOK)
                {
                    if ( mCurrentFrame.mnId<=(mnLastRelocFrameId+mnFramesToResetIMU) &&
                         (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD))
                    {
                        // 条件：刚完成重定位且在IMU重置帧数内，且使用IMU传感器
                        mState = LOST; // 直接标记为完全丢失
                    }
                    else if(pCurrentMap->KeyFramesInMap()>10)
                    {
                        // 条件：地图中有足够的关键帧（>10）
                        // cout << "KF in map: " << pCurrentMap->KeyFramesInMap() << endl;
                        mState = RECENTLY_LOST; // 标记为最近丢失（可恢复状态）
                        mTimeStampLost = mCurrentFrame.mTimeStamp; // 记录丢失时间戳
                    }
                    else
                    {
                        // 其他情况：地图中关键帧不足
                        mState = LOST; // 标记为完全丢失
                    }
                }
            }
            else // 状态不是OK的情况
            {
                // 最近丢失状态处理：跟踪丢失但可能恢复
                if (mState == RECENTLY_LOST) //跟踪丢失
                {
                    Verbose::PrintMess("Lost for a short time", Verbose::VERBOSITY_NORMAL);

                    bOK = true; // 假设可以恢复
                    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))
                    {
                        // IMU传感器处理：使用IMU预测状态
                        if(pCurrentMap->isImuInitialized())
                            PredictStateIMU(); // IMU已初始化，预测状态
                        else
                            bOK = false; // IMU未初始化，无法恢复

                        // 检查丢失时间是否超过阈值（5秒）
                        if (mCurrentFrame.mTimeStamp-mTimeStampLost>time_recently_lost)
                        {
                            mState = LOST; // 超过时间阈值，标记为完全丢失
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK=false;
                        }
                    }
                    else
                    {
                        // 非IMU传感器：尝试重定位
                        bOK = Relocalization(); // 执行重定位
                        //std::cout << "mCurrentFrame.mTimeStamp:" << to_string(mCurrentFrame.mTimeStamp) << std::endl;
                        //std::cout << "mTimeStampLost:" << to_string(mTimeStampLost) << std::endl;
                        // 振动自适应：振动大时延长恢复窗口（基础3秒，振动大时最长6秒）
                        float lostTimeout = 3.0f + std::min(mfPrevVibrationLevel, 2.0f) * 1.5f;
                        if(mCurrentFrame.mTimeStamp-mTimeStampLost>lostTimeout && !bOK)
                        {
                            // 重定位失败且超过时间阈值，标记为完全丢失
                            mState = LOST;
                            Verbose::PrintMess("Track Lost...", Verbose::VERBOSITY_NORMAL);
                            bOK=false;
                        }
                    }
                }
                else if (mState == LOST)
                {
                    // 完全丢失状态处理：需要创建新地图
                    Verbose::PrintMess("A new map is started...", Verbose::VERBOSITY_NORMAL);

                    // 根据当前地图中关键帧数量决定处理策略
                    if (pCurrentMap->KeyFramesInMap()<10)
                    {
                        // 关键帧不足：重置当前地图
                        mpSystem->ResetActiveMap();
                        Verbose::PrintMess("Reseting current map...", Verbose::VERBOSITY_NORMAL);
                    }else
                        // 关键帧足够：在地图集中创建新地图
                        CreateMapInAtlas();

                    // 清空上一关键帧指针
                    if(mpLastKeyFrame)
                        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);

                    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

                    return; // 直接返回，重新开始跟踪流程
                }
            }

        }
        else
        {
            // 纯跟踪模式：局部建图被禁用（TODO 惯性模式下不可用）
            // 仅定位模式：只进行相机位姿跟踪，不建图
            // 局部建图停用：不会创建新的关键帧和地图点
            // 地图固定：使用已有的地图进行定位
            // 应用场景：在已有地图上进行导航或定位
            if(mState==LOST)
            {
                // 丢失状态处理：仅尝试重定位
                if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                {
                    Verbose::PrintMess("IMU. State LOST", Verbose::VERBOSITY_NORMAL);
                }
                bOK = Relocalization(); // 执行重定位
            }
            else
            {
                // 非丢失状态：根据是否处于视觉里程计模式选择跟踪策略
                if(!mbVO)
                {
                    // 有速度信息：使用运动模型跟踪
                    if(mbVelocity)
                    {
                        bOK = TrackWithMotionModel(); // 使用运动模型跟踪
                    }
                    else
                    {
                        bOK = TrackReferenceKeyFrame(); // 使用参考关键帧跟踪
                    }
                }
                else
                {
                    // 视觉里程计模式：上一帧主要跟踪"视觉里程计"点（临时地图点）
                    // 计算两个相机位姿：一个来自运动模型，一个来自重定位
                    // 如果重定位成功，选择重定位解；否则保留"视觉里程计"解
                    bool bOKMM = false; // 运动模型跟踪结果
                    bool bOKReloc = false; // 重定位结果
                    vector<MapPoint*> vpMPsMM; // 运动模型跟踪的地图点
                    vector<bool> vbOutMM; // 运动模型跟踪的外点标记
                    Sophus::SE3f TcwMM; // 运动模型计算的位姿
                    
                    // 如果有速度信息，尝试运动模型跟踪
                    if(mbVelocity)
                    {
                        bOKMM = TrackWithMotionModel();
                        vpMPsMM = mCurrentFrame.mvpMapPoints; // 保存跟踪结果
                        vbOutMM = mCurrentFrame.mvbOutlier;
                        TcwMM = mCurrentFrame.GetPose();
                    }
                    
                    bOKReloc = Relocalization(); // 同时尝试重定位

                    // 结果融合策略：优先选择重定位结果
                    // std::cout << "[DEBUG] Tracking: 结果融合 - bOKMM: " << bOKMM << ", bOKReloc: " << bOKReloc << std::endl;
                    if(bOKMM && !bOKReloc)
                    {
                        // 运动模型成功但重定位失败：使用运动模型结果
                        mCurrentFrame.SetPose(TcwMM);
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        // 如果仍处于视觉里程计模式，增加地图点的发现次数
                        if(mbVO)
                        {
                            for(int i =0; i<mCurrentFrame.N; i++)
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                {
                                    // 添加安全检查
                                    if(mCurrentFrame.mvpMapPoints[i] != nullptr)
                                    {
                                        mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                    }
                                    else
                                    {
                                        std::cout << "[WARNING] Tracking: 无效的地图点指针，索引: " << i << std::endl;
                                    }
                                }
                            }
                        }
                    }
                    else if(bOKReloc)
                    {
                        // 重定位成功：退出视觉里程计模式，切换到正常跟踪
                        mbVO = false;
                    }
                    else
                    {
                        // std::cout << "[ERROR] Tracking: 两种方法都失败" << std::endl;
                    }

                    // 最终结果：重定位或运动模型任一成功即可
                    bOK = bOKReloc || bOKMM;
                }
            }
        }

        // 确保当前帧有参考关键帧
        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

#ifdef REGISTER_TIMES
        // 记录位姿预测耗时统计
        std::chrono::steady_clock::time_point time_EndPosePred = std::chrono::steady_clock::now();

        double timePosePred = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndPosePred - time_StartPosePred).count();
        vdPosePred_ms.push_back(timePosePred); // 存储到位姿预测耗时向量
#endif
#ifdef REGISTER_TIMES
        // 开始记录局部地图跟踪耗时
        std::chrono::steady_clock::time_point time_StartLMTrack = std::chrono::steady_clock::now();
#endif
        // 如果有相机位姿和匹配的初始估计，跟踪局部地图
        if(!mbOnlyTracking) // 正常模式（包含局部建图）
        {
            if(bOK) // 如果初始跟踪成功
            {
                bOK = TrackLocalMap(); // 执行局部地图跟踪
            }
            if(!bOK)
            {
                cout << "[ERROR] Tracking::Track: Fail to track local map!" << endl; // 局部地图跟踪失败
                std::cout << "[DEBUG] Tracking::Track: 局部地图跟踪失败，当前状态: " << mState << std::endl;
                std::cout << "[DEBUG] Tracking::Track: 当前帧ID: " << mCurrentFrame.mnId 
                          << ", 地图关键帧数量: " << (pCurrentMap ? pCurrentMap->KeyFramesInMap() : 0) << std::endl;
            }
        }
        else // 纯定位模式
        {
            // mbVO为true表示与地图中地图点的匹配较少。我们无法检索局部地图，
            // 因此不执行TrackLocalMap()。一旦系统重定位相机，我们将再次使用局部地图。
            if(bOK && !mbVO) // 跟踪成功且不处于视觉里程计模式
                bOK = TrackLocalMap(); // 执行局部地图跟踪
        }

        // 更新跟踪状态：根据跟踪结果设置状态
        if(bOK)
            mState = OK; // 跟踪成功，状态保持OK
        else if (mState == OK) // 当前状态为OK但跟踪失败
        {
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            {
                // IMU传感器处理：跟踪丢失但时间小于1秒
                Verbose::PrintMess("Track lost for less than one second...", Verbose::VERBOSITY_NORMAL);
                if(!pCurrentMap->isImuInitialized() || !pCurrentMap->GetIniertialBA2())
                {
                    // IMU未初始化或最近初始化：重置活动地图
                    cout << "IMU is not or recently initialized. Reseting active map..." << endl;
                    mpSystem->ResetActiveMap();
                }

                mState=RECENTLY_LOST; // 设置为最近丢失状态
            }
            else
                mState=RECENTLY_LOST; // 视觉传感器：从正常状态转为丢失状态

            // 记录丢失时间戳（注释掉的代码是更复杂的条件判断）
            /*if(mCurrentFrame.mnId>mnLastRelocFrameId+mMaxFrames)
            {*/
                mTimeStampLost = mCurrentFrame.mTimeStamp;
            //}
        }

        // 如果最近进行了重定位，保存帧（用于IMU重置，因为我们在复制，应该在mCurrFrame完全修改后执行）
        if((mCurrentFrame.mnId<(mnLastRelocFrameId+mnFramesToResetIMU)) && (mCurrentFrame.mnId > mnFramesToResetIMU) &&
           (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && pCurrentMap->isImuInitialized())
        {
            // TODO 检查这种情况
            Verbose::PrintMess("Saving pointer to frame. imu needs reset...", Verbose::VERBOSITY_NORMAL);
            Frame* pF = new Frame(mCurrentFrame); // 创建当前帧的副本
            pF->mpPrevFrame = new Frame(mLastFrame); // 创建上一帧的副本

            // 加载预积分数据
            pF->mpImuPreintegratedFrame = new IMU::Preintegrated(mCurrentFrame.mpImuPreintegratedFrame);
        }

        // IMU初始化后的特殊处理
        if(pCurrentMap->isImuInitialized())
        {
            if(bOK) // 跟踪成功
            {
                if(mCurrentFrame.mnId==(mnLastRelocFrameId+mnFramesToResetIMU))
                {
                    // 达到重置帧数：执行IMU帧重置
                    cout << "RESETING FRAME!!!" << endl;
                    ResetFrameIMU();
                }
                else if(mCurrentFrame.mnId>(mnLastRelocFrameId+30))
                    // 超过30帧：更新IMU偏置
                    mLastBias = mCurrentFrame.mImuBias;
            }
        }

#ifdef REGISTER_TIMES
        // 记录局部地图跟踪耗时统计
        std::chrono::steady_clock::time_point time_EndLMTrack = std::chrono::steady_clock::now();

        double timeLMTrack = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndLMTrack - time_StartLMTrack).count();
        vdLMTrack_ms.push_back(timeLMTrack); // 存储到局部地图跟踪耗时向量
#endif

        // ── 检测框绘制 + 图像更新 ──
        // Viewer 模式：在 Pangolin 窗口中绘制；Qt 模式：仅更新 mImColor 供 FrameDrawer 使用
        if (mbShowDynamicVis)
        {
            // 动态一致性模式：VisualizeSemanticPoints 已写入 mImColor，不需要额外绘制
        }
        else
        {
            if (mpViewer && !mpViewer->isStopped())
            {
                mpDetector->draw(mpDetector->mImg, mpDetector->objects);
                DrawDynamicSemanticPoints();
            }

            // mImColor 赋值移出 Viewer 检查：Qt/ROS 模式下 Viewer 不运行，但仍需要
            // 将检测结果传递给 FrameDrawer 以生成带标注的彩色图像
            if (this->mpDetector != nullptr && !this->mpDetector->mImg.empty())
            {
                this->mImColor = this->mpDetector->mImg;
            }
        }
        mpFrameDrawer->Update(this); // 更新帧绘制器
        if(mCurrentFrame.isSet())
            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose()); // 设置地图绘制器的当前相机位姿
        // 跟踪成功或处于最近丢失状态时，进行后续处理
        if(bOK || mState==RECENTLY_LOST)
        {
            // 更新运动模型：计算当前帧与上一帧之间的相对运动
            if(mLastFrame.isSet() && mCurrentFrame.isSet())
            {
                Sophus::SE3f LastTwc = mLastFrame.GetPose().inverse(); // 上一帧的逆位姿（世界到相机）
                mVelocity = mCurrentFrame.GetPose() * LastTwc; // 计算速度（当前帧到上一帧的变换）
                mbVelocity = true; // 标记速度信息可用
            }
            else {
                mbVelocity = false; // 帧未设置，速度信息不可用
            }

            // 对于IMU传感器，确保地图绘制器有当前相机位姿
            if(mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

            // 清理视觉里程计匹配：移除观测次数不足的地图点
            for(int i=0; i<mCurrentFrame.N; i++)
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                    if(pMP->Observations()<1) // 地图点被观测次数少于1次
                    {
                        mCurrentFrame.mvbOutlier[i] = false; // 重置外点标记
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL); // 移除地图点引用
                    }
            }

            // 删除临时地图点：释放临时创建的地图点内存
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit;
                delete pMP; // 释放地图点内存
            }
            mlpTemporalPoints.clear(); // 清空临时地图点列表

#ifdef REGISTER_TIMES
            // 开始记录新关键帧创建耗时
            std::chrono::steady_clock::time_point time_StartNewKF = std::chrono::steady_clock::now();
#endif
            bool bNeedKF = NeedNewKeyFrame(); // 检查是否需要创建新关键帧

            // 检查是否需要插入新关键帧
            // if(bNeedKF && bOK) // 原始条件：需要关键帧且跟踪成功
            // 扩展条件：允许在丢失状态下为IMU传感器创建关键帧（如果启用mInsertKFsLost）
            if(bNeedKF && (bOK || (mInsertKFsLost && mState==RECENTLY_LOST &&
                                   (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD))))
                CreateNewKeyFrame(); // 创建新关键帧

#ifdef REGISTER_TIMES
            // 记录新关键帧创建耗时统计
            std::chrono::steady_clock::time_point time_EndNewKF = std::chrono::steady_clock::now();

            double timeNewKF = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndNewKF - time_StartNewKF).count();
            vdNewKF_ms.push_back(timeNewKF); // 存储到新关键帧创建耗时向量
#endif

            // 允许高创新值点（被Huber函数视为外点）传递给新关键帧，
            // 这样束调整将最终决定它们是否是外点。我们不希望下一帧使用这些点估计位姿，
            // 因此在帧中丢弃它们。仅在上—帧被跟踪时有效
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL); // 移除外点引用
            }
        }

        // 如果相机在初始化后很快丢失，重置系统
        if(mState==LOST)
        {
            // 关键帧数量不足：直接重置地图
            if(pCurrentMap->KeyFramesInMap()<=10)
            {
                mpSystem->ResetActiveMap();
                return; // 直接返回，重新开始
            }
            // IMU传感器特殊处理：如果IMU未初始化就丢失，重置系统
            if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
                if (!pCurrentMap->isImuInitialized())
                {
                    Verbose::PrintMess("Track lost before IMU initialisation, reseting...", Verbose::VERBOSITY_QUIET);
                    mpSystem->ResetActiveMap();
                    return;
                }

            // 关键帧足够且IMU已初始化：在地图集中创建新地图
            CreateMapInAtlas();

            return; // 返回，开始新地图的跟踪
        }

        // 确保当前帧有参考关键帧
        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // 保存当前帧到上一帧，为下一帧处理做准备
        mLastFrame = Frame(mCurrentFrame);
    }
    if(mState==OK || mState==RECENTLY_LOST)
    {
        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        if(mCurrentFrame.isSet())
        {
            
            // mCurrentFrame.GetPose()：当前帧相对于世界坐标系的位姿（Tcw）
            // mCurrentFrame.mpReferenceKF->GetPoseInverse()：参考关键帧相对于世界坐标系的逆位姿（Twr）
            // Tcr_ = Tcw × Twr：当前帧相对于参考关键帧的相对位姿
            Sophus::SE3f Tcr_ = mCurrentFrame.GetPose() * mCurrentFrame.mpReferenceKF->GetPoseInverse();

            mlRelativeFramePoses.push_back(Tcr_);
            mlpReferences.push_back(mCurrentFrame.mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState==LOST);
        }
        else
        {
            // This can happen if tracking is lost for a long time.
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState==LOST);

        }

    }

#ifdef REGISTER_LOOP
    if (Stop()) {

        // Safe area to stop
        while(isStopped())
        {
            usleep(3000);
        }
    }
#endif

    UpdateFrameStatistics();
}
void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500)
    {
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            if (!mCurrentFrame.mpImuPreintegrated || !mLastFrame.mpImuPreintegrated)
            {
                cout << "not IMU meas" << endl;
                return;
            }

            if (!mFastInit && (mCurrentFrame.mpImuPreintegratedFrame->avgA-mLastFrame.mpImuPreintegratedFrame->avgA).norm()<0.5)
            {
                cout << "not enough acceleration" << endl;
                return;
            }

            if(mpImuPreintegratedFromLastKF)
                delete mpImuPreintegratedFromLastKF;

            mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
            mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
        }

        // Set Frame pose to the origin (In case of inertial SLAM to imu)
        if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            Eigen::Matrix3f Rwb0 = mCurrentFrame.mImuCalib.mTcb.rotationMatrix();
            Eigen::Vector3f twb0 = mCurrentFrame.mImuCalib.mTcb.translation();
            Eigen::Vector3f Vwb0;
            Vwb0.setZero();
            mCurrentFrame.SetImuPoseVelocity(Rwb0, twb0, Vwb0);
        }
        else
            mCurrentFrame.SetPose(Sophus::SE3f());

        // Create KeyFrame
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);

        // Insert KeyFrame in the map
        mpAtlas->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        if(!mpCamera2){
            for(int i=0; i<mCurrentFrame.N;i++)
            {
                float z = mCurrentFrame.mvDepth[i];
                if(z>0)
                {
                    Eigen::Vector3f x3D;
                    mCurrentFrame.UnprojectStereo(i, x3D);
                    MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());
                    pNewMP->AddObservation(pKFini,i);
                    pKFini->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                }
            }
        } else{
            for(int i = 0; i < mCurrentFrame.Nleft; i++){
                int rightIndex = mCurrentFrame.mvLeftToRightMatch[i];
                if(rightIndex != -1){
                    Eigen::Vector3f x3D = mCurrentFrame.mvStereo3Dpoints[i];

                    MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpAtlas->GetCurrentMap());

                    pNewMP->AddObservation(pKFini,i);
                    pNewMP->AddObservation(pKFini,rightIndex + mCurrentFrame.Nleft);

                    pKFini->AddMapPoint(pNewMP,i);
                    pKFini->AddMapPoint(pNewMP,rightIndex + mCurrentFrame.Nleft);

                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    mCurrentFrame.mvpMapPoints[rightIndex + mCurrentFrame.Nleft]=pNewMP;
                }
            }
        }

        Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);

        //cout << "Active map: " << mpAtlas->GetCurrentMap()->GetId() << endl;

        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;
        //mnLastRelocFrameId = mCurrentFrame.mnId;

        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints=mpAtlas->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

        mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.GetPose());

        mState=OK;
    }
}
/**
 * 单目相机初始化方法
 * 通过两帧图像的特征点匹配和三角化，建立初始地图和相机位姿
 * 采用两阶段初始化策略：第一帧作为参考帧，第二帧进行匹配和重建
 */
void Tracking::MonocularInitialization()
{
    // 第一阶段：准备初始化（设置参考帧）
    if(!mbReadyToInitializate)
    {
        // 检查当前帧特征点数量是否足够初始化（至少100个特征点）
        if(mCurrentFrame.mvKeys.size()>100)
        {
            // 设置初始帧和上一帧为当前帧的副本
            mInitialFrame = Frame(mCurrentFrame);  // 初始参考帧
            mLastFrame = Frame(mCurrentFrame);     // 上一帧
            
            // 初始化匹配向量：存储前一帧特征点的位置
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;  // 记录特征点位置

            // 初始化匹配索引向量，全部设为-1（表示未匹配）
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            // IMU单目传感器特殊处理
            if (mSensor == System::IMU_MONOCULAR)
            {
                // 清理之前的IMU预积分数据
                if(mpImuPreintegratedFromLastKF)
                {
                    delete mpImuPreintegratedFromLastKF;
                }
                // 创建新的IMU预积分对象（使用零偏和标定参数）
                mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);
                mCurrentFrame.mpImuPreintegrated = mpImuPreintegratedFromLastKF;
            }

            // 标记为准备好初始化，等待下一帧
            mbReadyToInitializate = true;

            return;  // 第一阶段完成，等待下一帧
        }
    }
    // 第二阶段：执行初始化（匹配和重建）
    else
    {
        // 检查初始化条件是否仍然满足
        if (((int)mCurrentFrame.mvKeys.size()<=100)||((mSensor == System::IMU_MONOCULAR)&&(mLastFrame.mTimeStamp-mInitialFrame.mTimeStamp>1.0)))
        {
            // 条件不满足：特征点不足或IMU模式下时间间隔过长
            mbReadyToInitializate = false;
            return;
        }

        // 寻找特征点对应关系
        ORBmatcher matcher(0.9,true);  // 初始化匹配器（最小距离比0.9，启用交叉检查）
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // 自适应匹配数阈值：取 min(200, 当前帧特征数*0.2)，扑翼场景放宽
        int minMatches = std::min(200, static_cast<int>(mCurrentFrame.mvKeys.size() * 0.2));
        minMatches = std::max(minMatches, 80);  // 绝对下限
        if(nmatches < minMatches)
        {
            // cout << "[Init] 匹配数不足: " << nmatches << " < " << minMatches
            //      << " (当前帧特征=" << mCurrentFrame.mvKeys.size() << ")" << endl;
            mbReadyToInitializate = false;  // 匹配不足，重选参考帧
            return;
        }

        Sophus::SE3f Tcw;  // 当前帧相对于世界坐标系的位姿
        vector<bool> vbTriangulated; // 标记哪些匹配点被成功三角化

        // 使用两视图重建方法计算相机位姿和3D点
        if(mpCamera->ReconstructWithTwoViews(mInitialFrame.mvKeysUn,mCurrentFrame.mvKeysUn,mvIniMatches,Tcw,mvIniP3D,vbTriangulated))
        {
            // 清理未成功三角化的匹配点
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;  // 标记为无效匹配
                    nmatches--;          // 减少匹配计数
                }
            }

            // 设置帧位姿
            mInitialFrame.SetPose(Sophus::SE3f());  // 初始帧设为世界坐标系原点
            mCurrentFrame.SetPose(Tcw);             // 当前帧设为计算得到的位姿

            // ===== 前置初始化质量门控 =====
            // A. 有效三角化点阈值（取 min(100, 匹配数×0.5)，扑翼场景放宽）
            int nValid3D = 0;
            for(size_t i = 0; i < mvIniMatches.size(); i++)
                if(mvIniMatches[i] >= 0 && vbTriangulated[i]) nValid3D++;
            int minValid3D = std::min(100, static_cast<int>(nmatches * 0.5));
            minValid3D = std::max(minValid3D, 40);  // 绝对下限
            if(nValid3D < minValid3D) {
                cout << "[InitCheck] FAIL: 有效3D点 " << nValid3D << " < " << minValid3D
                     << " (匹配=" << nmatches << " 特征=" << mCurrentFrame.mvKeys.size() << ")" << endl;
                mbReadyToInitializate = false;
                return;
            }

            // B. 三角化点的网格覆盖率（10×10网格，借鉴MF-SLAM思路）
            {
                const int nGrid = 10;
                std::vector<bool> vGrid(nGrid * nGrid, false);
                float cellW = (mCurrentFrame.mnMaxX - mCurrentFrame.mnMinX) / (float)nGrid;
                float cellH = (mCurrentFrame.mnMaxY - mCurrentFrame.mnMinY) / (float)nGrid;
                int nInGrid = 0;
                for(size_t i = 0; i < mvIniMatches.size(); i++) {
                    if(mvIniMatches[i] < 0 || !vbTriangulated[i]) continue;
                    const cv::KeyPoint &kp = mCurrentFrame.mvKeysUn[mvIniMatches[i]];
                    int gx = static_cast<int>((kp.pt.x - mCurrentFrame.mnMinX) / cellW);
                    int gy = static_cast<int>((kp.pt.y - mCurrentFrame.mnMinY) / cellH);
                    gx = std::max(0, std::min(gx, nGrid - 1));
                    gy = std::max(0, std::min(gy, nGrid - 1));
                    if(!vGrid[gy * nGrid + gx]) { vGrid[gy * nGrid + gx] = true; nInGrid++; }
                }
                float coverage = (float)nInGrid / (nGrid * nGrid);
                const float MIN_GRID_COVERAGE = 0.08f;  // 扑翼场景放宽到8%
                if(coverage < MIN_GRID_COVERAGE) {
                    cout << "[InitCheck] FAIL: 网格覆盖率 " << coverage << " < " << MIN_GRID_COVERAGE
                         << " (3D点=" << nValid3D << " 占格=" << nInGrid << "/" << (nGrid*nGrid) << ")" << endl;
                    mbReadyToInitializate = false;
                    return;
                }
            }

            // C. 地面平面验证：扑翼飞行场景下，绝大部分特征点应位于相机下方
            {
                // 收集有效3D点
                std::vector<cv::Point3f> vGroundPts;
                for(size_t i = 0; i < mvIniMatches.size(); i++) {
                    if(mvIniMatches[i] < 0 || !vbTriangulated[i]) continue;
                    const cv::Point3f& p = mvIniP3D[i];
                    vGroundPts.push_back(p);
                }
                if(vGroundPts.size() < 30) {
                    cout << "[InitCheck] FAIL: 3D点不足(" << vGroundPts.size() << " < 30)，跳过平面验证" << endl;
                    // 不在此处失败，点太少时平面拟合不可靠，让后续BA判断
                } else {
                    // 计算质心
                    cv::Point3f centroid3f(0,0,0);
                    for(const auto& pt : vGroundPts) centroid3f += pt;
                    centroid3f *= (1.0f / vGroundPts.size());
                    Eigen::Vector3f centroid(centroid3f.x, centroid3f.y, centroid3f.z);

                    // 质心必须在相机前方和下方（相机坐标系：X→右, Y→↓, Z→前）
                    // 扑翼飞行场景：地面在下方，Z>0（前方）且 Y>0（下方）
                    // if(centroid.z() < 0.2f || centroid.y() < 0.0f) {
                    if(centroid.z() < 0.2f) {    
                        cout << "[InitCheck] FAIL: 点云质心位置异常 "
                             << "Y=" << centroid.y() << " (需>0, 点在相机上方)"
                             << " Z=" << centroid.z() << " (需>0.2)"
                             << " 总计" << vGroundPts.size() << "点" << endl;
                        mbReadyToInitializate = false;
                        return;
                    }

                    // 用SVD拟合平面：法向量 = V的最后一列（最小奇异值方向）
                    cv::Mat ptsMat(vGroundPts.size(), 3, CV_32F);
                    for(size_t i = 0; i < vGroundPts.size(); i++) {
                        ptsMat.at<float>(i, 0) = vGroundPts[i].x - centroid3f.x;
                        ptsMat.at<float>(i, 1) = vGroundPts[i].y - centroid3f.y;
                        ptsMat.at<float>(i, 2) = vGroundPts[i].z - centroid3f.z;
                    }
                    cv::Mat covar, meanTmp;
                    cv::calcCovarMatrix(ptsMat, covar, meanTmp,
                        cv::COVAR_NORMAL | cv::COVAR_ROWS);
                    cv::SVD svd(covar);
                    cv::Mat normal = svd.vt.row(2);  // 3×1, 最小奇异值 → 平面法向量

                    // SVD 质量检查：最小/最大奇异值比太小时点云近退化，法向量不可靠
                    cv::Mat singVals = svd.w;  // 奇异值（降序）
                    float svRatio = singVals.at<float>(2) / (singVals.at<float>(0) + 1e-10f);
                    bool bSVDValid = (svRatio > 0.05f);  // 最小奇异值不低于最大的5%

                    // 归一化并统一方向
                    float nx = normal.at<float>(0);
                    float ny = normal.at<float>(1);
                    float nz = normal.at<float>(2);
                    float nlen = std::sqrt(nx*nx + ny*ny + nz*nz);
                    if(nlen > 1e-6f) { nx /= nlen; ny /= nlen; nz /= nlen; }
                    if(ny < 0) { nx = -nx; ny = -ny; nz = -nz; }

                    if(bSVDValid) {
                        // 可靠平面：检查法向量方向
                        float yScore = ny;
                        float zScore = nz;
                        const float MIN_NORMAL_Y = 0.2f;            // 放宽至0.2（斜向下看地面时法向量Y分量会减小）
                        const float MIN_NORMAL_Z = -0.3f;           // 放宽至-0.3

                        if(yScore < MIN_NORMAL_Y || zScore < MIN_NORMAL_Z) {
                            cout << "[InitCheck] FAIL: 平面法向量不正常 n=("
                                 << nx << "," << ny << "," << nz
                                 << ") yScore=" << yScore << " < " << MIN_NORMAL_Y
                                 << " (质心Y=" << centroid.y() << " Z=" << centroid.z()
                                 << " SV比值=" << svRatio << ")"
                                 << endl;
                            mbReadyToInitializate = false;
                            return;
                        }
                        cout << "[InitCheck] 地面平面 PASS: centroid=("
                             << centroid.x() << "," << centroid.y() << "," << centroid.z()
                             << ") normal=(" << nx << "," << ny << "," << nz
                             << ") SV比值=" << svRatio << " 点数=" << vGroundPts.size() << endl;
                    } else {
                        // SVD退化：平面拟合不可靠，仅依赖质心位置判断
                        // 质心Y>0、Z>0 已在上方检查通过，此处仅放松通过
                        cout << "[InitCheck] 地面平面 PASS(退化): centroid=("
                             << centroid.x() << "," << centroid.y() << "," << centroid.z()
                             << ") SV比值=" << svRatio << " 点数=" << vGroundPts.size()
                             << " (SVD退化，跳过法向量检查)" << endl;
                    }
                }
            }

            cout << "[InitCheck] PASS: 有效3D=" << nValid3D
                 << "/" << minValid3D << " 匹配=" << nmatches
                 << " 覆盖率=合格 (帧ID=" << mCurrentFrame.mnId << ")" << endl;

            // 创建初始单目地图
            CreateInitialMapMonocular();
        }
        // else
        // {
        //     cout << "[Init] TwoView重建失败 (匹配=" << nmatches << " 帧ID=" << mCurrentFrame.mnId << ")" << endl;
        // }
    }
}

/*
 * @brief 检查SLAM初始化的健康状态
 * 
 * 该函数对SLAM系统初始化阶段的两个关键帧进行全面的健康检查，确保初始化质量可靠。
 * 通过多项统计指标评估初始化结果，防止因错误初始化导致后续跟踪失败。
 * 
 * @param pKF1 第一个关键帧（参考帧）
 * @param pKF2 第二个关键帧（当前帧）
 * @return true 初始化健康，false 初始化失败，需要重新初始化
 */
bool Tracking::CheckInitializationHealth(KeyFrame* pKF1, KeyFrame* pKF2)
{
    // 获取第一帧的所有地图点匹配
    vector<MapPoint*> vpAllMPs = pKF1->GetMapPointMatches();
    int nTotalMPs = 0;
    // 统计有效地图点数量（非空且未标记为坏点）
    for(MapPoint* pMP : vpAllMPs) {
        if(pMP && !pMP->isBad()) nTotalMPs++;
    }

    // 检查1：有效地图点数量阈值
    // 地图点过少意味着初始化不稳定，需要至少50个有效地图点
    if(nTotalMPs < 50) {
        // Verbose::PrintMess("[HealthCheck] FAIL: too few valid map points: " + to_string(nTotalMPs), Verbose::VERBOSITY_NORMAL);
        cout << "有效地图点数量不足50个" << endl;
        return false;
    }

    // 获取两帧的位姿（世界坐标系到相机坐标系的变换）
    Sophus::SE3f T1w = pKF1->GetPose();
    Sophus::SE3f T2w = pKF2->GetPose();

    // 分解位姿为旋转矩阵和平移向量
    Eigen::Matrix3f R1w = T1w.rotationMatrix();
    Eigen::Vector3f t1w = T1w.translation();
    Eigen::Matrix3f R2w = T2w.rotationMatrix();
    Eigen::Vector3f t2w = T2w.translation();

    // 获取两帧的相机内参
    const float fx1 = pKF1->fx, fy1 = pKF1->fy, cx1 = pKF1->cx, cy1 = pKF1->cy;
    const float fx2 = pKF2->fx, fy2 = pKF2->fy, cx2 = pKF2->cx, cy2 = pKF2->cy;

    // 计算帧间变换 T21 = T2w * T1w^{-1}
    // 表示从第一帧相机坐标系到第二帧相机坐标系的变换
    Sophus::SE3f T21 = T2w * T1w.inverse();
    Eigen::Matrix3f R21 = T21.rotationMatrix();
    Eigen::Vector3f t21 = T21.translation();
    // 计算基线长度（两相机光心之间的距离）
    float baseline = t21.norm();

    // 获取两帧的相机光心位置（世界坐标系下）
    Eigen::Vector3f Ow1 = pKF1->GetCameraCenter();
    Eigen::Vector3f Ow2 = pKF2->GetCameraCenter();

    // 初始化统计数据容器
    vector<float> vReprojErrors;      // 重投影误差（像素平方）
    vector<float> vDepths1, vDepths2; // 两帧观测到的深度值
    vector<float> vParallaxAngles;    // 视差角（度）
    vector<float> vDepthRatios;       // 两帧深度比值

    // 遍历所有地图点，计算统计数据
    for(size_t i = 0; i < vpAllMPs.size(); i++)
    {
        MapPoint* pMP = vpAllMPs[i];
        // 跳过无效地图点
        if(!pMP || pMP->isBad())
            continue;

        // 获取地图点的世界坐标
        Eigen::Vector3f Pw = pMP->GetWorldPos();

        // 将世界坐标投影到第一帧相机坐标系
        Eigen::Vector3f Pc1 = R1w * Pw + t1w;
        float z1 = Pc1(2);
        // 深度必须为正（在相机前方）
        if(z1 <= 0) continue;
        vDepths1.push_back(z1);

        // 将世界坐标投影到第二帧相机坐标系
        Eigen::Vector3f Pc2 = R2w * Pw + t2w;
        float z2 = Pc2(2);
        // 深度必须为正
        if(z2 <= 0) continue;
        vDepths2.push_back(z2);

        // 计算两帧深度的比值（取较大值除以较小值）
        float ratio = (z1 > z2) ? z1 / z2 : z2 / z1;
        vDepthRatios.push_back(ratio);

        // 计算第一帧的重投影误差（像素坐标差的平方和）
        float u1 = fx1 * Pc1(0) / z1 + cx1;
        float v1 = fy1 * Pc1(1) / z1 + cy1;
        const cv::KeyPoint& kp1 = pKF1->mvKeysUn[i];
        float err1 = (u1 - kp1.pt.x) * (u1 - kp1.pt.x) + (v1 - kp1.pt.y) * (v1 - kp1.pt.y);
        vReprojErrors.push_back(err1);

        // 计算视差角：从两个相机光心指向地图点的向量之间的夹角
        Eigen::Vector3f normal1 = Pw - Ow1;
        Eigen::Vector3f normal2 = Pw - Ow2;
        float d1 = normal1.norm(), d2 = normal2.norm();
        if(d1 > 1e-6 && d2 > 1e-6) {
            float cosPA = normal1.dot(normal2) / (d1 * d2);
            // 使用clamp防止数值误差导致acos参数越界
            float parallaxDeg = acos(std::max(-1.0f, std::min(1.0f, cosPA))) * 180.0f / M_PI;
            vParallaxAngles.push_back(parallaxDeg);
        }
    }

    // 检查统计数据的有效性：需要至少40个有效点
    if(vReprojErrors.size() < 40 || vDepths1.size() < 40) {
        // Verbose::PrintMess("[HealthCheck] FAIL: insufficient valid points for statistics: " + to_string(vReprojErrors.size()), Verbose::VERBOSITY_NORMAL);
        cout << "统计数据无效，有效点数量不足40个" << endl;
        return false;
    }

    // 定义统计计算lambda函数
    // 计算中位数（对数据排序后取中间值）
    auto CalcMedian = [](vector<float>& v) -> float {
        std::sort(v.begin(), v.end());
        size_t n = v.size();
        return (n % 2 == 0) ? (v[n/2 - 1] + v[n/2]) * 0.5f : v[n/2];
    };
    // 计算指定百分位数
    auto CalcPercentile = [](vector<float>& v, float pct) -> float {
        std::sort(v.begin(), v.end());
        return v[static_cast<size_t>(v.size() * pct)];
    };
    // 计算标准差（已知均值的情况下）
    auto CalcStd = [](const vector<float>& v, float mean) -> float {
        float var = 0;
        for(float x : v) var += (x - mean) * (x - mean);
        return sqrt(var / v.size());
    };

    // --- 检查1：重投影误差中位数 ---
    // 阈值设为健康值的 ~15x 和 ~12x
    float medianErr = CalcMedian(vReprojErrors);
    float meanErr = 0;
    for(float e : vReprojErrors) meanErr += e;
    meanErr /= vReprojErrors.size();
    float stdErr = CalcStd(vReprojErrors, meanErr);
    float limitErrMedian = 2.0f;
    float limitErrStd = 0.8f;

    if(medianErr > limitErrMedian || stdErr > limitErrStd) {
        cout << "[HealthCheck] FAIL: reprojection error median=" << medianErr << " std=" << stdErr << endl;
        return false;
    }

    // --- 检查2：深度一致性 ---
    // 同一地图点在两帧中的深度比中位数
    // 健康值应接近1.0，>2.0说明结构扭曲
    float medianDepthRatio = CalcMedian(vDepthRatios);
    if(medianDepthRatio > 2.0f) {
        cout << "[HealthCheck] FAIL: depth inconsistency ratio=" << medianDepthRatio << endl;
        return false;
    }

    // --- 检查3：基线/深度比 ---
    // 健康数据: blRatio=0.035
    // 下限 0.01: 视差极小导致三角化退化
    // 上限 2.0:  基线过大可能运动估计错误
    float medianDepth1 = CalcMedian(vDepths1);
    float medianDepth2 = CalcMedian(vDepths2);
    float sceneMedianDepth = (medianDepth1 + medianDepth2) * 0.5f;
    float baselineRatio = baseline / sceneMedianDepth;

    if(baselineRatio < 0.025f || baselineRatio > 2.0f) {
        cout << "[HealthCheck] FAIL: baseline/depth ratio=" << baselineRatio << endl;
        return false;
    }

    // --- 检查4：旋转角度合理性 ---
    // 健康数据: rot=9.98°
    // 扑翼场景下帧间可能叠加振动，阈值 25° 给 2.5x 余量
    Eigen::AngleAxisf aa(R21);
    float rotDeg = aa.angle() * 180.0f / M_PI;

    if(rotDeg > 8.0f) {
        cout << "[HealthCheck] FAIL: excessive rotation=" << rotDeg << " deg" << endl;
        return false;
    }

    // --- 检查5：视差角分布 ---
    // 健康场景应有足够的视差角用于三角化
    if(!vParallaxAngles.empty()) {
        float medianParallax = CalcMedian(vParallaxAngles);
        if(medianParallax < 0.3f) {
            cout << "[HealthCheck] FAIL: insufficient parallax=" << medianParallax << " deg" << endl;
            return false;
        }
    }

    // --- 检查6：深度分位数比 ---
    // >8.0 说明场景深度极度不均匀或存在错误深度估计
    float depthP20 = CalcPercentile(vDepths1, 0.2f);
    float depthP80 = CalcPercentile(vDepths1, 0.8f);
    float depthRatio = (depthP20 > 1e-6f) ? depthP80 / depthP20 : 999.0f;

    if(depthRatio > 2.0f) {
        cout << "[HealthCheck] FAIL: extreme depth spread p80/p20=" << depthRatio << endl;
        return false;
    }

    // 所有检查通过，输出健康报告
    cout << "[HealthCheck] PASS: 有效地图点=" << nTotalMPs
        << " 重投影误差中位数=" << to_string(medianErr).substr(0,5)
        << " 重投影误差标准差=" << to_string(stdErr).substr(0,5)
        << " 深度分位数比=" << to_string(depthRatio).substr(0,5)
        << " 基线/深度比=" << to_string(baselineRatio).substr(0,5)
        << " 旋转角度=" << to_string(rotDeg).substr(0,5) << "deg" 
        << " 当前帧ID：" << mCurrentFrame.mnId
        << endl;


    return true;
}

/**
 * 创建单目初始地图：完成单目SLAM的初始化过程
 * 这是单目SLAM系统启动的关键步骤，将初始化帧和当前帧转换为关键帧，并构建初始地图
 * 包括地图点创建、位姿优化、尺度归一化等核心操作
 */
void Tracking::CreateInitialMapMonocular()
{
    // 第一步：创建关键帧
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);  // 初始化帧关键帧
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);  // 当前帧关键帧

    // IMU单目模式特殊处理：初始化帧的IMU预积分设为空
    if(mSensor == System::IMU_MONOCULAR)
        pKFini->mpImuPreintegrated = (IMU::Preintegrated*)(NULL);

    // 计算关键帧的词袋向量（用于快速匹配）
    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // 将关键帧插入地图
    mpAtlas->AddKeyFrame(pKFini);
    mpAtlas->AddKeyFrame(pKFcur);

    // 第二步：创建地图点并建立观测关系
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)  // 跳过无效匹配
            continue;

        // 创建地图点：使用三角化得到的世界坐标
        Eigen::Vector3f worldPos;
        worldPos << mvIniP3D[i].x, mvIniP3D[i].y, mvIniP3D[i].z;
        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpAtlas->GetCurrentMap());

        // 将地图点关联到两个关键帧
        pKFini->AddMapPoint(pMP,i);           // 初始化帧的第i个特征点
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);  // 当前帧的匹配特征点

        // 建立地图点的观测关系
        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        // 计算地图点的描述子和几何属性
        pMP->ComputeDistinctiveDescriptors();  // 计算代表性描述子
        pMP->UpdateNormalAndDepth();           // 更新法向量和深度信息

        // 填充当前帧的地图点结构
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;  // 标记为内点

        // 将地图点添加到地图中
        mpAtlas->AddMapPoint(pMP);
    }

    // 第三步：更新关键帧间的连接关系
    pKFini->UpdateConnections();  // 更新初始化帧的连接
    pKFcur->UpdateConnections();  // 更新当前帧的连接

    std::set<MapPoint*> sMPs;
    sMPs = pKFini->GetMapPoints();  // 获取初始化帧的所有地图点

    // 第四步：执行全局Bundle Adjustment优化
    Verbose::PrintMess("New Map created with " + to_string(mpAtlas->MapPointsInMap()) + " points", Verbose::VERBOSITY_QUIET);
    Optimizer::GlobalBundleAdjustemnt(mpAtlas->GetCurrentMap(),20);  // 20次迭代的全局BA

    // 初始化健康检查：验证BA优化后的地图结构质量
    // if(!CheckInitializationHealth(pKFini, pKFcur))
    // {
    //     mpSystem->ResetActiveMap();  // 重置地图
    //     return;
    // }

    // 第五步：尺度归一化处理（单目SLAM的尺度不确定性）
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);  // 计算场景中值深度
    float invMedianDepth;
    if(mSensor == System::IMU_MONOCULAR)
        invMedianDepth = 4.0f/medianDepth; // IMU单目：使用4.0作为尺度因子
    else
        invMedianDepth = 1.0f/medianDepth;  // 纯单目：使用中值深度的倒数

    // 检查初始化质量
    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<50) // 深度无效或跟踪点太少（原版为100）
    {
        Verbose::PrintMess("Wrong initialization, reseting...", Verbose::VERBOSITY_QUIET);
        mpSystem->ResetActiveMap();  // 重置地图
        return;
    }

    // 缩放初始基线（平移部分）
    Sophus::SE3f Tc2w = pKFcur->GetPose();
    Tc2w.translation() *= invMedianDepth;  // 缩放平移向量
    pKFcur->SetPose(Tc2w);

    // 缩放所有地图点的世界坐标
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);  // 缩放地图点位置
            pMP->UpdateNormalAndDepth();  // 更新法向量和深度
        }
    }

    // IMU单目模式：设置关键帧间的IMU连接
    if (mSensor == System::IMU_MONOCULAR)
    {
        pKFcur->mPrevKF = pKFini;  // 当前帧的前一关键帧
        pKFini->mNextKF = pKFcur;  // 初始化帧的后一关键帧
        pKFcur->mpImuPreintegrated = mpImuPreintegratedFromLastKF;  // 设置IMU预积分

        // 创建新的IMU预积分对象
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKFcur->mpImuPreintegrated->GetUpdatedBias(),pKFcur->mImuCalib);
    }

    // 第六步：将关键帧插入局部建图器
    mpLocalMapper->InsertKeyFrame(pKFini);  // 插入初始化关键帧
    mpLocalMapper->InsertKeyFrame(pKFcur);  // 插入当前关键帧
    mpLocalMapper->mFirstTs=pKFcur->mTimeStamp;  // 记录第一个时间戳

    // 第七步：更新跟踪状态和参考信息
    mCurrentFrame.SetPose(pKFcur->GetPose());  // 设置当前帧位姿
    mnLastKeyFrameId=mCurrentFrame.mnId;        // 记录最后关键帧ID
    mpLastKeyFrame = pKFcur;                    // 设置最后关键帧指针
    //mnLastRelocFrameId = mInitialFrame.mnId;  // 重定位帧ID（已注释）

    // 更新局部关键帧和地图点
    mvpLocalKeyFrames.push_back(pKFcur);        // 添加当前关键帧到局部关键帧
    mvpLocalKeyFrames.push_back(pKFini);        // 添加初始化关键帧到局部关键帧
    mvpLocalMapPoints=mpAtlas->GetAllMapPoints();  // 获取所有地图点作为局部地图点
    mpReferenceKF = pKFcur;                     // 设置参考关键帧
    mCurrentFrame.mpReferenceKF = pKFcur;       // 设置当前帧的参考关键帧

    // 第八步：计算初始速度（用于运动模型）
    vector<KeyFrame*> vKFs = mpAtlas->GetAllKeyFrames();  // 获取所有关键帧

    Sophus::SE3f deltaT = vKFs.back()->GetPose() * vKFs.front()->GetPoseInverse();  // 计算位姿变化
    mbVelocity = false;  // 重置速度标志
    Eigen::Vector3f phi = deltaT.so3().log();  // 提取旋转部分的对数映射

    // 时间缩放因子：基于时间戳计算速度
    double aux = (mCurrentFrame.mTimeStamp-mLastFrame.mTimeStamp)/(mCurrentFrame.mTimeStamp-mInitialFrame.mTimeStamp);
    phi *= aux;  // 缩放旋转向量

    mLastFrame = Frame(mCurrentFrame);  // 更新上一帧

    // 第九步：设置参考点和可视化
    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);  // 设置参考地图点
    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());  // 设置可视化相机位姿

    // 记录地图的起源关键帧
    mpAtlas->GetCurrentMap()->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;  // 设置跟踪状态为正常

    initID = pKFcur->mnId;  // 记录初始化ID
    mMonoInitFrameId = mCurrentFrame.mnId;  // 记录初始化时的帧ID，用于地面平面拟合延迟计时
}
/**
 * 在地图集中创建新地图：重置跟踪状态并准备新的SLAM会话
 * 这个函数在系统需要创建新地图时调用，比如重置、重定位失败或地图切换时
 * 它会清理所有跟踪相关的状态变量，为新的地图构建做准备
 */
void Tracking::CreateMapInAtlas()
{
    // 第一步：记录初始化信息并创建新地图
    mnLastInitFrameId = mCurrentFrame.mnId;  // 记录最后初始化帧ID（当前帧）
    mpAtlas->CreateNewMap();  // 在地图集中创建新的地图
    
    // 设置IMU传感器标志（如果使用IMU传感器）
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
        mpAtlas->SetInertialSensor();  // 标记地图使用惯性传感器
    
    mbSetInit=false;  // 重置初始化设置标志

    // 第二步：重置跟踪状态变量
    mnInitialFrameId = mCurrentFrame.mnId+1;  // 设置初始帧ID为下一帧
    mState = NO_IMAGES_YET;  // 设置跟踪状态为"尚未接收图像"

    // 重置与最后关键帧相关的变量
    mbVelocity = false;  // 重置速度标志
    //mnLastRelocFrameId = mnLastInitFrameId; // 最后重定位关键帧ID设为当前ID（新地图起点）
    
    // 输出新地图的第一帧ID信息
    Verbose::PrintMess("First frame id in map: " + to_string(mnLastInitFrameId+1), Verbose::VERBOSITY_NORMAL);
    
    mbVO = false; // 初始化值，用于判断最后关键帧是否有足够的地图点
    
    // 单目模式特殊处理：重置初始化准备标志
    if(mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR)
    {
        mbReadyToInitializate = false;  // 单目模式需要重新初始化
    }

    // 第三步：IMU相关清理和重置
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && mpImuPreintegratedFromLastKF)
    {
        delete mpImuPreintegratedFromLastKF;  // 删除旧的IMU预积分对象
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(IMU::Bias(),*mpImuCalib);  // 创建新的IMU预积分对象
    }

    // 第四步：清理关键帧指针
    if(mpLastKeyFrame)
        mpLastKeyFrame = static_cast<KeyFrame*>(NULL);  // 清空最后关键帧指针

    if(mpReferenceKF)
        mpReferenceKF = static_cast<KeyFrame*>(NULL);  // 清空参考关键帧指针

    // 第五步：重置帧数据
    mLastFrame = Frame();      // 重置上一帧为空帧
    mCurrentFrame = Frame();    // 重置当前帧为空帧
    mvIniMatches.clear();       // 清空初始化匹配数据

    // 第六步：设置地图创建完成标志
    mbCreatedMap = true;  // 标记地图已创建完成
}
void Tracking::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}
/**
 * 基于参考关键帧的跟踪方法
 * 使用词袋模型快速匹配当前帧与参考关键帧，然后通过PnP优化计算位姿
 * 主要用于运动模型不可用或重定位后的稳定跟踪
 * 
 * @return bool 跟踪是否成功
 */
bool Tracking::TrackReferenceKeyFrame()
{
    // 计算当前帧的词袋向量（用于快速特征匹配）
    mCurrentFrame.ComputeBoW();

    // 首先使用词袋模型与参考关键帧进行ORB匹配
    // 如果找到足够的匹配点，则设置PnP求解器
    ORBmatcher matcher(0.7,true);  // 初始化匹配器（最小距离比0.7，启用交叉检查）
    vector<MapPoint*> vpMapPointMatches;  // 存储匹配的地图点

    // 使用词袋模型进行快速匹配
    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches);

    // 振动自适应：降低参考帧匹配失败阈值（基础15，最低降至9）
    int nMinRefMatches = std::max(9, static_cast<int>(15 - 3 * std::min(mfPrevVibrationLevel, 2.0f)));

    // 检查匹配数量是否足够
    if(nmatches<nMinRefMatches)
    {
        cout << "TRACK_REF_KF: Less than " << nMinRefMatches << " matches!!\n";
        return false;  // 匹配不足，跟踪失败
    }

    // 将匹配结果关联到当前帧
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    // 使用上一帧的位姿作为当前帧的初始位姿估计
    mCurrentFrame.SetPose(mLastFrame.GetPose());

    // ============================================================================
    // 收集帧间特征点匹配结果（用于动态点检测）—— O(N) 哈希表版本
    // ============================================================================
    CollectFrameMatches();
    // ============================================================================

    if(mCurrentFrame.mnId != mnLastDynamicCheckFrameId)
    {
        // 计算振动指标
        CalculateVibrationMetrics();

        // 检测动态语义特征点
        DetectDynamicPoints();

        // 3D检测框提升（可选，默认关闭）
        if(mbEnable3DBoxDetection)
            Lift2DBoxesTo3D();

        // 更新最后动态检查帧ID,每一帧只运行一次振动指标计算和动态点检测检测
        mnLastDynamicCheckFrameId = mCurrentFrame.mnId;
    // }else{
    //     cout << "当前帧已进行过振动指标计算和动态点检测，无需重复检测" << endl;
    }

    
    // 使用所有匹配点优化当前帧的位姿（PnP优化）
    Optimizer::PoseOptimization(&mCurrentFrame);
   

    // 剔除异常点（外点）
    int nmatchesMap = 0;  // 有效地图点匹配数量
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        //if(i >= mCurrentFrame.Nleft) break;  // 跳过右视图检查（已注释）
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(mCurrentFrame.mvbOutlier[i])  // 如果是异常点
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                // 清除当前帧的地图点关联
                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;  // 重置异常点标记
                
                // 更新地图点的可见性状态
                if(i < mCurrentFrame.Nleft){
                    pMP->mbTrackInView = false;  // 左视图不可见
                }
                else{
                    pMP->mbTrackInViewR = false; // 右视图不可见
                }
                pMP->mbTrackInView = false;      // 确保跟踪标志重置
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;  // 记录最后被看到的帧ID
                nmatches--;  // 减少总匹配计数
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)  // 有效的地图点（被多次观测）
                nmatchesMap++;  // 增加有效匹配计数
        }
    }

    // 根据传感器类型返回跟踪结果
    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        return true;  // IMU传感器：即使匹配较少也返回成功（依赖IMU数据）
    else
    {
        // 振动自适应：放宽有效地图点匹配成功条件（当前帧振动已计算完毕）
        int nMinMapMatches = std::max(6, static_cast<int>(10 - 2 * std::min(mfCurrentVibrationLevel, 2.0f)));
        return nmatchesMap>=nMinMapMatches;
    }
}
void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    Sophus::SE3f Tlr = mlRelativeFramePoses.back();
    mLastFrame.SetPose(Tlr * pRef->GetPose());

    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    const int Nfeat = mLastFrame.Nleft == -1? mLastFrame.N : mLastFrame.Nleft;
    vDepthIdx.reserve(Nfeat);
    for(int i=0; i<Nfeat;i++)
    {
        float z = mLastFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++)
    {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(!pMP)
            bCreateNew = true;
        else if(pMP->Observations()<1)
            bCreateNew = true;

        if(bCreateNew)
        {
            Eigen::Vector3f x3D;

            if(mLastFrame.Nleft == -1){
                mLastFrame.UnprojectStereo(i, x3D);
            }
            else{
                x3D = mLastFrame.UnprojectStereoFishEye(i);
            }

            MapPoint* pNewMP = new MapPoint(x3D,mpAtlas->GetCurrentMap(),&mLastFrame,i);
            mLastFrame.mvpMapPoints[i]=pNewMP;

            mlpTemporalPoints.push_back(pNewMP);
            nPoints++;
        }
        else
        {
            nPoints++;
        }

        if(vDepthIdx[j].first>mThDepth && nPoints>100)
            break;

    }
}
/**
 * 基于运动模型的跟踪方法
 * 使用上一帧的位姿和运动速度来预测当前帧的位姿，然后通过特征点匹配进行优化
 * 支持IMU和纯视觉两种运动模型
 * 
 * @return bool 跟踪是否成功
 */
bool Tracking::TrackWithMotionModel()
{
    // 初始化ORB特征点匹配器，设置最小距离比阈值为0.9，启用交叉检查
    ORBmatcher matcher(0.9,true);

    // 更新上一帧的位姿（根据其参考关键帧）
    // 如果在定位模式下，创建"视觉里程计"点（临时地图点）
    UpdateLastFrame();

    // IMU运动模型：如果IMU已初始化且不需要重置
    if (mpAtlas->isImuInitialized() && (mCurrentFrame.mnId>mnLastRelocFrameId+mnFramesToResetIMU))
    {
        // 使用IMU预测当前帧的状态（位姿和速度）
        PredictStateIMU();
        return true;  // IMU跟踪总是返回成功（即使匹配较少）
    }
    else
    {
        // 视觉运动模型：使用速度模型预测当前帧位姿
        // 当前帧位姿 = 运动速度 × 上一帧位姿
        mCurrentFrame.SetPose(mVelocity * mLastFrame.GetPose());
    }

    // 清空当前帧的地图点关联（准备重新匹配）
    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

    // 投影上一帧中可见的地图点到当前帧进行匹配
    int th;  // 搜索半径阈值
    int thBase;  // 基础搜索半径

    // 根据传感器类型设置不同的搜索阈值
    if(mSensor==System::STEREO)
        thBase=7;   // 双目相机：阈值较小（特征点更密集）
    else
        thBase=15;  // 单目相机：阈值较大（特征点较稀疏）

    // 振动自适应：使用上一帧振动等级调整搜索半径
    // 阈值为中位数，50%帧会超出，因此用超出百分比渐进缩放
    // vibrationLevel=0 → 不放大; =1.0 → 放大50%; =2.0 → 放大100%(封顶)
    float vibScale = std::min(mfPrevVibrationLevel, 2.0f) * 0.5f;  // [0, 1.0]
    th = static_cast<int>(thBase * (1.0f + vibScale));  // 单目: 15~30, 双目: 7~14

    // 振动自适应：降低匹配失败阈值（振动大时允许更少匹配也继续尝试）
    // 基础20，最低降至12
    int nMinMatches = std::max(12, static_cast<int>(20 - 4 * std::min(mfPrevVibrationLevel, 2.0f)));

    // 通过投影匹配在当前帧中搜索与上一帧地图点对应的特征点
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);

    // 如果匹配数量较少，使用更大的搜索窗口重新搜索
    if(nmatches<nMinMatches)
    {
        Verbose::PrintMess("Not enough matches, wider window search!!", Verbose::VERBOSITY_NORMAL);
        // 清空当前匹配结果
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));

        // 使用双倍搜索半径重新匹配
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR || mSensor==System::IMU_MONOCULAR);
        Verbose::PrintMess("Matches with wider search: " + to_string(nmatches), Verbose::VERBOSITY_NORMAL);

    }

    // 如果匹配数量仍然不足，跟踪失败
    if(nmatches<nMinMatches)
    {
        Verbose::PrintMess("Not enough matches!!", Verbose::VERBOSITY_NORMAL);
        // IMU传感器：即使匹配较少也返回成功（依赖IMU数据）
        if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            return true;
        else
            return false;  // 纯视觉传感器：匹配不足则跟踪失败
    }

    // ============================================================================
    // 收集帧间特征点匹配结果（用于动态点检测）—— O(N) 哈希表版本
    // ============================================================================
    CollectFrameMatches();
    // ============================================================================

    if(mCurrentFrame.mnId != mnLastDynamicCheckFrameId)
    {
        // 计算振动指标
        CalculateVibrationMetrics();
        
        // 检测动态语义特征点
        DetectDynamicPoints();

        // 3D检测框提升（可选，默认关闭）
        if(mbEnable3DBoxDetection)
            Lift2DBoxesTo3D();

        // 更新最后动态检查帧ID,每一帧只运行一次振动指标计算和动态点检测检测
        mnLastDynamicCheckFrameId = mCurrentFrame.mnId;
    }
    
    // 使用所有匹配点优化当前帧的位姿
    Optimizer::PoseOptimization(&mCurrentFrame);
    
    // std::vector<std::pair<int, int>> frameMatches;  // 存储帧间匹配对
    // // 遍历当前帧的所有特征点，构建匹配关系
    // for(int i = 0; i < mCurrentFrame.N; i++)
    // {
    //     if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
    //     {
    //         MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];      
    //         // 查找这个地图点在上一帧中的观测
    //         auto observations = pMP->GetObservations();
    //         for(auto& obs : observations)
    //         {
    //             KeyFrame* pKF = obs.first;
    //             if(pKF == &mLastFrame)  // 检查是否是上一帧的观测
    //             {
    //                 int lastIdx = get<0>(obs.second);  // 上一帧中的特征点索引
    //                 frameMatches.push_back({lastIdx, i});  // 保存匹配对
    //                 break;
    //             }
    //         }
    //     }
    // }
    // // 计算当前帧相对于上一帧的相对位姿
    // Sophus::SE3f relativePose = mCurrentFrame.GetPose() * mLastFrame.GetPose().inverse();
    // // 保存匹配结果和相对位姿到当前帧
    // mCurrentFrame.SetFrameMatches(frameMatches);
    // mCurrentFrame.SetRelativePose(relativePose);
    // ============================================================================

    // 剔除异常点（外点）
    int nmatchesMap = 0;  // 有效地图点匹配数量
    for(int i =0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            // 如果是异常点，移除关联关系
            if(mCurrentFrame.mvbOutlier[i])
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

                // 清空当前帧的地图点关联
                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                mCurrentFrame.mvbOutlier[i]=false;
                
                // 更新地图点的可见性状态
                if(i < mCurrentFrame.Nleft){
                    pMP->mbTrackInView = false;  // 左视图不可见
                }
                else{
                    pMP->mbTrackInViewR = false; // 右视图不可见
                }
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;  // 记录最后被看到的帧ID
                nmatches--;  // 减少匹配计数
            }
            // 如果是有效的地图点（被多次观测），增加有效匹配计数
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)
                nmatchesMap++;
        }
    }

    // 如果是纯定位模式（不建图）
    if(mbOnlyTracking)
    {
        // 设置视觉里程计标志：有效地图点匹配少于10个时为VO模式
        mbVO = nmatchesMap<10;
        return nmatches>20;  // 总匹配数 大于20则跟踪成功
    }

    // 正常建图模式下的跟踪成功条件
    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
        return true;  // IMU传感器：总是返回成功
    else
    {
        // 振动自适应：放宽有效地图点匹配成功条件
        // 基础10，振动大时最低降至6（当前帧振动已计算完毕）
        int nMinMapMatches = std::max(6, static_cast<int>(10 - 2 * std::min(mfCurrentVibrationLevel, 2.0f)));
        return nmatchesMap>=nMinMapMatches;
    }
}
/**
 * 局部地图跟踪：使用局部地图中的地图点优化当前帧的位姿
 * 这是跟踪流程中的关键步骤，通过局部地图提供更多的约束来提高位姿估计的精度
 * 
 * @return bool 局部地图跟踪是否成功
 */
bool Tracking::TrackLocalMap()
{
    mTrackedFr++;  // 增加跟踪帧计数器
    
    // 更新局部地图：获取当前帧周围的局部关键帧和地图点
    UpdateLocalMap();
    // 在局部地图中搜索与当前帧特征点对应的地图点
    SearchLocalPoints();

    // 优化前检查异常点统计（调试用）
    int aux1 = 0, aux2=0;  // aux1: 总地图点数, aux2: 异常点数
    for(int i=0; i<mCurrentFrame.N; i++)
        if( mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            if(mCurrentFrame.mvbOutlier[i])
                aux2++;
        }

    int inliers;  // 内点数量
    // 根据IMU初始化状态选择不同的优化方法
    if (!mpAtlas->isImuInitialized())
    {
        // IMU未初始化：使用纯视觉位姿优化
        Optimizer::PoseOptimization(&mCurrentFrame);
    }
    else
    {
        // IMU已初始化：根据重定位历史选择优化策略
        if(mCurrentFrame.mnId<=mnLastRelocFrameId+mnFramesToResetIMU)
        {
            // 刚完成重定位（在重置帧数内）：使用纯视觉优化
            Verbose::PrintMess("TLM: PoseOptimization ", Verbose::VERBOSITY_DEBUG);
            Optimizer::PoseOptimization(&mCurrentFrame);
        }
        else
        {
            // 正常IMU跟踪：根据地图是否更新选择优化方法
            // if(!mbMapUpdated && mState == OK) //  && (mnMatchesInliers>30))
            if(!mbMapUpdated) // 地图未更新
            {
                // 使用上一帧的IMU信息进行惯性优化
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastFrame ", Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
            else
            {
                // 使用上一关键帧的IMU信息进行惯性优化
                Verbose::PrintMess("TLM: PoseInertialOptimizationLastKeyFrame ", Verbose::VERBOSITY_DEBUG);
                inliers = Optimizer::PoseInertialOptimizationLastKeyFrame(&mCurrentFrame); // , !mpLastKeyFrame->GetMap()->GetIniertialBA1());
            }
        }
    }

    // 优化后再次检查异常点统计（调试用）
    aux1 = 0, aux2 = 0;
    for(int i=0; i<mCurrentFrame.N; i++)
        if( mCurrentFrame.mvpMapPoints[i])
        {
            aux1++;
            if(mCurrentFrame.mvbOutlier[i])
                aux2++;
        }

    mnMatchesInliers = 0;  // 重置内点计数器

    // 更新地图点统计信息
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            if(!mCurrentFrame.mvbOutlier[i])  // 非异常点
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound();  // 增加地图点的发现次数
                
                // 根据跟踪模式统计内点
                if(!mbOnlyTracking)  // 非纯跟踪模式（正常建图模式）
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0)  // 地图点被观测次数大于0
                        mnMatchesInliers++;  // 统计为内点
                }
                else  // 纯跟踪模式
                    mnMatchesInliers++;  // 所有非异常点都统计为内点
            }
            else if(mSensor==System::STEREO)  // 双目传感器：移除异常点关联
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        }
    }

    // 判断跟踪是否成功
    // 如果最近进行了重定位，使用更严格的标准
    mpLocalMapper->mnMatchesInliers=mnMatchesInliers;  // 将内点数传递给局部建图器
    
    // 重定位后的严格检查：在重定位后的mMaxFrames帧内，内点数必须大于50
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50)
        return false;

    // 最近丢失状态下的宽松检查：振动大时进一步降低恢复门槛
    int nRecentlyLostMin = std::max(6, static_cast<int>(10 - 2 * std::min(mfCurrentVibrationLevel, 2.0f)));
    if((mnMatchesInliers>nRecentlyLostMin)&&(mState==RECENTLY_LOST))
        return true;

    // 根据传感器类型设置不同的成功阈值
    if (mSensor == System::IMU_MONOCULAR)  // IMU单目相机
    {
        if((mnMatchesInliers<15 && mpAtlas->isImuInitialized())||(mnMatchesInliers<50 && !mpAtlas->isImuInitialized()))
        {
            return false;  // IMU初始化后要求15个内点，未初始化要求50个
        }
        else
            return true;
    }
    else if (mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)  // IMU双目或RGBD
    {
        if(mnMatchesInliers<15)  // 要求15个内点
        {
            return false;
        }
        else
            return true;
    }
    else  // 纯视觉传感器（单目、双目、RGBD）
    {
        // 振动自适应：放宽内点数要求（基础30，振动大时最低降至15）
        int nMinInliers = std::max(15, static_cast<int>(30 - 8 * std::min(mfCurrentVibrationLevel, 2.0f)));
        if(mnMatchesInliers<nMinInliers)
            return false;
        else
            return true;
    }
}

bool Tracking::NeedNewKeyFrame()
{
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && !mpAtlas->GetCurrentMap()->isImuInitialized())
    {
        if (mSensor == System::IMU_MONOCULAR && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25)
            return true;
        else if ((mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) && (mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.25)
            return true;
        else
            return false;
    }

    if(mbOnlyTracking)
        return false;

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested()) {
        /*if(mSensor == System::MONOCULAR)
        {
            std::cout << "NeedNewKeyFrame: localmap stopped" << std::endl;
        }*/
        return false;
    }

    const int nKFs = mpAtlas->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
    {
        return false;
    }

    // 振动抑制：振动幅度过大时不插入关键帧（使用统一的振动等级）
    if(mbVibrationThresholdsInitialized && mfCurrentVibrationLevel > 2.0f) {
        // cout << "振动抑制：振动等级=" << mfCurrentVibrationLevel << "，不插入关键帧" << endl;
        return false;
    }

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3;
    if(nKFs<=2)
        nMinObs=2;
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;

    if(mSensor!=System::MONOCULAR && mSensor!=System::IMU_MONOCULAR)
    {
        int N = (mCurrentFrame.Nleft == -1) ? mCurrentFrame.N : mCurrentFrame.Nleft;
        for(int i =0; i<N; i++)
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth)
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    nTrackedClose++;
                else
                    nNonTrackedClose++;

            }
        }
        //Verbose::PrintMess("[NEEDNEWKF]-> closed points: " + to_string(nTrackedClose) + "; non tracked closed points: " + to_string(nNonTrackedClose), Verbose::VERBOSITY_NORMAL);// Verbose::VERBOSITY_DEBUG);
    }

    bool bNeedToInsertClose;
    bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2)
        thRefRatio = 0.4f;

    /*int nClosedPoints = nTrackedClose + nNonTrackedClose;
    const int thStereoClosedPoints = 15;
    if(nClosedPoints < thStereoClosedPoints && (mSensor==System::STEREO || mSensor==System::IMU_STEREO))
    {
        //Pseudo-monocular, there are not enough close points to be confident about the stereo observations.
        thRefRatio = 0.9f;
    }*/

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    if(mpCamera2) thRefRatio = 0.75f;

    if(mSensor==System::IMU_MONOCULAR)
    {
        if(mnMatchesInliers>350) // Points tracked from the local map
            thRefRatio = 0.75f;
        else
            thRefRatio = 0.90f;
    }

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = ((mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames) && bLocalMappingIdle); //mpLocalMapper->KeyframesInQueue() < 2);
    //Condition 1c: tracking is weak
    const bool c1c = mSensor!=System::MONOCULAR && mSensor!=System::IMU_MONOCULAR && mSensor!=System::IMU_STEREO && mSensor!=System::IMU_RGBD && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = (((mnMatchesInliers<nRefMatches*thRefRatio || bNeedToInsertClose)) && mnMatchesInliers>15);

    //std::cout << "NeedNewKF: c1a=" << c1a << "; c1b=" << c1b << "; c1c=" << c1c << "; c2=" << c2 << std::endl;
    // Temporal condition for Inertial cases
    bool c3 = false;
    if(mpLastKeyFrame)
    {
        if (mSensor==System::IMU_MONOCULAR)
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                c3 = true;
        }
        else if (mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD)
        {
            if ((mCurrentFrame.mTimeStamp-mpLastKeyFrame->mTimeStamp)>=0.5)
                c3 = true;
        }
    }

    bool c4 = false;
    if ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && (mSensor == System::IMU_MONOCULAR)) // MODIFICATION_2, originally ((((mnMatchesInliers<75) && (mnMatchesInliers>15)) || mState==RECENTLY_LOST) && ((mSensor == System::IMU_MONOCULAR)))
        c4=true;
    else
        c4=false;

    if(((c1a||c1b||c1c) && c2)||c3 ||c4)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle || mpLocalMapper->IsInitializing())
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if(mSensor!=System::MONOCULAR  && mSensor!=System::IMU_MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue()<3)
                    return true;
                else
                    return false;
            }
            else
            {
                //std::cout << "NeedNewKeyFrame: localmap is busy" << std::endl;
                return false;
            }
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    if(mpLocalMapper->IsInitializing() && !mpAtlas->isImuInitialized())
        return;

    if(!mpLocalMapper->SetNotStop(true))
        return;

    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpAtlas->GetCurrentMap(),mpKeyFrameDB);  // 1. 创建KeyFrame对象

    if(mpAtlas->isImuInitialized()) //  || mpLocalMapper->IsInitializing())
        pKF->bImu = true;

    pKF->SetNewBias(mCurrentFrame.mImuBias);

     // 2. 设置关键帧关系
    mpReferenceKF = pKF;                    // 设置参考关键帧
    mCurrentFrame.mpReferenceKF = pKF;      // 设置当前帧的参考关键帧
    
    // 3. 更新关键帧链
    if(mpLastKeyFrame)
    {
        pKF->mPrevKF = mpLastKeyFrame;  // 新关键帧指向上一个关键帧
        mpLastKeyFrame->mNextKF = pKF;  // 上一个关键帧指向新关键帧
    }
    else
        Verbose::PrintMess("No last KF in KF creation!!", Verbose::VERBOSITY_NORMAL);

    // Reset preintegration from last KF (Create new object)
    if (mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
    {
        mpImuPreintegratedFromLastKF = new IMU::Preintegrated(pKF->GetImuBias(),pKF->mImuCalib);
    }

    if(mSensor!=System::MONOCULAR && mSensor != System::IMU_MONOCULAR) // TODO check if incluide imu_stereo
    {
        mCurrentFrame.UpdatePoseMatrices();
        // cout << "create new MPs" << endl;
        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        int maxPoint = 100;
        if(mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD)
            maxPoint = 100;

        vector<pair<float,int> > vDepthIdx;
        int N = (mCurrentFrame.Nleft != -1) ? mCurrentFrame.Nleft : mCurrentFrame.N;
        vDepthIdx.reserve(mCurrentFrame.N);
        for(int i=0; i<N; i++)
        {
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                vDepthIdx.push_back(make_pair(z,i));
            }
        }

        if(!vDepthIdx.empty())
        {
            sort(vDepthIdx.begin(),vDepthIdx.end());

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                if(!pMP)
                    bCreateNew = true;
                else if(pMP->Observations()<1)
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
                }

                if(bCreateNew)
                {
                    Eigen::Vector3f x3D;

                    if(mCurrentFrame.Nleft == -1){
                        mCurrentFrame.UnprojectStereo(i, x3D);
                    }
                    else{
                        x3D = mCurrentFrame.UnprojectStereoFishEye(i);
                    }

                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpAtlas->GetCurrentMap());
                    pNewMP->AddObservation(pKF,i);

                    //Check if it is a stereo observation in order to not
                    //duplicate mappoints
                    if(mCurrentFrame.Nleft != -1 && mCurrentFrame.mvLeftToRightMatch[i] >= 0){
                        mCurrentFrame.mvpMapPoints[mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]]=pNewMP;
                        pNewMP->AddObservation(pKF,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                        pKF->AddMapPoint(pNewMP,mCurrentFrame.Nleft + mCurrentFrame.mvLeftToRightMatch[i]);
                    }

                    pKF->AddMapPoint(pNewMP,i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpAtlas->AddMapPoint(pNewMP);

                    mCurrentFrame.mvpMapPoints[i]=pNewMP;
                    nPoints++;
                }
                else
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>maxPoint)
                {
                    break;
                }
            }
            //Verbose::PrintMess("new mps for stereo KF: " + to_string(nPoints), Verbose::VERBOSITY_NORMAL);
        }
    }

    // 5. 发送给局部建图器
    mpLocalMapper->InsertKeyFrame(pKF);

    mpLocalMapper->SetNotStop(false);
    
    mnLastKeyFrameId = mCurrentFrame.mnId;

    // 4. 更新关键帧指针
    mpLastKeyFrame = pKF;   // 更新最后一个关键帧指针
}

/**
 * 搜索局部地图点：在局部地图中寻找与当前帧特征点对应的地图点
 * 这是局部地图跟踪的关键步骤，通过投影匹配增加跟踪的约束条件
 */
void Tracking::SearchLocalPoints()
{
    // 第一步：清理已匹配的地图点，避免重复搜索
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())  // 地图点已标记为坏点
            {
                *vit = static_cast<MapPoint*>(NULL);  // 清除无效的地图点引用
            }
            else
            {
                pMP->IncreaseVisible();           // 增加地图点的可见次数
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;  // 记录最后被看到的帧ID
                pMP->mbTrackInView = false;       // 重置左视图跟踪标志
                pMP->mbTrackInViewR = false;      // 重置右视图跟踪标志
            }
        }
    }

    int nToMatch=0;  // 需要匹配的地图点数量

    // 第二步：投影局部地图点到当前帧并检查可见性
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;

        // 跳过在当前帧已经处理过的地图点
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        // 跳过坏点
        if(pMP->isBad())
            continue;
        // 跳过高动态概率地图点
        if(pMP->IsDynamicMapPoint())
            continue;
        
        // 投影地图点到当前帧视锥体内（此操作会填充地图点的匹配变量）
        if(mCurrentFrame.isInFrustum(pMP,0.5))  // 0.5为视锥体缩放因子
        {
            pMP->IncreaseVisible();  // 增加可见次数
            nToMatch++;              // 增加待匹配计数
        }
        
        // 如果地图点在当前视锥体内可见，记录投影坐标
        if(pMP->mbTrackInView)
        {
            mCurrentFrame.mmProjectPoints[pMP->mnId] = cv::Point2f(pMP->mTrackProjX, pMP->mTrackProjY);
        }
    }

    // 第三步：如果有需要匹配的地图点，执行投影匹配
    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);  // 初始化匹配器，最小距离比阈值为0.8
        int th = 1;  // 默认搜索半径阈值
        
        // 根据传感器类型调整搜索阈值
        if(mSensor==System::RGBD || mSensor==System::IMU_RGBD)
            th=3;  // RGBD传感器：阈值较大（深度信息更可靠）
            
        // IMU初始化状态相关的阈值调整
        if(mpAtlas->isImuInitialized())
        {
            if(mpAtlas->GetCurrentMap()->GetIniertialBA2())  // 已完成惯性BA2
                th=2;  // 阈值较小（位姿估计更准确）
            else
                th=6;  // 阈值较大（位姿估计不确定性高）
        }
        else if(!mpAtlas->isImuInitialized() && (mSensor==System::IMU_MONOCULAR || mSensor==System::IMU_STEREO || mSensor == System::IMU_RGBD))
        {
            th=10;  // IMU未初始化：阈值最大（不确定性最高）
        }

        // 特殊状态下的阈值调整
        // 如果相机最近进行了重定位，执行更粗糙的搜索
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)  // 重定位后的2帧内
            th=5;  // 使用较大的搜索半径

        // 丢失状态下的阈值调整
        if(mState==LOST || mState==RECENTLY_LOST) // 丢失或最近丢失状态（1秒内）
            th=15; // 使用最大的搜索半径（15）

        // 执行投影匹配：在当前帧中搜索与局部地图点对应的特征点
        int matches = matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th, mpLocalMapper->mbFarPoints, mpLocalMapper->mThFarPoints);
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpAtlas->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    int count_pts = 0;

    for(vector<KeyFrame*>::const_reverse_iterator itKF=mvpLocalKeyFrames.rbegin(), itEndKF=mvpLocalKeyFrames.rend(); itKF!=itEndKF; ++itKF)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {

            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
                continue;
            if(!pMP->isBad())
            {
                count_pts++;
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}

void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter;
    if(!mpAtlas->isImuInitialized() || (mCurrentFrame.mnId<mnLastRelocFrameId+2))
    {
        for(int i=0; i<mCurrentFrame.N; i++)
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();
                    for(map<KeyFrame*,tuple<int,int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                        keyframeCounter[it->first]++;
                }
                else
                {
                    mCurrentFrame.mvpMapPoints[i]=NULL;
                }
            }
        }
    }
    else
    {
        for(int i=0; i<mLastFrame.N; i++)
        {
            // Using lastframe since current frame has not matches yet
            if(mLastFrame.mvpMapPoints[i])
            {
                MapPoint* pMP = mLastFrame.mvpMapPoints[i];
                if(!pMP)
                    continue;
                if(!pMP->isBad())
                {
                    const map<KeyFrame*,tuple<int,int>> observations = pMP->GetObservations();
                    for(map<KeyFrame*,tuple<int,int>>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                        keyframeCounter[it->first]++;
                }
                else
                {
                    // MODIFICATION
                    mLastFrame.mvpMapPoints[i]=NULL;
                }
            }
        }
    }
    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL);

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())
            continue;

        if(it->second>max)
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(pKF);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80) // 80
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);
        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }
    }

    // Add 10 last temporal KFs (mainly for IMU)
    if((mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_STEREO || mSensor == System::IMU_RGBD) &&mvpLocalKeyFrames.size()<80)
    {
        KeyFrame* tempKeyFrame = mCurrentFrame.mpLastKeyFrame;

        const int Nd = 20;
        for(int i=0; i<Nd; i++){
            if (!tempKeyFrame)
                break;
            if(tempKeyFrame->mnTrackReferenceForFrame!=mCurrentFrame.mnId)
            {
                mvpLocalKeyFrames.push_back(tempKeyFrame);
                tempKeyFrame->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                tempKeyFrame=tempKeyFrame->mPrevKF;
            }
        }
    }

    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    Verbose::PrintMess("Starting relocalization", Verbose::VERBOSITY_NORMAL);
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame, mpAtlas->GetCurrentMap());

    if(vpCandidateKFs.empty()) {
        Verbose::PrintMess("There are not candidates", Verbose::VERBOSITY_NORMAL);
        return false;
    }

    const int nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75,true);

    vector<MLPnPsolver*> vpMLPnPsolvers;
    vpMLPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates=0;

    for(int i=0; i<nKFs; i++)
    {
        KeyFrame* pKF = vpCandidateKFs[i];
        if(pKF->isBad())
            vbDiscarded[i] = true;
        else
        {
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]);
            if(nmatches<15)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                MLPnPsolver* pSolver = new MLPnPsolver(mCurrentFrame,vvpMapPointMatches[i]);
                pSolver->SetRansacParameters(0.99,10,300,6,0.5,5.991);  //This solver needs at least 6 points
                vpMLPnPsolvers[i] = pSolver;
                nCandidates++;
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true);

    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nKFs; i++)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            MLPnPsolver* pSolver = vpMLPnPsolvers[i];
            Eigen::Matrix4f eigTcw;
            bool bTcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers, eigTcw);

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(bTcw)
            {
                Sophus::SE3f Tcw(eigTcw);
                mCurrentFrame.SetPose(Tcw);
                // Tcw.copyTo(mCurrentFrame.mTcw);

                set<MapPoint*> sFound;

                const int np = vbInliers.size();

                for(int j=0; j<np; j++)
                {
                    if(vbInliers[j])
                    {
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]);
                    }
                    else
                        mCurrentFrame.mvpMapPoints[j]=NULL;
                }

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                if(nGood<10)
                    continue;

                for(int io =0; io<mCurrentFrame.N; io++)
                    if(mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL);

                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50)
                        {
                            sFound.clear();
                            for(int ip =0; ip<mCurrentFrame.N; ip++)
                                if(mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for(int io =0; io<mCurrentFrame.N; io++)
                                    if(mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io]=NULL;
                            }
                        }
                    }
                }
                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if(!bMatch)
    {
        return false;
    }
    else
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        cout << "Relocalized!!" << endl;
        return true;
    }

}

void Tracking::Reset(bool bLocMap)
{
    Verbose::PrintMess("System Reseting", Verbose::VERBOSITY_NORMAL);

    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    // Reset Local Mapping
    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_NORMAL);
        mpLocalMapper->RequestReset();
        Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);
    }
    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestReset();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database...", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clear();
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearAtlas();
    mpAtlas->CreateNewMap();
    if (mSensor==System::IMU_STEREO || mSensor == System::IMU_MONOCULAR || mSensor == System::IMU_RGBD)
        mpAtlas->SetInertialSensor();
    mnInitialFrameId = 0;

    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET;

    mbReadyToInitializate = false;
    mbSetInit=false;

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();
    mCurrentFrame = Frame();
    mnLastRelocFrameId = 0;
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

void Tracking::ResetActiveMap(bool bLocMap)
{
    Verbose::PrintMess("Active map Reseting", Verbose::VERBOSITY_NORMAL);
    if(mpViewer)
    {
        mpViewer->RequestStop();
        while(!mpViewer->isStopped())
            usleep(3000);
    }

    Map* pMap = mpAtlas->GetCurrentMap();

    if (!bLocMap)
    {
        Verbose::PrintMess("Reseting Local Mapper...", Verbose::VERBOSITY_VERY_VERBOSE);
        mpLocalMapper->RequestResetActiveMap(pMap);
        Verbose::PrintMess("done", Verbose::VERBOSITY_VERY_VERBOSE);
    }

    // Reset Loop Closing
    Verbose::PrintMess("Reseting Loop Closing...", Verbose::VERBOSITY_NORMAL);
    mpLoopClosing->RequestResetActiveMap(pMap);
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear BoW Database
    Verbose::PrintMess("Reseting Database", Verbose::VERBOSITY_NORMAL);
    mpKeyFrameDB->clearMap(pMap); // Only clear the active map references
    Verbose::PrintMess("done", Verbose::VERBOSITY_NORMAL);

    // Clear Map (this erase MapPoints and KeyFrames)
    mpAtlas->clearMap();
    //KeyFrame::nNextId = mpAtlas->GetLastInitKFid();
    //Frame::nNextId = mnLastInitFrameId;
    mnLastInitFrameId = Frame::nNextId;
    //mnLastRelocFrameId = mnLastInitFrameId;
    mState = NO_IMAGES_YET; //NOT_INITIALIZED;

    mbReadyToInitializate = false;

    list<bool> lbLost;
    // lbLost.reserve(mlbLost.size());
    unsigned int index = mnFirstFrameId;
    cout << "mnFirstFrameId = " << mnFirstFrameId << endl;
    for(Map* pMap : mpAtlas->GetAllMaps())
    {
        if(pMap->GetAllKeyFrames().size() > 0)
        {
            if(index > pMap->GetLowerKFID())
                index = pMap->GetLowerKFID();
        }
    }

    //cout << "First Frame id: " << index << endl;
    int num_lost = 0;
    cout << "mnInitialFrameId = " << mnInitialFrameId << endl;

    for(list<bool>::iterator ilbL = mlbLost.begin(); ilbL != mlbLost.end(); ilbL++)
    {
        if(index < mnInitialFrameId)
            lbLost.push_back(*ilbL);
        else
        {
            lbLost.push_back(true);
            num_lost += 1;
        }

        index++;
    }
    cout << num_lost << " Frames set to lost" << endl;

    mlbLost = lbLost;

    mnInitialFrameId = mCurrentFrame.mnId;
    mnLastRelocFrameId = mCurrentFrame.mnId;

    mCurrentFrame = Frame();
    mLastFrame = Frame();
    mpReferenceKF = static_cast<KeyFrame*>(NULL);
    mpLastKeyFrame = static_cast<KeyFrame*>(NULL);
    mvIniMatches.clear();

    mbVelocity = false;

    if(mpViewer)
        mpViewer->Release();

    Verbose::PrintMess("   End reseting! ", Verbose::VERBOSITY_NORMAL);
}

vector<MapPoint*> Tracking::GetLocalMapMPS()
{
    return mvpLocalMapPoints;
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    mK_.setIdentity();
    mK_(0,0) = fx;
    mK_(1,1) = fy;
    mK_(0,2) = cx;
    mK_(1,2) = cy;

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}

void Tracking::UpdateFrameIMU(const float s, const IMU::Bias &b, KeyFrame* pCurrentKeyFrame)
{
    Map * pMap = pCurrentKeyFrame->GetMap();
    unsigned int index = mnFirstFrameId;
    list<ORB_SLAM3::KeyFrame*>::iterator lRit = mlpReferences.begin();
    list<bool>::iterator lbL = mlbLost.begin();
    for(auto lit=mlRelativeFramePoses.begin(),lend=mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        while(pKF->isBad())
        {
            pKF = pKF->GetParent();
        }

        if(pKF->GetMap() == pMap)
        {
            (*lit).translation() *= s;
        }
    }

    mLastBias = b;

    mpLastKeyFrame = pCurrentKeyFrame;

    mLastFrame.SetNewBias(mLastBias);
    mCurrentFrame.SetNewBias(mLastBias);

    while(!mCurrentFrame.imuIsPreintegrated())
    {
        usleep(500);
    }
    if(mLastFrame.mnId == mLastFrame.mpLastKeyFrame->mnFrameId)
    {
        mLastFrame.SetImuPoseVelocity(mLastFrame.mpLastKeyFrame->GetImuRotation(),
                                      mLastFrame.mpLastKeyFrame->GetImuPosition(),
                                      mLastFrame.mpLastKeyFrame->GetVelocity());
    }
    else
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);
        const Eigen::Vector3f twb1 = mLastFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mLastFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mLastFrame.mpLastKeyFrame->GetVelocity();
        float t12 = mLastFrame.mpImuPreintegrated->dT;

        mLastFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mLastFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    if (mCurrentFrame.mpImuPreintegrated)
    {
        const Eigen::Vector3f Gz(0, 0, -IMU::GRAVITY_VALUE);

        const Eigen::Vector3f twb1 = mCurrentFrame.mpLastKeyFrame->GetImuPosition();
        const Eigen::Matrix3f Rwb1 = mCurrentFrame.mpLastKeyFrame->GetImuRotation();
        const Eigen::Vector3f Vwb1 = mCurrentFrame.mpLastKeyFrame->GetVelocity();
        float t12 = mCurrentFrame.mpImuPreintegrated->dT;

        mCurrentFrame.SetImuPoseVelocity(IMU::NormalizeRotation(Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaRotation()),
                                      twb1 + Vwb1*t12 + 0.5f*t12*t12*Gz+ Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaPosition(),
                                      Vwb1 + Gz*t12 + Rwb1*mCurrentFrame.mpImuPreintegrated->GetUpdatedDeltaVelocity());
    }

    mnFirstImuFrameId = mCurrentFrame.mnId;
}

void Tracking::NewDataset()
{
    mnNumDataset++;
}

int Tracking::GetNumberDataset()
{
    return mnNumDataset;
}

int Tracking::GetMatchesInliers()
{
    return mnMatchesInliers;
}

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, string strFolder)
{
    mpSystem->SaveTrajectoryEuRoC(strFolder + strNameFile_frames);
    //mpSystem->SaveKeyFrameTrajectoryEuRoC(strFolder + strNameFile_kf);
}

void Tracking::SaveSubTrajectory(string strNameFile_frames, string strNameFile_kf, Map* pMap)
{
    mpSystem->SaveTrajectoryEuRoC(strNameFile_frames, pMap);
    if(!strNameFile_kf.empty())
        mpSystem->SaveKeyFrameTrajectoryEuRoC(strNameFile_kf, pMap);
}

float Tracking::GetImageScale()
{
    return mImageScale;
}

#ifdef REGISTER_LOOP
void Tracking::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
}

bool Tracking::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Tracking STOP" << endl;
        return true;
    }

    return false;
}

bool Tracking::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

bool Tracking::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

void Tracking::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
    mbStopRequested = false;
}
#endif

// ============================================================================
// 特征点筛选和动态点检测功能实现
// ============================================================================

/**
 * 收集帧间特征点匹配结果（用于动态点检测）
 * 使用哈希表将原 O(N²) 嵌套循环降为 O(N)：
 * 1. 先遍历当前帧建 MapPoint→索引 哈希表
 * 2. 再遍历上一帧时 O(1) 查找匹配，避免内层全扫描
 */
void Tracking::CollectFrameMatches()
{
    std::vector<std::pair<int, int>> frameMatches;
    frameMatches.reserve(std::min(mLastFrame.N, mCurrentFrame.N));

    std::unordered_map<MapPoint*, int> currMPToIdx;
    for(int i = 0; i < mCurrentFrame.N; i++)
    {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
        if(pMP && !mCurrentFrame.mvbOutlier[i])
            currMPToIdx[pMP] = i;
    }

    for(int lastIdx = 0; lastIdx < mLastFrame.N; lastIdx++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[lastIdx];
        if(!pMP || mLastFrame.mvbOutlier[lastIdx])
            continue;

        auto it = currMPToIdx.find(pMP);
        if(it != currMPToIdx.end())
        {
            frameMatches.push_back({lastIdx, it->second});
            currMPToIdx.erase(it);
        }
    }

    Sophus::SE3f relativePose = mCurrentFrame.GetPose() * mLastFrame.GetPose().inverse();
    mCurrentFrame.SetFrameMatches(frameMatches);
    mCurrentFrame.SetRelativePose(relativePose);
}

/**
 * 每帧计算振动强度参数（旋转角、垂直位移）
 * 
 * 从帧间相对位姿中提取扑翼振动的两个关键指标：
 * - 旋转角度：反映俯仰/翻滚振动强度（弧度）
 * - 垂直位移比例：帧间归一化平移向量在相机Z轴上的投影分量，
 *   反映运动方向中竖直分量的占比。平稳前飞时接近0，振动时增大。
 * 
 * 参考静态姿态的计算方式：将平移向量归一化后投影到(0,0,1)方向，
 * 消除飞行速度对绝对值的影响，仅保留方向变化信息。
 */
void Tracking::CalculateVibrationMetrics()
{
    Sophus::SE3f relativePose = mCurrentFrame.GetRelativePose();
    
    Eigen::Vector3d t = relativePose.translation().cast<double>();
    Eigen::Matrix3d R = relativePose.rotationMatrix().cast<double>();
    
    Eigen::AngleAxisd angleAxis(R);
    double rotationAngle = angleAxis.angle();
    
    if(rotationAngle > M_PI/2) {
        rotationAngle = M_PI - rotationAngle;
    }
    
    // 垂直方向分量：归一化平移向量在相机Z轴上的投影（与staticPose方法一致）
    double verticalDisplacement = 0.0;
    if(t.norm() > 1e-6) {
        Eigen::Vector3d t_normalized = t / t.norm();
        Eigen::Vector3d verticalDirection(0, 0, 1);
        verticalDisplacement = std::abs(t_normalized.dot(verticalDirection));
    }
    
    mCurrentFrame.SetVibrationMetrics(static_cast<float>(rotationAngle), static_cast<float>(verticalDisplacement));
    
    // 计算当前帧振动等级并存储（供后续 TrackLocalMap/NeedNewKeyFrame 使用）
    mfCurrentVibrationLevel = ComputeVibrationLevel(static_cast<float>(rotationAngle), static_cast<float>(verticalDisplacement));
    
    if(!mbVibrationThresholdsInitialized) {
        mRotationHistory.push_back(rotationAngle);
        mVerticalHistory.push_back(verticalDisplacement);
        
        if(mRotationHistory.size() > INITIALIZATION_PHASE) mRotationHistory.erase(mRotationHistory.begin());
        if(mVerticalHistory.size() > INITIALIZATION_PHASE) mVerticalHistory.erase(mVerticalHistory.begin());
        
        if(mRotationHistory.size() >= INITIALIZATION_PHASE && mVerticalHistory.size() >= INITIALIZATION_PHASE) {
            InitializeVibrationThresholds();
            mbVibrationThresholdsInitialized = true;
        }
    }
    
    // 更新上一帧振动等级（供下一帧匹配前使用）
    mfPrevVibrationLevel = mfCurrentVibrationLevel;
}


/**
 * 根据目标检测结果筛选特征点
 * 将特征点分为静态队列和语义队列
 */
bool Tracking::FilterFeaturePointsByDetection()
{
    // 清空之前的分类结果
    mCurrentFrame.ClearSemanticData();
    
    std::vector<std::pair<int, int>> staticMatches;    // 静态特征点匹配队列
    std::vector<std::pair<int, int>> semanticMatches;  // 语义特征点匹配队列（含行人/车辆等所有检测框内目标）

    // 获取帧间匹配结果
    const auto& matches = mCurrentFrame.GetFrameMatches();
    
    if(matches.empty()) {
        return false;
    }
    if(mCurrentFrame.detectedBoxes.empty() && mLastFrame.detectedBoxes.empty()) {
        return false;
    }

    // ── 使用 ComputeSemanticClassForKeys() 的预计算结果，避免重复检测框判断 ──

    // 遍历所有匹配的特征点对，进行分类
    for(const auto& match : matches) {
        int lastIdx = match.first;
        int currIdx = match.second;
        
        if(lastIdx >= mLastFrame.N || currIdx >= mCurrentFrame.N) {
            continue;
        }
        
        // 继承上一帧匹配点的动态观测计数器
        if(currIdx < mCurrentFrame.mvpMapPoints.size() && lastIdx < mLastFrame.mvpMapPoints.size()) {
            MapPoint* pCurrMP = mCurrentFrame.mvpMapPoints[currIdx];
            MapPoint* pLastMP = mLastFrame.mvpMapPoints[lastIdx];
            if(pCurrMP && pLastMP && pCurrMP == pLastMP) {
                pCurrMP->mnObservedDynamic = pLastMP->mnObservedDynamic;
            }
        }
        
        // 复用预计算的语义分类结果（O(1)查询）
        int lastClassId = mLastFrame.GetSemanticClassAt(lastIdx);
        int currClassId = mCurrentFrame.GetSemanticClassAt(currIdx);
        
        bool lastInBox = (lastClassId >= 0);
        bool currInBox = (currClassId >= 0);
        
        if(lastInBox || currInBox) {
            semanticMatches.push_back(match);
            if(currIdx < mCurrentFrame.mvpMapPoints.size()) {
                MapPoint* pCurrMP = mCurrentFrame.mvpMapPoints[currIdx];
                if(pCurrMP) {
                    pCurrMP->mFeatureStatus = MapPoint::SEMANTIC;
                    if(currInBox && currClassId >= 0)
                        pCurrMP->mnSemanticClass = currClassId;
                    else if(lastInBox && lastClassId >= 0)
                        pCurrMP->mnSemanticClass = lastClassId;
                }
            }
        } else {
            staticMatches.push_back(match);
            if(currIdx < mCurrentFrame.mvpMapPoints.size()) {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[currIdx];
                if(pMP) {
                    pMP->mFeatureStatus = MapPoint::STATIC;
                }
            }
        }
    }
    
    // 保存分类结果到当前帧
    mCurrentFrame.SetStaticMatches(staticMatches);
    mCurrentFrame.SetSemanticMatches(semanticMatches);

    return true;
}


/**
 * 使用静态特征点计算位姿（不依赖地图点）
 * 只使用不在目标检测区域内的特征点进行位姿估计
 * 使用 2D-2D 匹配计算本质矩阵，然后分解得到相对位姿
 * 
 * @return Sophus::SE3f 使用静态点计算的相对位姿（上一帧 -> 当前帧）
 */
Sophus::SE3f Tracking::ComputePoseWithStaticPoints()
{
    // 获取静态特征点匹配队列
    const auto& staticMatches = mCurrentFrame.GetStaticMatches();
    
    if(staticMatches.empty()) {
        // Verbose::PrintMess("警告: 没有静态特征点可用于位姿计算", Verbose::VERBOSITY_NORMAL);
        cout << "[ComputePoseWithStaticPoints] 没有静态特征点可用于位姿计算" << endl;
        return Sophus::SE3f();  // 返回单位矩阵
    }
    
    // 准备特征点对
    std::vector<cv::Point2f> points1, points2;
    for(const auto& match : staticMatches) {
        int lastIdx = match.first;   // 上一帧特征点索引
        int currIdx = match.second;  // 当前帧特征点索引
        
        // 检查索引是否有效
        if(lastIdx >= mLastFrame.N || currIdx >= mCurrentFrame.N) {
            cout << "[ComputePoseWithStaticPoints] 地图点无效索引: " << lastIdx << " -> " << currIdx << endl;
            continue;
        }
        
        points1.push_back(mLastFrame.mvKeysUn[lastIdx].pt);
        points2.push_back(mCurrentFrame.mvKeysUn[currIdx].pt);
    }
    
    if(points1.size() < 8) {
        // Verbose::PrintMess("警告: 静态特征点数量不足（至少需要8个）", Verbose::VERBOSITY_NORMAL);
        cout << "[ComputePoseWithStaticPoints] 静态特征点数量不足（至少需要8个）" << endl;
        return Sophus::SE3f();  // 返回单位矩阵
    }
    
    // 大场景（50m高度）下静态点过多（300-500），RANSAC开销大
    // 均匀采样限制到200点以内，精度影响可忽略
    if(points1.size() > 200) {
        std::vector<cv::Point2f> sampled1, sampled2;
        float step = static_cast<float>(points1.size()) / 200.0f;
        for(float fi = 0.0f; static_cast<int>(fi) < static_cast<int>(points1.size()); fi += step) {
            int idx = static_cast<int>(fi);
            sampled1.push_back(points1[idx]);
            sampled2.push_back(points2[idx]);
        }
        points1.swap(sampled1);
        points2.swap(sampled2);
    }
    
    // 相机内参（直接使用成员变量，避免每帧 clone 3×3 矩阵的开销）
    const cv::Mat& K = mCurrentFrame.mK;
    
    // RANSAC 计算本质矩阵（置信度0.99，比0.999快约60%）
    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(points2, points1, K, cv::RANSAC, 0.99, 1.0, mask);
    
    if(E.empty()) {
        // Verbose::PrintMess("警告: 无法计算本质矩阵", Verbose::VERBOSITY_NORMAL);
        cout << "[ComputePoseWithStaticPoints] 无法计算本质矩阵" << endl;
        return Sophus::SE3f();
    }
    
    // 分解本质矩阵得到旋转和平移
    cv::Mat R, t;
    int inliers = cv::recoverPose(E, points2, points1, K, R, t, mask);
    
    // cout << "静态点 2D-2D 位姿计算完成: 内点=" << inliers << "/" << points1.size() << endl;
    
    // 将 cv::Mat 转换为 Eigen 矩阵
    Eigen::Matrix3f eigenR;
    Eigen::Vector3f eigent;
    
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            eigenR(i, j) = R.at<double>(i, j);
        }
        eigent(i) = t.at<double>(i);
    }
    
    // 构建 SE3 变换矩阵（上一帧 -> 当前帧）
    Sophus::SE3f staticPose(eigenR, eigent);
    
    return staticPose;
}

/**
 * 分类语义点（静态/动态）- 极线约束方法（抗振动优化）
 * 使用对极约束直接计算点到极线的距离，针对4Hz扑翼振动优化
 * 加入多帧一致性检测和自适应阈值
 * 
 * @param staticPose 使用静态点计算的相对位姿
 */
void Tracking::ClassifySemanticPoints(const Sophus::SE3f& staticPose)
{

    const auto& semanticMatches = mCurrentFrame.GetSemanticMatches();
    
    if(semanticMatches.empty()) {
        return;  // 没有语义点需要分类
    }
    
    // 从当前帧读取已计算的振动强度参数
    double rotationAngle = static_cast<double>(mCurrentFrame.GetRotationAngle());
    double verticalDisplacement = static_cast<double>(mCurrentFrame.GetVerticalDisplacement());
    
    double rotationThreshold = mOptimizedRotationThreshold;
    double dispThreshold = mOptimizedVerticalThreshold;
    
    double rotationExceedRatio = 0.0;
    if(rotationAngle > rotationThreshold) {
        rotationExceedRatio = (rotationAngle - rotationThreshold) / rotationThreshold;
    }

    double dispExceedRatio = 0.0;
    if(verticalDisplacement > dispThreshold) {
        dispExceedRatio = (verticalDisplacement - dispThreshold) / dispThreshold;
    }
    
    
    // 基于旋转和位移超出百分比的动态阈值调整
    bool isVibrationDominant = false;
    double maxExceedRatio = 0.0;
    
    if(rotationExceedRatio > 0) {
        isVibrationDominant = true;
        maxExceedRatio = std::max(maxExceedRatio, rotationExceedRatio);
    }
    
    if(dispExceedRatio > 0) {
        isVibrationDominant = true;
        maxExceedRatio = std::max(maxExceedRatio, dispExceedRatio);
    }
    
    
    // 计算本质矩阵 E = [t]_{×} * R（使用 double）
    Eigen::Vector3d t = staticPose.translation().cast<double>();
    Eigen::Matrix3d R = staticPose.rotationMatrix().cast<double>();
    Eigen::Matrix3d t_x;
    t_x << 0, -t(2), t(1),
           t(2), 0, -t(0),
          -t(1), t(0), 0;
    Eigen::Matrix3d E = t_x * R;
    
    // cout << "本质矩阵 E:\n" << E << endl;
    
    // K 逆矩阵预计算：首帧执行一次，后续复用（相机内参恒定不变）
    static Eigen::Matrix3d sK_inv = Eigen::Matrix3d::Zero();
    static Eigen::Matrix3d sK_inv_T = Eigen::Matrix3d::Zero();
    static bool sKPrecomputed = false;
    if(!sKPrecomputed)
    {
        Eigen::Matrix3d K_eigen = ORB_SLAM3::Converter::toMatrix3d(mCurrentFrame.mK);
        sK_inv = K_eigen.inverse();
        sK_inv_T = sK_inv.transpose();
        sKPrecomputed = true;
    }

    // 计算基础矩阵 F = K^{-T} * E * K^{-1}
    Eigen::Matrix3d F = sK_inv_T * E * sK_inv;
    
    // cout << "基础矩阵 F:\n" << F << endl;

    int count = 0;
    double totalDistance = 0.0;
    std::vector<double> epipolarDistances;  // 存储所有点的对极距离
    
    // 第一步：收集所有点的对极距离
    for(const auto& match : semanticMatches) {
        int lastIdx = match.first;   // 上一帧特征点索引
        int currIdx = match.second;  // 当前帧特征点索引
        
        // 检查索引是否有效
        if(lastIdx >= mLastFrame.N || currIdx >= mCurrentFrame.N) {
            continue;
        }
        
        // 获取特征点坐标（像素坐标）
        cv::Point2f lastPt = mLastFrame.mvKeysUn[lastIdx].pt;
        cv::Point2f currPt = mCurrentFrame.mvKeysUn[currIdx].pt;
        
        // 转换为齐次坐标（像素坐标系，double类型）
        Eigen::Vector3d p1(lastPt.x, lastPt.y, 1.0);
        Eigen::Vector3d p2(currPt.x, currPt.y, 1.0);
        
        // 使用基础矩阵 F 计算对极距离
        Eigen::Vector3d Fp1 = F * p1;
        double numerator = std::abs(p2.transpose() * Fp1);
        double denominator = std::sqrt(Fp1(0)*Fp1(0) + Fp1(1)*Fp1(1));
        
        double epipolarDistance = 0.0;
        if(denominator > 1e-8) {
            epipolarDistance = numerator / denominator;
        }
        
        epipolarDistances.push_back(epipolarDistance);
        totalDistance += epipolarDistance;
        count++;
    }
    
    // 自适应阈值计算
    double adaptiveThreshold = 2.0f;  // 基础阈值
    
    if(!epipolarDistances.empty()) {
        // 计算中位值（对异常值更鲁棒）
        std::vector<double> sortedDistances = epipolarDistances;
        std::sort(sortedDistances.begin(), sortedDistances.end());
        double medianDistance = sortedDistances[sortedDistances.size() / 2];    // 中位值
        
        // 计算平均值
        double meanDistance = totalDistance / count;
        
        // 计算标准差
        double variance = 0.0;
        for(double dist : epipolarDistances) {
            variance += (dist - meanDistance) * (dist - meanDistance);
        }
        double stdDev = std::sqrt(variance / count);
        
        // 统计阈值策略：中位值 + 2倍标准差
        adaptiveThreshold = 0.8 * medianDistance + 1.5 * stdDev;
        // adaptiveThreshold = medianDistance + 2.0 * stdDev;
        
        // 设置最小阈值，避免阈值过低
        adaptiveThreshold = std::max(adaptiveThreshold, 1.0);
        
        if(isVibrationDominant) {
        // 根据最大超出百分比调整动态阈值
        // 超出百分比越大，阈值调整幅度越大
        double adjustmentFactor = 1.0 + maxExceedRatio * 0.6f;  // 每超出100%，阈值增加50%

        adaptiveThreshold *= adjustmentFactor;
        
        // cout << "[振动检测] 检测到振动主导运动: " << (maxExceedRatio * 100) 
        //      << "%, 调整因子: " << adjustmentFactor 
        //      << ", 放宽阈值至: " << adaptiveThreshold << "像素" << endl;
        }
    }
    
    // 第三步：使用统计阈值进行分类
    // 重置计数器
    int dynamicPoints = 0;
    int staticPoints = 0;
    std::set<long unsigned int> setClassifiedMPs;  // 防止同一MapPoint重复分类

    float imageCenterX = (Frame::mnMaxX + Frame::mnMinX) * 0.5f;
    float imageCenterY = (Frame::mnMaxY + Frame::mnMinY) * 0.5f;
    float maxRadiusX = (Frame::mnMaxX - Frame::mnMinX) * 0.5f;
    float maxRadiusY = (Frame::mnMaxY - Frame::mnMinY) * 0.5f;

    
    for(size_t i = 0; i < semanticMatches.size(); i++) {
        const auto& match = semanticMatches[i];
        int lastIdx = match.first;   // 上一帧特征点索引
        int currIdx = match.second;  // 当前帧特征点索引
        
        // 检查索引是否有效
        if(lastIdx >= mLastFrame.N || currIdx >= mCurrentFrame.N) {
            continue;
        }

        // MapPoint级去重：同一帧内同一MapPoint只分类一次
        MapPoint* pMP = nullptr;
        if(currIdx < mCurrentFrame.mvpMapPoints.size()) {
            pMP = mCurrentFrame.mvpMapPoints[currIdx];
        }
        if(pMP && !setClassifiedMPs.insert(pMP->mnId).second) {
            continue;
        }
        
        // 获取特征点坐标（像素坐标）
        cv::Point2f lastPt = mLastFrame.mvKeysUn[lastIdx].pt;
        cv::Point2f currPt = mCurrentFrame.mvKeysUn[currIdx].pt;
        
        float dx = (currPt.x - imageCenterX) / maxRadiusX;
        float dy = (currPt.y - imageCenterY) / maxRadiusY;
        float edgeFactor = std::max(std::abs(dx), std::abs(dy));
        
        double adaptiveEdgeThreshold = adaptiveThreshold * (1.0f + 0.4f * edgeFactor);
        
        double epipolarDistance = epipolarDistances[i];
        
        // 按语义类别应用阈值比例（common.h THRESHOLD_RATIOS），不同类别运动幅度不同
        double classRatio = 1.0;  // 默认不变
        if(pMP && pMP->mnSemanticClass >= 0 && pMP->mnSemanticClass < static_cast<int>(THRESHOLD_RATIOS.size())) {
            classRatio = static_cast<double>(THRESHOLD_RATIOS[pMP->mnSemanticClass]);
        }
        double actualThreshold = adaptiveEdgeThreshold * classRatio;
        const double MIN_DYNAMIC_THRESHOLD = 1.0;  // 绝对下限，避免阈值过低导致噪声误触发
        if(actualThreshold < MIN_DYNAMIC_THRESHOLD) actualThreshold = MIN_DYNAMIC_THRESHOLD;
        bool isDynamic = (epipolarDistance > actualThreshold);

        // 回退逻辑：极线约束未判定为动态，但历史动态观测计数器 >= 3，依然判定为动态
        if(!isDynamic && pMP && pMP->mnObservedDynamic >= 3) {
            // cout << "[ClassifySemantic] MapPoint " << pMP->mnId << " mnObservedDynamic=" << pMP->mnObservedDynamic << " 触发回退判定" << endl;
            isDynamic = true;
        }

        if(pMP) {
            if(isDynamic) {
                pMP->mFeatureStatus = MapPoint::DYNAMIC;
            }
        }
        
        if(isDynamic) {
            dynamicPoints++;
        } else {
            staticPoints++;
        }
    }

    // 第四步：精化动态状态
    RefineDynamicStatusFromSemanticRatio();
    
    // 第五步：可视化语义点（仅在动态一致性显示模式下）
    if (mbShowDynamicVis)
        VisualizeSemanticPoints(semanticMatches, rotationAngle, verticalDisplacement, adaptiveThreshold);
    
}


/**
 * 地面平面拟合 —— 收集语义地面类(车/卡车/公交/面包车)检测框中MapPoint指针
 *
 * SLAM初始化完成后延迟500帧触发，在此期间仅在新关键帧创建时收集：
 *  - 语义地面点：落在car(3)/van(4)/truck(5)/bus(8)检测框内的非动态MapPoint指针
 *  - 所有点：全部MapPoint指针（用于第二遍精化）
 *
 * 使用 std::unordered_set 自动去重（同一MapPoint在多个KF中只存一次）。
 * 存储指针而非坐标副本，确保 FitGroundPlane 能从指针读取BA更新后的实时坐标。
 *
 * 5秒后调用 FitGroundPlane() 一次性拟合。
 */
void Tracking::CollectGroundPlaneData()
{
    if(mbPlaneFitted) return;
    if(mState != OK) return;
    if(mMonoInitFrameId == 0) return;

    long unsigned int framesSinceInit = mCurrentFrame.mnId - mMonoInitFrameId;
    // 自适应：5秒后触发平面拟合（500帧@100fps, 150帧@30fps, 50帧@10fps）
    if(framesSinceInit > 5 * static_cast<long unsigned int>(mMaxFrames)) {
        FitGroundPlane();
        return;
    }

    // 仅在新关键帧创建时收集（数据质量高）
    if(!mpLastKeyFrame || mpLastKeyFrame->mnId == mnLastPlaneCollectKFId)
        return;
    mnLastPlaneCollectKFId = mpLastKeyFrame->mnId;

    // 收集当前关键帧的所有非动态 MapPoint 指针
    const set<MapPoint*>& kfMPs = mpLastKeyFrame->GetMapPoints();
    for(MapPoint* pMP : kfMPs) {
        if(!pMP || pMP->isBad() || pMP->mFeatureStatus == MapPoint::DYNAMIC)
            continue;
        mspAllPoints.insert(pMP);
    }

    // 语义地面点：只收集落在检测框内的 MapPoint 指针
    const auto& boxes = mCurrentFrame.detectedBoxes;
    if(boxes.empty()) return;

    auto isGroundClass = [](int cid) -> bool {
        return (cid == 3 || cid == 4 || cid == 5 || cid == 8);
    };

    for(const auto& det : boxes) {
        if(!isGroundClass(det.class_id)) continue;

        for(int i = 0; i < mCurrentFrame.N; i++) {
            if(!mCurrentFrame.mvpMapPoints[i] || mCurrentFrame.mvpMapPoints[i]->isBad())
                continue;
            if(mCurrentFrame.mvpMapPoints[i]->mFeatureStatus == MapPoint::DYNAMIC)
                continue;
            if(!det.bbox.contains(mCurrentFrame.mvKeysUn[i].pt))
                continue;

            mspGroundPoints.insert(mCurrentFrame.mvpMapPoints[i]);
        }
    }
}

/**
 * 拟合地面平面0（初始化后500帧时一次性执行）
 *
 * 两遍SVD拟合：
 *   第1遍：全点SVD → initNormal（5590+点，条件数远好于140车辆点）
 *   第2遍：底部60%精化SVD → planeNormal（25°门控，超过则回退initNormal）
 *   偏移量 = planeNormal·lowCentroid（SVD中心化建模的数学必然结果）
 *   紧阈值标记平面点（1.5%场景尺度），BA 通过 EdgePlaneConstraint 软约束引导
 */
void Tracking::FitGroundPlane()
{
    Map* pMap = mpAtlas->GetCurrentMap();
    if(!pMap) { mbPlaneFitted = false; return; }

    // 从指针集读取实时坐标（BA已更新）
    vector<Eigen::Vector3f> vGroundPoints, vAllPoints;
    vGroundPoints.reserve(mspGroundPoints.size());
    for(MapPoint* pMP : mspGroundPoints) {
        if(pMP && !pMP->isBad() && pMP->mFeatureStatus != MapPoint::DYNAMIC)
            vGroundPoints.push_back(pMP->GetWorldPos());
    }
    vAllPoints.reserve(mspAllPoints.size());
    for(MapPoint* pMP : mspAllPoints) {
        if(pMP && !pMP->isBad() && pMP->mFeatureStatus != MapPoint::DYNAMIC)
            vAllPoints.push_back(pMP->GetWorldPos());
    }

    if(vAllPoints.size() < 500) {
        cout << "[FitGroundPlane] 数据不足: 总点=" << vAllPoints.size() << ", 放弃拟合" << endl;
        mbPlaneFitted = false;
        mspGroundPoints.clear();
        mspAllPoints.clear();
        return;
    }

    // === 第1遍：全点SVD获取初始法向量 ===
    // 全场景5590+点分布广，SVD条件数远好于140个车辆点
    Eigen::Vector3f allCentroid(0,0,0);
    for(const auto& p : vAllPoints) allCentroid += p;
    allCentroid /= static_cast<float>(vAllPoints.size());

    Eigen::MatrixXf A_all(vAllPoints.size(), 3);
    for(size_t i = 0; i < vAllPoints.size(); i++)
        A_all.row(i) = (vAllPoints[i] - allCentroid).transpose();
    Eigen::JacobiSVD<Eigen::MatrixXf> svdAll(A_all, Eigen::ComputeFullV);
    Eigen::Vector3f initNormal = svdAll.matrixV().col(2);
    initNormal.normalize();

    // 确保initNormal指向相机一侧（上空方向）
    Eigen::Vector3f camCenter = mCurrentFrame.GetCameraCenter();
    if(initNormal.dot(camCenter - allCentroid) < 0) initNormal = -initNormal;

    cout << "[FitGroundPlane] 全点SVD初始法向量=(" << initNormal.transpose() << ")" << endl;

    // 场景尺度
    float sceneScale = 0;
    for(const auto& p : vAllPoints) sceneScale = max(sceneScale, (p - allCentroid).norm());

    // === 第2遍：用initNormal排序所有点，取底部60%做SVD精化 ===
    vector<pair<float, size_t>> vProjIdx;
    vProjIdx.reserve(vAllPoints.size());
    for(size_t i = 0; i < vAllPoints.size(); i++)
        vProjIdx.emplace_back(initNormal.dot(vAllPoints[i]), i);
    sort(vProjIdx.begin(), vProjIdx.end());

    size_t lowCount = max(size_t(100), vAllPoints.size() * 60 / 100);
    vector<Eigen::Vector3f> vLowPoints;
    vLowPoints.reserve(lowCount);
    for(size_t i = 0; i < lowCount; i++)
        vLowPoints.push_back(vAllPoints[vProjIdx[i].second]);

    Eigen::Vector3f lowCentroid(0,0,0);
    for(const auto& p : vLowPoints) lowCentroid += p;
    lowCentroid /= static_cast<float>(lowCount);

    Eigen::MatrixXf A_low(lowCount, 3);
    for(size_t i = 0; i < lowCount; i++)
        A_low.row(i) = (vLowPoints[i] - lowCentroid).transpose();
    Eigen::JacobiSVD<Eigen::MatrixXf> svdLow(A_low, Eigen::ComputeFullV);
    Eigen::Vector3f planeNormal = svdLow.matrixV().col(2);
    planeNormal.normalize();
    if(planeNormal.dot(initNormal) < 0) planeNormal = -planeNormal;

    // === 角度门控：精化法向量偏离initNormal >25° 则拒绝 ===
    float angleCos = std::abs(initNormal.dot(planeNormal));
    if(angleCos < 0.9063f) {  // cos(25°)
        cout << "[FitGroundPlane] 精化法向量偏离全点方向 " << acos(angleCos)*180.0/M_PI
             << "°, 回退到全点方向" << endl;
        planeNormal = initNormal;
    }

    // === 偏移量：SVD中心化数据拟合的平面必经 lowCentroid ===
    // planeNormal · (P - lowCentroid) = 0  →  offset = planeNormal · lowCentroid
    float mainOffset = planeNormal.dot(lowCentroid);

    // === 相机上方验证 ===
    float camHeight = planeNormal.dot(camCenter) - mainOffset;
    if(camHeight < 0) {
        planeNormal = -planeNormal;
        mainOffset = -mainOffset;
        camHeight = -camHeight;
    }

    // === 写入Map ===
    pMap->SetPlaneModel(planeNormal, mainOffset);
    pMap->SetPlaneRefHeight(camHeight);
    pMap->SetPlaneRefOffset(mainOffset);  // 原始偏移量，永不修改，用于尺度检测

    cout << "[FitGroundPlane] 完成! 语义点=" << vGroundPoints.size()
         << " 总点=" << vAllPoints.size()
         << " 低部点=" << lowCount
         << " 场景尺度=" << sceneScale
         << " 法向量=(" << planeNormal(0) <<","<< planeNormal(1) <<","<< planeNormal(2) <<")"
         << " 偏移=" << mainOffset
         << " 相机高度(SLAM单位)=" << camHeight << endl;

    // 宽阈值标记平面点（8%场景尺度）：标足够的点让 BA 尺度假感受力
    // info=50 意味着 1σ≈0.14 单位，即 ±10% d 的容忍度，不会锁死正常高度波动
    const float depthThreshold = max(0.08f * sceneScale, 0.08f);
    {
        vector<MapPoint*> vpAllMPs = pMap->GetAllMapPoints();
        int nPlanePts = 0;
        for(MapPoint* pMP : vpAllMPs) {
            if(!pMP || pMP->isBad()) continue;
            Eigen::Vector3f Pw = pMP->GetWorldPos();
            float signedD = planeNormal.dot(Pw);
            float minD = std::abs(signedD - mainOffset);
            pMP->mfPlaneDistance = minD;
            pMP->mnPlaneID = (minD < depthThreshold) ? 0 : -1;
            if(pMP->mnPlaneID >= 0) pMP->mfPlaneInfo = 1.0f;
            if(pMP->mnPlaneID >= 0) nPlanePts++;
        }
        cout << "[FitGroundPlane] 平面点=" << nPlanePts << " 阈值=" << depthThreshold << endl;
    }

    // 清空收集缓冲区
    mspGroundPoints.clear();
    mspAllPoints.clear();

    mbPlaneFitted = true;
}

/**
 * 平面点硬投影（备用工具，当前不自动调用）
 *
 * 遍历所有地图点，对 mnPlaneID>=0 的点强制投影到归属平面上。
 * 注意：此函数会暴力移动点坐标，可能导致跟踪丢失。
 * 当前策略已在 BA 中使用 EdgePlaneConstraint 软约束，不再需要此函数。
 * 保留作为备用调试工具。
 */
void Tracking::ProjectPlanePoints()
{
    Map* pMap = mpAtlas->GetCurrentMap();
    if(!pMap || !pMap->IsPlaneEstimated()) return;

    const Eigen::Vector3f& normal = pMap->GetPlaneNormal();
    const vector<float>& offsets = pMap->GetPlaneOffsets();
    if(offsets.empty()) return;

    float tightThreshold = max(0.015f * std::abs(offsets[0]), 0.015f);

    vector<MapPoint*> vpAllMPs = pMap->GetAllMapPoints();
    int nProjected = 0;

    for(MapPoint* pMP : vpAllMPs) {
        if(!pMP || pMP->isBad()) continue;

        Eigen::Vector3f Pw = pMP->GetWorldPos();
        float signedDist = normal.dot(Pw);

        // 寻找最近的平面
        float bestDist = std::abs(signedDist - offsets[0]);
        int bestPlaneID = 0;
        for(size_t k = 1; k < offsets.size(); k++) {
            float d = std::abs(signedDist - offsets[k]);
            if(d < bestDist) { bestDist = d; bestPlaneID = (int)k; }
        }
        float deviation = std::abs(signedDist - offsets[bestPlaneID]);
        if(deviation < 1e-4f) continue;

        bool shouldProject = (pMP->mnPlaneID >= 0) || (deviation < tightThreshold);
        if(shouldProject) {
            pMP->SetWorldPos(Pw - normal * (signedDist - offsets[bestPlaneID]));
            pMP->mnPlaneID = bestPlaneID;
            pMP->mfPlaneDistance = 0.0f;
            nProjected++;
        }
    }

    if(nProjected > 0) {
        cout << "[ProjectPlanePoints] 投影 " << nProjected
             << " 个点到平面, 阈值=" << tightThreshold << endl;
    }
}

/**
 * 2D 检测框 → 3D 检测框 (特征点驱动版本)
 *
 * 核心思路: 以BA优化过的地图点作为定位来源，而非bbox像素射线求交
 * 去重机制: 足迹重合比例 + 数量守恒 + 观测冻结
 * 频率控制: 每30帧执行一次
 */
void Tracking::Lift2DBoxesTo3D()
{
    Map* pMap = mpAtlas->GetCurrentMap();
    if(!pMap || !pMap->IsPlaneEstimated()) return;

    // ===== 频率控制: 每30帧执行一次 =====
    if(mCurrentFrame.mnId - mnLastLift3DFrameId < 30)
        return;
    mnLastLift3DFrameId = mCurrentFrame.mnId;

    const Eigen::Vector3f& n_w = pMap->GetPlaneNormal();
    float d_ref = pMap->GetPlaneRefOffset();
    float slamHeight = pMap->GetPlaneRefHeight();  // SLAM单位下的相机高度
    if(std::abs(d_ref) < 1e-6f || slamHeight < 0.01f) return;

    const float HEIGHT = 50.0f;
    float scaleToSlam = slamHeight / HEIGHT;

    // 类别默认尺寸 (物理米), 通过 scaleToSlam 转为 SLAM 单位
    // VisDrone: 0、1=pedestrian(跳过), 2=bicycle, 3=car, 4=van, 5=truck, 6=tricycle, 7=awning, 8=bus, 9=motor
    const float classW_m[10] = {0, 0.5f, 0.5f, 1.8f, 2.0f, 2.5f, 1.2f, 1.2f, 3.0f, 0.5f};
    const float classD_m[10] = {0, 0.5f, 0.5f, 4.5f, 5.5f, 10.0f, 2.5f, 2.5f, 12.0f, 0.5f};
    const float classH_m[10] = {0, 1.7f, 1.7f, 1.5f, 2.2f, 3.0f, 2.0f, 2.2f, 3.2f, 1.7f};

    const int MIN_OBJECT_POINTS = 4;  // 最低物体特征点数量
    const float MAX_HEIGHT_RATIO = 0.5f;  // 物体点最大高度 = 相机高度 * 50%

    // ===== 预统计: 每类2D检测数量 与 当前视野内持久3D框数量 =====
    int nDet2D[10] = {0};
    for(const auto& det : mCurrentFrame.detectedBoxes) {
        if(det.class_id < 2 || det.class_id > 9 || det.is_dynamic) continue;
        if(det.bbox.width < 4 || det.bbox.height < 4) continue;
        nDet2D[det.class_id]++;
    }

    // 视野判断: 3D框中心投影到当前帧图像范围内(与isInFrustum同理)
    const float fx = mCurrentFrame.fx, fy = mCurrentFrame.fy;
    const float cx = mCurrentFrame.cx, cy = mCurrentFrame.cy;
    const Sophus::SE3f Tcw = mCurrentFrame.GetPose();
    auto isInFOV = [&](const Eigen::Vector3f& Pw) -> bool {
        Eigen::Vector3f Pc = Tcw * Pw;
        if(Pc.z() < 0.05f) return false;
        float u = fx * Pc.x() / Pc.z() + cx;
        float v = fy * Pc.y() / Pc.z() + cy;
        return u >= mCurrentFrame.mnMinX && u < mCurrentFrame.mnMaxX &&
               v >= mCurrentFrame.mnMinY && v < mCurrentFrame.mnMaxY;
    };

    int nBox3D[10] = {0};
    bool bAllFrozen[10];
    std::fill(bAllFrozen, bAllFrozen + 10, true);
    for(const auto& b : pMap->GetPersistentBoxes()) {
        if(b.class_id < 2 || b.class_id > 9) continue;
        if(!isInFOV(b.center)) continue;
        nBox3D[b.class_id]++;
        if(!b.bFrozen) bAllFrozen[b.class_id] = false;
    }

    mvDetection3Ds.clear();
    mvDetection3Ds.reserve(mCurrentFrame.detectedBoxes.size());

    // 保存文件
    static std::string saveFile = "Detection3D.txt";
    static bool fileInitialized = false;
    std::ofstream f;
    if(!fileInitialized) {
        f.open(saveFile);
        f << "# frame_id class_id cx cy cz width depth height nObjPts nObs\n";
        fileInitialized = true;
    } else {
        f.open(saveFile, std::ios::app);
    }

    for(const auto& det : mCurrentFrame.detectedBoxes) {
        Detection3D det3d;
        det3d.class_id = det.class_id;
        det3d.bValid = false;
        det3d.nObservations = 0;

        // 跳过人类(0=pedestrian, 1=people)和动态目标
        if(det.class_id <= 1 || det.is_dynamic) {
            mvDetection3Ds.push_back(det3d); continue;
        }
        if(det.bbox.width < 4 || det.bbox.height < 4) {
            mvDetection3Ds.push_back(det3d); continue;
        }

        float bx = det.bbox.x, by = det.bbox.y;
        float bw = det.bbox.width, bh = det.bbox.height;
        int ci = (det.class_id >= 0 && det.class_id <= 9) ? det.class_id : 0;

        // ===== 数量门控: 视野内该类3D框已达2D检测数量且全部冻结 → 跳过计算 =====
        if(nBox3D[ci] >= nDet2D[ci] && bAllFrozen[ci]) {
            mvDetection3Ds.push_back(det3d);
            continue;
        }
        // 仅当视野内数量不足时才允许新建框(合并更新不受限)
        bool bAllowNew = nBox3D[ci] < nDet2D[ci];

        // ===== 步骤1: 收集框内的物体特征点 =====
        std::vector<Eigen::Vector3f> vObjectPts;  // 平面之上的点
        std::vector<float> vHeights;              // 各点相对平面的高度
        vObjectPts.reserve(32);
        vHeights.reserve(32);

        for(int i = 0; i < mCurrentFrame.N; i++) {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP || pMP->isBad()) continue;
            if(pMP->mFeatureStatus == MapPoint::DYNAMIC) continue;

            const cv::KeyPoint& kp = mCurrentFrame.mvKeysUn[i];
            if(kp.pt.x < bx || kp.pt.x > bx + bw ||
               kp.pt.y < by || kp.pt.y > by + bh)
                continue;

            Eigen::Vector3f Pw = pMP->GetWorldPos();
            float height = n_w.dot(Pw) - d_ref;

            // 物体点: 必须位于平面之上，且高度合理
            if(height > 0.001f && height < slamHeight * MAX_HEIGHT_RATIO) {
                vObjectPts.push_back(Pw);
                vHeights.push_back(height);
            }
        }

        // 数量门控
        if((int)vObjectPts.size() < MIN_OBJECT_POINTS) {
            mvDetection3Ds.push_back(det3d);
            continue;
        }

        // ===== 步骤2: 位置估计 - 中位数投影到平面 =====
        int nPts = (int)vObjectPts.size();

        // 将点投影到平面上（去除高度分量）
        std::vector<Eigen::Vector3f> vPlanePts(nPts);
        for(int i = 0; i < nPts; i++)
            vPlanePts[i] = vObjectPts[i] - n_w * (n_w.dot(vObjectPts[i]) - d_ref);

        // 中位数抗离群
        std::vector<float> coordX(nPts), coordY(nPts), coordZ(nPts);
        for(int i = 0; i < nPts; i++) {
            coordX[i] = vPlanePts[i].x();
            coordY[i] = vPlanePts[i].y();
            coordZ[i] = vPlanePts[i].z();
        }
        std::sort(coordX.begin(), coordX.end());
        std::sort(coordY.begin(), coordY.end());
        std::sort(coordZ.begin(), coordZ.end());
        Eigen::Vector3f P_ctr(coordX[nPts/2], coordY[nPts/2], coordZ[nPts/2]);
        // 确保精确在平面上
        P_ctr -= n_w * (n_w.dot(P_ctr) - d_ref);

        // ===== 步骤3: 尺寸估计 - 特征点分布 + 先验融合 =====
        // 平面局部坐标系
        Eigen::Vector3f V_dir = P_ctr - mCurrentFrame.GetCameraCenter();
        V_dir -= n_w * n_w.dot(V_dir);
        float Vn = V_dir.norm();
        if(Vn < 1e-6f) { mvDetection3Ds.push_back(det3d); continue; }
        V_dir /= Vn;
        Eigen::Vector3f U_dir = n_w.cross(V_dir).normalized();

        // 计算各点在局部坐标系下的分布
        std::vector<float> vU(nPts), vV(nPts);
        for(int i = 0; i < nPts; i++) {
            Eigen::Vector3f rel = vPlanePts[i] - P_ctr;
            vU[i] = rel.dot(U_dir);
            vV[i] = rel.dot(V_dir);
        }
        std::sort(vU.begin(), vU.end());
        std::sort(vV.begin(), vV.end());

        // 5%~95% 百分位范围（去除离群点）
        int iLo = std::max(0, (int)(nPts * 0.05f));
        int iHi = std::min(nPts - 1, (int)(nPts * 0.95f));
        float spreadU = vU[iHi] - vU[iLo];
        float spreadV = vV[iHi] - vV[iLo];
        // 确保最小尺寸
        spreadU = std::max(spreadU, 0.005f);
        spreadV = std::max(spreadV, 0.005f);

        // 高度中位数
        std::sort(vHeights.begin(), vHeights.end());
        float medHeight = vHeights[nPts / 2];

        // 先验尺寸 (SLAM单位)
        float W_prior = classW_m[ci] * scaleToSlam;
        float D_prior = classD_m[ci] * scaleToSlam;
        float H_prior = classH_m[ci] * scaleToSlam;

        // 融合权重: 点越多越信任特征点分布
        float alpha = std::min(1.0f, (float)nPts / 20.0f);
        float W = alpha * spreadU + (1.0f - alpha) * W_prior;
        float D = alpha * spreadV + (1.0f - alpha) * D_prior;
        float H = alpha * medHeight + (1.0f - alpha) * H_prior;

        // 尺寸合理性约束
        W = std::max(W_prior * 0.2f, std::min(W, W_prior * 3.0f));
        D = std::max(D_prior * 0.2f, std::min(D, D_prior * 3.0f));
        H = std::max(H_prior * 0.2f, std::min(H, slamHeight * MAX_HEIGHT_RATIO));

        det3d.center = P_ctr;
        det3d.width  = W;
        det3d.depth  = D;
        det3d.height = H;
        det3d.bValid = true;
        mvDetection3Ds.push_back(det3d);

        // ===== 步骤4: 加入持久地图（去重+数量配额） =====
        if(pMap->AddOrUpdateDetection3D(det3d, bAllowNew)) {
            nBox3D[ci]++;              // 新建成功, 更新视野内计数
            bAllFrozen[ci] = false;    // 新框未冻结
        }

        if(f.is_open())
            f << mCurrentFrame.mnId << " " << det3d.class_id << " "
              << det3d.center.x() << " " << det3d.center.y() << " " << det3d.center.z() << " "
              << W << " " << D << " " << H
              << " " << nPts << " 1\n";
    }
    if(f.is_open()) f.close();

    // ===== 数量清理: 视野内某类3D框多于2D检测数量 → 删除多余的未冻结框 =====
    {
        std::vector<Detection3D> boxes = pMap->GetPersistentBoxes();
        bool bChanged = false;
        for(int cid = 2; cid <= 9; cid++) {
            std::vector<int> vInFov;
            for(size_t i = 0; i < boxes.size(); i++)
                if(boxes[i].class_id == cid && isInFOV(boxes[i].center))
                    vInFov.push_back((int)i);
            int excess = (int)vInFov.size() - nDet2D[cid];
            if(excess <= 0) continue;
            // 只删未冻结的框(冻结框已有≥3次观测, 防止检测器漏检误删真实目标)
            std::vector<int> vRemovable;
            for(int idx : vInFov)
                if(!boxes[idx].bFrozen) vRemovable.push_back(idx);
            // 观测次数少的优先删除
            std::sort(vRemovable.begin(), vRemovable.end(), [&boxes](int a, int b){
                return boxes[a].nObservations < boxes[b].nObservations;
            });
            int nDel = std::min(excess, (int)vRemovable.size());
            for(int k = 0; k < nDel; k++) {
                boxes[vRemovable[k]].bValid = false;
                bChanged = true;
            }
        }
        if(bChanged) {
            std::vector<Detection3D> vKept;
            vKept.reserve(boxes.size());
            for(const auto& b : boxes)
                if(b.bValid) vKept.push_back(b);
            pMap->ReplacePersistentBoxes(vKept);
        }
    }

    // ===== RANSAC 直线对齐: 同类目标倾向排列（停车场/马路边缘） =====
    {
        std::vector<Detection3D> boxes = pMap->GetPersistentBoxes();
        int nSnapped = 0;

        Eigen::Vector3f ref2 = (std::abs(n_w.x()) < 0.9f) ? Eigen::Vector3f::UnitX() : Eigen::Vector3f::UnitZ();
        Eigen::Vector3f e1 = n_w.cross(ref2).normalized();
        Eigen::Vector3f e2 = n_w.cross(e1).normalized();

        for(int cid = 3; cid <= 8; cid++) {
            vector<int> indices;
            for(size_t i = 0; i < boxes.size(); i++)
                if(boxes[i].class_id == cid && boxes[i].bValid) indices.push_back((int)i);
            if(indices.size() < 3) continue;

            struct P { float u,v; };
            vector<P> pts(indices.size());
            for(size_t k = 0; k < indices.size(); k++) {
                Eigen::Vector3f p = boxes[indices[k]].center;
                pts[k] = {e1.dot(p), e2.dot(p)};
            }

            float bestR=0, bestA=0, bestB=0, bestC=0; int bestN=0;
            const float th = 0.06f;
            for(int t=0; t<std::min(30,(int)indices.size()*5); t++) {
                int i1=std::rand()%indices.size(), i2=std::rand()%indices.size();
                if(i1==i2) continue;
                float dx=pts[i2].u-pts[i1].u, dy=pts[i2].v-pts[i1].v, L=std::sqrt(dx*dx+dy*dy);
                if(L<0.005f) continue;
                float a=-dy/L, b=dx/L, c=-(a*pts[i1].u+b*pts[i1].v);
                int in=0; for(auto& p:pts) if(std::abs(a*p.u+b*p.v+c)<th) in++;
                float r=(float)in/indices.size();
                if(r>bestR) { bestR=r; bestA=a; bestB=b; bestC=c; bestN=in; }
            }
            if(bestR<0.6f || bestN<3) continue;

            // 仅对齐未冻结的框
            for(size_t k=0; k<indices.size(); k++) {
                int idx=indices[k];
                if(boxes[idx].bFrozen) continue;  // 冻结框不参与对齐
                float d=bestA*pts[k].u+bestB*pts[k].v+bestC;
                float su=pts[k].u-bestA*d*0.3f, sv=pts[k].v-bestB*d*0.3f;
                boxes[idx].center += e1*(su-pts[k].u) + e2*(sv-pts[k].v);
                boxes[idx].center -= n_w * (n_w.dot(boxes[idx].center) - d_ref);
                nSnapped++;
            }
        }
        if(nSnapped>0) pMap->ReplacePersistentBoxes(boxes);
    }
}

/**
 * 完整的动态点检测流程
 */
void Tracking::DetectDynamicPoints()
{
    // 步骤1：根据目标检测结果筛选特征点
    bool isFiltered = FilterFeaturePointsByDetection();

    if(isFiltered == false) {
        return;
    }
    
    // 步骤2：使用静态位姿分类语义点（内含精化检测框动态状态与可视化）
    Sophus::SE3f classifyPose;
    int nSemantic = static_cast<int>(mCurrentFrame.GetSemanticMatches().size());
    int nTotal = static_cast<int>(mCurrentFrame.GetFrameMatches().size());
    float semanticRatio = (nTotal > 0) ? static_cast<float>(nSemantic) / nTotal : 0.0f;
    if(semanticRatio > 0.2f)
    {
        // cout << "[ComputePoseWithStaticPoint] 语义点比例高于20%，使用静态点重新计算位姿" << endl;
        Sophus::SE3f staticPose = ComputePoseWithStaticPoints();
        classifyPose = staticPose;
    }
    else
    {
        classifyPose = mCurrentFrame.GetRelativePose();
    }

    // 步骤3：使用极线约束区分特征点的运动状态
    ClassifySemanticPoints(classifyPose);

    // 步骤4：更新各个类别的分类统计
    UpdateCategoryStatistics();

    // 步骤5：更新典型动态概率
    UpdateDynamicProbabilities();
    
    for(auto& detection : mCurrentFrame.detectedBoxes) {
        if(!detection.is_dynamic) {
            detection.moving_prob = detection.class_id < static_cast<int>(mvDynamicProbabilities.size()) 
                                    ? mvDynamicProbabilities[detection.class_id] : 0.5f;
        }
    }

    if(mpDetector) 
    {
        mpDetector->objects = mCurrentFrame.detectedBoxes;
    }
    
    // 步骤6：标记动态语义特征点为异常点
    MarkDynamicMapPointsAsOutliers();

    // 地面平面数据收集（初始化后每帧执行，500帧后自动触发拟合）
    CollectGroundPlaneData();

    // 每30帧刷新平面点标记 + 动态偏移量更新
    PlaneRemark();
}

/**
 * 平面标记刷新：每30帧执行一次，包含5个步骤
 *   1. 从已标记平面点估算当前动态偏移量 d_dynamic = median(n·P)
 *   2. 用冻结参考平面(d_ref)重新标记平面点，打破自引用回路
 *   3. 重新计算 d_dynamic（步骤2更新了平面点成员后）
 *   4. 三角化高度比 → λ = median(t_triang/t_frozen)
 *   5. 用 λ 驱动 BA 平面约束强度（mfPlaneInfo）
 */
void Tracking::PlaneRemark()
{
    static int sReMarkCounter = 0;
    Map* pMap = mpAtlas->GetCurrentMap();
    if(!pMap || !pMap->IsPlaneEstimated() || !mbPlaneFitted || ++sReMarkCounter < 30)
        return;

    sReMarkCounter = 0;
    Eigen::Vector3f n = pMap->GetPlaneNormal();
    float d_ref = pMap->GetPlaneRefOffset();
    Eigen::Vector3f camCenter = mCurrentFrame.GetCameraCenter();
    vector<MapPoint*> vpAllMPs = pMap->GetAllMapPoints();
    int nPlanePts = 0;    // 标记为平面点的数量（调试）
    int nRatiosOut = 0;   // 本次参与λ统计的样本数（调试）

    // 步骤1: 从已有标记点估算当前动态偏移量 d_dynamic = median(n·P)
    {
        vector<float> vDyn;
        for(MapPoint* pMP : vpAllMPs) {
            if(!pMP || pMP->isBad() || pMP->mnPlaneID < 0) continue;
            vDyn.push_back(n.dot(pMP->GetWorldPos()));
        }
        if(vDyn.size() >= 10) {
            sort(vDyn.begin(), vDyn.end());
            pMap->SetPlaneDynamicOffset(vDyn[vDyn.size() / 2]);
        }
    }
    float d_dynamic = pMap->GetPlaneDynamicOffset();

    // 步骤2: 用冻结参考平面(d_ref)重新标记平面点
    // 阈值收窄：原 0.6×|d_ref| ≈ 60%相机高度，会把楼/树误标为平面点；
    // 改为 5%×H_cam（尺度不变，米制下约2.5m@50m），只保留地面+车辆层。
    const float camHeight = pMap->GetPlaneRefHeight();
    float reMarkThreshold = std::max(0.05f * camHeight, 0.05f);
    int nNew = 0, nRemoved = 0;
    for(MapPoint* pMP : vpAllMPs) {
        if(!pMP || pMP->isBad()) continue;
        Eigen::Vector3f Pw = pMP->GetWorldPos();
        float dist = std::abs(n.dot(Pw) - d_ref);
        pMP->mfPlaneDistance = dist;
        if(dist >= reMarkThreshold) {
            if(pMP->mnPlaneID >= 0) { pMP->mnPlaneID = -1; nRemoved++; }
            continue;
        }
        Eigen::Vector3f dir = Pw - camCenter;
        float t_triang = dir.norm(); if(t_triang < 1e-6f) continue;
        dir /= t_triang; float ndir = n.dot(dir); if(std::abs(ndir) < 1e-6f) continue;
        float t_plane = (d_ref - n.dot(camCenter)) / ndir;
        if(t_plane <= 0) { if(pMP->mnPlaneID >= 0) { pMP->mnPlaneID = -1; nRemoved++; } continue; }
        if(t_triang / t_plane < 0.20f) { if(pMP->mnPlaneID >= 0) { pMP->mnPlaneID = -1; nRemoved++; } continue; }
        if(pMP->mnPlaneID < 0) { pMP->mnPlaneID = 0; pMP->mfPlaneInfo = 1.0f; nNew++; }
    }

    // 步骤3: 重新计算 d_dynamic（步骤2更新了平面点成员后）
    {
        vector<float> vDyn;
        for(MapPoint* pMP : vpAllMPs) {
            if(!pMP || pMP->isBad() || pMP->mnPlaneID < 0) continue;
            vDyn.push_back(n.dot(pMP->GetWorldPos()));
        }
        if(vDyn.size() >= 10) {
            sort(vDyn.begin(), vDyn.end());
            pMap->SetPlaneDynamicOffset(vDyn[vDyn.size() / 2]);
        }
    }
    d_dynamic = pMap->GetPlaneDynamicOffset();

    // 统计当前平面点数量（调试）
    for(MapPoint* pMP : vpAllMPs) {
        if(pMP && !pMP->isBad() && pMP->mnPlaneID >= 0) nPlanePts++;
    }

    // 步骤4: 三角化高度比 → λ = median(t_triang/t_frozen)
    float lambda = pMap->GetPlaneScaleLambda();
    {
        const vector<float>& vRatios = pMap->GetTriangRatios();
        int nRatios = (int)vRatios.size();
        nRatiosOut = nRatios;
        if(nRatios >= 10) {
            vector<float> vSorted = vRatios;
            sort(vSorted.begin(), vSorted.end());
            float medRatio = vSorted[vSorted.size() / 2];
            lambda = std::max(0.3f, std::min(1.2f, medRatio));
            pMap->SetPlaneScaleLambda(lambda);
            mvPlaneLambdaHist.push_back(lambda);
            if(mvPlaneLambdaHist.size() > 50) mvPlaneLambdaHist.erase(mvPlaneLambdaHist.begin());

            // 步骤4.5: 全局尺度回拉——绕圆/旋转主导场景下单目尺度快速退化时，
            // 对当前地图整体×1/λ拉回冻结平面参考尺度。
            // 两个间隔门控：成功回拉间隔>100帧（迟滞防振荡）；
            // 尝试间隔>90帧（失败/被跳过时不要每30帧空转重试，避免反复3秒停建图）
            if(mSensor == System::MONOCULAR && lambda < 0.85f
               && mCurrentFrame.mnId - mnLastScalePullbackFrameId > 100
               && mCurrentFrame.mnId - mnLastPullbackAttemptFrameId > 90)
            {
                mnLastPullbackAttemptFrameId = mCurrentFrame.mnId;
                cout << "[PlaneRemark] ★登记尺度回拉请求: λ=" << lambda
                     << " d_ref=" << d_ref << " H_cam=" << camHeight
                     << " 距上次回拉=" << (mCurrentFrame.mnId - mnLastScalePullbackFrameId)
                     << " 帧" << endl;
                // 非阻塞登记：由LocalMapping在安全点执行（ApplyScalePullback内部重置λ）
                RequestScalePullback(lambda);
            }
        }
        pMap->ClearTriangRatios();
    }

    // 步骤5: 用 λ 驱动 BA 平面约束强度
    {
        float boost = 1.0f;
        if(lambda < 0.5f)       boost = 5.0f;
        else if(lambda < 0.7f)  boost = 4.0f;
        else if(lambda < 0.85f) boost = 2.0f;
        else if(lambda < 0.95f) boost = 1.5f;

        for(MapPoint* pMP : vpAllMPs) {
            if(!pMP || pMP->isBad() || pMP->mnPlaneID < 0) continue;
            pMP->mfPlaneInfo = boost;
        }

        // 调试输出（每30帧一行）：λ、参考偏移、平面点状态、约束强度
        cout << "[PlaneRemark] frame=" << mCurrentFrame.mnId
             << " λ=" << lambda
             << " d_ref=" << d_ref
             << " d_dyn=" << d_dynamic
             << " H_cam=" << camHeight
             << " 平面点=" << nPlanePts
             << " new=" << nNew << " rem=" << nRemoved
             << " ratios=" << nRatiosOut
             << " boost=" << boost
             << " 回拉次数=" << mnScalePullbackCount << endl;

        // λ 历史（最近10个），观察退化趋势
        cout << "[PlaneRemark] λ历史:";
        for(size_t i = (mvPlaneLambdaHist.size() > 10 ? mvPlaneLambdaHist.size() - 10 : 0);
            i < mvPlaneLambdaHist.size(); i++)
            cout << " " << mvPlaneLambdaHist[i];
        cout << endl;

        // 写入 plane_scale_debug.txt（CSV，每30帧一行，便于离线分析）
        {
            static bool bHeader = false;
            std::ofstream fDbg("plane_scale_debug.txt", std::ios::out);  // 每次运行重新生成，便于对比
            if(fDbg.is_open()) {
                if(!bHeader) {
                    fDbg << "frame_id,time_s,lambda,d_ref,d_dynamic,cam_height,n_plane_pts,n_new,n_removed,n_ratios,boost,n_pullbacks\n";
                    bHeader = true;
                }
                fDbg << mCurrentFrame.mnId << "," << mCurrentFrame.mTimeStamp << ","
                     << lambda << "," << d_ref << "," << d_dynamic << "," << camHeight << ","
                     << nPlanePts << "," << nNew << "," << nRemoved << "," << nRatiosOut << ","
                     << boost << "," << mnScalePullbackCount << "\n";
            }
        }
    }

    // 步骤6: 语义车辆高度统计与约束
    // VisDrone2019类别: 0=pedestrian, 1=people, 2=bicycle, 3=car, 4=van,
    //                   5=truck, 6=tricycle, 7=awning-tricycle, 8=bus, 9=motor
    // 车辆类别(非人): 2~9
    {
        const float camHeight = pMap->GetPlaneRefHeight();  // 相机到平面的SLAM高度
        const float maxVehicleHeight = camHeight * 0.10f;   // 上界: 相机高度的10%

        if(!mbVehicleHeightFrozen) {
            // 收集阶段: 尺度健康时收集车辆语义点到平面的距离
            for(MapPoint* pMP : vpAllMPs) {
                if(!pMP || pMP->isBad()) continue;
                if(pMP->mnSemanticClass < 2 || pMP->mnSemanticClass > 9) continue;  // 仅车辆类别
                if(pMP->mFeatureStatus == MapPoint::DYNAMIC) continue;              // 排除动态点
                float height = n.dot(pMP->GetWorldPos()) - d_ref;  // 相对平面的有符号高度
                if(height > 0.0f && height < maxVehicleHeight) {
                    mvVehicleHeightSamples.push_back(height);
                }
            }

            // 冻结条件: lambda > 0.85 且样本充足
            if(lambda > 0.85f && mvVehicleHeightSamples.size() >= 20) {
                std::sort(mvVehicleHeightSamples.begin(), mvVehicleHeightSamples.end());
                mfVehicleRefHeight = mvVehicleHeightSamples[mvVehicleHeightSamples.size() / 2];
                // 安全截断
                if(mfVehicleRefHeight > maxVehicleHeight)
                    mfVehicleRefHeight = maxVehicleHeight;
                mbVehicleHeightFrozen = true;
                cout << "[PlaneRemark] 车辆参考高度冻结: " << mfVehicleRefHeight
                     << " (样本数=" << mvVehicleHeightSamples.size()
                     << ", 上界=" << maxVehicleHeight << ")" << endl;
            }
        }

        if(mbVehicleHeightFrozen) {
            // 约束阶段: 对车辆语义点设置平面高度偏移和增强权重
            for(MapPoint* pMP : vpAllMPs) {
                if(!pMP || pMP->isBad()) continue;
                if(pMP->mnSemanticClass < 2 || pMP->mnSemanticClass > 9) continue;
                if(pMP->mFeatureStatus == MapPoint::DYNAMIC) continue;

                // 设置语义高度偏移（BA约束目标: n·P = d_ref + offset）
                pMP->mfSemanticHeightOffset = mfVehicleRefHeight;
                // 确保标记为平面点并增强约束权重
                pMP->mnPlaneID = 0;
                float currentInfo = pMP->mfPlaneInfo;
                pMP->mfPlaneInfo = std::max(currentInfo, 3.0f);  // 车辆至少3倍权重
            }
        }
    }
}

/**
 * 尺度回拉请求/执行（Tracking请求 → LocalMapping安全点执行）
 *
 * 背景：早期版本在Tracking线程里"暂停LocalMapping→独占改图"，实测与
 * LocalMapping/LoopClosing的停止协议互踩——LocalMapping忙于处理关键帧队列
 * 数秒停不下来、地图锁被环回/合并/GBA长期占用，导致回拉几乎从未成功执行。
 *
 * 新方案：PlaneRemark只在Tracking线程登记请求（非阻塞，无任何等待）；
 * LocalMapping在Run循环的安全点（空闲、无并发写图）取出请求并执行缩放。
 * 该安全点天然满足独占性——LoopClosing/GBA改写地图前都会先暂停LocalMapping，
 * 因此执行时不会有其他线程同时改图。
 */
void Tracking::RequestScalePullback(float lambda)
{
    unique_lock<mutex> lock(mMutexScalePullback);
    mbScalePullbackPending = true;
    mfScalePullbackLambda = lambda;
}

bool Tracking::ConsumeScalePullback(float& lambda)
{
    unique_lock<mutex> lock(mMutexScalePullback);
    if(!mbScalePullbackPending) return false;
    mbScalePullbackPending = false;
    lambda = mfScalePullbackLambda;
    return true;
}

void Tracking::ApplyScalePullback(float lambda)
{
    Map* pMap = mpAtlas->GetCurrentMap();
    if(!pMap || !pMap->IsPlaneEstimated()) return;

    const float s = 1.0f / std::max(lambda, 0.3f);   // λ封底0.3，单次放大不超过3.33倍
    if(s < 1.001f) return;

    // 与LoopClosing/GBA的地图更新串行化。在LocalMapping线程的安全点调用，
    // 持锁时间短（纯算术缩放），不会长期阻塞其他线程
    unique_lock<mutex> mapLock(pMap->mMutexMapUpdate);

    // 1. 缩放所有关键帧平移（旋转不变）——先于点，供UpdateNormalAndDepth读取新位姿
    vector<KeyFrame*> vKFs = pMap->GetAllKeyFrames();
    for(KeyFrame* pKF : vKFs)
    {
        Sophus::SE3f Tcw = pKF->GetPose();
        Tcw.translation() *= s;
        pKF->SetPose(Tcw);
    }

    // 2. 缩放所有地图点并刷新尺度相关量
    vector<MapPoint*> vMPs = pMap->GetAllMapPoints();
    for(MapPoint* pMP : vMPs)
    {
        if(!pMP || pMP->isBad()) continue;
        pMP->SetWorldPos(pMP->GetWorldPos() * s);
        pMP->UpdateNormalAndDepth();
    }

    // 3. 动态偏移量随尺度缩放（d_ref冻结不变）
    pMap->SetPlaneDynamicOffset(pMap->GetPlaneDynamicOffset() * s);

    // 4. 清空退化期收集的车辆高度样本（已被尺度污染，须在健康期重新统计）
    mvVehicleHeightSamples.clear();

    // 5. λ重置为健康值
    pMap->SetPlaneScaleLambda(1.0f);

    mnScalePullbackCount++;

    // 6. 通知Tracking在Track()开头同步当前/上一帧位姿（保证与地图尺度一致）
    mfPendingFrameScale.store(s);

    cout << "[ScalePullback] #" << mnScalePullbackCount
         << " lambda=" << lambda << " s=" << s
         << " points=" << vMPs.size() << " kfs=" << vKFs.size()
         << " d_ref(冻结)=" << pMap->GetPlaneRefOffset()
         << " H_cam(冻结)=" << pMap->GetPlaneRefHeight()
         << " d_dyn->" << pMap->GetPlaneDynamicOffset() << endl;
}

void Tracking::SyncCurrentFrameScale()
{
    const float s = mfPendingFrameScale.exchange(1.0f);
    if(s < 0.999f || s > 1.001f)
    {
        // 点与位姿同因子绕原点缩放，重投影不变，跟踪无缝
        Sophus::SE3f TcwCur = mCurrentFrame.GetPose();
        TcwCur.translation() *= s;
        mCurrentFrame.SetPose(TcwCur);

        Sophus::SE3f TcwLast = mLastFrame.GetPose();
        TcwLast.translation() *= s;
        mLastFrame.SetPose(TcwLast);

        mnLastScalePullbackFrameId = mCurrentFrame.mnId;
    }
}

// 标记动态语义特征点为异常点
void Tracking::MarkDynamicMapPointsAsOutliers()
{
    mvDynamicSemanticPointIndices.clear();
    
    for(int i = 0; i < mCurrentFrame.N; i++)
    {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
        if(pMP && pMP->IsDynamicMapPoint())
        {
            mCurrentFrame.mvbOutlier[i] = true;
            mvDynamicSemanticPointIndices.push_back(i);
        }
    }
}


// 在ORB-SLAM的可视化界面内绘制动态语义特征点
void Tracking::DrawDynamicSemanticPoints()
{
    if(!mpDetector || mpDetector->mImg.empty())
        return;
    
    for(int idx : mvDynamicSemanticPointIndices)
    {
        if(idx >= mCurrentFrame.N)
            continue;
        cv::Point2f pt = mCurrentFrame.mvKeysUn[idx].pt;
        cv::circle(mpDetector->mImg, pt, 8, cv::Scalar(0, 0, 255), -1);
    }
}


/**
 * 可视化语义点（蓝色=静态，红色=动态）
 */
void Tracking::VisualizeSemanticPoints(const std::vector<std::pair<int, int>>& semanticMatches,
                                      double rotationAngle, double verticalDisplacement,
                                      double dynamicThreshold)
{
    if (!mImGray.empty()) {
        cv::Mat visImage;
        cv::cvtColor(mImGray, visImage, cv::COLOR_GRAY2BGR);
        
        // 在左上角显示运动参数和阈值信息
        cv::Scalar textColor;
        int fontFace = cv::FONT_HERSHEY_SIMPLEX;
        double fontScale = 0.5;
        int thickness = 1;
        int baseline = 0;
        
        // 根据初始化状态设置颜色：未初始化=红色，已初始化=绿色
        if (mbVibrationThresholdsInitialized) {
            textColor = cv::Scalar(0, 255, 0);  // 绿色（BGR格式）
        } else {
            textColor = cv::Scalar(0, 0, 255);  // 红色（BGR格式）
        }
        // 第一行：旋转角度信息
        std::string rotationText = "Rotate: " + std::to_string(rotationAngle * 180/M_PI).substr(0, 5) + " rad";
        cv::Size textSize = cv::getTextSize(rotationText, fontFace, fontScale, thickness, &baseline);
        cv::putText(visImage, rotationText, cv::Point(10, 25), fontFace, fontScale, textColor, thickness);
        
        // 第二行：旋转阈值信息
        std::string rotationThresholdText = "Rotate Threshold: " + std::to_string(mOptimizedRotationThreshold * 180/M_PI).substr(0, 5) + " rad";
        cv::putText(visImage, rotationThresholdText, cv::Point(10, 45), fontFace, fontScale, textColor, thickness);
        
        // 第三行：垂直位移信息
        std::string verticalText = "Vertical: " + std::to_string(verticalDisplacement).substr(0, 5) + " pixel";
        cv::putText(visImage, verticalText, cv::Point(10, 65), fontFace, fontScale, textColor, thickness);
        
        // 第四行：垂直阈值信息
        std::string verticalThresholdText = "Vertical Threshold: " + std::to_string(mOptimizedVerticalThreshold).substr(0, 5) + " pixel";
        cv::putText(visImage, verticalThresholdText, cv::Point(10, 85), fontFace, fontScale, textColor, thickness);
        
        // 第五行：动态阈值状态
        std::string dynamicThresholdText = "Dynamic Threshold: " + std::to_string(dynamicThreshold).substr(0, 5);
        cv::putText(visImage, dynamicThresholdText, cv::Point(10, 105), fontFace, fontScale, textColor, thickness);
        
        
        // 绘制静态语义特征点（蓝色）和动态语义特征点（红色）
        for(const auto& match : semanticMatches) {
            int currIdx = match.second;  // 当前帧特征点索引
            
            if(currIdx >= mCurrentFrame.N) {
                continue;
            }

            // 从MapPoint的mFeatureStatus读取动态状态
            bool isDynamic = false;
            if(currIdx < mCurrentFrame.mvpMapPoints.size()) {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[currIdx];
                if(pMP && pMP->mFeatureStatus == MapPoint::DYNAMIC) {
                    isDynamic = true;
                }
            }
            
            // 获取特征点坐标
            cv::Point2f currPt = mCurrentFrame.mvKeysUn[currIdx].pt;
            cv::Scalar color = isDynamic ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0);
            
            // 绘制当前特征点（大圆）
            cv::circle(visImage, currPt, 4, color, -1);
        }
        
        // 显示结果 → 写入 mImColor 供 FrameDrawer/Qt 使用
        mImColor = visImage.clone();

        // cv::imshow("极线约束 语义点", visImage);
        // cv::waitKey(1);
    }
}



/**
 * 基于语义点动态占比精化目标检测框的动态状态
 * 
 * 在语义点分类完成后执行：
 * 1. 遍历非行人的目标检测框
 * 2. 统计每个框内的语义点总数和通过mFeatureStatus判定的动态点占比，收集框内MapPoint
 * 3. 若动态点占比超过阈值，将该框判定为动态
 * 4. 对框内所有语义关联MapPoint设置MovingProbability=1.0，同时更新mFeatureStatus
 * 5. 清除动态框内所有特征点的地图点关联，使其不参与后续优化
 */
void Tracking::RefineDynamicStatusFromSemanticRatio()
{
    const auto& semanticMatches = mCurrentFrame.GetSemanticMatches();
    
    if(semanticMatches.empty()) {
        return;
    }
    
    const int nBoxes = static_cast<int>(mCurrentFrame.detectedBoxes.size());
    if(nBoxes == 0) return;
    
    const double DYNAMIC_RATIO_THRESHOLD = 0.4;
    
    // 优化：单次遍历语义匹配点，将每个点分配到其所在的检测框
    // 复杂度从 O(boxes × matches) 降为 O(matches × boxes)，但内层循环极短（通常<10个框）
    // 且避免了外层大循环的重复 contains 检查
    std::vector<int> vTotalPoints(nBoxes, 0);
    std::vector<int> vDynamicPoints(nBoxes, 0);
    std::vector<std::vector<MapPoint*>> vBoxMapPoints(nBoxes);
    std::vector<std::set<long unsigned int>> vSetBoxMPs(nBoxes);
    
    for(size_t i = 0; i < semanticMatches.size(); i++) {
        int currIdx = semanticMatches[i].second;
        if(currIdx >= mCurrentFrame.N) continue;
        
        const cv::Point2f& pt = mCurrentFrame.mvKeysUn[currIdx].pt;
        
        // 检查该点属于哪个检测框（通常只有少量框，内层循环极短）
        for(int b = 0; b < nBoxes; b++) {
            if(!mCurrentFrame.detectedBoxes[b].bbox.contains(pt)) continue;
            
            vTotalPoints[b]++;
            if(currIdx < static_cast<int>(mCurrentFrame.mvpMapPoints.size())) {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[currIdx];
                if(pMP && vSetBoxMPs[b].insert(pMP->mnId).second) {
                    vBoxMapPoints[b].push_back(pMP);
                    if(pMP->mFeatureStatus == MapPoint::DYNAMIC) {
                        vDynamicPoints[b]++;
                    }
                }
            }
            break;  // 一个点只属于一个框（假设检测框不重叠）
        }
    }
    
    // 根据统计结果判定每个检测框的动态状态
    for(int b = 0; b < nBoxes; b++) {
        auto& detection = mCurrentFrame.detectedBoxes[b];
        int totalPoints = vTotalPoints[b];
        int dynamicPoints = vDynamicPoints[b];
        auto& boxMapPoints = vBoxMapPoints[b];
        
        if(totalPoints > 0) {
            double dynamicRatio = static_cast<double>(dynamicPoints) / totalPoints;

            if(dynamicRatio >= DYNAMIC_RATIO_THRESHOLD) {
                detection.is_dynamic = true;
                detection.moving_prob = 1.0f;
                mCurrentFrame.detectedBoxes_dynamic.push_back(detection);
                for(MapPoint* pMP : boxMapPoints) {
                    pMP->SetMovingProbability(detection.moving_prob);
                }
            }
            else if(dynamicPoints > 0 && totalPoints < 3) {
                detection.is_dynamic = true;
                detection.moving_prob = 1.0f;
                mCurrentFrame.detectedBoxes_dynamic.push_back(detection);
                for(MapPoint* pMP : boxMapPoints) {
                    pMP->SetMovingProbability(detection.moving_prob);
                }
            }
            else {
                detection.is_dynamic = false;
                detection.moving_prob = detection.class_id < static_cast<int>(mvDynamicProbabilities.size()) 
                                        ? mvDynamicProbabilities[detection.class_id] : 0.4f;
                for(MapPoint* pMP : boxMapPoints) {
                    pMP->SetMovingProbability(detection.moving_prob);
                }
            }
        }
    }
}


// 更新分类统计, 统计每个类别的目标数量和动态数量
void Tracking::UpdateCategoryStatistics()
{
    for(const auto& detection : mCurrentFrame.detectedBoxes) {
        int cid = detection.class_id;
        if(cid < 0 || cid >= static_cast<int>(CLASS_NAMES.size())) {
            continue;
        }
        
        bool found = false;
        for(auto& stat : mvCategoryStats) {
            if(stat.class_id == cid) {
                stat.total_count++;
                if(detection.is_dynamic) {
                    stat.dynamic_count++;
                }
                found = true;
                break;
            }
        }
        if(!found) {
            CategoryStatistics stat;
            stat.class_id = cid;
            stat.class_name = CLASS_NAMES[cid];
            stat.total_count = 1;
            stat.dynamic_count = detection.is_dynamic ? 1 : 0;
            mvCategoryStats.push_back(stat);
        }
    }
}


// 更新典型动态概率
void Tracking::UpdateDynamicProbabilities()
{
    mnProbUpdateCounter++;
    if(mnProbUpdateCounter < PROB_UPDATE_INTERVAL) {
        return;
    }
    mnProbUpdateCounter = 0;
    
    const float MIN_PROB = 0.30f;          // 提高下限，避免无动态目标时概率过低
    const float MAX_PROB = 0.60f;
    const float PRIOR_STRENGTH = 200.0f;  // 更强的先验，降低新观测的冲击
    const float EMA_ALPHA = 0.25f;        // 指数滑动平均系数：越小更新越慢
    const float DYNAMIC_OBS_WEIGHT = 3.0f; // 动态观测权重：1个动态 ≈ 3个静态的信息量

    // cout << "[更新动态概率] 基于近" << PROB_UPDATE_INTERVAL << "帧统计数据:" << endl;
    
    for(const auto& stat : mvCategoryStats) {
        if(stat.class_id >= static_cast<int>(mvDynamicProbabilities.size())) {
            continue;
        }
        
        float prior = mvDynamicProbabilities[stat.class_id];
        
        float alpha = PRIOR_STRENGTH * prior;
        float beta  = PRIOR_STRENGTH * (1.0f - prior);
        
        // 动态观测加权：少量动态目标在大量静态目标中不应被淹没
        float staticCount = stat.total_count - stat.dynamic_count;
        float effectiveDynamic = stat.dynamic_count * DYNAMIC_OBS_WEIGHT;
        float effectiveTotal = staticCount + effectiveDynamic;
        
        float rawPosterior = (alpha + effectiveDynamic) / (alpha + beta + effectiveTotal);
        float targetPosterior = MIN_PROB + (MAX_PROB - MIN_PROB) * rawPosterior;
        
        // EMA 平滑：只向目标移动一小步，避免剧烈跳动
        float posterior = prior * (1.0f - EMA_ALPHA) + targetPosterior * EMA_ALPHA;
        
        mvDynamicProbabilities[stat.class_id] = posterior;
        
        // cout << "  " << stat.class_name << ": 总数=" << stat.total_count
        //      << " 动态=" << stat.dynamic_count
        //      << " raw=" << rawPosterior
        //      << " target=" << targetPosterior
        //      << " prior=" << prior << " → posterior=" << posterior << endl;
    }
    
    mvCategoryStats.clear();
    // cout << endl;
}
/**
 * 初始化振动检测阈值（使用中位数策略）
 * 在初始化阶段结束后调用，计算优化的旋转和垂直位移阈值
 */
void Tracking::InitializeVibrationThresholds()
{
    // 优化策略：使用中位数，对异常值更鲁棒
    // 中位数比80%分位数更适合扑翼场景，因为：
    // 1. 对异常值更鲁棒（避免被偶尔的大振动影响）
    // 2. 更能反映典型的振动水平
    // 3. 在振动数据分布不均匀时更稳定
    
    // 计算旋转角阈值（中位数）- 对异常值更鲁棒
    std::vector<double> sortedRotation = mRotationHistory;
    std::sort(sortedRotation.begin(), sortedRotation.end());
    // 正确的中位数计算：考虑奇偶性
    if(sortedRotation.size() % 2 == 0) {
        // 偶数个数据：取中间两个数的平均值
        mOptimizedRotationThreshold = (sortedRotation[sortedRotation.size() / 2 - 1] + sortedRotation[sortedRotation.size() / 2]) / 2.0;
    } else {
        // 奇数个数据：取中间值
        mOptimizedRotationThreshold = sortedRotation[sortedRotation.size() / 2];
    }
    
    // 计算垂直位移阈值（中位数）- 对异常值更鲁棒
    std::vector<double> sortedVertical = mVerticalHistory;
    std::sort(sortedVertical.begin(), sortedVertical.end());
    // 正确的中位数计算：考虑奇偶性
    if(sortedVertical.size() % 2 == 0) {
        // 偶数个数据：取中间两个数的平均值
        mOptimizedVerticalThreshold = (sortedVertical[sortedVertical.size() / 2 - 1] + sortedVertical[sortedVertical.size() / 2]) / 2.0;
    } else {
        // 奇数个数据：取中间值
        mOptimizedVerticalThreshold = sortedVertical[sortedVertical.size() / 2];
    }

    cout << "[InitializeVibrationThresholds] 初始化完成！ 优化后的旋转阈值：" << mOptimizedRotationThreshold << " 优化后的垂直位阈值：" << mOptimizedVerticalThreshold << endl;
    

}

/**
 * 计算振动等级：取旋转和垂直位移超出阈值百分比的最大值
 * 阈值为中位数，因此50%的帧会超出阈值，超出百分比反映的是"比典型振动强多少"
 * @return 振动等级：0=低于中位数，1.0=超出中位数100%，2.0=超出200%
 */
float Tracking::ComputeVibrationLevel(float rotAngle, float vertDisp)
{
    if(!mbVibrationThresholdsInitialized)
        return 0.0f;
    
    // 防止除零
    if(mOptimizedRotationThreshold < 1e-8 && mOptimizedVerticalThreshold < 1e-8)
        return 0.0f;
    
    // 计算旋转超出比例
    float rotExceed = 0.0f;
    if(mOptimizedRotationThreshold > 1e-8 && rotAngle > mOptimizedRotationThreshold)
        rotExceed = (rotAngle - mOptimizedRotationThreshold) / mOptimizedRotationThreshold;
    
    // 计算垂直位移超出比例
    float vertExceed = 0.0f;
    if(mOptimizedVerticalThreshold > 1e-8 && vertDisp > mOptimizedVerticalThreshold)
        vertExceed = (vertDisp - mOptimizedVerticalThreshold) / mOptimizedVerticalThreshold;
    
    // 取超出百分比最大的一项（与原有代码逻辑一致，更科学）
    return std::max(rotExceed, vertExceed);
}
} //namespace ORB_SLAM3
