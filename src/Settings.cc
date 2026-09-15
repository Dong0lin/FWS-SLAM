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
// 无修改
#include "Settings.h"

#include "CameraModels/Pinhole.h"
#include "CameraModels/KannalaBrandt8.h"

#include "System.h"

#include <opencv2/core/persistence.hpp>
#include <opencv2/core/eigen.hpp>

#include <iostream>
#include <algorithm>
#include <limits>

using namespace std;

namespace ORB_SLAM3 {

    namespace {
        // 兼容两套 yaml 写法：优先 Camera1.<key>（ORB-SLAM3 新版），
        // 回退到 Camera.<key>（ORB-SLAM2/MF-SLAM 旧版，长短焦标定文件使用）
        float readCamera1Param(cv::FileStorage& fSettings, const std::string& key, bool& found, const bool required = true){
            cv::FileNode node = fSettings["Camera1." + key];
            if(node.empty())
                node = fSettings["Camera." + key];
            if(node.empty()){
                if(required){
                    std::cerr << "Camera1." << key << "/Camera." << key
                              << " required parameter does not exist, aborting..." << std::endl;
                    exit(-1);
                }
                found = false;
                return 0.f;
            }
            found = true;
            return node.real();
        }
    }

    template<>
    float Settings::readParameter<float>(cv::FileStorage& fSettings, const std::string& name, bool& found, const bool required){
        cv::FileNode node = fSettings[name];
        if(node.empty()){
            if(required){
                std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
                exit(-1);
            }
            else{
                std::cerr << name << " optional parameter does not exist..." << std::endl;
                found = false;
                return 0.0f;
            }
        }
        else if(!node.isReal() && !node.isInt()){
            std::cerr << name << " parameter must be a real number, aborting..." << std::endl;
            exit(-1);
        }
        else{
            found = true;
            return node.real();
        }
    }

    template<>
    int Settings::readParameter<int>(cv::FileStorage& fSettings, const std::string& name, bool& found, const bool required){
        cv::FileNode node = fSettings[name];
        if(node.empty()){
            if(required){
                std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
                exit(-1);
            }
            else{
                std::cerr << name << " optional parameter does not exist..." << std::endl;
                found = false;
                return 0;
            }
        }
        else if(!node.isInt()){
            std::cerr << name << " parameter must be an integer number, aborting..." << std::endl;
            exit(-1);
        }
        else{
            found = true;
            return node.operator int();
        }
    }

    template<>
    string Settings::readParameter<string>(cv::FileStorage& fSettings, const std::string& name, bool& found, const bool required){
        cv::FileNode node = fSettings[name];
        if(node.empty()){
            if(required){
                std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
                exit(-1);
            }
            else{
                std::cerr << name << " optional parameter does not exist..." << std::endl;
                found = false;
                return string();
            }
        }
        else if(!node.isString()){
            std::cerr << name << " parameter must be a string, aborting..." << std::endl;
            exit(-1);
        }
        else{
            found = true;
            return node.string();
        }
    }

    template<>
    cv::Mat Settings::readParameter<cv::Mat>(cv::FileStorage& fSettings, const std::string& name, bool& found, const bool required){
        cv::FileNode node = fSettings[name];
        if(node.empty()){
            if(required){
                std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
                exit(-1);
            }
            else{
                std::cerr << name << " optional parameter does not exist..." << std::endl;
                found = false;
                return cv::Mat();
            }
        }
        else{
            found = true;
            return node.mat();
        }
    }

    Settings::Settings(const std::string &configFile, const int& sensor) :
    bNeedToUndistort_(false), bNeedToRectify_(false), bNeedToResize1_(false), bNeedToResize2_(false),
    bMultiFocal_(false), combine_(""),
    calibration1_(nullptr), calibration2_(nullptr),
    originalCalib1_(nullptr), originalCalib2_(nullptr),
    leftFocalMm_(4.f), rightFocalMm_(6.f), bFocalMmSet_(false),
    minDepth_(0.f), maxDepth_(1e9f) {
        sensor_ = sensor;

        //Open settings file
        cv::FileStorage fSettings(configFile, cv::FileStorage::READ);
        if (!fSettings.isOpened()) {
            cerr << "[ERROR]: could not open configuration file at: " << configFile << endl;
            cerr << "Aborting..." << endl;

            exit(-1);
        }
        else{
            cout << "Loading settings from " << configFile << endl;
        }

        //Read first camera
        readCamera1(fSettings);
        cout << "\t-Loaded camera 1" << endl;

        //Read second camera if stereo (not rectified)
        if(sensor_ == System::STEREO || sensor_ == System::IMU_STEREO){
            bool bHasCombine = !fSettings["Camera.combine"].empty();
            if(bHasCombine){
                readMultiFocal(fSettings);
                cout << "\t-Loaded multi-focal stereo calibration" << endl;
            }
            else{
                readCamera2(fSettings);
                cout << "\t-Loaded camera 2" << endl;
            }
        }

        //Read image info
        readImageInfo(fSettings);
        cout << "\t-Loaded image info" << endl;

        if(sensor_ == System::IMU_MONOCULAR || sensor_ == System::IMU_STEREO || sensor_ == System::IMU_RGBD){
            readIMU(fSettings);
            cout << "\t-Loaded IMU calibration" << endl;
        }

        if(sensor_ == System::RGBD || sensor_ == System::IMU_RGBD){
            readRGBD(fSettings);
            cout << "\t-Loaded RGB-D calibration" << endl;
        }

        readORB(fSettings);
        cout << "\t-Loaded ORB settings" << endl;
        readViewer(fSettings);
        cout << "\t-Loaded viewer settings" << endl;
        readLoadAndSave(fSettings);
        cout << "\t-Loaded Atlas settings" << endl;
        readOtherParameters(fSettings);
        cout << "\t-Loaded misc parameters" << endl;

        if(bNeedToRectify_){
            precomputeRectificationMaps();
            cout << "\t-Computed rectification maps" << endl;
        }

        cout << "----------------------------------" << endl;
    }

    void Settings::readCamera1(cv::FileStorage &fSettings) {
        bool found;

        //Read camera model
        string cameraModel = readParameter<string>(fSettings,"Camera.type",found);

        vector<float> vCalibration;
        if (cameraModel == "PinHole") {
            cameraType_ = PinHole;

            //Read intrinsic parameters
            float fx = readCamera1Param(fSettings,"fx",found);
            float fy = readCamera1Param(fSettings,"fy",found);
            float cx = readCamera1Param(fSettings,"cx",found);
            float cy = readCamera1Param(fSettings,"cy",found);

            vCalibration = {fx, fy, cx, cy};

            calibration1_ = new Pinhole(vCalibration);
            originalCalib1_ = new Pinhole(vCalibration);

            //Check if it is a distorted PinHole
            readCamera1Param(fSettings,"k1",found,false);
            if(found){
                readCamera1Param(fSettings,"k3",found,false);
                if(found){
                    vPinHoleDistorsion1_.resize(5);
                    vPinHoleDistorsion1_[4] = readCamera1Param(fSettings,"k3",found);
                }
                else{
                    vPinHoleDistorsion1_.resize(4);
                }
                vPinHoleDistorsion1_[0] = readCamera1Param(fSettings,"k1",found);
                vPinHoleDistorsion1_[1] = readCamera1Param(fSettings,"k2",found);
                vPinHoleDistorsion1_[2] = readCamera1Param(fSettings,"p1",found);
                vPinHoleDistorsion1_[3] = readCamera1Param(fSettings,"p2",found);
            }

            //Check if we need to correct distortion from the images
            if((sensor_ == System::MONOCULAR || sensor_ == System::IMU_MONOCULAR) && vPinHoleDistorsion1_.size() != 0){
                bNeedToUndistort_ = true;
            }
        }
        else if(cameraModel == "Rectified"){
            cameraType_ = Rectified;

            //Read intrinsic parameters
            float fx = readCamera1Param(fSettings,"fx",found);
            float fy = readCamera1Param(fSettings,"fy",found);
            float cx = readCamera1Param(fSettings,"cx",found);
            float cy = readCamera1Param(fSettings,"cy",found);

            vCalibration = {fx, fy, cx, cy};

            calibration1_ = new Pinhole(vCalibration);
            originalCalib1_ = new Pinhole(vCalibration);

            //Rectified images are assumed to be ideal PinHole images (no distortion)
        }
        else if(cameraModel == "KannalaBrandt8"){
            cameraType_ = KannalaBrandt;

            //Read intrinsic parameters
            float fx = readCamera1Param(fSettings,"fx",found);
            float fy = readCamera1Param(fSettings,"fy",found);
            float cx = readCamera1Param(fSettings,"cx",found);
            float cy = readCamera1Param(fSettings,"cy",found);

            float k0 = readCamera1Param(fSettings,"k1",found);
            float k1 = readCamera1Param(fSettings,"k2",found);
            float k2 = readCamera1Param(fSettings,"k3",found);
            float k3 = readCamera1Param(fSettings,"k4",found);

            vCalibration = {fx,fy,cx,cy,k0,k1,k2,k3};

            calibration1_ = new KannalaBrandt8(vCalibration);
            originalCalib1_ = new KannalaBrandt8(vCalibration);

            if(sensor_ == System::STEREO || sensor_ == System::IMU_STEREO){
                int colBegin = readParameter<int>(fSettings,"Camera1.overlappingBegin",found);
                int colEnd = readParameter<int>(fSettings,"Camera1.overlappingEnd",found);
                vector<int> vOverlapping = {colBegin, colEnd};

                static_cast<KannalaBrandt8*>(calibration1_)->mvLappingArea = vOverlapping;
            }
        }
        else{
            cerr << "Error: " << cameraModel << " not known" << endl;
            exit(-1);
        }
    }

    void Settings::readCamera2(cv::FileStorage &fSettings) {
        bool found;
        vector<float> vCalibration;
        if (cameraType_ == PinHole) {
            bNeedToRectify_ = true;

            //Read intrinsic parameters
            float fx = readParameter<float>(fSettings,"Camera2.fx",found);
            float fy = readParameter<float>(fSettings,"Camera2.fy",found);
            float cx = readParameter<float>(fSettings,"Camera2.cx",found);
            float cy = readParameter<float>(fSettings,"Camera2.cy",found);


            vCalibration = {fx, fy, cx, cy};

            calibration2_ = new Pinhole(vCalibration);
            originalCalib2_ = new Pinhole(vCalibration);

            //Check if it is a distorted PinHole
            readParameter<float>(fSettings,"Camera2.k1",found,false);
            if(found){
                readParameter<float>(fSettings,"Camera2.k3",found,false);
                if(found){
                    vPinHoleDistorsion2_.resize(5);
                    vPinHoleDistorsion2_[4] = readParameter<float>(fSettings,"Camera2.k3",found);
                }
                else{
                    vPinHoleDistorsion2_.resize(4);
                }
                vPinHoleDistorsion2_[0] = readParameter<float>(fSettings,"Camera2.k1",found);
                vPinHoleDistorsion2_[1] = readParameter<float>(fSettings,"Camera2.k2",found);
                vPinHoleDistorsion2_[2] = readParameter<float>(fSettings,"Camera2.p1",found);
                vPinHoleDistorsion2_[3] = readParameter<float>(fSettings,"Camera2.p2",found);
            }
        }
        else if(cameraType_ == KannalaBrandt){
            //Read intrinsic parameters
            float fx = readParameter<float>(fSettings,"Camera2.fx",found);
            float fy = readParameter<float>(fSettings,"Camera2.fy",found);
            float cx = readParameter<float>(fSettings,"Camera2.cx",found);
            float cy = readParameter<float>(fSettings,"Camera2.cy",found);

            float k0 = readParameter<float>(fSettings,"Camera1.k1",found);
            float k1 = readParameter<float>(fSettings,"Camera1.k2",found);
            float k2 = readParameter<float>(fSettings,"Camera1.k3",found);
            float k3 = readParameter<float>(fSettings,"Camera1.k4",found);


            vCalibration = {fx,fy,cx,cy,k0,k1,k2,k3};

            calibration2_ = new KannalaBrandt8(vCalibration);
            originalCalib2_ = new KannalaBrandt8(vCalibration);

            int colBegin = readParameter<int>(fSettings,"Camera2.overlappingBegin",found);
            int colEnd = readParameter<int>(fSettings,"Camera2.overlappingEnd",found);
            vector<int> vOverlapping = {colBegin, colEnd};

            static_cast<KannalaBrandt8*>(calibration2_)->mvLappingArea = vOverlapping;
        }

        //Load stereo extrinsic calibration
        if(cameraType_ == Rectified){
            b_ = readParameter<float>(fSettings,"Stereo.b",found);
            bf_ = b_ * calibration1_->getParameter(0);
        }
        else{
            cv::Mat cvTlr = readParameter<cv::Mat>(fSettings,"Stereo.T_c1_c2",found);
            Tlr_ = Converter::toSophus(cvTlr);

            //TODO: also search for Trl and invert if necessary

            b_ = Tlr_.translation().norm();
            bf_ = b_ * calibration1_->getParameter(0);
        }

        thDepth_ = readParameter<float>(fSettings,"Stereo.ThDepth",found);


    }

    void Settings::readMultiFocal(cv::FileStorage &fSettings) {
        bool found;

        // 焦距组合类型 "01"/"02"/"12"（本移植支持 "01"：宽短焦左目 + 长焦右目）
        combine_ = readParameter<std::string>(fSettings,"Camera.combine",found);

        // 右目（长焦）原始内参
        rfx_ = readParameter<float>(fSettings,"rightCamera.fx",found);
        rfy_ = readParameter<float>(fSettings,"rightCamera.fy",found);
        rcx_ = readParameter<float>(fSettings,"rightCamera.cx",found);
        rcy_ = readParameter<float>(fSettings,"rightCamera.cy",found);

        // 立体校正输出（multi-focal-stereo-calib 格式）
        leftD_  = readParameter<cv::Mat>(fSettings,"LEFT.D",found);
        leftK_  = readParameter<cv::Mat>(fSettings,"LEFT.K",found);
        leftR_  = readParameter<cv::Mat>(fSettings,"LEFT.R",found);
        leftP_  = readParameter<cv::Mat>(fSettings,"LEFT.P",found);
        rightD_ = readParameter<cv::Mat>(fSettings,"RIGHT.D",found);
        rightK_ = readParameter<cv::Mat>(fSettings,"RIGHT.K",found);
        rightR_ = readParameter<cv::Mat>(fSettings,"RIGHT.R",found);
        rightP_ = readParameter<cv::Mat>(fSettings,"RIGHT.P",found);

        if(leftK_.empty() || leftD_.empty() || leftR_.empty() || leftP_.empty() ||
           rightK_.empty() || rightD_.empty() || rightR_.empty() || rightP_.empty()){
            cerr << "ERROR: multi-focal rectification parameters (LEFT.*/RIGHT.*) are missing!" << endl;
            exit(-1);
        }

        // 校正后的公共内参取 LEFT.P 的 K（左右 P 共享同一 fx/fy/cx/cy）
        const double fxF = leftP_.at<double>(0,0);
        const double fyF = leftP_.at<double>(1,1);
        const double cxF = leftP_.at<double>(0,2);
        const double cyF = leftP_.at<double>(1,2);

        // 焦距比：右(长焦)fx / 左(短焦)fx（取原始内参）
        const float fx1 = calibration1_->getParameter(0);
        if(fx1 <= 0.f || rfx_ <= 0.f){
            cerr << "ERROR: invalid focal lengths in multi-focal calibration" << endl;
            exit(-1);
        }
        fFscale_ = rfx_ / fx1;

        // 可选焦距参数 Stereo.LeftFocal / Stereo.RightFocal（mm）：
        // 若两者都填了有效值，则以其比值（右/左）作为焦距比，
        // 后续长短焦模式的匹配区域（ROI）与金字塔层补偿（Leyermis）都由它决定；
        // 未填时回退到标定内参比（右fx/左fx）。
        leftFocalMm_ = 4.f;
        rightFocalMm_ = 6.f;
        bFocalMmSet_ = false;
        {
            cv::FileNode nLF = fSettings["Stereo.LeftFocal"];
            cv::FileNode nRF = fSettings["Stereo.RightFocal"];
            if(!nLF.empty() && !nRF.empty()){
                const float lf = (float)nLF.real();
                const float rf = (float)nRF.real();
                if(lf > 0.f && rf > 0.f){
                    leftFocalMm_ = lf;
                    rightFocalMm_ = rf;
                    fFscale_ = rf / lf;
                    bFocalMmSet_ = true;
                }
            }
        }

        // 用校正后的相机覆盖 camera1，使后续 mK/mpCamera/网格/投影全部使用公共校正坐标系
        delete calibration1_;
        calibration1_ = new Pinhole({(float)fxF, (float)fyF, (float)cxF, (float)cyF});

        // bf 直接用 yaml 里的 Camera.bf（校正后 bf），ThDepth 直接作为深度阈值
        bf_ = readParameter<float>(fSettings,"Camera.bf",found);
        b_ = bf_ / (float)fxF;
        thDepth_ = (float)readParameter<int>(fSettings,"ThDepth",found);

        // 左图重叠视场 ROI：根据标定（LEFT/RIGHT 的 K/D/R/P）精确推出长焦视场
        // 落在左目原始图像上的区域（替代 MF-SLAM 的近似公式——该公式只用了
        // rightCamera.cx/cy 和焦距比，未考虑畸变/旋转，在 4mm/6mm 标定下
        // 与真实区域偏差约 25px 且区域偏窄，表现为“匹配区域中心偏左、边缘目标超框”）。
        // 流程：右图角点+边中点 -> 校正公共坐标（undistortPoints）-> 左目原始坐标（projectPoints）
        int imW = readParameter<int>(fSettings,"Camera.width",found);
        int imH = readParameter<int>(fSettings,"Camera.height",found);
        bool bRoiFromCalib = false;
        // 有标定矩阵（LEFT/RIGHT 的 K/D/R/P）时始终用标定精算重叠视场 ROI——
        // 真实长焦镜头桶形畸变大，焦距比近似公式会把匹配区域算小（丢右半边缘可匹配特征）；
        // 焦距参数（Stereo.LeftFocal/RightFocal）只用于金字塔层差（Leyermis）。
        if(!leftK_.empty() && !leftD_.empty() && !leftR_.empty() && !leftP_.empty() &&
           !rightK_.empty() && !rightD_.empty() && !rightR_.empty() && !rightP_.empty())
        {
            std::vector<cv::Point2f> vRc;
            vRc.push_back(cv::Point2f(0.f, 0.f));
            vRc.push_back(cv::Point2f((float)imW, 0.f));
            vRc.push_back(cv::Point2f(0.f, (float)imH));
            vRc.push_back(cv::Point2f((float)imW, (float)imH));
            vRc.push_back(cv::Point2f((float)imW * 0.5f, 0.f));
            vRc.push_back(cv::Point2f((float)imW * 0.5f, (float)imH));
            vRc.push_back(cv::Point2f(0.f, (float)imH * 0.5f));
            vRc.push_back(cv::Point2f((float)imW, (float)imH * 0.5f));

            cv::Mat pts((int)vRc.size(), 2, CV_32F);
            for(size_t k = 0; k < vRc.size(); k++)
            {
                pts.at<float>(k, 0) = vRc[k].x;
                pts.at<float>(k, 1) = vRc[k].y;
            }
            pts = pts.reshape(2);
            cv::undistortPoints(pts, pts, rightK_, rightD_, rightR_, rightP_);
            pts = pts.reshape(1);

            float minx = std::numeric_limits<float>::max(), miny = std::numeric_limits<float>::max();
            float maxx = -std::numeric_limits<float>::max(), maxy = -std::numeric_limits<float>::max();
            cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
            cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
            for(int k = 0; k < pts.rows; k++)
            {
                const double u = pts.at<float>(k, 0);
                const double v = pts.at<float>(k, 1);
                // 校正像素 -> 公共归一化坐标
                cv::Mat xn(3, 1, CV_64F);
                xn.at<double>(0, 0) = (u - cxF) / fxF;
                xn.at<double>(1, 0) = (v - cyF) / fyF;
                xn.at<double>(2, 0) = 1.0;
                // -> 左目原始相机归一化坐标（LEFT.P 平移为 0）
                cv::Mat xnL = leftR_.t() * xn;
                // 注意：本机 OpenCV 构建的 projectPoints 不接受 CV_64FC3 输入，统一用 CV_32FC3
                cv::Mat objPts(1, 1, CV_32FC3);
                objPts.at<cv::Vec3f>(0, 0) = cv::Vec3f((float)xnL.at<double>(0, 0),
                                                       (float)xnL.at<double>(1, 0),
                                                       (float)xnL.at<double>(2, 0));
                std::vector<cv::Point2f> vOut;
                cv::projectPoints(objPts, rvec, tvec, leftK_, leftD_, vOut);
                minx = std::min(minx, vOut[0].x); miny = std::min(miny, vOut[0].y);
                maxx = std::max(maxx, vOut[0].x); maxy = std::max(maxy, vOut[0].y);
            }

            roiLeftUp_.x = std::max(minx, 0.f);
            roiLeftUp_.y = std::max(miny, 0.f);
            roiRightBottom_.x = std::min(maxx, (float)imW - 1.f);
            roiRightBottom_.y = std::min(maxy, (float)imH - 1.f);
            bRoiFromCalib = true;
        }
        if(!bRoiFromCalib)
        {
            // 回退：无标定矩阵（LEFT/RIGHT）时用 MF-SLAM Calibfind 近似公式
            const float invFscale = 1.0f / fFscale_;
            roiLeftUp_.x = std::max(rcx_ * (1.f - invFscale), 0.f);
            roiLeftUp_.y = std::max(rcy_ * (1.f - invFscale), 0.f);
            roiRightBottom_.x = std::min(roiLeftUp_.x + invFscale * imW, (float)imW - 1.f);
            roiRightBottom_.y = std::min(roiLeftUp_.y + invFscale * imH, (float)imH - 1.f);
        }

        // MF 模式不做整图 remap（特征点在 Frame 内自行校正），也不需要去畸变
        bNeedToRectify_ = false;
        bNeedToUndistort_ = false;
        bMultiFocal_ = true;

        // 深度合理性门控（50m 航拍：太近/太远的立体匹配基本是假匹配）
        {
            cv::FileNode node = fSettings["Stereo.MinDepth"];
            if(!node.empty())
                minDepth_ = (float)node.real();
            else
                minDepth_ = 10.0f;
            node = fSettings["Stereo.MaxDepth"];
            if(!node.empty())
                maxDepth_ = (float)node.real();
            else
                maxDepth_ = 300.0f;
        }

        cout << "\t-Multi-focal: combine=" << combine_
             << " Fscale=" << fFscale_
             << (bFocalMmSet_ ? " (from focal mm)" : " (from calibration)")
             << " focalL=" << leftFocalMm_ << "mm focalR=" << rightFocalMm_ << "mm"
             << " bf=" << bf_
             << " rectified fx=" << fxF
             << " ROI=[" << roiLeftUp_ << " -> " << roiRightBottom_ << "]" << endl;
    }

    void Settings::readImageInfo(cv::FileStorage &fSettings) {
        bool found;
        //Read original and desired image dimensions
        int originalRows = readParameter<int>(fSettings,"Camera.height",found);
        int originalCols = readParameter<int>(fSettings,"Camera.width",found);
        originalImSize_.width = originalCols;
        originalImSize_.height = originalRows;

        newImSize_ = originalImSize_;
        int newHeigh = readParameter<int>(fSettings,"Camera.newHeight",found,false);
        if(found){
            bNeedToResize1_ = true;
            newImSize_.height = newHeigh;

            if(!bNeedToRectify_){
                //Update calibration
                float scaleRowFactor = (float)newImSize_.height / (float)originalImSize_.height;
                calibration1_->setParameter(calibration1_->getParameter(1) * scaleRowFactor, 1);
                calibration1_->setParameter(calibration1_->getParameter(3) * scaleRowFactor, 3);


                if((sensor_ == System::STEREO || sensor_ == System::IMU_STEREO) && cameraType_ != Rectified){
                    calibration2_->setParameter(calibration2_->getParameter(1) * scaleRowFactor, 1);
                    calibration2_->setParameter(calibration2_->getParameter(3) * scaleRowFactor, 3);
                }
            }
        }

        int newWidth = readParameter<int>(fSettings,"Camera.newWidth",found,false);
        if(found){
            bNeedToResize1_ = true;
            newImSize_.width = newWidth;

            if(!bNeedToRectify_){
                //Update calibration
                float scaleColFactor = (float)newImSize_.width /(float) originalImSize_.width;
                calibration1_->setParameter(calibration1_->getParameter(0) * scaleColFactor, 0);
                calibration1_->setParameter(calibration1_->getParameter(2) * scaleColFactor, 2);

                if((sensor_ == System::STEREO || sensor_ == System::IMU_STEREO) && cameraType_ != Rectified){
                    calibration2_->setParameter(calibration2_->getParameter(0) * scaleColFactor, 0);
                    calibration2_->setParameter(calibration2_->getParameter(2) * scaleColFactor, 2);

                    if(cameraType_ == KannalaBrandt){
                        static_cast<KannalaBrandt8*>(calibration1_)->mvLappingArea[0] *= scaleColFactor;
                        static_cast<KannalaBrandt8*>(calibration1_)->mvLappingArea[1] *= scaleColFactor;

                        static_cast<KannalaBrandt8*>(calibration2_)->mvLappingArea[0] *= scaleColFactor;
                        static_cast<KannalaBrandt8*>(calibration2_)->mvLappingArea[1] *= scaleColFactor;
                    }
                }
            }
        }

        fps_ = readParameter<float>(fSettings,"Camera.fps",found);
        bRGB_ = (bool) readParameter<int>(fSettings,"Camera.RGB",found);
    }

    void Settings::readIMU(cv::FileStorage &fSettings) {
        bool found;
        noiseGyro_ = readParameter<float>(fSettings,"IMU.NoiseGyro",found);
        noiseAcc_ = readParameter<float>(fSettings,"IMU.NoiseAcc",found);
        gyroWalk_ = readParameter<float>(fSettings,"IMU.GyroWalk",found);
        accWalk_ = readParameter<float>(fSettings,"IMU.AccWalk",found);
        imuFrequency_ = readParameter<float>(fSettings,"IMU.Frequency",found);

        cv::Mat cvTbc = readParameter<cv::Mat>(fSettings,"IMU.T_b_c1",found);
        Tbc_ = Converter::toSophus(cvTbc);

        readParameter<int>(fSettings,"IMU.InsertKFsWhenLost",found,false);
        if(found){
            insertKFsWhenLost_ = (bool) readParameter<int>(fSettings,"IMU.InsertKFsWhenLost",found,false);
        }
        else{
            insertKFsWhenLost_ = true;
        }
    }

    void Settings::readRGBD(cv::FileStorage& fSettings) {
        bool found;

        depthMapFactor_ = readParameter<float>(fSettings,"RGBD.DepthMapFactor",found);
        thDepth_ = readParameter<float>(fSettings,"Stereo.ThDepth",found);
        b_ = readParameter<float>(fSettings,"Stereo.b",found);
        bf_ = b_ * calibration1_->getParameter(0);
    }

    void Settings::readORB(cv::FileStorage &fSettings) {
        bool found;

        nFeatures_ = readParameter<int>(fSettings,"ORBextractor.nFeatures",found);
        scaleFactor_ = readParameter<float>(fSettings,"ORBextractor.scaleFactor",found);
        nLevels_ = readParameter<int>(fSettings,"ORBextractor.nLevels",found);
        initThFAST_ = readParameter<int>(fSettings,"ORBextractor.iniThFAST",found);
        minThFAST_ = readParameter<int>(fSettings,"ORBextractor.minThFAST",found);
    }

    void Settings::readViewer(cv::FileStorage &fSettings) {
        bool found;

        keyFrameSize_ = readParameter<float>(fSettings,"Viewer.KeyFrameSize",found);
        keyFrameLineWidth_ = readParameter<float>(fSettings,"Viewer.KeyFrameLineWidth",found);
        graphLineWidth_ = readParameter<float>(fSettings,"Viewer.GraphLineWidth",found);
        pointSize_ = readParameter<float>(fSettings,"Viewer.PointSize",found);
        cameraSize_ = readParameter<float>(fSettings,"Viewer.CameraSize",found);
        cameraLineWidth_ = readParameter<float>(fSettings,"Viewer.CameraLineWidth",found);
        viewPointX_ = readParameter<float>(fSettings,"Viewer.ViewpointX",found);
        viewPointY_ = readParameter<float>(fSettings,"Viewer.ViewpointY",found);
        viewPointZ_ = readParameter<float>(fSettings,"Viewer.ViewpointZ",found);
        viewPointF_ = readParameter<float>(fSettings,"Viewer.ViewpointF",found);
        imageViewerScale_ = readParameter<float>(fSettings,"Viewer.imageViewScale",found,false);

         if(!found)
            imageViewerScale_ = 1.0f;
    }

    void Settings::readLoadAndSave(cv::FileStorage &fSettings) {
        bool found;

        sLoadFrom_ = readParameter<string>(fSettings,"System.LoadAtlasFromFile",found,false);
        sSaveto_ = readParameter<string>(fSettings,"System.SaveAtlasToFile",found,false);
    }

    void Settings::readOtherParameters(cv::FileStorage& fSettings) {
        bool found;

        thFarPoints_ = readParameter<float>(fSettings,"System.thFarPoints",found,false);
    }

    void Settings::precomputeRectificationMaps() {
        //Precompute rectification maps, new calibrations, ...
        cv::Mat K1 = static_cast<Pinhole*>(calibration1_)->toK();
        K1.convertTo(K1,CV_64F);
        cv::Mat K2 = static_cast<Pinhole*>(calibration2_)->toK();
        K2.convertTo(K2,CV_64F);

        cv::Mat cvTlr;
        cv::eigen2cv(Tlr_.inverse().matrix3x4(),cvTlr);
        cv::Mat R12 = cvTlr.rowRange(0,3).colRange(0,3);
        R12.convertTo(R12,CV_64F);
        cv::Mat t12 = cvTlr.rowRange(0,3).col(3);
        t12.convertTo(t12,CV_64F);

        cv::Mat R_r1_u1, R_r2_u2;
        cv::Mat P1, P2, Q;

        cv::stereoRectify(K1,camera1DistortionCoef(),K2,camera2DistortionCoef(),newImSize_,
                          R12, t12,
                          R_r1_u1,R_r2_u2,P1,P2,Q,
                          cv::CALIB_ZERO_DISPARITY,-1,newImSize_);
        cv::initUndistortRectifyMap(K1, camera1DistortionCoef(), R_r1_u1, P1.rowRange(0, 3).colRange(0, 3),
                                    newImSize_, CV_32F, M1l_, M2l_);
        cv::initUndistortRectifyMap(K2, camera2DistortionCoef(), R_r2_u2, P2.rowRange(0, 3).colRange(0, 3),
                                    newImSize_, CV_32F, M1r_, M2r_);

        //Update calibration
        calibration1_->setParameter(P1.at<double>(0,0), 0);
        calibration1_->setParameter(P1.at<double>(1,1), 1);
        calibration1_->setParameter(P1.at<double>(0,2), 2);
        calibration1_->setParameter(P1.at<double>(1,2), 3);

        //Update bf
        bf_ = b_ * P1.at<double>(0,0);

        //Update relative pose between camera 1 and IMU if necessary
        if(sensor_ == System::IMU_STEREO){
            Eigen::Matrix3f eigenR_r1_u1;
            cv::cv2eigen(R_r1_u1,eigenR_r1_u1);
            Sophus::SE3f T_r1_u1(eigenR_r1_u1,Eigen::Vector3f::Zero());
            Tbc_ = Tbc_ * T_r1_u1.inverse();
        }
    }

    ostream &operator<<(std::ostream& output, const Settings& settings){
        output << "SLAM settings: " << endl;

        output << "\t-Camera 1 parameters (";
        if(settings.cameraType_ == Settings::PinHole || settings.cameraType_ ==  Settings::Rectified){
            output << "Pinhole";
        }
        else{
            output << "Kannala-Brandt";
        }
        output << ")" << ": [";
        for(size_t i = 0; i < settings.originalCalib1_->size(); i++){
            output << " " << settings.originalCalib1_->getParameter(i);
        }
        output << " ]" << endl;

        if(!settings.vPinHoleDistorsion1_.empty()){
            output << "\t-Camera 1 distortion parameters: [ ";
            for(float d : settings.vPinHoleDistorsion1_){
                output << " " << d;
            }
            output << " ]" << endl;
        }

        if(settings.sensor_ == System::STEREO || settings.sensor_ == System::IMU_STEREO){
            if(settings.originalCalib2_){
                output << "\t-Camera 2 parameters (";
                if(settings.cameraType_ == Settings::PinHole || settings.cameraType_ ==  Settings::Rectified){
                    output << "Pinhole";
                }
                else{
                    output << "Kannala-Brandt";
                }
                output << "" << ": [";
                for(size_t i = 0; i < settings.originalCalib2_->size(); i++){
                    output << " " << settings.originalCalib2_->getParameter(i);
                }
                output << " ]" << endl;

                if(!settings.vPinHoleDistorsion2_.empty()){
                    output << "\t-Camera 2 distortion parameters: [ ";
                    for(float d : settings.vPinHoleDistorsion2_){
                        output << " " << d;
                    }
                    output << " ]" << endl;
                }
            }
        }

        output << "\t-Original image size: [ " << settings.originalImSize_.width << " , " << settings.originalImSize_.height << " ]" << endl;
        output << "\t-Current image size: [ " << settings.newImSize_.width << " , " << settings.newImSize_.height << " ]" << endl;

        if(settings.bNeedToRectify_){
            output << "\t-Camera 1 parameters after rectification: [ ";
            for(size_t i = 0; i < settings.calibration1_->size(); i++){
                output << " " << settings.calibration1_->getParameter(i);
            }
            output << " ]" << endl;
        }
        else if(settings.bNeedToResize1_){
            output << "\t-Camera 1 parameters after resize: [ ";
            for(size_t i = 0; i < settings.calibration1_->size(); i++){
                output << " " << settings.calibration1_->getParameter(i);
            }
            output << " ]" << endl;

            if((settings.sensor_ == System::STEREO || settings.sensor_ == System::IMU_STEREO) &&
                settings.cameraType_ == Settings::KannalaBrandt){
                output << "\t-Camera 2 parameters after resize: [ ";
                for(size_t i = 0; i < settings.calibration2_->size(); i++){
                    output << " " << settings.calibration2_->getParameter(i);
                }
                output << " ]" << endl;
            }
        }

        output << "\t-Sequence FPS: " << settings.fps_ << endl;

        //Stereo stuff
        if(settings.sensor_ == System::STEREO || settings.sensor_ == System::IMU_STEREO){
            output << "\t-Stereo baseline: " << settings.b_ << endl;
            output << "\t-Stereo depth threshold : " << settings.thDepth_ << endl;

            if(settings.cameraType_ == Settings::KannalaBrandt){
                auto vOverlapping1 = static_cast<KannalaBrandt8*>(settings.calibration1_)->mvLappingArea;
                auto vOverlapping2 = static_cast<KannalaBrandt8*>(settings.calibration2_)->mvLappingArea;
                output << "\t-Camera 1 overlapping area: [ " << vOverlapping1[0] << " , " << vOverlapping1[1] << " ]" << endl;
                output << "\t-Camera 2 overlapping area: [ " << vOverlapping2[0] << " , " << vOverlapping2[1] << " ]" << endl;
            }
        }

        if(settings.sensor_ == System::IMU_MONOCULAR || settings.sensor_ == System::IMU_STEREO || settings.sensor_ == System::IMU_RGBD) {
            output << "\t-Gyro noise: " << settings.noiseGyro_ << endl;
            output << "\t-Accelerometer noise: " << settings.noiseAcc_ << endl;
            output << "\t-Gyro walk: " << settings.gyroWalk_ << endl;
            output << "\t-Accelerometer walk: " << settings.accWalk_ << endl;
            output << "\t-IMU frequency: " << settings.imuFrequency_ << endl;
        }

        if(settings.sensor_ == System::RGBD || settings.sensor_ == System::IMU_RGBD){
            output << "\t-RGB-D depth map factor: " << settings.depthMapFactor_ << endl;
        }

        output << "\t-Features per image: " << settings.nFeatures_ << endl;
        output << "\t-ORB scale factor: " << settings.scaleFactor_ << endl;
        output << "\t-ORB number of scales: " << settings.nLevels_ << endl;
        output << "\t-Initial FAST threshold: " << settings.initThFAST_ << endl;
        output << "\t-Min FAST threshold: " << settings.minThFAST_ << endl;
        
        return output;
    }
};
