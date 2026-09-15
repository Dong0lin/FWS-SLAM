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

#ifndef ORB_SLAM3_SETTINGS_H
#define ORB_SLAM3_SETTINGS_H


// Flag to activate the measurement of time in each process (track,localmap, place recognition).
//#define REGISTER_TIMES

#include "CameraModels/GeometricCamera.h"

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

namespace ORB_SLAM3 {

    class System;

    //TODO: change to double instead of float

    class Settings {
    public:
        /*
         * Enum for the different camera types implemented
         */
        enum CameraType {
            PinHole = 0,
            Rectified = 1,
            KannalaBrandt = 2
        };

        /*
         * Delete default constructor
         */
        Settings() = delete;

        /*
         * Constructor from file
         */
        Settings(const std::string &configFile, const int& sensor);

        /*
         * Ostream operator overloading to dump settings to the terminal
         */
        friend std::ostream &operator<<(std::ostream &output, const Settings &s);

        /*
         * Getter methods
         */
        CameraType cameraType() {return cameraType_;}
        GeometricCamera* camera1() {return calibration1_;}
        GeometricCamera* camera2() {return calibration2_;}
        cv::Mat camera1DistortionCoef() {return cv::Mat(vPinHoleDistorsion1_.size(),1,CV_32F,vPinHoleDistorsion1_.data());}
        cv::Mat camera2DistortionCoef() {return cv::Mat(vPinHoleDistorsion2_.size(),1,CV_32F,vPinHoleDistorsion1_.data());}

        Sophus::SE3f Tlr() {return Tlr_;}
        float bf() {return bf_;}
        float b() {return b_;}
        float thDepth() {return thDepth_;}

        bool needToUndistort() {return bNeedToUndistort_;}

        cv::Size newImSize() {return newImSize_;}
        float fps() {return fps_;}
        bool rgb() {return bRGB_;}
        bool needToResize() {return bNeedToResize1_;}
        bool needToRectify() {return bNeedToRectify_;}

        float noiseGyro() {return noiseGyro_;}
        float noiseAcc() {return noiseAcc_;}
        float gyroWalk() {return gyroWalk_;}
        float accWalk() {return accWalk_;}
        float imuFrequency() {return imuFrequency_;}
        Sophus::SE3f Tbc() {return Tbc_;}
        bool insertKFsWhenLost() {return insertKFsWhenLost_;}

        float depthMapFactor() {return depthMapFactor_;}

        int nFeatures() {return nFeatures_;}
        int nLevels() {return nLevels_;}
        float initThFAST() {return initThFAST_;}
        float minThFAST() {return minThFAST_;}
        float scaleFactor() {return scaleFactor_;}

        float keyFrameSize() {return keyFrameSize_;}
        float keyFrameLineWidth() {return keyFrameLineWidth_;}
        float graphLineWidth() {return graphLineWidth_;}
        float pointSize() {return pointSize_;}
        float cameraSize() {return cameraSize_;}
        float cameraLineWidth() {return cameraLineWidth_;}
        float viewPointX() {return viewPointX_;}
        float viewPointY() {return viewPointY_;}
        float viewPointZ() {return viewPointZ_;}
        float viewPointF() {return viewPointF_;}
        float imageViewerScale() {return imageViewerScale_;}

        std::string atlasLoadFile() {return sLoadFrom_;}
        std::string atlasSaveFile() {return sSaveto_;}

        float thFarPoints() {return thFarPoints_;}

        cv::Mat M1l() {return M1l_;}
        cv::Mat M2l() {return M2l_;}
        cv::Mat M1r() {return M1r_;}
        cv::Mat M2r() {return M2r_;}

        /*
         * Multi-focal (long-short focal) stereo stuff
         */
        bool isMultiFocal() {return bMultiFocal_;}
        std::string combine() {return combine_;}
        cv::Mat leftK() {return leftK_;}
        cv::Mat leftD() {return leftD_;}
        cv::Mat leftR() {return leftR_;}
        cv::Mat leftP() {return leftP_;}
        cv::Mat rightK() {return rightK_;}
        cv::Mat rightD() {return rightD_;}
        cv::Mat rightR() {return rightR_;}
        cv::Mat rightP() {return rightP_;}
        float rightFocalX() {return rfx_;}
        float rightFocalY() {return rfy_;}
        float rightCx() {return rcx_;}
        float rightCy() {return rcy_;}
        float focalScale() {return fFscale_;}
        float leftFocal() {return leftFocalMm_;}
        float rightFocal() {return rightFocalMm_;}
        bool focalMmSet() {return bFocalMmSet_;}
        cv::Point2f roiLeftUp() {return roiLeftUp_;}
        cv::Point2f roiRightBottom() {return roiRightBottom_;}
        float minDepth() {return minDepth_;}
        float maxDepth() {return maxDepth_;}

    private:
        template<typename T>
        T readParameter(cv::FileStorage& fSettings, const std::string& name, bool& found,const bool required = true){
            cv::FileNode node = fSettings[name];
            if(node.empty()){
                if(required){
                    std::cerr << name << " required parameter does not exist, aborting..." << std::endl;
                    exit(-1);
                }
                else{
                    std::cerr << name << " optional parameter does not exist..." << std::endl;
                    found = false;
                    return T();
                }

            }
            else{
                found = true;
                return (T) node;
            }
        }

        void readCamera1(cv::FileStorage& fSettings);
        void readCamera2(cv::FileStorage& fSettings);
        void readMultiFocal(cv::FileStorage& fSettings);
        void readImageInfo(cv::FileStorage& fSettings);
        void readIMU(cv::FileStorage& fSettings);
        void readRGBD(cv::FileStorage& fSettings);
        void readORB(cv::FileStorage& fSettings);
        void readViewer(cv::FileStorage& fSettings);
        void readLoadAndSave(cv::FileStorage& fSettings);
        void readOtherParameters(cv::FileStorage& fSettings);

        void precomputeRectificationMaps();

        int sensor_;
        CameraType cameraType_;     //Camera type

        /*
         * Visual stuff
         */
        GeometricCamera* calibration1_, *calibration2_;   //Camera calibration
        GeometricCamera* originalCalib1_, *originalCalib2_;
        std::vector<float> vPinHoleDistorsion1_, vPinHoleDistorsion2_;

        cv::Size originalImSize_, newImSize_;
        float fps_;
        bool bRGB_;

        bool bNeedToUndistort_;
        bool bNeedToRectify_;
        bool bNeedToResize1_, bNeedToResize2_;

        bool bMultiFocal_;                 // 长短焦（多焦距）双目模式
        std::string combine_;              // 焦距组合 "01"/"02"/"12"
        cv::Mat leftK_, leftD_, leftR_, leftP_;
        cv::Mat rightK_, rightD_, rightR_, rightP_;
        float rfx_, rfy_, rcx_, rcy_;      // 右目（长焦）原始内参
        float fFscale_;                    // 焦距比 右fx/左fx
        float leftFocalMm_, rightFocalMm_; // 左右目焦距（mm），yaml 可选参数 Stereo.LeftFocal/RightFocal
        bool bFocalMmSet_;                 // 是否由 yaml 焦距参数驱动（true → ROI/金字塔层数按焦距比算）
        cv::Point2f roiLeftUp_, roiRightBottom_;   // 校正后左图重叠视场ROI
        float minDepth_, maxDepth_;        // 立体深度合理性门控（假近点/假远点过滤）

        Sophus::SE3f Tlr_;
        float thDepth_;
        float bf_, b_;

        /*
         * Rectification stuff
         */
        cv::Mat M1l_, M2l_;
        cv::Mat M1r_, M2r_;

        /*
         * Inertial stuff
         */
        float noiseGyro_, noiseAcc_;
        float gyroWalk_, accWalk_;
        float imuFrequency_;
        Sophus::SE3f Tbc_;
        bool insertKFsWhenLost_;

        /*
         * RGBD stuff
         */
        float depthMapFactor_;

        /*
         * ORB stuff
         */
        int nFeatures_;
        float scaleFactor_;
        int nLevels_;
        int initThFAST_, minThFAST_;

        /*
         * Viewer stuff
         */
        float keyFrameSize_;
        float keyFrameLineWidth_;
        float graphLineWidth_;
        float pointSize_;
        float cameraSize_;
        float cameraLineWidth_;
        float viewPointX_, viewPointY_, viewPointZ_, viewPointF_;
        float imageViewerScale_;

        /*
         * Save & load maps
         */
        std::string sLoadFrom_, sSaveto_;

        /*
         * Other stuff
         */
        float thFarPoints_;

    };
};


#endif //ORB_SLAM3_SETTINGS_H
