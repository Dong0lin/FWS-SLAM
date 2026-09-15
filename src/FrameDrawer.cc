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

#include "FrameDrawer.h"
#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include<mutex>

namespace ORB_SLAM3
{

FrameDrawer::FrameDrawer(Atlas* pAtlas):both(false),mpAtlas(pAtlas)
{
    mState=Tracking::SYSTEM_NOT_READY;
    mIm = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
    mImRight = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));

    // mIm = cv::Mat(720,1280,CV_8UC3, cv::Scalar(0,0,0));
    // mImRight = cv::Mat(720,1280,CV_8UC3, cv::Scalar(0,0,0));
}

cv::Mat FrameDrawer::DrawFrame(float imageScale)
{
    cv::Mat im;
    vector<cv::KeyPoint> vIniKeys;
    vector<int> vMatches;
    vector<cv::KeyPoint> vCurrentKeys;
    vector<bool> vbVO, vbMap;
    vector<bool> vbOutlier;
    vector<float> vMovingProb;
    vector<pair<cv::Point2f, cv::Point2f> > vTracks;
    int state; // Tracking state
    vector<float> vCurrentDepth;
    float thDepth;

    Frame currentFrame;
    vector<MapPoint*> vpLocalMap;
    vector<cv::KeyPoint> vMatchesKeys;
    vector<MapPoint*> vpMatchedMPs;
    vector<cv::KeyPoint> vOutlierKeys;
    vector<MapPoint*> vpOutlierMPs;
    map<long unsigned int, cv::Point2f> mProjectPoints;
    map<long unsigned int, cv::Point2f> mMatchedInImage;

    cv::Scalar standardColor(0,255,0);
    cv::Scalar odometryColor(255,0,0);

    //Copy variables within scoped mutex
    {
        unique_lock<mutex> lock(mMutex);
        state=mState;
        if(mState==Tracking::SYSTEM_NOT_READY)
            mState=Tracking::NO_IMAGES_YET;

        mIm.copyTo(im);

        if(mState==Tracking::NOT_INITIALIZED)
        {
            vCurrentKeys = mvCurrentKeys;
            vIniKeys = mvIniKeys;
            vMatches = mvIniMatches;
            vTracks = mvTracks;
        }
        else if(mState==Tracking::OK)
        {
            vCurrentKeys = mvCurrentKeys;
            vbVO = mvbVO;
            vbMap = mvbMap;
            vMovingProb = mvMovingProbability;
            vbOutlier = mCurrentFrame.mvbOutlier;

            currentFrame = mCurrentFrame;
            vbOutlier = currentFrame.mvbOutlier;
            vpLocalMap = mvpLocalMap;
            vMatchesKeys = mvMatchedKeys;
            vpMatchedMPs = mvpMatchedMPs;
            vOutlierKeys = mvOutlierKeys;
            vpOutlierMPs = mvpOutlierMPs;
            mProjectPoints = mmProjectPoints;
            mMatchedInImage = mmMatchedInImage;

            vCurrentDepth = mvCurrentDepth;
            thDepth = mThDepth;

        }
        else if(mState==Tracking::LOST)
        {
            vCurrentKeys = mvCurrentKeys;
        }
    }

    if(imageScale != 1.f)
    {
        int imWidth = im.cols / imageScale;
        int imHeight = im.rows / imageScale;
        cv::resize(im, im, cv::Size(imWidth, imHeight));
    }

    if(im.channels()<3) //this should be always true
        cvtColor(im,im,cv::COLOR_GRAY2BGR);

    // 动态一致性模式：mIm 已由 VisualizeSemanticPoints 写入语义点+检测框，
    // 直接返回，不叠加 ORB 特征点（避免与语义点混淆）
    if(mbShowDynamicVis)
    {
        cv::Mat imWithInfo;
        DrawTextInfo(im, state, imWithInfo);
        return imWithInfo;
    }

    //Draw
    if(state==Tracking::NOT_INITIALIZED)
    {
        for(unsigned int i=0; i<vMatches.size(); i++)
        {
            if(vMatches[i]>=0)
            {
                cv::Point2f pt1,pt2;
                if(imageScale != 1.f)
                {
                    pt1 = vIniKeys[i].pt / imageScale;
                    pt2 = vCurrentKeys[vMatches[i]].pt / imageScale;
                }
                else
                {
                    pt1 = vIniKeys[i].pt;
                    pt2 = vCurrentKeys[vMatches[i]].pt;
                }
                cv::line(im,pt1,pt2,standardColor);
            }
        }
        for(vector<pair<cv::Point2f, cv::Point2f> >::iterator it=vTracks.begin(); it!=vTracks.end(); it++)
        {
            cv::Point2f pt1,pt2;
            if(imageScale != 1.f)
            {
                pt1 = (*it).first / imageScale;
                pt2 = (*it).second / imageScale;
            }
            else
            {
                pt1 = (*it).first;
                pt2 = (*it).second;
            }
            cv::line(im,pt1,pt2, standardColor,5);
        }

    }
    else if(state==Tracking::OK) //TRACKING
    {
        mnTracked=0;
        mnTrackedVO=0;
        const float r = 5;
        int n = vCurrentKeys.size();
        for(int i=0;i<n;i++)
        {
            bool isOutlier = (i < static_cast<int>(vbOutlier.size())) && vbOutlier[i];
            bool bTracked = (i < static_cast<int>(vbVO.size())) && (i < static_cast<int>(vbMap.size())) &&
                            (vbVO[i] || vbMap[i]);
            // 本帧是否与长焦图像立体匹配成功（有深度）
            bool bStereoMatched = (i < static_cast<int>(vCurrentDepth.size())) && vCurrentDepth[i] > 0.f;
            float prob = (i < static_cast<int>(vMovingProb.size())) ? vMovingProb[i] : 0.0f;

            // 颜色统一：动态语义=红，静态语义=蓝，
            // 与长焦匹配成功的非语义点=紫，已跟踪但本帧无立体匹配=绿
            cv::Scalar color;
            if(prob > 0.5f)
                color = cv::Scalar(0, 0, 255);       // 红 → 动态
            else if(prob > 0.0f)
                color = cv::Scalar(255, 0, 0);       // 蓝 → 静态语义
            else if(bStereoMatched)
                color = cv::Scalar(255, 0, 255);     // 紫 → 与长焦匹配成功（非语义）
            else if(bTracked || isOutlier)
                color = cv::Scalar(0, 255, 0);       // 绿 → 已跟踪但无立体匹配
            else
                continue;

            if(bTracked)
            {
                cv::Point2f pt1,pt2;
                cv::Point2f point;
                if(imageScale != 1.f)
                {
                    point = vCurrentKeys[i].pt / imageScale;
                    float px = vCurrentKeys[i].pt.x / imageScale;
                    float py = vCurrentKeys[i].pt.y / imageScale;
                    pt1.x=px-r;
                    pt1.y=py-r;
                    pt2.x=px+r;
                    pt2.y=py+r;
                }
                else
                {
                    point = vCurrentKeys[i].pt;
                    pt1.x=vCurrentKeys[i].pt.x-r;
                    pt1.y=vCurrentKeys[i].pt.y-r;
                    pt2.x=vCurrentKeys[i].pt.x+r;
                    pt2.y=vCurrentKeys[i].pt.y+r;
                }

                if(isOutlier)
                    cv::circle(im, point, 2, color, -1);  // 外点：空心小圆
                else
                {
                    cv::rectangle(im, pt1, pt2, color);
                    cv::circle(im, point, 2, color, -1);  // 跟踪点：方块+圆
                }

                if(vbMap[i])
                    mnTracked++;
                else
                    mnTrackedVO++;
            }
            else
            {
                // 仅立体匹配成功（未入地图）：小圆点
                cv::circle(im, vCurrentKeys[i].pt, 2, color, -1);
            }
        }

        // 长短焦模式：画出重叠视场ROI框（细紫框，加"ROI"标签，
        // 避免与目标检测框混淆——它代表长焦视场在左图的对应区域）
        if(both)
        {
            if(ORB_SLAM3::Frame::mbMultiFocal &&
               ORB_SLAM3::Frame::mROIRightBottom.x > ORB_SLAM3::Frame::mROILeftUp.x)
            {
                cv::rectangle(im,
                              cv::Point2f(ORB_SLAM3::Frame::mROILeftUp.x, ORB_SLAM3::Frame::mROILeftUp.y),
                              cv::Point2f(ORB_SLAM3::Frame::mROIRightBottom.x, ORB_SLAM3::Frame::mROIRightBottom.y),
                              cv::Scalar(255,0,255), 1);
                cv::putText(im, "ROI", cv::Point2f(ORB_SLAM3::Frame::mROILeftUp.x + 3,
                                                   ORB_SLAM3::Frame::mROILeftUp.y + 14),
                            cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 0, 255), 1);
            }
        }
    }

    cv::Mat imWithInfo;
    DrawTextInfo(im,state, imWithInfo);

    return imWithInfo;
}

cv::Mat FrameDrawer::DrawRightFrame(float imageScale)
{
    cv::Mat im;
    vector<cv::KeyPoint> vIniKeys; // Initialization: KeyPoints in reference frame
    vector<int> vMatches; // Initialization: correspondeces with reference keypoints
    vector<cv::KeyPoint> vCurrentKeys; // KeyPoints in current frame
    vector<bool> vbVO, vbMap;
    vector<bool> vbOutlier;
    vector<float> vMovingProb;
    int state; // Tracking state

    //Copy variables within scoped mutex
    {
        unique_lock<mutex> lock(mMutex);
        state=mState;
        if(mState==Tracking::SYSTEM_NOT_READY)
            mState=Tracking::NO_IMAGES_YET;

        mImRight.copyTo(im);

        if(mState==Tracking::NOT_INITIALIZED)
        {
            vCurrentKeys = mvCurrentKeysRight;
            vIniKeys = mvIniKeys;
            vMatches = mvIniMatches;
        }
        else if(mState==Tracking::OK)
        {
            vCurrentKeys = mvCurrentKeysRight;
            vbVO = mvbVO;
            vbMap = mvbMap;
            vMovingProb = mvMovingProbability;
            vbOutlier = mCurrentFrame.mvbOutlier;
        }
        else if(mState==Tracking::LOST)
        {
            vCurrentKeys = mvCurrentKeysRight;
        }
    } // destroy scoped mutex -> release mutex

    if(imageScale != 1.f)
    {
        int imWidth = im.cols / imageScale;
        int imHeight = im.rows / imageScale;
        cv::resize(im, im, cv::Size(imWidth, imHeight));
    }

    if(im.channels()<3) //this should be always true
        cvtColor(im,im,cv::COLOR_GRAY2BGR);

    //Draw
    if(state==Tracking::NOT_INITIALIZED) //INITIALIZING
    {
        for(unsigned int i=0; i<vMatches.size(); i++)
        {
            if(vMatches[i]>=0)
            {
                cv::Point2f pt1,pt2;
                if(imageScale != 1.f)
                {
                    pt1 = vIniKeys[i].pt / imageScale;
                    pt2 = vCurrentKeys[vMatches[i]].pt / imageScale;
                }
                else
                {
                    pt1 = vIniKeys[i].pt;
                    pt2 = vCurrentKeys[vMatches[i]].pt;
                }

                cv::line(im,pt1,pt2,cv::Scalar(0,255,0));
            }
        }
    }
    else if(state==Tracking::OK) //TRACKING
    {
        mnTracked=0;
        mnTrackedVO=0;
        const float r = 5;
        const int n = mvCurrentKeysRight.size();
        const int Nleft = mvCurrentKeys.size();

        for(int i=0;i<n;i++)
        {
            int idx = i + Nleft;
            bool isOutlier = (idx < static_cast<int>(vbOutlier.size())) && vbOutlier[idx];
            // 长短焦模式：右目特征点若与左目成功立体匹配（RightIdToLeftId>=0），也绘制出来
            bool bStereoMatched = (mCurrentFrame.RightIdToLeftId.size() > (size_t)i) &&
                                  (mCurrentFrame.RightIdToLeftId[i] >= 0);
            bool bTracked = (idx < static_cast<int>(vbVO.size())) && (idx < static_cast<int>(vbMap.size())) &&
                            (vbVO[idx] || vbMap[idx]);
            float prob = (idx < static_cast<int>(vMovingProb.size())) ? vMovingProb[idx] : 0.0f;

            // 颜色统一：动态语义=红，静态语义=蓝，
            // 与左目匹配成功的非语义点=紫，已跟踪非语义=绿，未匹配=灰
            cv::Scalar color;
            if(prob > 0.5f)
                color = cv::Scalar(0, 0, 255);       // 红 → 动态
            else if(prob > 0.0f)
                color = cv::Scalar(255, 0, 0);       // 蓝 → 静态语义
            else if(bStereoMatched)
                color = cv::Scalar(255, 0, 255);     // 紫 → 与左目匹配成功（非语义）
            else if(bTracked || isOutlier)
                color = cv::Scalar(0, 255, 0);       // 绿 → 已跟踪（非语义）
            else
            {
                // 未匹配的右目特征点：灰色小点（便于观察长焦特征提取情况）
                if(ORB_SLAM3::Frame::mbMultiFocal)
                    cv::circle(im, mvCurrentKeysRight[i].pt, 1, cv::Scalar(128,128,128), -1);
                continue;
            }

            cv::Point2f pt1,pt2;
            cv::Point2f point;
            if(imageScale != 1.f)
            {
                point = mvCurrentKeysRight[i].pt / imageScale;
                float px = mvCurrentKeysRight[i].pt.x / imageScale;
                float py = mvCurrentKeysRight[i].pt.y / imageScale;
                pt1.x=px-r;
                pt1.y=py-r;
                pt2.x=px+r;
                pt2.y=py+r;
            }
            else
            {
                point = mvCurrentKeysRight[i].pt;
                pt1.x=mvCurrentKeysRight[i].pt.x-r;
                pt1.y=mvCurrentKeysRight[i].pt.y-r;
                pt2.x=mvCurrentKeysRight[i].pt.x+r;
                pt2.y=mvCurrentKeysRight[i].pt.y+r;
            }

            if(bTracked && !isOutlier)
            {
                cv::rectangle(im, pt1, pt2, color);
                cv::circle(im, point, 2, color, -1);     // 跟踪点：方块+圆
            }
            else
            {
                cv::circle(im, point, 2, color, -1);     // 匹配成功/外点：小圆
            }

            if(vbMap[idx])
                mnTracked++;
            else
                mnTrackedVO++;
        }
    }

    cv::Mat imWithInfo;
    DrawTextInfo(im,state, imWithInfo);

    return imWithInfo;
}



void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
{
    stringstream s;
    if(nState==Tracking::NO_IMAGES_YET)
        s << " WAITING FOR IMAGES";
    else if(nState==Tracking::NOT_INITIALIZED)
        s << " TRYING TO INITIALIZE ";
    else if(nState==Tracking::OK)
    {
        if(!mbOnlyTracking)
            s << "SLAM MODE |  ";
        else
            s << "LOCALIZATION | ";
        int nMaps = mpAtlas->CountMaps();
        int nKFs = mpAtlas->KeyFramesInMap();
        int nMPs = mpAtlas->MapPointsInMap();
        s << "Maps: " << nMaps << ", KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
        if(mnTrackedVO>0)
            s << ", + VO matches: " << mnTrackedVO;
    }
    else if(nState==Tracking::LOST)
    {
        s << " TRACK LOST. TRYING TO RELOCALIZE ";
    }
    else if(nState==Tracking::SYSTEM_NOT_READY)
    {
        s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
    }

    // Qt 模式下不拼接底部信息栏 (信息已在界面状态栏显示)
    if(mbQtMode)
    {
        imText = im;
    }
    else
    {
        int baseline=0;
        cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);

        imText = cv::Mat(im.rows+textSize.height+10,im.cols,im.type());
        im.copyTo(imText.rowRange(0,im.rows).colRange(0,im.cols));
        imText.rowRange(im.rows,imText.rows) = cv::Mat::zeros(textSize.height+10,im.cols,im.type());
        cv::putText(imText,s.str(),cv::Point(5,imText.rows-5),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);
    }

}

void FrameDrawer::Update(Tracking *pTracker)
{
    unique_lock<mutex> lock(mMutex);

    // 动态一致性模式：mIm 已包含语义点+检测框可视化，DrawFrame 不再叠加 ORB 特征点
    mbShowDynamicVis = pTracker->IsShowDynamicVis();

    // 优先使用Detector绘制检测框后的彩色图像
    if(pTracker-> mImColor.data != nullptr)
    {
        pTracker->mImColor.copyTo(mIm);
        // cout << "拷贝带检测框的图像到mIm" << endl;
        pTracker->mImColor.release(); // 释放mImColor的内存，避免占用过多资源
    }
    else
    {
        pTracker->mImGray.copyTo(mIm);
    }
    // 长短焦模式：显示的是原始（未校正）图像，特征点必须用原始坐标绘制，
    // 否则与检测框/图像内容错位（校正坐标相对原始坐标有缩放平移）
    if(ORB_SLAM3::Frame::mbMultiFocal &&
       pTracker->mCurrentFrame.refermvKeys.size() == (size_t)pTracker->mCurrentFrame.N)
        mvCurrentKeys = pTracker->mCurrentFrame.refermvKeys;
    else
        mvCurrentKeys = pTracker->mCurrentFrame.mvKeys;
    mThDepth = pTracker->mCurrentFrame.mThDepth;
    mvCurrentDepth = pTracker->mCurrentFrame.mvDepth;

    if(both){
        if(ORB_SLAM3::Frame::mbMultiFocal &&
           pTracker->mCurrentFrame.refermvKeysRight.size() == pTracker->mCurrentFrame.mvKeysRight.size())
            mvCurrentKeysRight = pTracker->mCurrentFrame.refermvKeysRight;
        else
            mvCurrentKeysRight = pTracker->mCurrentFrame.mvKeysRight;
        // 优先使用带右目检测框的彩色图像（Tracking 已把右目检测结果画在 mImColorRight 上）
        if(pTracker->mImColorRight.data != nullptr)
        {
            pTracker->mImColorRight.copyTo(mImRight);
            pTracker->mImColorRight.release(); // 释放引用，避免占用过多内存
        }
        else
        {
            pTracker->mImRight.copyTo(mImRight);
        }
        N = mvCurrentKeys.size() + mvCurrentKeysRight.size();
    }
    else{
        N = mvCurrentKeys.size();
    }

    mvbVO = vector<bool>(N,false);
    mvbMap = vector<bool>(N,false);
    mvMovingProbability = vector<float>(N, 0.0f);
    mbOnlyTracking = pTracker->mbOnlyTracking;

    //Variables for the new visualization
    mCurrentFrame = pTracker->mCurrentFrame;
    mmProjectPoints = mCurrentFrame.mmProjectPoints;
    mmMatchedInImage.clear();

    mvpLocalMap = pTracker->GetLocalMapMPS();
    mvMatchedKeys.clear();
    mvMatchedKeys.reserve(N);
    mvpMatchedMPs.clear();
    mvpMatchedMPs.reserve(N);
    mvOutlierKeys.clear();
    mvOutlierKeys.reserve(N);
    mvpOutlierMPs.clear();
    mvpOutlierMPs.reserve(N);

    if(pTracker->mLastProcessedState==Tracking::NOT_INITIALIZED)
    {
        mvIniKeys=pTracker->mInitialFrame.mvKeys;
        mvIniMatches=pTracker->mvIniMatches;
    }
    else if(pTracker->mLastProcessedState==Tracking::OK)
    {
        // 修复：旧式双目（含长短焦模式）Frame::N 是左目特征数，
        // 而这里的 N 是 左+右 合并数，直接循环会越界读取 mvpMapPoints/mvbOutlier
        const int nTrackedKeys = pTracker->mCurrentFrame.N;
        for(int i=0; i<nTrackedKeys && i<N; i++)
        {
            MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                mvMovingProbability[i] = pMP->GetMovingProbability();
                if(!pTracker->mCurrentFrame.mvbOutlier[i])
                {
                    if(pMP->Observations()>0)
                        mvbMap[i]=true;
                    else
                        mvbVO[i]=true;

                    mmMatchedInImage[pMP->mnId] = mvCurrentKeys[i].pt;
                }
                else
                {
                    mvpOutlierMPs.push_back(pMP);
                    mvOutlierKeys.push_back(mvCurrentKeys[i]);
                }
            }
        }

    }
    mState=static_cast<int>(pTracker->mLastProcessedState);
}

} //namespace ORB_SLAM
