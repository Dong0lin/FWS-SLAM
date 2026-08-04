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


#include "Map.h"

#include<mutex>
#include<cmath>
#include<algorithm>

namespace ORB_SLAM3
{

long unsigned int Map::nNextId=0;

Map::Map():mnMaxKFid(0),mnBigChangeIdx(0), mbImuInitialized(false), mnMapChange(0), mpFirstRegionKF(static_cast<KeyFrame*>(NULL)),
mbFail(false), mIsInUse(false), mHasTumbnail(false), mbBad(false), mnMapChangeNotified(0), mbIsInertial(false), mbIMU_BA1(false), mbIMU_BA2(false),
mPlaneNormal(Eigen::Vector3f::UnitZ()), mbPlaneEstimated(false), mPlaneRefHeight(-1.0f), mPlaneRefOffset(0.0f), mPlaneDynamicOffset(0.0f), mPlaneScaleLambda(1.0f)
{
    mnId=nNextId++;
    mThumbnail = static_cast<unsigned char*>(nullptr);
}

Map::Map(int initKFid):mnInitKFid(initKFid), mnMaxKFid(initKFid),/*mnLastLoopKFid(initKFid),*/ mnBigChangeIdx(0), mIsInUse(false),
                       mHasTumbnail(false), mbBad(false), mbImuInitialized(false), mpFirstRegionKF(static_cast<KeyFrame*>(NULL)),
                       mnMapChange(0), mbFail(false), mnMapChangeNotified(0), mbIsInertial(false), mbIMU_BA1(false), mbIMU_BA2(false),
                       mPlaneNormal(Eigen::Vector3f::UnitZ()), mbPlaneEstimated(false), mPlaneRefHeight(-1.0f), mPlaneRefOffset(0.0f), mPlaneDynamicOffset(0.0f), mPlaneScaleLambda(1.0f)
{
    mnId=nNextId++;
    mThumbnail = static_cast<unsigned char*>(nullptr);
}

Map::~Map()
{
    //TODO: erase all points from memory
    mspMapPoints.clear();

    //TODO: erase all keyframes from memory
    mspKeyFrames.clear();

    if(mThumbnail)
        delete mThumbnail;
    mThumbnail = static_cast<unsigned char*>(nullptr);

    mvpReferenceMapPoints.clear();
    mvpKeyFrameOrigins.clear();
}

void Map::AddKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    if(mspKeyFrames.empty()){
        cout << "First KF:" << pKF->mnId << "; Map init KF:" << mnInitKFid << endl;
        mnInitKFid = pKF->mnId;
        mpKFinitial = pKF;
        mpKFlowerID = pKF;
    }
    mspKeyFrames.insert(pKF);
    if(pKF->mnId>mnMaxKFid)
    {
        mnMaxKFid=pKF->mnId;
    }
    if(pKF->mnId<mpKFlowerID->mnId)
    {
        mpKFlowerID = pKF;
    }
}

void Map::AddMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

void Map::SetImuInitialized()
{
    unique_lock<mutex> lock(mMutexMap);
    mbImuInitialized = true;
}

bool Map::isImuInitialized()
{
    unique_lock<mutex> lock(mMutexMap);
    return mbImuInitialized;
}

void Map::EraseMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::EraseKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);
    if(mspKeyFrames.size()>0)
    {
        if(pKF->mnId == mpKFlowerID->mnId)
        {
            vector<KeyFrame*> vpKFs = vector<KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());
            sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);
            mpKFlowerID = vpKFs[0];
        }
    }
    else
    {
        mpKFlowerID = 0;
    }

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs)
{
    unique_lock<mutex> lock(mMutexMap);
    mvpReferenceMapPoints = vpMPs;
}

void Map::InformNewBigChange()
{
    unique_lock<mutex> lock(mMutexMap);
    mnBigChangeIdx++;
}

int Map::GetLastBigChangeIdx()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnBigChangeIdx;
}

vector<KeyFrame*> Map::GetAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());
}

vector<MapPoint*> Map::GetAllMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapPoint*>(mspMapPoints.begin(),mspMapPoints.end());
}

long unsigned int Map::MapPointsInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

long unsigned int Map::KeyFramesInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}

vector<MapPoint*> Map::GetReferenceMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return mvpReferenceMapPoints;
}

long unsigned int Map::GetId()
{
    return mnId;
}
long unsigned int Map::GetInitKFid()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnInitKFid;
}

void Map::SetInitKFid(long unsigned int initKFif)
{
    unique_lock<mutex> lock(mMutexMap);
    mnInitKFid = initKFif;
}

long unsigned int Map::GetMaxKFid()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnMaxKFid;
}

KeyFrame* Map::GetOriginKF()
{
    return mpKFinitial;
}

void Map::SetCurrentMap()
{
    mIsInUse = true;
}

void Map::SetStoredMap()
{
    mIsInUse = false;
}

void Map::clear()
{
//    for(set<MapPoint*>::iterator sit=mspMapPoints.begin(), send=mspMapPoints.end(); sit!=send; sit++)
//        delete *sit;

    for(set<KeyFrame*>::iterator sit=mspKeyFrames.begin(), send=mspKeyFrames.end(); sit!=send; sit++)
    {
        KeyFrame* pKF = *sit;
        pKF->UpdateMap(static_cast<Map*>(NULL));
//        delete *sit;
    }

    mspMapPoints.clear();
    mspKeyFrames.clear();
    mnMaxKFid = mnInitKFid;
    mbImuInitialized = false;
    mvpReferenceMapPoints.clear();
    mvpKeyFrameOrigins.clear();
    mbIMU_BA1 = false;
    mbIMU_BA2 = false;
}

bool Map::IsInUse()
{
    return mIsInUse;
}

void Map::SetBad()
{
    mbBad = true;
}

bool Map::IsBad()
{
    return mbBad;
}


void Map::ApplyScaledRotation(const Sophus::SE3f &T, const float s, const bool bScaledVel)
{
    unique_lock<mutex> lock(mMutexMap);

    // Body position (IMU) of first keyframe is fixed to (0,0,0)
    Sophus::SE3f Tyw = T;
    Eigen::Matrix3f Ryw = Tyw.rotationMatrix();
    Eigen::Vector3f tyw = Tyw.translation();

    for(set<KeyFrame*>::iterator sit=mspKeyFrames.begin(); sit!=mspKeyFrames.end(); sit++)
    {
        KeyFrame* pKF = *sit;
        Sophus::SE3f Twc = pKF->GetPoseInverse();
        Twc.translation() *= s;
        Sophus::SE3f Tyc = Tyw*Twc;
        Sophus::SE3f Tcy = Tyc.inverse();
        pKF->SetPose(Tcy);
        Eigen::Vector3f Vw = pKF->GetVelocity();
        if(!bScaledVel)
            pKF->SetVelocity(Ryw*Vw);
        else
            pKF->SetVelocity(Ryw*Vw*s);

    }
    for(set<MapPoint*>::iterator sit=mspMapPoints.begin(); sit!=mspMapPoints.end(); sit++)
    {
        MapPoint* pMP = *sit;
        pMP->SetWorldPos(s * Ryw * pMP->GetWorldPos() + tyw);
        pMP->UpdateNormalAndDepth();
    }
    mnMapChange++;
}

void Map::SetInertialSensor()
{
    unique_lock<mutex> lock(mMutexMap);
    mbIsInertial = true;
}

bool Map::IsInertial()
{
    unique_lock<mutex> lock(mMutexMap);
    return mbIsInertial;
}

void Map::SetIniertialBA1()
{
    unique_lock<mutex> lock(mMutexMap);
    mbIMU_BA1 = true;
}

void Map::SetIniertialBA2()
{
    unique_lock<mutex> lock(mMutexMap);
    mbIMU_BA2 = true;
}

bool Map::GetIniertialBA1()
{
    unique_lock<mutex> lock(mMutexMap);
    return mbIMU_BA1;
}

bool Map::GetIniertialBA2()
{
    unique_lock<mutex> lock(mMutexMap);
    return mbIMU_BA2;
}

void Map::ChangeId(long unsigned int nId)
{
    mnId = nId;
}

unsigned int Map::GetLowerKFID()
{
    unique_lock<mutex> lock(mMutexMap);
    if (mpKFlowerID) {
        return mpKFlowerID->mnId;
    }
    return 0;
}

int Map::GetMapChangeIndex()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnMapChange;
}

void Map::IncreaseChangeIndex()
{
    unique_lock<mutex> lock(mMutexMap);
    mnMapChange++;
}

int Map::GetLastMapChange()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnMapChangeNotified;
}

void Map::SetLastMapChange(int currentChangeId)
{
    unique_lock<mutex> lock(mMutexMap);
    mnMapChangeNotified = currentChangeId;
}

void Map::PreSave(std::set<GeometricCamera*> &spCams)
{
    int nMPWithoutObs = 0;
    for(MapPoint* pMPi : mspMapPoints)
    {
        if(!pMPi || pMPi->isBad())
            continue;

        if(pMPi->GetObservations().size() == 0)
        {
            nMPWithoutObs++;
        }
        map<KeyFrame*, std::tuple<int,int>> mpObs = pMPi->GetObservations();
        for(map<KeyFrame*, std::tuple<int,int>>::iterator it= mpObs.begin(), end=mpObs.end(); it!=end; ++it)
        {
            if(it->first->GetMap() != this || it->first->isBad())
            {
                pMPi->EraseObservation(it->first);
            }

        }
    }

    // Saves the id of KF origins
    mvBackupKeyFrameOriginsId.clear();
    mvBackupKeyFrameOriginsId.reserve(mvpKeyFrameOrigins.size());
    for(int i = 0, numEl = mvpKeyFrameOrigins.size(); i < numEl; ++i)
    {
        mvBackupKeyFrameOriginsId.push_back(mvpKeyFrameOrigins[i]->mnId);
    }


    // Backup of MapPoints
    mvpBackupMapPoints.clear();
    for(MapPoint* pMPi : mspMapPoints)
    {
        if(!pMPi || pMPi->isBad())
            continue;

        mvpBackupMapPoints.push_back(pMPi);
        pMPi->PreSave(mspKeyFrames,mspMapPoints);
    }

    // Backup of KeyFrames
    mvpBackupKeyFrames.clear();
    for(KeyFrame* pKFi : mspKeyFrames)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        mvpBackupKeyFrames.push_back(pKFi);
        pKFi->PreSave(mspKeyFrames,mspMapPoints, spCams);
    }

    mnBackupKFinitialID = -1;
    if(mpKFinitial)
    {
        mnBackupKFinitialID = mpKFinitial->mnId;
    }

    mnBackupKFlowerID = -1;
    if(mpKFlowerID)
    {
        mnBackupKFlowerID = mpKFlowerID->mnId;
    }

}

void Map::PostLoad(KeyFrameDatabase* pKFDB, ORBVocabulary* pORBVoc/*, map<long unsigned int, KeyFrame*>& mpKeyFrameId*/, map<unsigned int, GeometricCamera*> &mpCams)
{
    std::copy(mvpBackupMapPoints.begin(), mvpBackupMapPoints.end(), std::inserter(mspMapPoints, mspMapPoints.begin()));
    std::copy(mvpBackupKeyFrames.begin(), mvpBackupKeyFrames.end(), std::inserter(mspKeyFrames, mspKeyFrames.begin()));

    map<long unsigned int,MapPoint*> mpMapPointId;
    for(MapPoint* pMPi : mspMapPoints)
    {
        if(!pMPi || pMPi->isBad())
            continue;

        pMPi->UpdateMap(this);
        mpMapPointId[pMPi->mnId] = pMPi;
    }

    map<long unsigned int, KeyFrame*> mpKeyFrameId;
    for(KeyFrame* pKFi : mspKeyFrames)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        pKFi->UpdateMap(this);
        pKFi->SetORBVocabulary(pORBVoc);
        pKFi->SetKeyFrameDatabase(pKFDB);
        mpKeyFrameId[pKFi->mnId] = pKFi;
    }

    // References reconstruction between different instances
    for(MapPoint* pMPi : mspMapPoints)
    {
        if(!pMPi || pMPi->isBad())
            continue;

        pMPi->PostLoad(mpKeyFrameId, mpMapPointId);
    }

    for(KeyFrame* pKFi : mspKeyFrames)
    {
        if(!pKFi || pKFi->isBad())
            continue;

        pKFi->PostLoad(mpKeyFrameId, mpMapPointId, mpCams);
        pKFDB->add(pKFi);
    }


    if(mnBackupKFinitialID != -1)
    {
        mpKFinitial = mpKeyFrameId[mnBackupKFinitialID];
    }

    if(mnBackupKFlowerID != -1)
    {
        mpKFlowerID = mpKeyFrameId[mnBackupKFlowerID];
    }

    mvpKeyFrameOrigins.clear();
    mvpKeyFrameOrigins.reserve(mvBackupKeyFrameOriginsId.size());
    for(int i = 0; i < mvBackupKeyFrameOriginsId.size(); ++i)
    {
        mvpKeyFrameOrigins.push_back(mpKeyFrameId[mvBackupKeyFrameOriginsId[i]]);
    }

    mvpBackupMapPoints.clear();
}


void Map::SetPlaneModel(const Eigen::Vector3f& normal, float offset)
{
    unique_lock<mutex> lock(mMutexMap);
    mPlaneNormal = normal;
    mvpPlaneOffsets.clear();
    mvpPlaneOffsets.push_back(offset);
    mPlaneDynamicOffset = offset;
    mbPlaneEstimated = true;
}

const Eigen::Vector3f& Map::GetPlaneNormal() const
{
    return mPlaneNormal;
}

const std::vector<float>& Map::GetPlaneOffsets() const
{
    return mvpPlaneOffsets;
}

float Map::GetDistanceToNearestPlane(const Eigen::Vector3f& Pw) const
{
    if(!mbPlaneEstimated || mvpPlaneOffsets.empty()) return 0.0f;
    float signedDist = mPlaneNormal.dot(Pw);
    float minDist = std::abs(signedDist - mvpPlaneOffsets[0]);
    for(size_t i = 1; i < mvpPlaneOffsets.size(); i++) {
        float d = std::abs(signedDist - mvpPlaneOffsets[i]);
        if(d < minDist) minDist = d;
    }
    return minDist;
}

bool Map::IsPlaneEstimated()
{
    unique_lock<mutex> lock(mMutexMap);
    return mbPlaneEstimated;
}

void Map::SetPlaneRefHeight(float height)
{
    unique_lock<mutex> lock(mMutexMap);
    mPlaneRefHeight = height;
    // cout << "设置平面高度为： " << mPlaneRefHeight << endl;
}

float Map::GetPlaneRefHeight() const
{
    return mPlaneRefHeight;
}

void Map::SetPlaneRefOffset(float offset)
{
    unique_lock<mutex> lock(mMutexMap);
    mPlaneRefOffset = offset;
}

float Map::GetPlaneRefOffset() const
{
    return mPlaneRefOffset;
}

void Map::SetPlaneDynamicOffset(float offset)
{
    unique_lock<mutex> lock(mMutexMap);
    mPlaneDynamicOffset = offset;
}

float Map::GetPlaneDynamicOffset() const
{
    return mPlaneDynamicOffset;
}

void Map::SetPlaneScaleLambda(float lambda)
{
    unique_lock<mutex> lock(mMutexMap);
    mPlaneScaleLambda = lambda;
}

float Map::GetPlaneScaleLambda() const
{
    return mPlaneScaleLambda;
}

void Map::PushTriangRatio(float ratio)
{
    unique_lock<mutex> lock(mMutexMap);
    mvTriangRatios.push_back(ratio);
}

const std::vector<float>& Map::GetTriangRatios() const
{
    return mvTriangRatios;
}

void Map::ClearTriangRatios()
{
    unique_lock<mutex> lock(mMutexMap);
    mvTriangRatios.clear();
}

bool Map::AddOrUpdateDetection3D(const Detection3D& box, bool bAllowNew)
{
    unique_lock<mutex> lock(mMutexMap);

    // 足迹近似为等面积圆, 计算重合比例 (旋转无关; Detection3D未存朝向, 无法做精确矩形IoU)
    // 返回: 相交面积 / 较小圆面积, 范围[0, 1]
    auto footprintOverlapRatio = [](const Detection3D& a, const Detection3D& b) -> float {
        const float PI = 3.14159265f;
        float r1 = std::sqrt(std::max(a.width * a.depth, 1e-8f) / PI);
        float r2 = std::sqrt(std::max(b.width * b.depth, 1e-8f) / PI);
        float d = (a.center - b.center).norm();
        if(d >= r1 + r2) return 0.0f;                       // 无重合
        float rmin = std::min(r1, r2), rmax = std::max(r1, r2);
        float areaMin = PI * rmin * rmin;
        if(d <= rmax - rmin) return 1.0f;                   // 小圆完全被包含
        // 圆-圆相交面积 (透镜公式)
        float d2 = d*d, r1sq = r1*r1, r2sq = r2*r2;
        float ca = std::max(-1.0f, std::min(1.0f, (d2 + r1sq - r2sq) / (2.0f*d*r1)));
        float cb = std::max(-1.0f, std::min(1.0f, (d2 + r2sq - r1sq) / (2.0f*d*r2)));
        float alpha = std::acos(ca), beta = std::acos(cb);
        float area = r1sq*(alpha - 0.5f*std::sin(2.0f*alpha)) + r2sq*(beta - 0.5f*std::sin(2.0f*beta));
        return area / areaMin;
    };

    const float MERGE_RATIO = 0.3f;  // 重合比例超过此值且同类 → 判定为同一目标

    // 先遍历所有历史框, 找同类最佳重合候选, 同时记录是否存在任意重合
    int bestIdx = -1;
    float bestRatio = 0.0f;
    bool bAnyOverlap = false;
    for(size_t i = 0; i < mvPersistentBoxes.size(); i++) {
        float ratio = footprintOverlapRatio(mvPersistentBoxes[i], box);
        if(ratio <= 1e-3f) continue;
        bAnyOverlap = true;
        if(mvPersistentBoxes[i].class_id == box.class_id && ratio > bestRatio) {
            bestRatio = ratio;
            bestIdx = (int)i;
        }
    }

    if(bestIdx >= 0 && bestRatio > MERGE_RATIO) {
        // 同一目标: 平均融合位置和尺寸
        Detection3D& existing = mvPersistentBoxes[bestIdx];
        existing.nObservations++;
        if(!existing.bFrozen) {
            // 移动平均: 观测越多新观测权重越低
            float w = 1.0f / existing.nObservations;
            existing.center += w * (box.center - existing.center);
            existing.width  += w * (box.width  - existing.width);
            existing.depth  += w * (box.depth  - existing.depth);
            existing.height += w * (box.height - existing.height);
            // 观测达到3次后冻结
            if(existing.nObservations >= 3)
                existing.bFrozen = true;
        }
        return false;
    }

    if(bAnyOverlap) {
        // 有重合但比例不足(或类别不同): 丢弃新框, 保留历史框
        return false;
    }

    // 数量配额已满: 不允许新建
    if(!bAllowNew)
        return false;

    // 与所有历史框均无重合 → 新目标
    Detection3D newBox = box;
    newBox.nObservations = 1;
    newBox.bFrozen = false;
    mvPersistentBoxes.push_back(newBox);
    return true;
}

void Map::ReplacePersistentBoxes(const std::vector<Detection3D>& boxes)
{
    unique_lock<mutex> lock(mMutexMap);
    mvPersistentBoxes = boxes;
}

} //namespace ORB_SLAM3
