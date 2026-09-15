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

// 有向矩形（地面足迹）相交面积比：相交面积 / 较小矩形面积，返回 [0,1]。
// Detection3D 已带朝向，长条目标(车/公交)比等面积圆近似精确得多；
// 朝向无效时退化为平面内轴对齐矩形兜底。
static float OrientedFootprintOverlapRatio(const Detection3D& a, const Detection3D& b,
                                           const Eigen::Vector3f& nIn)
{
    Eigen::Vector3f n = nIn;
    if(n.norm() < 1e-6f) n = Eigen::Vector3f::UnitZ();
    n.normalize();
    const Eigen::Vector3f ref = (std::abs(n.x()) < 0.9f) ? Eigen::Vector3f::UnitX()
                                                         : Eigen::Vector3f::UnitZ();
    const Eigen::Vector3f e1 = n.cross(ref).normalized();
    const Eigen::Vector3f e2 = n.cross(e1).normalized();

    auto rect = [&](const Detection3D& d, Eigen::Vector2f out[4]) {
        Eigen::Vector3f h = d.heading;
        h -= n * n.dot(h);
        if(h.squaredNorm() < 0.25f) h = e1;          // 无有效朝向 → 轴对齐兜底
        else h.normalize();
        const Eigen::Vector3f w = n.cross(h).normalized();
        const float hd = std::max(d.depth, 1e-4f) * 0.5f;
        const float hw = std::max(d.width, 1e-4f) * 0.5f;
        const Eigen::Vector3f c = d.center - n * n.dot(d.center);   // 投到平面
        const Eigen::Vector3f cs[4] = {
            c - h*hd - w*hw, c + h*hd - w*hw, c + h*hd + w*hw, c - h*hd + w*hw};
        for(int i = 0; i < 4; i++)
            out[i] = Eigen::Vector2f(e1.dot(cs[i]), e2.dot(cs[i]));
    };

    Eigen::Vector2f A[4], B[4];
    rect(a, A);
    rect(b, B);

    auto area = [](const Eigen::Vector2f p[4]) {
        float s = 0.f;
        for(int i = 0; i < 4; i++) {
            const Eigen::Vector2f& q = p[(i+1) % 4];
            s += p[i].x()*q.y() - q.x()*p[i].y();
        }
        return std::abs(s) * 0.5f;
    };

    // 裁剪多边形 B 保证逆时针（inside 以"有向边左侧"为正）
    float sb = 0.f;
    for(int i = 0; i < 4; i++) {
        const Eigen::Vector2f& q = B[(i+1) % 4];
        sb += B[i].x()*q.y() - q.x()*B[i].y();
    }
    Eigen::Vector2f Bc[4];
    for(int i = 0; i < 4; i++) Bc[i] = (sb < 0.f) ? B[3-i] : B[i];

    auto inside = [](const Eigen::Vector2f& p, const Eigen::Vector2f& a0, const Eigen::Vector2f& a1) {
        return (a1.x()-a0.x())*(p.y()-a0.y()) - (a1.y()-a0.y())*(p.x()-a0.x()) >= 0.f;
    };
    auto intersect = [](const Eigen::Vector2f& p, const Eigen::Vector2f& q,
                        const Eigen::Vector2f& a0, const Eigen::Vector2f& a1) {
        const float x1=p.x(), y1=p.y(), x2=q.x(), y2=q.y();
        const float x3=a0.x(), y3=a0.y(), x4=a1.x(), y4=a1.y();
        const float den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4);
        if(std::abs(den) < 1e-12f) return q;
        const float t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / den;
        return Eigen::Vector2f(x1 + t*(x2-x1), y1 + t*(y2-y1));
    };

    // Sutherland–Hodgman：用凸多边形 B 裁剪 A
    std::vector<Eigen::Vector2f> poly(A, A+4);
    for(int i = 0; i < 4 && !poly.empty(); i++) {
        const Eigen::Vector2f a0 = Bc[i], a1 = Bc[(i+1) % 4];
        std::vector<Eigen::Vector2f> out;
        out.reserve(poly.size() + 4);
        for(size_t j = 0; j < poly.size(); j++) {
            const Eigen::Vector2f cur = poly[j];
            const Eigen::Vector2f prv = poly[(j + poly.size() - 1) % poly.size()];
            const bool curIn = inside(cur, a0, a1);
            const bool prvIn = inside(prv, a0, a1);
            if(curIn) {
                if(!prvIn) out.push_back(intersect(prv, cur, a0, a1));
                out.push_back(cur);
            } else if(prvIn) {
                out.push_back(intersect(prv, cur, a0, a1));
            }
        }
        poly.swap(out);
    }
    if(poly.size() < 3) return 0.f;

    float sInter = 0.f;
    for(size_t i = 0; i < poly.size(); i++) {
        const Eigen::Vector2f& q = poly[(i+1) % poly.size()];
        sInter += poly[i].x()*q.y() - q.x()*poly[i].y();
    }
    const float interArea = std::abs(sInter) * 0.5f;
    const float minArea = std::min(area(A), area(B));
    if(minArea <= 1e-9f) return 0.f;
    return std::min(1.f, interArea / minArea);
}

bool Map::AddOrUpdateDetection3D(const Detection3D& box, bool bAllowNew)
{
    unique_lock<mutex> lock(mMutexMap);

    // ===== P0 融合参数 =====
    const float MERGE_RATIO        = 0.3f;   // 足迹重合比例超过此值且同类 → 同一目标
    const int   MIN_FREEZE_OBS     = 5;      // 冻结前最少观测次数（原为 3，太易锁死坏值）
    const float FREEZE_STD_FACTOR  = 0.5f;   // 中心标准差 < 系数×min(W,D) 才冻结
    const float MIN_OBS_W          = 0.05f;  // 观测置信度权重下限（防 0 权重）
    const float OUTLIER_SIGMA      = 3.0f;   // 离群门：3σ
    const float OUTLIER_MIN_FACTOR = 1.5f;   // 离群门下限：1.5×min(W,D)
    // =======================

    // 先遍历所有历史框, 找同类最佳重合候选, 同时记录是否存在任意重合
    int bestIdx = -1;
    float bestRatio = 0.0f;
    bool bAnyOverlap = false;
    for(size_t i = 0; i < mvPersistentBoxes.size(); i++) {
        float ratio = OrientedFootprintOverlapRatio(mvPersistentBoxes[i], box, mPlaneNormal);
        if(ratio <= 1e-3f) continue;
        bAnyOverlap = true;
        if(mvPersistentBoxes[i].class_id == box.class_id && ratio > bestRatio) {
            bestRatio = ratio;
            bestIdx = (int)i;
        }
    }

    if(bestIdx >= 0 && bestRatio > MERGE_RATIO) {
        Detection3D& existing = mvPersistentBoxes[bestIdx];

        // 离群观测丢弃：成熟框(n>=3)且新中心偏离超过门限时忽略本次观测，
        // 避免个别远距离坏估计把已经稳定的框拽走。
        const float scaleRef = std::max(std::min(existing.width, existing.depth), 1e-4f);
        const float stdC = std::sqrt(std::max(existing.centerVar, 0.f));
        const float outlierGate = std::max(OUTLIER_SIGMA * stdC, OUTLIER_MIN_FACTOR * scaleRef);
        if(existing.nObservations >= 3 && (box.center - existing.center).norm() > outlierGate)
            return false;

        // 置信度加权增量均值：w = conf_new / (Σconf)，高置信观测影响更大
        const float wNew = std::max(box.confidence, MIN_OBS_W);
        if(existing.wSum <= 0.f) existing.wSum = std::max(existing.confidence, MIN_OBS_W);
        existing.wSum += wNew;
        const float alpha = wNew / existing.wSum;

        // 指数加权方差（用于稳定性冻结判据），恒非负
        const Eigen::Vector3f delta = box.center - existing.center;
        existing.center += alpha * delta;
        existing.centerVar = (1.f - alpha) * existing.centerVar
                           + alpha * delta.dot(box.center - existing.center);
        if(existing.centerVar < 0.f) existing.centerVar = 0.f;

        existing.width  += alpha * (box.width  - existing.width);
        existing.depth  += alpha * (box.depth  - existing.depth);
        existing.height += alpha * (box.height - existing.height);
        existing.confidence += alpha * (box.confidence - existing.confidence);

        // 朝向（方向无关轴，模π一致后加权平均）
        if(box.heading.squaredNorm() > 0.5f) {
            if(existing.heading.squaredNorm() <= 0.5f)
                existing.heading = box.heading.normalized();
            else {
                Eigen::Vector3f hNew = box.heading.normalized();
                if(hNew.dot(existing.heading) < 0.f) hNew = -hNew;
                existing.heading = (existing.heading + alpha * (hNew - existing.heading)).normalized();
            }
        }

        existing.nObservations++;

        // 冻结：观测足够 且 中心已稳定（标准差相对于目标尺寸足够小）
        if(!existing.bFrozen && existing.nObservations >= MIN_FREEZE_OBS) {
            const float stdNow = std::sqrt(std::max(existing.centerVar, 0.f));
            const float ref = std::max(std::min(existing.width, existing.depth), 1e-4f);
            if(stdNow < FREEZE_STD_FACTOR * ref)
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
    newBox.wSum = std::max(newBox.confidence, MIN_OBS_W);
    newBox.centerVar = 0.f;
    mvPersistentBoxes.push_back(newBox);
    return true;
}

void Map::ReplacePersistentBoxes(const std::vector<Detection3D>& boxes)
{
    unique_lock<mutex> lock(mMutexMap);
    mvPersistentBoxes = boxes;
}

} //namespace ORB_SLAM3
