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

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "common.h"
#include <mutex>

namespace ORB_SLAM3
{


MapDrawer::MapDrawer(Atlas* pAtlas, const string &strSettingPath, Settings* settings):mpAtlas(pAtlas)
{
    if(settings){
        newParameterLoader(settings);
    }
    else{
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        bool is_correct = ParseViewerParamFile(fSettings);

        if(!is_correct)
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
}

void MapDrawer::newParameterLoader(Settings *settings) {
    mKeyFrameSize = settings->keyFrameSize();
    mKeyFrameLineWidth = settings->keyFrameLineWidth();
    mGraphLineWidth = settings->graphLineWidth();
    mPointSize = settings->pointSize();
    mCameraSize = settings->cameraSize();
    mCameraLineWidth  = settings->cameraLineWidth();
}

bool MapDrawer::ParseViewerParamFile(cv::FileStorage &fSettings)
{
    bool b_miss_params = false;

    cv::FileNode node = fSettings["Viewer.KeyFrameSize"];
    if(!node.empty())
    {
        mKeyFrameSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.KeyFrameSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.KeyFrameLineWidth"];
    if(!node.empty())
    {
        mKeyFrameLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.KeyFrameLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.GraphLineWidth"];
    if(!node.empty())
    {
        mGraphLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.GraphLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.PointSize"];
    if(!node.empty())
    {
        mPointSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.PointSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.CameraSize"];
    if(!node.empty())
    {
        mCameraSize = node.real();
    }
    else
    {
        std::cerr << "*Viewer.CameraSize parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    node = fSettings["Viewer.CameraLineWidth"];
    if(!node.empty())
    {
        mCameraLineWidth = node.real();
    }
    else
    {
        std::cerr << "*Viewer.CameraLineWidth parameter doesn't exist or is not a real number*" << std::endl;
        b_miss_params = true;
    }

    return !b_miss_params;
}

void MapDrawer::DrawMapPoints()
{
    Map* pActiveMap = mpAtlas->GetCurrentMap();
    if(!pActiveMap)
        return;

    const vector<MapPoint*> &vpMPs = pActiveMap->GetAllMapPoints();
    const vector<MapPoint*> &vpRefMPs = pActiveMap->GetReferenceMapPoints();

    set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    if(vpMPs.empty())
        return;

    // ── 第一遍：非参考点（已固定的点），按语义着色 ──
    const float semanticPointSize = mPointSize * 3.0f;  // 语义点显著大于普通点

    // 子批次 1a：非语义非参考点 → 蓝色，普通大小
    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0, 0.0, 1.0);
    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
        MapPoint* pMP = vpMPs[i];
        if(pMP->isBad() || spRefMPs.count(pMP))
            continue;
        if(pMP->mnSemanticClass >= 0)
            continue;  // 语义点在下一批次绘制
        Eigen::Matrix<float,3,1> pos = pMP->GetWorldPos();
        glVertex3f(pos(0),pos(1),pos(2));
    }
    glEnd();

    // 子批次 1b：语义非参考点 → 按语义着色，大号
    glPointSize(semanticPointSize);
    glBegin(GL_POINTS);
    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
        MapPoint* pMP = vpMPs[i];
        if(pMP->isBad() || spRefMPs.count(pMP))
            continue;
        int semClass = pMP->mnSemanticClass;
        if(semClass < 0)
            continue;  // 非语义点已在上一批次绘制

        if(pMP->IsDynamicMapPoint())
        {
            // 动态语义点 → 红色
            glColor3f(1.0f, 0.0f, 0.0f);
        }
        else
        {
            // 静态语义点 → 按 common.h COLORS 着色
            const auto& c = COLORS[semClass % COLORS.size()];
            glColor3f(c[0] / 255.0f, c[1] / 255.0f, c[2] / 255.0f);
        }
        Eigen::Matrix<float,3,1> pos = pMP->GetWorldPos();
        glVertex3f(pos(0),pos(1),pos(2));
    }
    glEnd();

    // ── 第二遍：参考点（正在优化的点）──
    // 非语义参考点 → 黑色/普通大小；语义参考点 → 同语义着色/大号
    // 先画非语义参考点
    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0, 0.0, 0.0);
    for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
    {
        MapPoint* pMP = *sit;
        if(pMP->isBad() || pMP->mnSemanticClass >= 0)
            continue;
        Eigen::Matrix<float,3,1> pos = pMP->GetWorldPos();
        glVertex3f(pos(0),pos(1),pos(2));
    }
    glEnd();

    // 再画语义参考点（大号，按类别着色，保持与非参考语义点一致）
    glPointSize(semanticPointSize);
    glBegin(GL_POINTS);
    for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
    {
        MapPoint* pMP = *sit;
        if(pMP->isBad() || pMP->mnSemanticClass < 0)
            continue;

        if(pMP->IsDynamicMapPoint())
        {
            glColor3f(1.0f, 0.0f, 0.0f);
        }
        else
        {
            const auto& c = COLORS[pMP->mnSemanticClass % COLORS.size()];
            glColor3f(c[0] / 255.0f, c[1] / 255.0f, c[2] / 255.0f);
        }
        Eigen::Matrix<float,3,1> pos = pMP->GetWorldPos();
        glVertex3f(pos(0),pos(1),pos(2));
    }
    glEnd();
}

void MapDrawer::DrawPlane()
{
    Map* pActiveMap = mpAtlas->GetCurrentMap();
    if(!pActiveMap || !pActiveMap->IsPlaneEstimated())
        return;

    const Eigen::Vector3f& N = pActiveMap->GetPlaneNormal();
    float d = pActiveMap->GetPlaneOffsets()[0];

    Eigen::Vector3f camCenter = mCameraPose.inverse().translation();
    // 相机在地面上的投影点
    Eigen::Vector3f camProj = camCenter - N * (N.dot(camCenter) - d);

    // 构建水平面上的两个正交方向
    Eigen::Vector3f ref = (std::abs(N.x()) < 0.9f) ? Eigen::Vector3f::UnitX() : Eigen::Vector3f::UnitZ();
    Eigen::Vector3f U = N.cross(ref).normalized();
    Eigen::Vector3f V = N.cross(U).normalized();

    // 面片半径：先随相机高度扩大确保覆盖视野，
    // 再扩展到已建图区域（关键帧在平面上的投影分布范围）。
    // 长短焦模式地图尺度大（轨迹跨度大），单按相机高度的面片会显得太小；
    // 用关键帧分布做上界，尺度多大面片就多大。
    float camHeight = std::abs(N.dot(camCenter) - d);
    float radius = std::max(camHeight * 4.0f, 50.0f);
    {
        const std::vector<KeyFrame*>& vpKFs = pActiveMap->GetAllKeyFrames();
        for(KeyFrame* pKF : vpKFs)
        {
            if(!pKF || pKF->isBad()) continue;
            Eigen::Vector3f kfProj = pKF->GetCameraCenter() - N * (N.dot(pKF->GetCameraCenter()) - d);
            float dist = (kfProj - camProj).norm();
            if(dist > radius) radius = dist;
        }
        radius *= 1.3f;   // 留 30% 边距
    }

    // 细分网格
    const int steps = 40;
    const float stepSize = radius * 2.0f / steps;

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // 半透明绿色面片
    glColor4f(0.0f, 0.8f, 0.0f, 0.25f);
    glBegin(GL_QUADS);
    for(int i = 0; i < steps; i++) {
        for(int j = 0; j < steps; j++) {
            float u0 = -radius +  i    * stepSize;
            float u1 = -radius + (i+1) * stepSize;
            float v0 = -radius +  j    * stepSize;
            float v1 = -radius + (j+1) * stepSize;

            Eigen::Vector3f p00 = camProj + U*u0 + V*v0;
            Eigen::Vector3f p10 = camProj + U*u1 + V*v0;
            Eigen::Vector3f p11 = camProj + U*u1 + V*v1;
            Eigen::Vector3f p01 = camProj + U*u0 + V*v1;

            glVertex3f(p00(0), p00(1), p00(2));
            glVertex3f(p10(0), p10(1), p10(2));
            glVertex3f(p11(0), p11(1), p11(2));
            glVertex3f(p01(0), p01(1), p01(2));
        }
    }
    glEnd();

    // 边框网格线（深绿色）
    glColor4f(0.0f, 0.5f, 0.0f, 0.4f);
    glLineWidth(1.0f);
    for(int i = 0; i <= steps; i++) {
        float offset = -radius + i * stepSize;
        glBegin(GL_LINE_STRIP);
        for(int j = 0; j <= steps; j++) {
            float v = -radius + j * stepSize;
            Eigen::Vector3f p = camProj + U*offset + V*v;
            glVertex3f(p(0), p(1), p(2));
        }
        glEnd();
        glBegin(GL_LINE_STRIP);
        for(int j = 0; j <= steps; j++) {
            float u = -radius + j * stepSize;
            Eigen::Vector3f p = camProj + U*u + V*offset;
            glVertex3f(p(0), p(1), p(2));
        }
        glEnd();
    }

    // 法向量箭头（向上，长度 = 面片半径的 20%）
    float arrowLen = radius * 0.2f;
    Eigen::Vector3f arrowBase = camProj;
    Eigen::Vector3f arrowTip  = arrowBase + N * arrowLen;
    glColor4f(0.0f, 1.0f, 0.0f, 0.8f);
    glLineWidth(3.0f);
    glBegin(GL_LINES);
    glVertex3f(arrowBase(0), arrowBase(1), arrowBase(2));
    glVertex3f(arrowTip(0),  arrowTip(1),  arrowTip(2));
    glEnd();

    glDisable(GL_BLEND);
}

void MapDrawer::DrawDetection3Ds()
{
    Map* pMap = mpAtlas->GetCurrentMap();
    if(!pMap || !pMap->IsPlaneEstimated()) return;

    const std::vector<Detection3D>& boxes = pMap->GetPersistentBoxes();
    if(boxes.empty()) return;

    const Eigen::Vector3f& N = pMap->GetPlaneNormal();
    // 世界固定基准（仅对未设置朝向的旧框兜底）
    Eigen::Vector3f ref = (std::abs(N.x()) < 0.9f) ? Eigen::Vector3f::UnitX() : Eigen::Vector3f::UnitZ();
    Eigen::Vector3f U0 = N.cross(ref).normalized();
    Eigen::Vector3f V0 = N.cross(U0).normalized();

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    for(const auto& box : boxes) {
        if(!box.bValid) continue;
        int ci = (box.class_id >= 0 && box.class_id < (int)COLORS.size()) ? box.class_id : 0;
        float alphaF = std::min(1.0f, box.nObservations * 0.3f + 0.35f);

        // 使用 common.h 中的 COLORS（与语义点云颜色一致）
        float r = COLORS[ci][0] / 255.0f;
        float g = COLORS[ci][1] / 255.0f;
        float b = COLORS[ci][2] / 255.0f;

        // 有朝向的框：width 沿车宽方向（N×heading），depth 沿车长方向（heading）
        Eigen::Vector3f hU, hV;
        if(box.heading.squaredNorm() > 0.5f)
        {
            Eigen::Vector3f V = box.heading.normalized();
            Eigen::Vector3f U = N.cross(V).normalized();
            hU = U * (box.width * 0.5f);
            hV = V * (box.depth * 0.5f);
        }
        else
        {
            hU = U0 * (box.width * 0.5f);
            hV = V0 * (box.depth * 0.5f);
        }
        Eigen::Vector3f up  = N * box.height;

        Eigen::Vector3f b1 = box.center - hU - hV;
        Eigen::Vector3f b2 = box.center + hU - hV;
        Eigen::Vector3f b3 = box.center + hU + hV;
        Eigen::Vector3f b4 = box.center - hU + hV;
        Eigen::Vector3f t1 = b1 + up, t2 = b2 + up, t3 = b3 + up, t4 = b4 + up;

        glColor4f(r, g, b, alphaF * 0.45f);
        glBegin(GL_QUADS);
        glVertex3f(b1(0),b1(1),b1(2)); glVertex3f(b2(0),b2(1),b2(2));
        glVertex3f(b3(0),b3(1),b3(2)); glVertex3f(b4(0),b4(1),b4(2));
        glEnd();
        glBegin(GL_QUADS);
        glVertex3f(t1(0),t1(1),t1(2)); glVertex3f(t2(0),t2(1),t2(2));
        glVertex3f(t3(0),t3(1),t3(2)); glVertex3f(t4(0),t4(1),t4(2));
        glEnd();

        glColor4f(r, g, b, alphaF);
        glLineWidth(2.5f);
        glBegin(GL_LINE_LOOP);
        glVertex3f(b1(0),b1(1),b1(2)); glVertex3f(b2(0),b2(1),b2(2));
        glVertex3f(b3(0),b3(1),b3(2)); glVertex3f(b4(0),b4(1),b4(2));
        glEnd();
        glBegin(GL_LINE_LOOP);
        glVertex3f(t1(0),t1(1),t1(2)); glVertex3f(t2(0),t2(1),t2(2));
        glVertex3f(t3(0),t3(1),t3(2)); glVertex3f(t4(0),t4(1),t4(2));
        glEnd();
        glBegin(GL_LINES);
        glVertex3f(b1(0),b1(1),b1(2)); glVertex3f(t1(0),t1(1),t1(2));
        glVertex3f(b2(0),b2(1),b2(2)); glVertex3f(t2(0),t2(1),t2(2));
        glVertex3f(b3(0),b3(1),b3(2)); glVertex3f(t3(0),t3(1),t3(2));
        glVertex3f(b4(0),b4(1),b4(2)); glVertex3f(t4(0),t4(1),t4(2));
        glEnd();
        glLineWidth(1.0f);
    }

    glDisable(GL_BLEND);
}

void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph, const bool bDrawInertialGraph, const bool bDrawOptLba)
{
    const float &w = mKeyFrameSize;
    const float h = w*0.75;
    const float z = w*0.6;

    Map* pActiveMap = mpAtlas->GetCurrentMap();
    // DEBUG LBA
    std::set<long unsigned int> sOptKFs = pActiveMap->msOptKFs;
    std::set<long unsigned int> sFixedKFs = pActiveMap->msFixedKFs;

    if(!pActiveMap)
        return;

    const vector<KeyFrame*> vpKFs = pActiveMap->GetAllKeyFrames();

    if(bDrawKF)
    {
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKF = vpKFs[i];
            Eigen::Matrix4f Twc = pKF->GetPoseInverse().matrix();
            unsigned int index_color = pKF->mnOriginMapId;

            glPushMatrix();

            glMultMatrixf((GLfloat*)Twc.data());

            if(!pKF->GetParent()) // It is the first KF in the map
            {
                glLineWidth(mKeyFrameLineWidth*5);
                glColor3f(1.0f,0.0f,0.0f);
                glBegin(GL_LINES);
            }
            else
            {
                //cout << "Child KF: " << vpKFs[i]->mnId << endl;
                glLineWidth(mKeyFrameLineWidth);
                if (bDrawOptLba) {
                    if(sOptKFs.find(pKF->mnId) != sOptKFs.end())
                    {
                        glColor3f(0.0f,1.0f,0.0f); // Green -> Opt KFs
                    }
                    else if(sFixedKFs.find(pKF->mnId) != sFixedKFs.end())
                    {
                        glColor3f(1.0f,0.0f,0.0f); // Red -> Fixed KFs
                    }
                    else
                    {
                        glColor3f(0.0f,0.0f,1.0f); // Basic color
                    }
                }
                else
                {
                    glColor3f(0.0f,0.0f,1.0f); // Basic color
                }
                glBegin(GL_LINES);
            }

            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();

            glPopMatrix();

            glEnd();
        }
    }

    if(bDrawGraph)
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(0.0f,1.0f,0.0f,0.6f);
        glBegin(GL_LINES);

        // cout << "-----------------Draw graph-----------------" << endl;
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            // Covisibility Graph
            const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
            Eigen::Vector3f Ow = vpKFs[i]->GetCameraCenter();
            if(!vCovKFs.empty())
            {
                for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                {
                    if((*vit)->mnId<vpKFs[i]->mnId)
                        continue;
                    Eigen::Vector3f Ow2 = (*vit)->GetCameraCenter();
                    glVertex3f(Ow(0),Ow(1),Ow(2));
                    glVertex3f(Ow2(0),Ow2(1),Ow2(2));
                }
            }

            // Spanning tree
            KeyFrame* pParent = vpKFs[i]->GetParent();
            if(pParent)
            {
                Eigen::Vector3f Owp = pParent->GetCameraCenter();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owp(0),Owp(1),Owp(2));
            }

            // Loops
            set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
            for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
            {
                if((*sit)->mnId<vpKFs[i]->mnId)
                    continue;
                Eigen::Vector3f Owl = (*sit)->GetCameraCenter();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owl(0),Owl(1),Owl(2));
            }
        }

        glEnd();
    }

    if(bDrawInertialGraph && pActiveMap->isImuInitialized())
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(1.0f,0.0f,0.0f,0.6f);
        glBegin(GL_LINES);

        //Draw inertial links
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKFi = vpKFs[i];
            Eigen::Vector3f Ow = pKFi->GetCameraCenter();
            KeyFrame* pNext = pKFi->mNextKF;
            if(pNext)
            {
                Eigen::Vector3f Owp = pNext->GetCameraCenter();
                glVertex3f(Ow(0),Ow(1),Ow(2));
                glVertex3f(Owp(0),Owp(1),Owp(2));
            }
        }

        glEnd();
    }

    vector<Map*> vpMaps = mpAtlas->GetAllMaps();

    if(bDrawKF)
    {
        for(Map* pMap : vpMaps)
        {
            if(pMap == pActiveMap)
                continue;

            vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();

            for(size_t i=0; i<vpKFs.size(); i++)
            {
                KeyFrame* pKF = vpKFs[i];
                Eigen::Matrix4f Twc = pKF->GetPoseInverse().matrix();
                unsigned int index_color = pKF->mnOriginMapId;

                glPushMatrix();

                glMultMatrixf((GLfloat*)Twc.data());

                if(!vpKFs[i]->GetParent()) // It is the first KF in the map
                {
                    glLineWidth(mKeyFrameLineWidth*5);
                    glColor3f(1.0f,0.0f,0.0f);
                    glBegin(GL_LINES);
                }
                else
                {
                    glLineWidth(mKeyFrameLineWidth);
                    glColor3f(mfFrameColors[index_color][0],mfFrameColors[index_color][1],mfFrameColors[index_color][2]);
                    glBegin(GL_LINES);
                }

                glVertex3f(0,0,0);
                glVertex3f(w,h,z);
                glVertex3f(0,0,0);
                glVertex3f(w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,h,z);

                glVertex3f(w,h,z);
                glVertex3f(w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(-w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(w,h,z);

                glVertex3f(-w,-h,z);
                glVertex3f(w,-h,z);
                glEnd();

                glPopMatrix();
            }
        }
    }
}

void MapDrawer::DrawCurrentCamera(const Eigen::Matrix4f &Twc)
{
    const float &w = mCameraSize;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();
    glMultMatrixf(Twc.data());

    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f,1.0f,0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}


void MapDrawer::SetCurrentCameraPose(const Sophus::SE3f &Tcw)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.inverse();
}

Eigen::Matrix4f MapDrawer::GetCurrentCameraPose()
{
    unique_lock<mutex> lock(mMutexCamera);
    return mCameraPose.matrix();
}
} //namespace ORB_SLAM
