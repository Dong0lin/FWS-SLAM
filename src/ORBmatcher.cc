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


#include "ORBmatcher.h"

#include <map>

#include<limits.h>

#include<opencv2/core/core.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<stdint-gcc.h>

using namespace std;

namespace ORB_SLAM3
{

    const int ORBmatcher::TH_HIGH = 100;
    const int ORBmatcher::TH_LOW = 50;
    const int ORBmatcher::HISTO_LENGTH = 30;

    ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
    {
    }

    /**
 * 通过投影匹配在当前帧中搜索与地图点对应的特征点
 * 将地图点投影到当前帧图像平面，然后在投影点周围搜索匹配的特征点
 * 
 * @param F 当前帧
 * @param vpMapPoints 待匹配的地图点集合
 * @param th 搜索半径阈值因子（通常为1.0，大于1.0时扩大搜索范围）
 * @param bFarPoints 是否忽略远点
 * @param thFarPoints 远点距离阈值
 * @return int 成功匹配的数量
 */
int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th, const bool bFarPoints, const float thFarPoints)
    {
        int nmatches=0, left = 0, right = 0;  // 匹配计数器和左右视图匹配数

        const bool bFactor = th!=1.0;  // 检查是否需要调整搜索半径

        // 遍历所有待匹配的地图点
        for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
        {
            MapPoint* pMP = vpMapPoints[iMP];
            
            // 跳过不在当前视图中的地图点（左视图和右视图都不可见）
            if(!pMP->mbTrackInView && !pMP->mbTrackInViewR)
                continue;

            // 如果启用远点过滤且地图点深度超过阈值，跳过该点
            if(bFarPoints && pMP->mTrackDepth>thFarPoints)
                continue;

            // 跳过已标记为坏点的地图点
            if(pMP->isBad())
                continue;

            // 处理左视图可见的地图点
            if(pMP->mbTrackInView)
            {
                const int &nPredictedLevel = pMP->mnTrackScaleLevel;  // 预测的金字塔层级

                // 根据观察方向计算搜索窗口半径：观察角度越小（正对相机），窗口越小
                float r = RadiusByViewingCos(pMP->mTrackViewCos);

                // 如果设置了阈值因子，调整搜索半径
                if(bFactor)
                    r*=th;

                // 在投影点周围获取候选特征点索引
                const vector<size_t> vIndices =
                        F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

                // 如果存在候选特征点，进行特征匹配
                if(!vIndices.empty()){
                    const cv::Mat MPdescriptor = pMP->GetDescriptor();  // 获取地图点的描述子

                    // 初始化最佳匹配和次佳匹配的变量
                    int bestDist=256;      // 最佳匹配距离（256为最大可能距离）
                    int bestLevel= -1;     // 最佳匹配特征点的金字塔层级
                    int bestDist2=256;     // 次佳匹配距离
                    int bestLevel2 = -1;   // 次佳匹配特征点的金字塔层级
                    int bestIdx =-1 ;      // 最佳匹配特征点的索引

                    // 在候选特征点中寻找最佳和次佳匹配
                    for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                    {
                        const size_t idx = *vit;  // 候选特征点索引

                        // 跳过已有关联地图点且被多次观测的特征点
                        if(F.mvpMapPoints[idx])
                            if(F.mvpMapPoints[idx]->Observations()>0)
                                continue;

                        // 双目相机：检查右视图的视差约束
                        if(F.Nleft == -1 && F.mvuRight[idx]>0)
                        {
                            const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
                            if(er>r*F.mvScaleFactors[nPredictedLevel])
                                continue;
                        }

                        // ── 语义一致性约束 ──
                        // 已分类的语义地图点只能匹配同类语义特征点
                        {
                            int kpSemClass = F.GetSemanticClassAt(idx);
                            if(pMP->mnSemanticClass >= 0 && kpSemClass != pMP->mnSemanticClass)
                                continue;
                        }

                        const cv::Mat &d = F.mDescriptors.row(idx);  // 获取特征点描述子

                        const int dist = DescriptorDistance(MPdescriptor,d);  // 计算描述子距离

                        // 更新最佳匹配
                        if(dist<bestDist)
                        {
                            bestDist2=bestDist;      // 原最佳变为次佳
                            bestDist=dist;           // 更新最佳距离
                            bestLevel2 = bestLevel;  // 原最佳层级变为次佳
                            
                            // 根据相机类型获取特征点层级
                            bestLevel = (F.Nleft == -1) ? F.mvKeysUn[idx].octave
                                                        : (idx < F.Nleft) ? F.mvKeys[idx].octave
                                                                          : F.mvKeysRight[idx - F.Nleft].octave;
                            bestIdx=idx;  // 更新最佳匹配索引
                        }
                        else if(dist<bestDist2)
                        {
                            // 更新次佳匹配
                            bestLevel2 = (F.Nleft == -1) ? F.mvKeysUn[idx].octave
                                                         : (idx < F.Nleft) ? F.mvKeys[idx].octave
                                                                           : F.mvKeysRight[idx - F.Nleft].octave;
                            bestDist2=dist;  // 更新次佳距离
                        }
                    }

                                        // 应用比率测试到次佳匹配（仅当最佳和次佳在同一金字塔层级时）
                    if(bestDist<=TH_HIGH)  // 最佳匹配距离小于高阈值（通常为100）
                    {
                        // 比率测试：如果最佳和次佳在同一层级，且最佳距离大于比率×次佳距离，则拒绝
                        if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                            continue;  // 匹配不可靠，跳过

                        // 如果满足以下条件之一，则接受匹配：
                        // 1. 最佳和次佳在不同层级（无法进行比率测试）
                        // 2. 最佳距离小于等于比率×次佳距离（通过比率测试）
                        if(bestLevel!=bestLevel2 || bestDist<=mfNNratio*bestDist2){
                            // 建立匹配关系：将地图点关联到当前帧的特征点
                            F.mvpMapPoints[bestIdx]=pMP;

                            // 双目相机处理：如果存在左到右的匹配关系，同时关联右视图
                            if(F.Nleft != -1 && F.mvLeftToRightMatch[bestIdx] != -1){ 
                                // 同时匹配右相机的立体观测
                                F.mvpMapPoints[F.mvLeftToRightMatch[bestIdx] + F.Nleft] = pMP;
                                nmatches++;  // 总匹配数增加
                                right++;     // 右视图匹配数增加
                            }

                            nmatches++;  // 总匹配数增加
                            left++;      // 左视图匹配数增加
                        }
                    }
                }
            }

            // 处理右视图可见的地图点（双目相机）
            if(F.Nleft != -1 && pMP->mbTrackInViewR){
                const int &nPredictedLevel = pMP->mnTrackScaleLevelR;  // 右视图预测的金字塔层级
                if(nPredictedLevel != -1){  // 确保有有效的预测层级
                    // 根据右视图的观察方向计算搜索半径
                    float r = RadiusByViewingCos(pMP->mTrackViewCosR);

                    // 在右视图投影点周围获取候选特征点索引
                    const vector<size_t> vIndices =
                            F.GetFeaturesInArea(pMP->mTrackProjXR,pMP->mTrackProjYR,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel,true);

                    if(vIndices.empty())
                        continue;  // 没有候选特征点，跳过

                    const cv::Mat MPdescriptor = pMP->GetDescriptor();  // 获取地图点描述子

                    // 初始化匹配变量
                    int bestDist=256;
                    int bestLevel= -1;
                    int bestDist2=256;
                    int bestLevel2 = -1;
                    int bestIdx =-1 ;

                    // 在候选特征点中寻找最佳和次佳匹配
                    for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
                    {
                        const size_t idx = *vit;  // 候选特征点索引

                        // 跳过已有关联地图点且被多次观测的特征点
                        if(F.mvpMapPoints[idx + F.Nleft])
                            if(F.mvpMapPoints[idx + F.Nleft]->Observations()>0)
                                continue;

                        // ── 语义一致性约束（右视图）──
                        {
                            int kpSemClass = F.GetSemanticClassAt(idx + F.Nleft);
                            if(pMP->mnSemanticClass >= 0 && kpSemClass != pMP->mnSemanticClass)
                                continue;
                        }

                        const cv::Mat &d = F.mDescriptors.row(idx + F.Nleft);  // 获取右视图特征点描述子

                        const int dist = DescriptorDistance(MPdescriptor,d);  // 计算描述子距离

                        // 更新最佳匹配
                        if(dist<bestDist)
                        {
                            bestDist2=bestDist;
                            bestDist=dist;
                            bestLevel2 = bestLevel;
                            bestLevel = F.mvKeysRight[idx].octave;  // 右视图特征点层级
                            bestIdx=idx;
                        }
                        else if(dist<bestDist2)
                        {
                            bestLevel2 = F.mvKeysRight[idx].octave;  // 更新次佳层级
                            bestDist2=dist;  // 更新次佳距离
                        }
                    }

                    // 应用比率测试到次佳匹配
                    if(bestDist<=TH_HIGH)  // 最佳匹配距离小于高阈值
                    {
                        // 比率测试：如果最佳和次佳在同一层级，且最佳距离大于比率×次佳距离，则拒绝
                        if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                            continue;  // 匹配不可靠，跳过

                        // 双目相机处理：如果存在右到左的匹配关系，同时关联左视图
                        if(F.Nleft != -1 && F.mvRightToLeftMatch[bestIdx] != -1){ 
                            // 同时匹配左相机的立体观测
                            F.mvpMapPoints[F.mvRightToLeftMatch[bestIdx]] = pMP;
                            nmatches++;  // 总匹配数增加
                            left++;      // 左视图匹配数增加
                        }

                        // 建立右视图匹配关系
                        F.mvpMapPoints[bestIdx + F.Nleft]=pMP;
                        nmatches++;  // 总匹配数增加
                        right++;     // 右视图匹配数增加
                    }
                }
            }
        }
        return nmatches;  // 返回成功匹配的总数量
    }

    float ORBmatcher::RadiusByViewingCos(const float &viewCos)
    {
        if(viewCos>0.998)
            return 2.5;
        else
            return 4.0;
    }

    /**
 * 基于词袋模型的特征匹配方法
 * 通过比较关键帧和当前帧在词袋空间中的相似性，快速找到对应的特征点匹配
 * 
 * @param pKF 参考关键帧
 * @param F 当前帧
 * @param vpMapPointMatches 输出参数，存储匹配的地图点
 * @return int 成功匹配的数量
 */
int ORBmatcher::SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)
    {
        // 获取关键帧的地图点集合
        const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();

        // 初始化匹配结果向量，大小为当前帧特征点数量，初始值为NULL
        vpMapPointMatches = vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));

        // 获取关键帧和当前帧的词袋特征向量
        const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

        int nmatches=0;  // 匹配计数器

        // 创建方向直方图用于后续的旋转一致性检查
        vector<int> rotHist[HISTO_LENGTH];  // 30个bin的直方图
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);  // 预分配内存
        const float factor = 1.0f/HISTO_LENGTH;  // 角度到bin的转换因子

        // 在属于同一词汇节点（特定层级）的ORB特征之间执行匹配
        DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();  // 关键帧特征向量迭代器
        DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();   // 当前帧特征向量迭代器
        DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();   // 关键帧结束迭代器
        DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();    // 当前帧结束迭代器

        // 同时遍历关键帧和当前帧的特征向量
        while(KFit != KFend && Fit != Fend)
        {
            // 如果两个特征向量在同一词汇节点
            if(KFit->first == Fit->first)
            {
                // 获取该词汇节点下的特征点索引
                const vector<unsigned int> vIndicesKF = KFit->second;  // 关键帧特征点索引
                const vector<unsigned int> vIndicesF = Fit->second;    // 当前帧特征点索引

                // 遍历关键帧在该节点下的所有特征点
                for(size_t iKF=0; iKF<vIndicesKF.size(); iKF++)
                {
                    const unsigned int realIdxKF = vIndicesKF[iKF];  // 关键帧特征点真实索引

                    // 获取该特征点对应的地图点
                    MapPoint* pMP = vpMapPointsKF[realIdxKF];

                    if(!pMP)
                        continue;  // 跳过没有关联地图点的特征点

                    if(pMP->isBad())
                        continue;  // 跳过已标记为坏点的地图点

                    // 获取关键帧特征点的描述子
                    const cv::Mat &dKF= pKF->mDescriptors.row(realIdxKF);

                    // 初始化匹配变量（左视图和右视图分别处理）
                    int bestDist1=256;      // 左视图最佳匹配距离
                    int bestIdxF =-1 ;      // 左视图最佳匹配索引
                    int bestDist2=256;      // 左视图次佳匹配距离

                    int bestDist1R=256;     // 右视图最佳匹配距离
                    int bestIdxFR =-1 ;     // 右视图最佳匹配索引
                    int bestDist2R=256;     // 右视图次佳匹配距离

                    // 在当前帧的同一词汇节点下寻找最佳匹配
                    for(size_t iF=0; iF<vIndicesF.size(); iF++)
                    {
                        // 单目相机处理
                        if(F.Nleft == -1){
                            const unsigned int realIdxF = vIndicesF[iF];

                            // 跳过已匹配的特征点
                            if(vpMapPointMatches[realIdxF])
                                continue;

                            // ── 语义一致性约束 ──
                            {
                                int kpSemClass = F.GetSemanticClassAt(realIdxF);
                                if(pMP->mnSemanticClass >= 0 && kpSemClass != pMP->mnSemanticClass)
                                    continue;
                            }

                            // 获取当前帧特征点描述子
                            const cv::Mat &dF = F.mDescriptors.row(realIdxF);

                            // 计算描述子距离
                            const int dist =  DescriptorDistance(dKF,dF);

                            // 更新最佳和次佳匹配
                            if(dist<bestDist1)
                            {
                                bestDist2=bestDist1;
                                bestDist1=dist;
                                bestIdxF=realIdxF;
                            }
                            else if(dist<bestDist2)
                            {
                                bestDist2=dist;
                            }
                        }
                        // 双目相机处理（分别处理左视图和右视图）
                        else{
                            const unsigned int realIdxF = vIndicesF[iF];

                            // 跳过已匹配的特征点
                            if(vpMapPointMatches[realIdxF])
                                continue;

                            // ── 语义一致性约束 ──
                            {
                                int kpSemClass = F.GetSemanticClassAt(realIdxF);
                                if(pMP->mnSemanticClass >= 0 && kpSemClass != pMP->mnSemanticClass)
                                    continue;
                            }

                            const cv::Mat &dF = F.mDescriptors.row(realIdxF);

                            const int dist =  DescriptorDistance(dKF,dF);

                            // 左视图特征点匹配
                            if(realIdxF < F.Nleft && dist<bestDist1){
                                bestDist2=bestDist1;
                                bestDist1=dist;
                                bestIdxF=realIdxF;
                            }
                            else if(realIdxF < F.Nleft && dist<bestDist2){
                                bestDist2=dist;
                            }

                            // 右视图特征点匹配
                            if(realIdxF >= F.Nleft && dist<bestDist1R){
                                bestDist2R=bestDist1R;
                                bestDist1R=dist;
                                bestIdxFR=realIdxF;
                            }
                            else if(realIdxF >= F.Nleft && dist<bestDist2R){
                                bestDist2R=dist;
                            }
                        }

                    }

                    // 检查左视图匹配是否满足条件
                    if(bestDist1<=TH_LOW)  // 最佳匹配距离小于低阈值（通常为50）
                    {
                        // 比率测试：最佳距离 < 比率 × 次佳距离
                        if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                        {
                            // 建立匹配关系
                            vpMapPointMatches[bestIdxF]=pMP;

                            // 获取关键帧特征点的关键点信息（用于方向检查）
                            const cv::KeyPoint &kp =
                                    (!pKF->mpCamera2) ? pKF->mvKeysUn[realIdxKF] :
                                    (realIdxKF >= pKF -> NLeft) ? pKF -> mvKeysRight[realIdxKF - pKF -> NLeft]
                                                                : pKF -> mvKeys[realIdxKF];

                            // 如果启用方向检查，记录特征点方向差异
                            if(mbCheckOrientation)
                            {
                                // 获取当前帧匹配特征点的关键点信息
                                cv::KeyPoint &Fkp =
                                        (!pKF->mpCamera2 || F.Nleft == -1) ? F.mvKeys[bestIdxF] :
                                        (bestIdxF >= F.Nleft) ? F.mvKeysRight[bestIdxF - F.Nleft]
                                                              : F.mvKeys[bestIdxF];

                                // 计算方向差异（0-360度）
                                float rot = kp.angle-Fkp.angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);  // 转换为bin索引
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(bestIdxF);  // 记录到直方图
                            }
                            nmatches++;  // 增加匹配计数
                        }

                        // 检查右视图匹配是否满足条件
                        if(bestDist1R<=TH_LOW)
                        {
                            // 右视图的比率测试条件更宽松（|| true表示总是接受）
                            if(static_cast<float>(bestDist1R)<mfNNratio*static_cast<float>(bestDist2R) || true)
                            {
                                vpMapPointMatches[bestIdxFR]=pMP;

                                const cv::KeyPoint &kp =
                                        (!pKF->mpCamera2) ? pKF->mvKeysUn[realIdxKF] :
                                        (realIdxKF >= pKF -> NLeft) ? pKF -> mvKeysRight[realIdxKF - pKF -> NLeft]
                                                                    : pKF -> mvKeys[realIdxKF];

                                if(mbCheckOrientation)
                                {
                                    cv::KeyPoint &Fkp =
                                            (!F.mpCamera2) ? F.mvKeys[bestIdxFR] :
                                            (bestIdxFR >= F.Nleft) ? F.mvKeysRight[bestIdxFR - F.Nleft]
                                                                   : F.mvKeys[bestIdxFR];

                                    float rot = kp.angle-Fkp.angle;
                                    if(rot<0.0)
                                        rot+=360.0f;
                                    int bin = round(rot*factor);
                                    if(bin==HISTO_LENGTH)
                                        bin=0;
                                    assert(bin>=0 && bin<HISTO_LENGTH);
                                    rotHist[bin].push_back(bestIdxFR);
                                }
                                nmatches++;
                            }
                        }
                    }

                }

                // 移动到下一个词汇节点
                KFit++;
                Fit++;
            }
            // 如果关键帧的词汇节点ID小于当前帧，跳转到当前帧的节点
            else if(KFit->first < Fit->first)
            {
                KFit = vFeatVecKF.lower_bound(Fit->first);
            }
            // 如果当前帧的词汇节点ID小于关键帧，跳转到关键帧的节点
            else
            {
                Fit = F.mFeatVec.lower_bound(KFit->first);
            }
        }

        // 方向一致性检查：移除方向不一致的匹配
        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            // 找到直方图中三个最大的bin
            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            // 移除不在前三个最大bin中的匹配
            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i==ind1 || i==ind2 || i==ind3)
                    continue;  // 保留前三个最大bin的匹配
                
                // 移除其他bin中的所有匹配
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    vpMapPointMatches[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                    nmatches--;  // 减少匹配计数
                }
            }
        }

        return nmatches;  // 返回最终的匹配数量
    }

    /**
 * 通过投影匹配在关键帧中搜索与地图点对应的特征点
 * 将地图点投影到关键帧图像平面，然后在投影点周围搜索匹配的特征点
 * 主要用于闭环检测和重定位中的地图点匹配
 * 
 * @param pKF 目标关键帧
 * @param Scw 相似变换矩阵（包含尺度信息）
 * @param vpPoints 待匹配的地图点集合
 * @param vpMatched 输出参数，存储匹配的地图点
 * @param th 搜索半径阈值因子
 * @param ratioHamming 汉明距离比率因子
 * @return int 成功匹配的数量
 */
int ORBmatcher::SearchByProjection(KeyFrame* pKF, Sophus::Sim3f &Scw, const vector<MapPoint*> &vpPoints,
                                       vector<MapPoint*> &vpMatched, int th, float ratioHamming)
    {
        // 获取相机内参用于后续的投影计算
        const float &fx = pKF->fx;  // 焦距x
        const float &fy = pKF->fy;  // 焦距y
        const float &cx = pKF->cx;  // 主点x
        const float &cy = pKF->cy;  // 主点y

        // 将相似变换转换为欧几里得变换（去除尺度因子）
        Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
        // 计算相机在世界坐标系中的位置
        Eigen::Vector3f Ow = Tcw.inverse().translation();

        // 创建已匹配地图点的集合（用于去重）
        set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
        spAlreadyFound.erase(static_cast<MapPoint*>(NULL));  // 移除NULL指针

        int nmatches=0;  // 匹配计数器

        // 遍历所有候选地图点，进行投影和匹配
        for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
        {
            MapPoint* pMP = vpPoints[iMP];  // 当前处理的地图点

            // 丢弃坏点和已匹配的地图点
            if(pMP->isBad() || spAlreadyFound.count(pMP))
                continue;

            // 获取地图点的3D世界坐标
            Eigen::Vector3f p3Dw = pMP->GetWorldPos();

            // 将世界坐标转换为相机坐标
            Eigen::Vector3f p3Dc = Tcw * p3Dw;

            // 深度必须为正（点在相机前方）
            if(p3Dc(2)<0.0)
                continue;

            // 投影到图像平面
            const Eigen::Vector2f uv = pKF->mpCamera->project(p3Dc);

            // 点必须在图像范围内
            if(!pKF->IsInImage(uv(0),uv(1)))
                continue;

            // 深度必须在尺度不变性区域内
            const float maxDistance = pMP->GetMaxDistanceInvariance();  // 最大距离
            const float minDistance = pMP->GetMinDistanceInvariance();  // 最小距离
            Eigen::Vector3f PO = p3Dw-Ow;  // 从相机到地图点的向量
            const float dist = PO.norm();   // 计算距离

            // 检查距离是否在有效范围内
            if(dist<minDistance || dist>maxDistance)
                continue;

            // 视角必须小于60度（cos(60°)=0.5）
            Eigen::Vector3f Pn = pMP->GetNormal();  // 地图点法向量

            // PO·Pn < 0.5*dist 等价于 cos(θ) < 0.5，即 θ > 60°
            if(PO.dot(Pn)<0.5*dist)
                continue;

            // 预测特征点应该在的金字塔层级
            int nPredictedLevel = pMP->PredictScale(dist,pKF);

            // 在投影点周围半径内搜索特征点
            const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

            // 获取半径内的候选特征点索引
            const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0),uv(1),radius);

            if(vIndices.empty())
                continue;  // 没有候选特征点，跳过

            // 在半径内寻找最相似的关键点
            const cv::Mat dMP = pMP->GetDescriptor();  // 地图点描述子

            int bestDist = 256;  // 最佳匹配距离（初始化为最大值）
            int bestIdx = -1;    // 最佳匹配索引
            
            // 遍历所有候选特征点
            for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
            {
                const size_t idx = *vit;  // 特征点索引
                
                // 跳过已匹配的特征点
                if(vpMatched[idx])
                    continue;

                // 获取特征点的金字塔层级
                const int &kpLevel= pKF->mvKeysUn[idx].octave;

                // 层级约束：特征点层级必须在预测层级附近（±1层）
                if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                    continue;

                // 获取关键帧特征点描述子
                const cv::Mat &dKF = pKF->mDescriptors.row(idx);

                // 计算描述子距离
                const int dist = DescriptorDistance(dMP,dKF);

                // 更新最佳匹配
                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            // 如果最佳匹配距离满足阈值条件，建立匹配关系
            if(bestDist<=TH_LOW*ratioHamming)  // TH_LOW通常为50，ratioHamming调整阈值
            {
                vpMatched[bestIdx]=pMP;  // 建立匹配关系
                nmatches++;              // 增加匹配计数
            }

        }

        return nmatches;  // 返回成功匹配的数量
    }

    /**
 * 通过投影匹配在关键帧中搜索与地图点对应的特征点（带关键帧信息版本）
 * 此版本额外记录每个匹配点对应的源关键帧，用于闭环检测中的一致性检查
 * 
 * @param pKF 目标关键帧
 * @param Scw 相似变换矩阵（包含尺度信息）
 * @param vpPoints 待匹配的地图点集合
 * @param vpPointsKFs 地图点对应的源关键帧集合
 * @param vpMatched 输出参数，存储匹配的地图点
 * @param vpMatchedKF 输出参数，存储匹配点对应的源关键帧
 * @param th 搜索半径阈值因子
 * @param ratioHamming 汉明距离比率因子
 * @return int 成功匹配的数量
 */
int ORBmatcher::SearchByProjection(KeyFrame* pKF, Sophus::Sim3<float> &Scw, const std::vector<MapPoint*> &vpPoints, const std::vector<KeyFrame*> &vpPointsKFs,
                                       std::vector<MapPoint*> &vpMatched, std::vector<KeyFrame*> &vpMatchedKF, int th, float ratioHamming)
    {
        // 获取相机内参用于后续的投影计算
        const float &fx = pKF->fx;  // 焦距x
        const float &fy = pKF->fy;  // 焦距y
        const float &cx = pKF->cx;  // 主点x
        const float &cy = pKF->cy;  // 主点y

        // 将相似变换转换为欧几里得变换（去除尺度因子）
        Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
        // 计算相机在世界坐标系中的位置
        Eigen::Vector3f Ow = Tcw.inverse().translation();

        // 创建已匹配地图点的集合（用于去重）
        set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
        spAlreadyFound.erase(static_cast<MapPoint*>(NULL));  // 移除NULL指针

        int nmatches=0;  // 匹配计数器

        // 遍历所有候选地图点，进行投影和匹配
        for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
        {
            MapPoint* pMP = vpPoints[iMP];   // 当前处理的地图点
            KeyFrame* pKFi = vpPointsKFs[iMP];  // 地图点对应的源关键帧

            // 丢弃坏点和已匹配的地图点
            if(pMP->isBad() || spAlreadyFound.count(pMP))
                continue;

            // 获取地图点的3D世界坐标
            Eigen::Vector3f p3Dw = pMP->GetWorldPos();

            // 将世界坐标转换为相机坐标
            Eigen::Vector3f p3Dc = Tcw * p3Dw;

            // 深度必须为正（点在相机前方）
            if(p3Dc(2)<0.0)
                continue;

            // 投影到图像平面（手动计算投影，不使用相机模型）
            const float invz = 1/p3Dc(2);  // 深度倒数
            const float x = p3Dc(0)*invz;  // 归一化x坐标
            const float y = p3Dc(1)*invz;  // 归一化y坐标

            // 计算像素坐标
            const float u = fx*x+cx;
            const float v = fy*y+cy;

            // 点必须在图像范围内
            if(!pKF->IsInImage(u,v))
                continue;

            // 深度必须在尺度不变性区域内
            const float maxDistance = pMP->GetMaxDistanceInvariance();  // 最大距离
            const float minDistance = pMP->GetMinDistanceInvariance();  // 最小距离
            Eigen::Vector3f PO = p3Dw-Ow;  // 从相机到地图点的向量
            const float dist = PO.norm();   // 计算距离

            // 检查距离是否在有效范围内
            if(dist<minDistance || dist>maxDistance)
                continue;

            // 视角必须小于60度（cos(60°)=0.5）
            Eigen::Vector3f Pn = pMP->GetNormal();  // 地图点法向量

            // PO·Pn < 0.5*dist 等价于 cos(θ) < 0.5，即 θ > 60°
            if(PO.dot(Pn)<0.5*dist)
                continue;

            // 预测特征点应该在的金字塔层级
            int nPredictedLevel = pMP->PredictScale(dist,pKF);

            // 在投影点周围半径内搜索特征点
            const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

            // 获取半径内的候选特征点索引
            const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

            if(vIndices.empty())
                continue;  // 没有候选特征点，跳过

            // 在半径内寻找最相似的关键点
            const cv::Mat dMP = pMP->GetDescriptor();  // 地图点描述子

            int bestDist = 256;  // 最佳匹配距离（初始化为最大值）
            int bestIdx = -1;    // 最佳匹配索引
            
            // 遍历所有候选特征点
            for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
            {
                const size_t idx = *vit;  // 特征点索引
                
                // 跳过已匹配的特征点
                if(vpMatched[idx])
                    continue;

                // 获取特征点的金字塔层级
                const int &kpLevel= pKF->mvKeysUn[idx].octave;

                // 层级约束：特征点层级必须在预测层级附近（±1层）
                if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                    continue;

                // 获取关键帧特征点描述子
                const cv::Mat &dKF = pKF->mDescriptors.row(idx);

                // 计算描述子距离
                const int dist = DescriptorDistance(dMP,dKF);

                // 更新最佳匹配
                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            // 如果最佳匹配距离满足阈值条件，建立匹配关系
            if(bestDist<=TH_LOW*ratioHamming)  // TH_LOW通常为50，ratioHamming调整阈值
            {
                vpMatched[bestIdx] = pMP;    // 记录匹配的地图点
                vpMatchedKF[bestIdx] = pKFi; // 记录匹配点对应的源关键帧
                nmatches++;                   // 增加匹配计数
            }

        }

        return nmatches;  // 返回成功匹配的数量
    }

    int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
    {
        int nmatches=0;
        vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);

        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f/HISTO_LENGTH;

        vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);
        vector<int> vnMatches21(F2.mvKeysUn.size(),-1);

        for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++)
        {
            cv::KeyPoint kp1 = F1.mvKeysUn[i1];
            int level1 = kp1.octave;
            if(level1>0)
                continue;

            vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);

            if(vIndices2.empty())
                continue;

            cv::Mat d1 = F1.mDescriptors.row(i1);

            int bestDist = INT_MAX;
            int bestDist2 = INT_MAX;
            int bestIdx2 = -1;

            for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
            {
                size_t i2 = *vit;

                cv::Mat d2 = F2.mDescriptors.row(i2);

                int dist = DescriptorDistance(d1,d2);

                if(vMatchedDistance[i2]<=dist)
                    continue;

                if(dist<bestDist)
                {
                    bestDist2=bestDist;
                    bestDist=dist;
                    bestIdx2=i2;
                }
                else if(dist<bestDist2)
                {
                    bestDist2=dist;
                }
            }

            if(bestDist<=TH_LOW)
            {
                if(bestDist<(float)bestDist2*mfNNratio)
                {
                    if(vnMatches21[bestIdx2]>=0)
                    {
                        vnMatches12[vnMatches21[bestIdx2]]=-1;
                        nmatches--;
                    }
                    vnMatches12[i1]=bestIdx2;
                    vnMatches21[bestIdx2]=i1;
                    vMatchedDistance[bestIdx2]=bestDist;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(i1);
                    }
                }
            }

        }

        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i==ind1 || i==ind2 || i==ind3)
                    continue;
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    int idx1 = rotHist[i][j];
                    if(vnMatches12[idx1]>=0)
                    {
                        vnMatches12[idx1]=-1;
                        nmatches--;
                    }
                }
            }

        }

        //Update prev matched
        for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
            if(vnMatches12[i1]>=0)
                vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;

        return nmatches;
    }

    int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
    {
        const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
        const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
        const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
        const cv::Mat &Descriptors1 = pKF1->mDescriptors;

        const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
        const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
        const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
        const cv::Mat &Descriptors2 = pKF2->mDescriptors;

        vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
        vector<bool> vbMatched2(vpMapPoints2.size(),false);

        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);

        const float factor = 1.0f/HISTO_LENGTH;

        int nmatches = 0;

        DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

        while(f1it != f1end && f2it != f2end)
        {
            if(f1it->first == f2it->first)
            {
                for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
                {
                    const size_t idx1 = f1it->second[i1];
                    if(pKF1 -> NLeft != -1 && idx1 >= pKF1 -> mvKeysUn.size()){
                        continue;
                    }

                    MapPoint* pMP1 = vpMapPoints1[idx1];
                    if(!pMP1)
                        continue;
                    if(pMP1->isBad())
                        continue;

                    const cv::Mat &d1 = Descriptors1.row(idx1);

                    int bestDist1=256;
                    int bestIdx2 =-1 ;
                    int bestDist2=256;

                    for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                    {
                        const size_t idx2 = f2it->second[i2];

                        if(pKF2 -> NLeft != -1 && idx2 >= pKF2 -> mvKeysUn.size()){
                            continue;
                        }

                        MapPoint* pMP2 = vpMapPoints2[idx2];

                        if(vbMatched2[idx2] || !pMP2)
                            continue;

                        if(pMP2->isBad())
                            continue;

                        const cv::Mat &d2 = Descriptors2.row(idx2);

                        int dist = DescriptorDistance(d1,d2);

                        if(dist<bestDist1)
                        {
                            bestDist2=bestDist1;
                            bestDist1=dist;
                            bestIdx2=idx2;
                        }
                        else if(dist<bestDist2)
                        {
                            bestDist2=dist;
                        }
                    }

                    if(bestDist1<TH_LOW)
                    {
                        if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                        {
                            vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                            vbMatched2[bestIdx2]=true;

                            if(mbCheckOrientation)
                            {
                                float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(idx1);
                            }
                            nmatches++;
                        }
                    }
                }

                f1it++;
                f2it++;
            }
            else if(f1it->first < f2it->first)
            {
                f1it = vFeatVec1.lower_bound(f2it->first);
            }
            else
            {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }

        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i==ind1 || i==ind2 || i==ind3)
                    continue;
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    vpMatches12[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                    nmatches--;
                }
            }
        }

        return nmatches;
    }

    int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2,
                                           vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo, const bool bCoarse)
    {
        const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
        const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

        //Compute epipole in second image
        Sophus::SE3f T1w = pKF1->GetPose();
        Sophus::SE3f T2w = pKF2->GetPose();
        Sophus::SE3f Tw2 = pKF2->GetPoseInverse(); // for convenience
        Eigen::Vector3f Cw = pKF1->GetCameraCenter();
        Eigen::Vector3f C2 = T2w * Cw;

        Eigen::Vector2f ep = pKF2->mpCamera->project(C2);
        Sophus::SE3f T12;
        Sophus::SE3f Tll, Tlr, Trl, Trr;
        Eigen::Matrix3f R12; // for fastest computation
        Eigen::Vector3f t12; // for fastest computation

        GeometricCamera* pCamera1 = pKF1->mpCamera, *pCamera2 = pKF2->mpCamera;

        if(!pKF1->mpCamera2 && !pKF2->mpCamera2){
            T12 = T1w * Tw2;
            R12 = T12.rotationMatrix();
            t12 = T12.translation();
        }
        else{
            Sophus::SE3f Tr1w = pKF1->GetRightPose();
            Sophus::SE3f Twr2 = pKF2->GetRightPoseInverse();
            Tll = T1w * Tw2;
            Tlr = T1w * Twr2;
            Trl = Tr1w * Tw2;
            Trr = Tr1w * Twr2;
        }

        Eigen::Matrix3f Rll = Tll.rotationMatrix(), Rlr  = Tlr.rotationMatrix(), Rrl  = Trl.rotationMatrix(), Rrr  = Trr.rotationMatrix();
        Eigen::Vector3f tll = Tll.translation(), tlr = Tlr.translation(), trl = Trl.translation(), trr = Trr.translation();

        // Find matches between not tracked keypoints
        // Matching speed-up by ORB Vocabulary
        // Compare only ORB that share the same node
        int nmatches=0;
        vector<bool> vbMatched2(pKF2->N,false);
        vector<int> vMatches12(pKF1->N,-1);

        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);

        const float factor = 1.0f/HISTO_LENGTH;

        DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
        DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
        DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
        DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

        while(f1it!=f1end && f2it!=f2end)
        {
            if(f1it->first == f2it->first)
            {
                for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
                {
                    const size_t idx1 = f1it->second[i1];

                    MapPoint* pMP1 = pKF1->GetMapPoint(idx1);

                    // If there is already a MapPoint skip
                    if(pMP1)
                    {
                        continue;
                    }

                    const bool bStereo1 = (!pKF1->mpCamera2 && pKF1->mvuRight[idx1]>=0);

                    if(bOnlyStereo)
                        if(!bStereo1)
                            continue;

                    const cv::KeyPoint &kp1 = (pKF1 -> NLeft == -1) ? pKF1->mvKeysUn[idx1]
                                                                    : (idx1 < pKF1 -> NLeft) ? pKF1 -> mvKeys[idx1]
                                                                                             : pKF1 -> mvKeysRight[idx1 - pKF1 -> NLeft];

                    const bool bRight1 = (pKF1 -> NLeft == -1 || idx1 < pKF1 -> NLeft) ? false
                                                                                       : true;

                    const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);

                    int bestDist = TH_LOW;
                    int bestIdx2 = -1;

                    for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                    {
                        size_t idx2 = f2it->second[i2];

                        MapPoint* pMP2 = pKF2->GetMapPoint(idx2);

                        // If we have already matched or there is a MapPoint skip
                        if(vbMatched2[idx2] || pMP2)
                            continue;

                        const bool bStereo2 = (!pKF2->mpCamera2 &&  pKF2->mvuRight[idx2]>=0);

                        if(bOnlyStereo)
                            if(!bStereo2)
                                continue;

                        const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);

                        const int dist = DescriptorDistance(d1,d2);

                        if(dist>TH_LOW || dist>bestDist)
                            continue;

                        const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[idx2]
                                                                        : (idx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[idx2]
                                                                                                 : pKF2 -> mvKeysRight[idx2 - pKF2 -> NLeft];
                        const bool bRight2 = (pKF2 -> NLeft == -1 || idx2 < pKF2 -> NLeft) ? false
                                                                                           : true;

                        if(!bStereo1 && !bStereo2 && !pKF1->mpCamera2)
                        {
                            const float distex = ep(0)-kp2.pt.x;
                            const float distey = ep(1)-kp2.pt.y;
                            if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave])
                            {
                                continue;
                            }
                        }

                        if(pKF1->mpCamera2 && pKF2->mpCamera2){
                            if(bRight1 && bRight2){
                                R12 = Rrr;
                                t12 = trr;
                                T12 = Trr;

                                pCamera1 = pKF1->mpCamera2;
                                pCamera2 = pKF2->mpCamera2;
                            }
                            else if(bRight1 && !bRight2){
                                R12 = Rrl;
                                t12 = trl;
                                T12 = Trl;

                                pCamera1 = pKF1->mpCamera2;
                                pCamera2 = pKF2->mpCamera;
                            }
                            else if(!bRight1 && bRight2){
                                R12 = Rlr;
                                t12 = tlr;
                                T12 = Tlr;

                                pCamera1 = pKF1->mpCamera;
                                pCamera2 = pKF2->mpCamera2;
                            }
                            else{
                                R12 = Rll;
                                t12 = tll;
                                T12 = Tll;

                                pCamera1 = pKF1->mpCamera;
                                pCamera2 = pKF2->mpCamera;
                            }

                        }

                        if(bCoarse || pCamera1->epipolarConstrain(pCamera2,kp1,kp2,R12,t12,pKF1->mvLevelSigma2[kp1.octave],pKF2->mvLevelSigma2[kp2.octave])) // MODIFICATION_2
                        {
                            bestIdx2 = idx2;
                            bestDist = dist;
                        }
                    }

                    if(bestIdx2>=0)
                    {
                        const cv::KeyPoint &kp2 = (pKF2 -> NLeft == -1) ? pKF2->mvKeysUn[bestIdx2]
                                                                        : (bestIdx2 < pKF2 -> NLeft) ? pKF2 -> mvKeys[bestIdx2]
                                                                                                     : pKF2 -> mvKeysRight[bestIdx2 - pKF2 -> NLeft];
                        vMatches12[idx1]=bestIdx2;
                        nmatches++;

                        if(mbCheckOrientation)
                        {
                            float rot = kp1.angle-kp2.angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                    }
                }

                f1it++;
                f2it++;
            }
            else if(f1it->first < f2it->first)
            {
                f1it = vFeatVec1.lower_bound(f2it->first);
            }
            else
            {
                f2it = vFeatVec2.lower_bound(f1it->first);
            }
        }

        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i==ind1 || i==ind2 || i==ind3)
                    continue;
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    vMatches12[rotHist[i][j]]=-1;
                    nmatches--;
                }
            }

        }

        vMatchedPairs.clear();
        vMatchedPairs.reserve(nmatches);

        for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
        {
            if(vMatches12[i]<0)
                continue;
            vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
        }

        return nmatches;
    }

    int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th, const bool bRight)
    {
        GeometricCamera* pCamera;
        Sophus::SE3f Tcw;
        Eigen::Vector3f Ow;

        if(bRight){
            Tcw = pKF->GetRightPose();
            Ow = pKF->GetRightCameraCenter();
            pCamera = pKF->mpCamera2;
        }
        else{
            Tcw = pKF->GetPose();
            Ow = pKF->GetCameraCenter();
            pCamera = pKF->mpCamera;
        }

        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;
        const float &bf = pKF->mbf;

        int nFused=0;

        const int nMPs = vpMapPoints.size();

        // For debbuging
        int count_notMP = 0, count_bad=0, count_isinKF = 0, count_negdepth = 0, count_notinim = 0, count_dist = 0, count_normal=0, count_notidx = 0, count_thcheck = 0;
        for(int i=0; i<nMPs; i++)
        {
            MapPoint* pMP = vpMapPoints[i];

            if(!pMP)
            {
                count_notMP++;
                continue;
            }

            if(pMP->isBad())
            {
                count_bad++;
                continue;
            }
            else if(pMP->IsInKeyFrame(pKF))
            {
                count_isinKF++;
                continue;
            }

            Eigen::Vector3f p3Dw = pMP->GetWorldPos();
            Eigen::Vector3f p3Dc = Tcw * p3Dw;

            // Depth must be positive
            if(p3Dc(2)<0.0f)
            {
                count_negdepth++;
                continue;
            }

            const float invz = 1/p3Dc(2);

            const Eigen::Vector2f uv = pCamera->project(p3Dc);

            // Point must be inside the image
            if(!pKF->IsInImage(uv(0),uv(1)))
            {
                count_notinim++;
                continue;
            }

            const float ur = uv(0)-bf*invz;

            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            Eigen::Vector3f PO = p3Dw-Ow;
            const float dist3D = PO.norm();

            // Depth must be inside the scale pyramid of the image
            if(dist3D<minDistance || dist3D>maxDistance) {
                count_dist++;
                continue;
            }

            // Viewing angle must be less than 60 deg
            Eigen::Vector3f Pn = pMP->GetNormal();

            if(PO.dot(Pn)<0.5*dist3D)
            {
                count_normal++;
                continue;
            }

            int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

            // Search in a radius
            const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0),uv(1),radius,bRight);

            if(vIndices.empty())
            {
                count_notidx++;
                continue;
            }

            // Match to the most similar keypoint in the radius

            const cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = 256;
            int bestIdx = -1;
            for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
            {
                size_t idx = *vit;
                const cv::KeyPoint &kp = (pKF -> NLeft == -1) ? pKF->mvKeysUn[idx]
                                                              : (!bRight) ? pKF -> mvKeys[idx]
                                                                          : pKF -> mvKeysRight[idx];

                const int &kpLevel= kp.octave;

                if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                    continue;

                if(pKF->mvuRight[idx]>=0)
                {
                    // Check reprojection error in stereo
                    const float &kpx = kp.pt.x;
                    const float &kpy = kp.pt.y;
                    const float &kpr = pKF->mvuRight[idx];
                    const float ex = uv(0)-kpx;
                    const float ey = uv(1)-kpy;
                    const float er = ur-kpr;
                    const float e2 = ex*ex+ey*ey+er*er;

                    if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8)
                        continue;
                }
                else
                {
                    const float &kpx = kp.pt.x;
                    const float &kpy = kp.pt.y;
                    const float ex = uv(0)-kpx;
                    const float ey = uv(1)-kpy;
                    const float e2 = ex*ex+ey*ey;

                    if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
                        continue;
                }

                if(bRight) idx += pKF->NLeft;

                const cv::Mat &dKF = pKF->mDescriptors.row(idx);

                const int dist = DescriptorDistance(dMP,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            // If there is already a MapPoint replace otherwise add new measurement
            if(bestDist<=TH_LOW)
            {
                MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
                if(pMPinKF)
                {
                    if(!pMPinKF->isBad())
                    {
                        if(pMPinKF->Observations()>pMP->Observations())
                            pMP->Replace(pMPinKF);
                        else
                            pMPinKF->Replace(pMP);
                    }
                }
                else
                {
                    pMP->AddObservation(pKF,bestIdx);
                    pKF->AddMapPoint(pMP,bestIdx);
                }
                nFused++;
            }
            else
                count_thcheck++;

        }

        return nFused;
    }

    int ORBmatcher::Fuse(KeyFrame *pKF, Sophus::Sim3f &Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
    {
        // Get Calibration Parameters for later projection
        const float &fx = pKF->fx;
        const float &fy = pKF->fy;
        const float &cx = pKF->cx;
        const float &cy = pKF->cy;

        // Decompose Scw
        Sophus::SE3f Tcw = Sophus::SE3f(Scw.rotationMatrix(),Scw.translation()/Scw.scale());
        Eigen::Vector3f Ow = Tcw.inverse().translation();

        // Set of MapPoints already found in the KeyFrame
        const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

        int nFused=0;

        const int nPoints = vpPoints.size();

        // For each candidate MapPoint project and match
        for(int iMP=0; iMP<nPoints; iMP++)
        {
            MapPoint* pMP = vpPoints[iMP];

            // Discard Bad MapPoints and already found
            if(pMP->isBad() || spAlreadyFound.count(pMP))
                continue;

            // Get 3D Coords.
            Eigen::Vector3f p3Dw = pMP->GetWorldPos();

            // Transform into Camera Coords.
            Eigen::Vector3f p3Dc = Tcw * p3Dw;

            // Depth must be positive
            if(p3Dc(2)<0.0f)
                continue;

            // Project into Image
            const Eigen::Vector2f uv = pKF->mpCamera->project(p3Dc);

            // Point must be inside the image
            if(!pKF->IsInImage(uv(0),uv(1)))
                continue;

            // Depth must be inside the scale pyramid of the image
            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            Eigen::Vector3f PO = p3Dw-Ow;
            const float dist3D = PO.norm();

            if(dist3D<minDistance || dist3D>maxDistance)
                continue;

            // Viewing angle must be less than 60 deg
            Eigen::Vector3f Pn = pMP->GetNormal();

            if(PO.dot(Pn)<0.5*dist3D)
                continue;

            // Compute predicted scale level
            const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

            // Search in a radius
            const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF->GetFeaturesInArea(uv(0),uv(1),radius);

            if(vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius

            const cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = INT_MAX;
            int bestIdx = -1;
            for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
            {
                const size_t idx = *vit;
                const int &kpLevel = pKF->mvKeysUn[idx].octave;

                if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF->mDescriptors.row(idx);

                int dist = DescriptorDistance(dMP,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            // If there is already a MapPoint replace otherwise add new measurement
            if(bestDist<=TH_LOW)
            {
                MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
                if(pMPinKF)
                {
                    if(!pMPinKF->isBad())
                        vpReplacePoint[iMP] = pMPinKF;
                }
                else
                {
                    pMP->AddObservation(pKF,bestIdx);
                    pKF->AddMapPoint(pMP,bestIdx);
                }
                nFused++;
            }
        }

        return nFused;
    }

    int ORBmatcher::SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches12, const Sophus::Sim3f &S12, const float th)
    {
        const float &fx = pKF1->fx;
        const float &fy = pKF1->fy;
        const float &cx = pKF1->cx;
        const float &cy = pKF1->cy;

        // Camera 1 & 2 from world
        Sophus::SE3f T1w = pKF1->GetPose();
        Sophus::SE3f T2w = pKF2->GetPose();

        //Transformation between cameras
        Sophus::Sim3f S21 = S12.inverse();

        const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
        const int N1 = vpMapPoints1.size();

        const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
        const int N2 = vpMapPoints2.size();

        vector<bool> vbAlreadyMatched1(N1,false);
        vector<bool> vbAlreadyMatched2(N2,false);

        for(int i=0; i<N1; i++)
        {
            MapPoint* pMP = vpMatches12[i];
            if(pMP)
            {
                vbAlreadyMatched1[i]=true;
                int idx2 = get<0>(pMP->GetIndexInKeyFrame(pKF2));
                if(idx2>=0 && idx2<N2)
                    vbAlreadyMatched2[idx2]=true;
            }
        }

        vector<int> vnMatch1(N1,-1);
        vector<int> vnMatch2(N2,-1);

        // Transform from KF1 to KF2 and search
        for(int i1=0; i1<N1; i1++)
        {
            MapPoint* pMP = vpMapPoints1[i1];

            if(!pMP || vbAlreadyMatched1[i1])
                continue;

            if(pMP->isBad())
                continue;

            Eigen::Vector3f p3Dw = pMP->GetWorldPos();
            Eigen::Vector3f p3Dc1 = T1w * p3Dw;
            Eigen::Vector3f p3Dc2 = S21 * p3Dc1;

            // Depth must be positive
            if(p3Dc2(2)<0.0)
                continue;

            const float invz = 1.0/p3Dc2(2);
            const float x = p3Dc2(0)*invz;
            const float y = p3Dc2(1)*invz;

            const float u = fx*x+cx;
            const float v = fy*y+cy;

            // Point must be inside the image
            if(!pKF2->IsInImage(u,v))
                continue;

            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            const float dist3D = p3Dc2.norm();

            // Depth must be inside the scale invariance region
            if(dist3D<minDistance || dist3D>maxDistance )
                continue;

            // Compute predicted octave
            const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);

            // Search in a radius
            const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);

            if(vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = INT_MAX;
            int bestIdx = -1;
            for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
            {
                const size_t idx = *vit;

                const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

                if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

                const int dist = DescriptorDistance(dMP,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if(bestDist<=TH_HIGH)
            {
                vnMatch1[i1]=bestIdx;
            }
        }

        // Transform from KF2 to KF2 and search
        for(int i2=0; i2<N2; i2++)
        {
            MapPoint* pMP = vpMapPoints2[i2];

            if(!pMP || vbAlreadyMatched2[i2])
                continue;

            if(pMP->isBad())
                continue;

            Eigen::Vector3f p3Dw = pMP->GetWorldPos();
            Eigen::Vector3f p3Dc2 = T2w * p3Dw;
            Eigen::Vector3f p3Dc1 = S12 * p3Dc2;

            // Depth must be positive
            if(p3Dc1(2)<0.0)
                continue;

            const float invz = 1.0/p3Dc1(2);
            const float x = p3Dc1(0)*invz;
            const float y = p3Dc1(1)*invz;

            const float u = fx*x+cx;
            const float v = fy*y+cy;

            // Point must be inside the image
            if(!pKF1->IsInImage(u,v))
                continue;

            const float maxDistance = pMP->GetMaxDistanceInvariance();
            const float minDistance = pMP->GetMinDistanceInvariance();
            const float dist3D = p3Dc1.norm();

            // Depth must be inside the scale pyramid of the image
            if(dist3D<minDistance || dist3D>maxDistance)
                continue;

            // Compute predicted octave
            const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

            // Search in a radius of 2.5*sigma(ScaleLevel)
            const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

            const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

            if(vIndices.empty())
                continue;

            // Match to the most similar keypoint in the radius
            const cv::Mat dMP = pMP->GetDescriptor();

            int bestDist = INT_MAX;
            int bestIdx = -1;
            for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
            {
                const size_t idx = *vit;

                const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

                if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                    continue;

                const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

                const int dist = DescriptorDistance(dMP,dKF);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }

            if(bestDist<=TH_HIGH)
            {
                vnMatch2[i2]=bestIdx;
            }
        }

        // Check agreement
        int nFound = 0;

        for(int i1=0; i1<N1; i1++)
        {
            int idx2 = vnMatch1[i1];

            if(idx2>=0)
            {
                int idx1 = vnMatch2[idx2];
                if(idx1==i1)
                {
                    vpMatches12[i1] = vpMapPoints2[idx2];
                    nFound++;
                }
            }
        }

        return nFound;
    }

    int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono)
    {
        int nmatches = 0;

        // Rotation Histogram (to check rotation consistency)
        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f/HISTO_LENGTH;

        const Sophus::SE3f Tcw = CurrentFrame.GetPose();
        const Eigen::Vector3f twc = Tcw.inverse().translation();

        const Sophus::SE3f Tlw = LastFrame.GetPose();
        const Eigen::Vector3f tlc = Tlw * twc;

        const bool bForward = tlc(2)>CurrentFrame.mb && !bMono;
        const bool bBackward = -tlc(2)>CurrentFrame.mb && !bMono;

        for(int i=0; i<LastFrame.N; i++)
        {
            MapPoint* pMP = LastFrame.mvpMapPoints[i];
            if(pMP)
            {
                if(!LastFrame.mvbOutlier[i])
                {
                    // Project
                    Eigen::Vector3f x3Dw = pMP->GetWorldPos();
                    Eigen::Vector3f x3Dc = Tcw * x3Dw;

                    const float xc = x3Dc(0);
                    const float yc = x3Dc(1);
                    const float invzc = 1.0/x3Dc(2);

                    if(invzc<0)
                        continue;

                    Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dc);

                    if(uv(0)<CurrentFrame.mnMinX || uv(0)>CurrentFrame.mnMaxX)
                        continue;
                    if(uv(1)<CurrentFrame.mnMinY || uv(1)>CurrentFrame.mnMaxY)
                        continue;

                    int nLastOctave = (LastFrame.Nleft == -1 || i < LastFrame.Nleft) ? LastFrame.mvKeys[i].octave
                                                                                     : LastFrame.mvKeysRight[i - LastFrame.Nleft].octave;

                    // Search in a window. Size depends on scale
                    float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

                    vector<size_t> vIndices2;

                    if(bForward)
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, nLastOctave);
                    else if(bBackward)
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, 0, nLastOctave);
                    else
                        vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, nLastOctave-1, nLastOctave+1);

                    if(vIndices2.empty())
                        continue;

                    const cv::Mat dMP = pMP->GetDescriptor();

                    int bestDist = 256;
                    int bestIdx2 = -1;

                    for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                    {
                        const size_t i2 = *vit;

                        if(CurrentFrame.mvpMapPoints[i2])
                            if(CurrentFrame.mvpMapPoints[i2]->Observations()>0)
                                continue;

                        if(CurrentFrame.Nleft == -1 && CurrentFrame.mvuRight[i2]>0)
                        {
                            const float ur = uv(0) - CurrentFrame.mbf*invzc;
                            const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                            if(er>radius)
                                continue;
                        }

                        // ── 语义一致性约束 ──
                        {
                            int kpSemClass = CurrentFrame.GetSemanticClassAt(i2);
                            if(pMP->mnSemanticClass >= 0 && kpSemClass != pMP->mnSemanticClass)
                                continue;
                        }

                        const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                        const int dist = DescriptorDistance(dMP,d);

                        if(dist<bestDist)
                        {
                            bestDist=dist;
                            bestIdx2=i2;
                        }
                    }

                    if(bestDist<=TH_HIGH)
                    {
                        CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                        nmatches++;

                        if(mbCheckOrientation)
                        {
                            cv::KeyPoint kpLF = (LastFrame.Nleft == -1) ? LastFrame.mvKeysUn[i]
                                                                        : (i < LastFrame.Nleft) ? LastFrame.mvKeys[i]
                                                                                                : LastFrame.mvKeysRight[i - LastFrame.Nleft];

                            cv::KeyPoint kpCF = (CurrentFrame.Nleft == -1) ? CurrentFrame.mvKeysUn[bestIdx2]
                                                                           : (bestIdx2 < CurrentFrame.Nleft) ? CurrentFrame.mvKeys[bestIdx2]
                                                                                                             : CurrentFrame.mvKeysRight[bestIdx2 - CurrentFrame.Nleft];
                            float rot = kpLF.angle-kpCF.angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdx2);
                        }
                    }
                    if(CurrentFrame.Nleft != -1){
                        Eigen::Vector3f x3Dr = CurrentFrame.GetRelativePoseTrl() * x3Dc;
                        Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dr);

                        int nLastOctave = (LastFrame.Nleft == -1 || i < LastFrame.Nleft) ? LastFrame.mvKeys[i].octave
                                             : LastFrame.mvKeysRight[i - LastFrame.Nleft].octave;

                        // Search in a window. Size depends on scale
                        float radius = th*CurrentFrame.mvScaleFactors[nLastOctave];

                        vector<size_t> vIndices2;

                        if(bForward)
                            vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, nLastOctave, -1,true);
                        else if(bBackward)
                            vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, 0, nLastOctave, true);
                        else
                            vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0),uv(1), radius, nLastOctave-1, nLastOctave+1, true);

                        const cv::Mat dMP = pMP->GetDescriptor();

                        int bestDist = 256;
                        int bestIdx2 = -1;

                        for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                        {
                            const size_t i2 = *vit;
                            if(CurrentFrame.mvpMapPoints[i2 + CurrentFrame.Nleft])
                                if(CurrentFrame.mvpMapPoints[i2 + CurrentFrame.Nleft]->Observations()>0)
                                    continue;

                            // ── 语义一致性约束（右视图）──
                            {
                                int kpSemClass = CurrentFrame.GetSemanticClassAt(i2 + CurrentFrame.Nleft);
                                if(pMP->mnSemanticClass >= 0 && kpSemClass != pMP->mnSemanticClass)
                                    continue;
                            }

                            const cv::Mat &d = CurrentFrame.mDescriptors.row(i2 + CurrentFrame.Nleft);

                            const int dist = DescriptorDistance(dMP,d);

                            if(dist<bestDist)
                            {
                                bestDist=dist;
                                bestIdx2=i2;
                            }
                        }

                        if(bestDist<=TH_HIGH)
                        {
                            CurrentFrame.mvpMapPoints[bestIdx2 + CurrentFrame.Nleft]=pMP;
                            nmatches++;
                            if(mbCheckOrientation)
                            {
                                cv::KeyPoint kpLF = (LastFrame.Nleft == -1) ? LastFrame.mvKeysUn[i]
                                                                            : (i < LastFrame.Nleft) ? LastFrame.mvKeys[i]
                                                                                                    : LastFrame.mvKeysRight[i - LastFrame.Nleft];

                                cv::KeyPoint kpCF = CurrentFrame.mvKeysRight[bestIdx2];

                                float rot = kpLF.angle-kpCF.angle;
                                if(rot<0.0)
                                    rot+=360.0f;
                                int bin = round(rot*factor);
                                if(bin==HISTO_LENGTH)
                                    bin=0;
                                assert(bin>=0 && bin<HISTO_LENGTH);
                                rotHist[bin].push_back(bestIdx2  + CurrentFrame.Nleft);
                            }
                        }

                    }
                }
            }
        }

        //Apply rotation consistency
        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i!=ind1 && i!=ind2 && i!=ind3)
                {
                    for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                    {
                        CurrentFrame.mvpMapPoints[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                        nmatches--;
                    }
                }
            }
        }

        // ── 语义匹配后验证：N帧同一框内的点，N+1帧也应在同一框内 ──
        if(!LastFrame.detectedBoxes.empty() && !CurrentFrame.detectedBoxes.empty())
        {
            std::map<MapPoint*, int> mpToCFidx;
            for(int j = 0; j < CurrentFrame.N; j++)
            {
                MapPoint* pMP = CurrentFrame.mvpMapPoints[j];
                if(pMP && pMP->Observations() > 0)
                    mpToCFidx[pMP] = j;
            }

            std::map<int, std::vector<int>> boxToCFindices;
            for(int i = 0; i < LastFrame.N; i++)
            {
                int boxIdx = LastFrame.GetBoxIndexAt(i);
                if(boxIdx < 0) continue;
                MapPoint* pMP = LastFrame.mvpMapPoints[i];
                if(!pMP || LastFrame.mvbOutlier[i]) continue;
                std::map<MapPoint*, int>::iterator it = mpToCFidx.find(pMP);
                if(it != mpToCFidx.end())
                    boxToCFindices[boxIdx].push_back(it->second);
            }

            for(std::map<int, std::vector<int>>::iterator kv = boxToCFindices.begin();
                kv != boxToCFindices.end(); ++kv)
            {
                const std::vector<int>& cfIndices = kv->second;
                if(cfIndices.size() < 2) continue;

                int commonBoxIdx = -1;
                bool consistent = true;
                for(size_t s = 0; s < cfIndices.size(); s++)
                {
                    int cbIdx = CurrentFrame.GetBoxIndexAt(cfIndices[s]);
                    if(cbIdx < 0) { consistent = false; break; }
                    if(commonBoxIdx < 0)
                        commonBoxIdx = cbIdx;
                    else if(cbIdx != commonBoxIdx)
                        { consistent = false; break; }
                }

                if(!consistent)
                {
                    for(size_t s = 0; s < cfIndices.size(); s++)
                    {
                        if(CurrentFrame.mvpMapPoints[cfIndices[s]])
                        {
                            CurrentFrame.mvpMapPoints[cfIndices[s]] = static_cast<MapPoint*>(NULL);
                            nmatches--;
                        }
                    }
                }
            }
        }

        return nmatches;
    }

    int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int ORBdist)
    {
        int nmatches = 0;

        const Sophus::SE3f Tcw = CurrentFrame.GetPose();
        Eigen::Vector3f Ow = Tcw.inverse().translation();

        // Rotation Histogram (to check rotation consistency)
        vector<int> rotHist[HISTO_LENGTH];
        for(int i=0;i<HISTO_LENGTH;i++)
            rotHist[i].reserve(500);
        const float factor = 1.0f/HISTO_LENGTH;

        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMPs[i];

            if(pMP)
            {
                if(!pMP->isBad() && !sAlreadyFound.count(pMP))
                {
                    //Project
                    Eigen::Vector3f x3Dw = pMP->GetWorldPos();
                    Eigen::Vector3f x3Dc = Tcw * x3Dw;

                    const Eigen::Vector2f uv = CurrentFrame.mpCamera->project(x3Dc);

                    if(uv(0)<CurrentFrame.mnMinX || uv(0)>CurrentFrame.mnMaxX)
                        continue;
                    if(uv(1)<CurrentFrame.mnMinY || uv(1)>CurrentFrame.mnMaxY)
                        continue;

                    // Compute predicted scale level
                    Eigen::Vector3f PO = x3Dw-Ow;
                    float dist3D = PO.norm();

                    const float maxDistance = pMP->GetMaxDistanceInvariance();
                    const float minDistance = pMP->GetMinDistanceInvariance();

                    // Depth must be inside the scale pyramid of the image
                    if(dist3D<minDistance || dist3D>maxDistance)
                        continue;

                    int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame);

                    // Search in a window
                    const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];

                    const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(uv(0), uv(1), radius, nPredictedLevel-1, nPredictedLevel+1);

                    if(vIndices2.empty())
                        continue;

                    const cv::Mat dMP = pMP->GetDescriptor();

                    int bestDist = 256;
                    int bestIdx2 = -1;

                    for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
                    {
                        const size_t i2 = *vit;
                        if(CurrentFrame.mvpMapPoints[i2])
                            continue;

                        // ── 语义一致性约束 ──
                        {
                            int kpSemClass = CurrentFrame.GetSemanticClassAt(i2);
                            if(pMP->mnSemanticClass >= 0 && kpSemClass != pMP->mnSemanticClass)
                                continue;
                        }

                        const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                        const int dist = DescriptorDistance(dMP,d);

                        if(dist<bestDist)
                        {
                            bestDist=dist;
                            bestIdx2=i2;
                        }
                    }

                    if(bestDist<=ORBdist)
                    {
                        CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                        nmatches++;

                        if(mbCheckOrientation)
                        {
                            float rot = pKF->mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdx2);
                        }
                    }

                }
            }
        }

        if(mbCheckOrientation)
        {
            int ind1=-1;
            int ind2=-1;
            int ind3=-1;

            ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

            for(int i=0; i<HISTO_LENGTH; i++)
            {
                if(i!=ind1 && i!=ind2 && i!=ind3)
                {
                    for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                    {
                        CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;
                        nmatches--;
                    }
                }
            }
        }

        return nmatches;
    }

    void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
    {
        int max1=0;
        int max2=0;
        int max3=0;

        for(int i=0; i<L; i++)
        {
            const int s = histo[i].size();
            if(s>max1)
            {
                max3=max2;
                max2=max1;
                max1=s;
                ind3=ind2;
                ind2=ind1;
                ind1=i;
            }
            else if(s>max2)
            {
                max3=max2;
                max2=s;
                ind3=ind2;
                ind2=i;
            }
            else if(s>max3)
            {
                max3=s;
                ind3=i;
            }
        }

        if(max2<0.1f*(float)max1)
        {
            ind2=-1;
            ind3=-1;
        }
        else if(max3<0.1f*(float)max1)
        {
            ind3=-1;
        }
    }


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
    int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
    {
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();

        int dist=0;

        for(int i=0; i<8; i++, pa++, pb++)
        {
            unsigned  int v = *pa ^ *pb;
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        return dist;
    }

} //namespace ORB_SLAM