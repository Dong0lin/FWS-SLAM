/**
* 长短焦（Multi-focal）双目匹配诊断工具：
*   不启动完整 SLAM，只用一两帧验证标定解析、特征校正、焦距比补偿匹配。
*   用法: ./stereo_mf_verify path_to_settings path_to_left_image path_to_right_image
*/

#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "Settings.h"
#include "System.h"
#include "ORBextractor.h"
#include "Frame.h"
#include "CameraModels/Pinhole.h"

using namespace std;

int main(int argc, char **argv)
{
    if(argc < 4)
    {
        cerr << "Usage: ./stereo_mf_verify path_to_settings path_to_left_image path_to_right_image" << endl;
        return 1;
    }

    ORB_SLAM3::Settings settings(argv[1], ORB_SLAM3::System::STEREO);
    if(!settings.isMultiFocal())
    {
        cerr << "ERROR: settings file is not a multi-focal yaml (missing Camera.combine)" << endl;
        return 1;
    }

    cv::Mat imLeft = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    cv::Mat imRight = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);
    if(imLeft.empty() || imRight.empty())
    {
        cerr << "ERROR: failed to load images" << endl;
        return 1;
    }

    ORB_SLAM3::ORBextractor extL(settings.nFeatures(), settings.scaleFactor(), settings.nLevels(),
                                 settings.initThFAST(), settings.minThFAST());
    ORB_SLAM3::ORBextractor extR(settings.nFeatures(), settings.scaleFactor(), settings.nLevels(),
                                 settings.initThFAST(), settings.minThFAST());

    ORB_SLAM3::Frame::SetMultiFocalCalib(settings.leftK(), settings.leftD(), settings.leftR(), settings.leftP(),
                                         settings.rightK(), settings.rightD(), settings.rightR(), settings.rightP(),
                                         settings.focalScale(), settings.roiLeftUp(), settings.roiRightBottom(),
                                         imLeft.cols, imLeft.rows,
                                         settings.minDepth(), settings.maxDepth());

    cv::Mat K = static_cast<ORB_SLAM3::Pinhole*>(settings.camera1())->toK();
    cv::Mat dist = cv::Mat::zeros(4, 1, CV_32F);

    // 单独计时左右目 ORB 提取（用于拆分 Frame 构造耗时：
    // Frame 构造 = max(左ORB, 右ORB) + 立体匹配）
    std::vector<cv::KeyPoint> vKpsL, vKpsR;
    cv::Mat descL, descR;
    std::vector<int> vLapL = {0, imLeft.cols};   // 计时用：整幅图作为重叠区
    std::vector<int> vLapR = {0, imRight.cols};
    auto tOrbL0 = std::chrono::steady_clock::now();
    extL(imLeft, cv::Mat(), vKpsL, descL, vLapL, 0);   // FLAG=0 与 Frame 左目一致
    auto tOrbL1 = std::chrono::steady_clock::now();
    extR(imRight, cv::Mat(), vKpsR, descR, vLapR, 1);  // FLAG=1 与 Frame 右目一致
    auto tOrbR1 = std::chrono::steady_clock::now();
    const double orbLMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(tOrbL1 - tOrbL0).count();
    const double orbRMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(tOrbR1 - tOrbL1).count();
    cout << "  ORB提取(左) = " << orbLMs << " ms   ORB提取(右) = " << orbRMs << " ms" << endl;

    auto tFrameStart = std::chrono::steady_clock::now();
    ORB_SLAM3::Frame F(imLeft, imRight, 0.0, &extL, &extR, nullptr,
                       K, dist, settings.bf(), settings.thDepth(), settings.camera1());
    auto tFrameEnd = std::chrono::steady_clock::now();
    const double frameMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(tFrameEnd - tFrameStart).count();
    cout << "  Frame构造(ORB左+右并行 + 长短焦立体匹配) = " << frameMs << " ms" << endl;

    int nMatched = 0;
    int nNear = 0;
    int nD0_10 = 0, nD10_50 = 0, nD50_150 = 0, nD150_300 = 0, nD300P = 0;
    int nLeftInROI = 0;
    int nRightInROI = 0;
    double sumDepth = 0;
    double minDepth = 1e9, maxDepth = 0;
    vector<double> vDepths;
    vector<double> vCamY;          // 相机坐标系 Y（=下方为正；负值表示方向在相机上方）
    double sumY = 0, sumZ = 0, sumYZ = 0, sumZ2 = 0;   // Y = a*Z + b 最小二乘（地面平面斜率估计）
    double sumZUpper = 0, sumZLower = 0;               // 上半(v<cy) vs 下半(v>cy) 平均深度
    int nUpper = 0, nLower = 0;
    int nYnegFar = 0, nYnegNear = 0, nYposFar = 0;     // y_cam<0 且 z>80m / y_cam<0 且 z<=80m / y_cam>=0 且 z>80m
    int nKeepAdaptive = 0, nKillAdaptive = 0;          // 自适应门控（v<cy 且 z>150 剔除）保留/剔除数
    int nKillAdaptive100 = 0;                          // 阈值100m时的剔除数（对比用）
    for(int i = 0; i < F.N; i++)
    {
        const cv::KeyPoint& kpL = F.mvKeys[i];
        if(kpL.pt.x > ORB_SLAM3::Frame::mROIRectLeftUp.x && kpL.pt.x < ORB_SLAM3::Frame::mROIRectRightBottom.x &&
           kpL.pt.y > ORB_SLAM3::Frame::mROIRectLeftUp.y && kpL.pt.y < ORB_SLAM3::Frame::mROIRectRightBottom.y)
            nLeftInROI++;

        if(F.mvDepth[i] > 0)
        {
            nMatched++;
            sumDepth += F.mvDepth[i];
            vDepths.push_back(F.mvDepth[i]);
            if(F.mvDepth[i] < minDepth) minDepth = F.mvDepth[i];
            if(F.mvDepth[i] > maxDepth) maxDepth = F.mvDepth[i];
            if(F.mvDepth[i] < F.mThDepth) nNear++;

            // 相机坐标系（mvKeys 为校正坐标）：Y = (v-cy)*z/fy, Z = depth
            const float vc = kpL.pt.y;
            const float camY = (vc - ORB_SLAM3::Frame::cy) * F.mvDepth[i] / ORB_SLAM3::Frame::fy;
            vCamY.push_back(camY);
            if(vc < ORB_SLAM3::Frame::cy) { sumZUpper += F.mvDepth[i]; nUpper++; }
            else { sumZLower += F.mvDepth[i]; nLower++; }
            if(camY < 0.0 && F.mvDepth[i] > 80.0) nYnegFar++;
            else if(camY < 0.0) nYnegNear++;
            else if(F.mvDepth[i] > 80.0) nYposFar++;
            if(kpL.pt.y < ORB_SLAM3::Frame::cy && F.mvDepth[i] > 150.0) nKillAdaptive++;
            if(kpL.pt.y < ORB_SLAM3::Frame::cy && F.mvDepth[i] > 100.0) nKillAdaptive100++;
            nKeepAdaptive++;
            sumY += camY;
            sumZ += F.mvDepth[i];
            sumYZ += camY * F.mvDepth[i];
            sumZ2 += F.mvDepth[i] * F.mvDepth[i];

            if(F.mvDepth[i] < 10.0) nD0_10++;
            else if(F.mvDepth[i] < 50.0) nD10_50++;
            else if(F.mvDepth[i] < 150.0) nD50_150++;
            else if(F.mvDepth[i] < 300.0) nD150_300++;
            else nD300P++;
        }
    }
    for(size_t i = 0; i < F.mvKeysRight.size(); i++)
    {
        if(F.mvKeysRight[i].pt.x > 0 && F.mvKeysRight[i].pt.x < 1280)
            nRightInROI++;
    }
    sort(vDepths.begin(), vDepths.end());
    const double medianDepth = vDepths.empty() ? 0.0 : vDepths[vDepths.size() / 2];
    sort(vCamY.begin(), vCamY.end());
    int nUnder500 = 0;
    for(size_t k = 0; k < vDepths.size(); k++)
        if(vDepths[k] < 500.0)
            nUnder500++;

    // 地面平面估计：y_cam = a*z + b
    double a = 0, b = 0;
    if(vDepths.size() > 1)
    {
        const double n = (double)vDepths.size();
        const double denom = n * sumZ2 - sumZ * sumZ;
        if(fabs(denom) > 1e-9)
        {
            a = (n * sumYZ - sumY * sumZ) / denom;
            b = (sumY - a * sumZ) / n;
        }
    }

    // 邻域深度一致性统计（与匹配器同一判据：45px 邻域内 log 深度偏差 > 0.6）
    int nInconsistent = 0;
    {
        vector<int> vIdx;
        for(int i = 0; i < F.N; i++)
            if(F.mvDepth[i] > 0)
                vIdx.push_back(i);
        for(size_t a = 0; a < vIdx.size(); a++)
        {
            const int iA = vIdx[a];
            vector<float> vN;
            for(size_t b = 0; b < vIdx.size(); b++)
            {
                if(a == b)
                    continue;
                const int iB = vIdx[b];
                const float dx = F.mvKeys[iA].pt.x - F.mvKeys[iB].pt.x;
                const float dy = F.mvKeys[iA].pt.y - F.mvKeys[iB].pt.y;
                if(dx * dx + dy * dy <= 80.f * 80.f)
                    vN.push_back(log(F.mvDepth[iB]));
            }
            if(vN.empty())
                continue;
            sort(vN.begin(), vN.end());
            const float med = vN[vN.size() / 2];
            if(fabs(log(F.mvDepth[iA]) - med) > 1.6f)
                nInconsistent++;
        }
    }

    cout << "----------------------------------------" << endl;
    cout << "Multi-focal verify:" << endl;
    cout << "  Fscale           = " << settings.focalScale() << endl;
    cout << "  ROI              = [" << settings.roiLeftUp() << " -> " << settings.roiRightBottom() << "]" << endl;
    cout << "  ROI(rectified)   = [" << ORB_SLAM3::Frame::mROIRectLeftUp << " -> "
         << ORB_SLAM3::Frame::mROIRectRightBottom << "]" << endl;
    cout << "  rectified fx     = " << K.at<float>(0,0) << "  bf = " << settings.bf() << endl;
    cout << "  left keys        = " << F.N << endl;
    cout << "  left keys in ROI = " << nLeftInROI << endl;
    cout << "  right keys       = " << (int)F.mvKeysRight.size() << endl;
    cout << "  right keys valid = " << nRightInROI << endl;
    cout << "  matched (depth>0)= " << nMatched << endl;
    cout << "  depth range      = [" << minDepth << ", " << maxDepth << "] m" << endl;
    cout << "  mean depth       = " << (nMatched ? sumDepth / nMatched : 0.0) << " m" << endl;
    cout << "  median depth     = " << medianDepth << " m" << endl;
    cout << "  depth hist       = [0,10):" << nD0_10
         << " [10,50):" << nD10_50
         << " [50,150):" << nD50_150
         << " [150,300):" << nD150_300
         << " [300,+):" << nD300P << endl;
    cout << "  depth<500m       = " << nUnder500 << " / " << nMatched << endl;
    cout << "  inconsistent(nbhd)= " << nInconsistent << endl;
    cout << "  close pts (<ThDepth=" << F.mThDepth << ") = " << nNear << endl;
    cout << "  camY range       = [" << (vCamY.empty() ? 0.0 : vCamY.front())
         << ", " << (vCamY.empty() ? 0.0 : vCamY.back()) << "] m (Y<0=方向在相机上方)" << endl;
    cout << "  camY<0 count     = " << (int)std::count_if(vCamY.begin(), vCamY.end(),
         [](double y){ return y < 0.0; }) << " / " << nMatched << endl;
    cout << "  mean depth upper-half(v<cy) = " << (nUpper ? sumZUpper/nUpper : 0.0)
         << " (n=" << nUpper << ")   lower-half(v>cy) = " << (nLower ? sumZLower/nLower : 0.0)
         << " (n=" << nLower << ")" << endl;
    cout << "  (y_cam<0 & z>80)=" << nYnegFar << "  (y_cam<0 & z<=80)=" << nYnegNear
         << "  (y_cam>=0 & z>80)=" << nYposFar << endl;
    cout << "  adaptive gate (v<cy & z>150): keep=" << (nKeepAdaptive - nKillAdaptive) << " kill=" << nKillAdaptive
         << "   (z>100 variant kill=" << nKillAdaptive100 << ")" << endl;
    cout << "  ground fit Y=aZ+b: a=" << a << " b=" << b
         << "  (pitch≈atan(a)=" << atan(a) * 180.0 / M_PI << "deg, 高度≈-b/sqrt(1+a^2))" << endl;
    cout << "----------------------------------------" << endl;

    // ================= 左右目检测框匹配自测（MF_VERIFY_MATCH=1 时执行）=================
    // 用合成检测框（同一辆车投影到左右原始图像）验证 RectifyDetectionBoxesRight +
    // MatchRightDetections 的匹配与深度计算逻辑，不需要 GPU/真实检测器。
    if(getenv("MF_VERIFY_MATCH"))
    {
        cout << "== MatchRightDetections self-test ==" << endl;

        // Z=40m 处一辆车（3.8x1.6x1.5m，中心在公共系 (2,6,40)）投影到左右原始图的框
        // （数值由投影几何独立算出，见长短焦匹配几何验证）
        ORB_SLAM3::Detection detL, detR;
        detL.class_id = 3;      // car
        detR.class_id = 3;
        detL.conf = detR.conf = 0.9f;
        detL.bbox = cv::Rect(651, 395, 85, 41);   // 左目原始框
        detR.bbox = cv::Rect(666, 422, 124, 60);  // 右目原始框

        // 另一辆不同类别的车（bus，位置偏移），应不参与匹配
        ORB_SLAM3::Detection detL2, detR2;
        detL2.class_id = 8;     // bus
        detR2.class_id = 8;
        detL2.conf = detR2.conf = 0.9f;
        detL2.bbox = cv::Rect(660, 355, 90, 31);    // Z=80m 的 bus（投影几何算出）
        detR2.bbox = cv::Rect(680, 364, 131, 46);

        // 行人（class_id=0）：应被排除
        ORB_SLAM3::Detection detL3, detR3;
        detL3.class_id = 0;
        detR3.class_id = 0;
        detL3.bbox = cv::Rect(800, 400, 20, 40);
        detR3.bbox = cv::Rect(812, 410, 28, 56);

        std::vector<ORB_SLAM3::Detection> vL = {detL, detL2, detL3};
        std::vector<ORB_SLAM3::Detection> vR = {detR, detR2, detR3};

        // 与 GrabImageStereo 一致：左目框先校正再 SetBoxes
        ORB_SLAM3::Frame::RectifyDetectionBoxes(vL);
        F.SetBoxes(vL);
        F.MatchRightDetections(vR);

        cout << "  matched[0](car)   -> right idx " << F.mvMatchedRightBoxIdx[0]
             << " (期望 0), depth=" << F.mvDetectionDepth[0] << "m" << endl;
        cout << "  matched[1](bus)   -> right idx " << F.mvMatchedRightBoxIdx[1]
             << " (期望 1), depth=" << F.mvDetectionDepth[1] << "m" << endl;
        cout << "  matched[2](person)-> right idx " << F.mvMatchedRightBoxIdx[2]
             << " (期望 -1, 人被排除)" << endl;
        cout << "----------------------------------------" << endl;
    }

    return 0;
}
