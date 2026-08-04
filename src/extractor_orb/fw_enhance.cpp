/**
 * FW-ORB-SLAM 第三章C: 局部自适应对比度增强
 *
 * 论文参考: "FW-ORB-SLAM: A Monocular Visual SLAM Algorithm for
 *            Flapping-Wing Flying Robots", IEEE RAL, 2025.
 *
 * 算法流程:
 *   1. 滑动窗口计算灰度局部均值 μ 和方差 σ² (Eq.9,10)
 *   2. α = 1.0 + k×σ/μ, 三通道共用同一组 α (保持色彩一致)
 *   3. L_EB = μ + min(α,β) × (L − μ)  (Eq.11)
 *   4. 引导滤波去噪保边 (Eq.12)
 *
 * 改进于原始论文:
 *   - α 从灰度图计算后三通道复用 → 消除色彩漂移 + 3×提速
 *   - α 基准设为 1.0 (非 0), 避免低纹理区域对比度压缩
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <dirent.h>
#include <sys/stat.h>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;

// ============================================================================
// 工具函数
// ============================================================================

double ComputeSharpness(const cv::Mat& gray) {
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar mu, sigma;
    cv::meanStdDev(lap, mu, sigma);
    return sigma.val[0] * sigma.val[0];
}

// ============================================================================
// 引导滤波 (手动实现, 不依赖 opencv_contrib)
//
// Eq.12: Lgf_E = a·I + b   其中 a = cov(I,p)/(var_I+eps), b = mean_p − a·mean_I
//        (mean_a, mean_b 是 a,b 的盒滤波平滑)
// ============================================================================
cv::Mat GuidedFilter(const cv::Mat& src, const cv::Mat& guide, int radius, float eps) {
    cv::Mat I;
    guide.convertTo(I, CV_32F);

    // 自引导优化: guide==src 时 mean_p=mean_I, mean_Ip=mean_II, 省 2 次 boxFilter
    cv::Mat p;
    bool selfGuided = (guide.data == src.data && guide.rows == src.rows && guide.cols == src.cols);
    if (selfGuided) {
        p = I;
    } else {
        src.convertTo(p, CV_32F);
    }

    cv::Mat mean_I, mean_p, mean_II, mean_Ip;
    cv::boxFilter(I,       mean_I,  -1, cv::Size(radius, radius));
    if (selfGuided) {
        mean_p  = mean_I;
        cv::boxFilter(I.mul(I), mean_II, -1, cv::Size(radius, radius));
        mean_Ip = mean_II;
    } else {
        cv::boxFilter(p,       mean_p,  -1, cv::Size(radius, radius));
        cv::boxFilter(I.mul(I), mean_II, -1, cv::Size(radius, radius));
        cv::boxFilter(I.mul(p), mean_Ip, -1, cv::Size(radius, radius));
    }

    cv::Mat var_I  = mean_II - mean_I.mul(mean_I);
    cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);

    cv::Mat a = cov_Ip / (var_I + eps);
    cv::Mat b = mean_p - a.mul(mean_I);

    cv::Mat mean_a, mean_b;
    cv::boxFilter(a, mean_a, -1, cv::Size(radius, radius));
    cv::boxFilter(b, mean_b, -1, cv::Size(radius, radius));

    cv::Mat q = mean_a.mul(I) + mean_b;
    cv::Mat result;
    q.convertTo(result, src.type());
    return result;
}

// ============================================================================
// 局部自适应对比度增强 — 单通道版
//
// 核心: 在灰度图上算出增益图 gain = min(α, β)
//       然后 BGR 三通道复用同一张增益图, 保证颜色一致性
//
// @param src     输入灰度图 (CV_8UC1)  — 用于计算 μ, σ, α
// @param gain    输出增益图 (CV_32FC1) — min(α, β), 供后续通道复用
// ============================================================================
void ComputeGainFromGray(const cv::Mat& src, cv::Mat& gain,
                          int blockW, int blockH,
                          float k, float beta, float lambda) {
    CV_Assert(src.type() == CV_8UC1);

    cv::Mat srcF;
    src.convertTo(srcF, CV_32F);

    // Eq.9: 局部均值 μ
    cv::Mat mu, mu2;
    cv::boxFilter(srcF, mu,  -1, cv::Size(blockW, blockH));
    cv::boxFilter(srcF.mul(srcF), mu2, -1, cv::Size(blockW, blockH));

    // Eq.10: 局部方差 σ² = E[X²] − E[X]²
    cv::Mat var = mu2 - mu.mul(mu);
    cv::max(var, 0, var);

    cv::Mat sigma;
    cv::sqrt(var, sigma);

    // 向量化: α = 1.0 + k×σ/μ, 平坦区保持 1.0, 整体截断到 β
    gain = cv::Mat::ones(src.size(), CV_32F);

    cv::Mat sigmaRel;       // σ/μ, 0 where μ≈0
    cv::divide(sigma, mu, sigmaRel);

    cv::Mat active = (mu > 1e-6f) & (sigmaRel > lambda);
    cv::Mat alpha = 1.0f + k * sigmaRel;
    cv::Mat alphaClamped;
    cv::min(alpha, beta, alphaClamped);
    alphaClamped.copyTo(gain, active);
}

// ============================================================================
// 三通道增强 (入口) — 优化版
//
// 优化要点:
//   - 灰度纹理增益 + 彩色显著性增益 → 合并为一张增益图
//   - 引导滤波只对合并增益图做 1 次 (原来 3 通道各做一次: 18→6 boxFilter)
//   - BGR 各通道局部均值 μ 只算一次, 颜色显著性和应用增益共用 (6→3 boxFilter)
//   - 三通道仍各自增强, 保持彩色信息完整, 不影响目标检测
//
// @param src        输入 BGR 彩色图
// @param dst        输出 BGR 彩色图
// @param saliency   颜色显著性增强强度 (0=关闭)
// ============================================================================
void EnhanceImage(const cv::Mat& src, cv::Mat& dst,
                  int blockSize, float k, float beta, float lambda,
                  int gfRadius, float saliencyBoost) {
    // === 阶段1: 灰度图 → 纹理增益 (Eq.9-11) ===
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    cv::Mat textureGain;
    ComputeGainFromGray(gray, textureGain, blockSize, blockSize, k, beta, lambda);

    // === 阶段2: BGR 三通道预处理 (split + 各通道局部均值 μ, 只算一次) ===
    cv::Mat srcF;
    src.convertTo(srcF, CV_32F);
    std::vector<cv::Mat> chSrc(3);
    cv::split(srcF, chSrc);

    cv::Mat muBGR[3];
    for (int c = 0; c < 3; c++)
        cv::boxFilter(chSrc[c], muBGR[c], -1, cv::Size(blockSize, blockSize));

    // === 阶段3: 颜色显著性增益 (复用阶段2的 μ, 零额外 boxFilter) ===
    cv::Mat combinedGain;
    if (saliencyBoost > 1e-6f) {
        // 逐通道 L2 颜色距离: dist² = Σ(B−μB)²+(G−μG)²+(R−μR)²
        cv::Mat distSq = cv::Mat::zeros(src.rows, src.cols, CV_32F);
        for (int c = 0; c < 3; c++) {
            cv::Mat diff = chSrc[c] - muBGR[c];
            distSq += diff.mul(diff);
        }
        cv::Mat colorDist;
        cv::sqrt(distSq, colorDist);

        // colorGain = 1.0 + boost × min(dist/30, 1.0)  截断到 beta
        cv::Mat s;
        cv::min(colorDist / 30.0f, 1.0f, s);
        cv::Mat colorGain;
        cv::min(1.0f + saliencyBoost * s, beta, colorGain);
        cv::multiply(textureGain, colorGain, combinedGain);
    } else {
        combinedGain = textureGain;
    }

    // === 阶段4: 引导滤波 → 只对增益图做 1 次 (原来 3 次, 核心加速点) ===
    if (gfRadius > 0)
        combinedGain = GuidedFilter(combinedGain, combinedGain, gfRadius, 0.01f);

    // === 阶段5: 三通道应用增益 (复用阶段2的 μ) ===
    //    output_c = μ_c + gain × (pixel_c − μ_c)
    std::vector<cv::Mat> chDst(3);
    for (int c = 0; c < 3; c++) {
        cv::Mat enhanced = muBGR[c] + combinedGain.mul(chSrc[c] - muBGR[c]);
        cv::min(cv::max(enhanced, 0), 255, enhanced);
        enhanced.convertTo(chDst[c], CV_8U);
    }
    cv::merge(chDst, dst);
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <folder> [k] [beta] [block] [gfR] [lambda] [saliency]" << endl;
        cerr << endl;
        cerr << "Parameters:" << endl;
        cerr << "  k         增强系数 (α=1.0+k×σ/μ). 0.1=微, 0.25=默认, 0.6=激进" << endl;
        cerr << "  beta      截断因子, 防止过增强.   2.0~4.0 (默认3.0)" << endl;
        cerr << "  block     局部块大小.              5~31 (默认15)" << endl;
        cerr << "  gfR       引导滤波半径.            0=关, 2~4 (默认2)" << endl;
        cerr << "  lambda    平坦区不增强阈值.        0.005~0.05 (默认0.02)" << endl;
        cerr << "  saliency  颜色显著性增强强度.      0=关, 0.5=默认, 1.0=激进" << endl;
        cerr << "            对与周围颜色差异大的色块(车辆等)额外增强" << endl;
        cerr << endl;
        cerr << "  输出: <folder>_fw_enhanced/" << endl;
        return -1;
    }

    string inputDir = argv[1];
    float k          = (argc >= 3) ? stof(argv[2]) : 0.25f;
    float beta       = (argc >= 4) ? stof(argv[3]) : 3.0f;
    int   blockSize  = (argc >= 5) ? stoi(argv[4]) : 15;
    int   gfRadius   = (argc >= 6) ? stoi(argv[5]) : 2;
    float lambda     = (argc >= 7) ? stof(argv[6]) : 0.02f;
    float saliency   = (argc >= 8) ? stof(argv[7]) : 0.25f;

    while (!inputDir.empty() && inputDir.back() == '/')
        inputDir.pop_back();

    string outputDir = inputDir + "_fw_enhanced";
    if (mkdir(outputDir.c_str(), 0755) != 0 && errno != EEXIST) {
        cerr << "Failed to create output dir: " << outputDir << endl;
        return -1;
    }

    // string compareDir = inputDir + "_fw_enhanced_compare";
    // if (mkdir(compareDir.c_str(), 0755) != 0 && errno != EEXIST) {
    //     cerr << "Failed to create compare dir: " << compareDir << endl;
    //     return -1;
    // }

    // 收集文件列表
    vector<string> filenames;
    {
        DIR* dir = opendir(inputDir.c_str());
        if (!dir) { cerr << "Cannot open: " << inputDir << endl; return -1; }
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            string name(entry->d_name);
            if (name == "." || name == "..") continue;
            size_t dot = name.find_last_of('.');
            if (dot == string::npos) continue;
            string ext = name.substr(dot);
            if (ext == ".jpg" || ext == ".JPG" || ext == ".jpeg" || ext == ".JPEG" ||
                ext == ".png" || ext == ".PNG" || ext == ".bmp"  || ext == ".BMP") {
                filenames.push_back(inputDir + "/" + name);
            }
        }
        closedir(dir);
    }
    sort(filenames.begin(), filenames.end());

    if (filenames.empty()) { cerr << "No images in " << inputDir << endl; return -1; }

    cout << "===== FW-ORB-SLAM Local Adaptive Contrast Enhancement =====" << endl;
    cout << "Input:   " << inputDir << " (" << filenames.size() << " images)" << endl;
    cout << "Output:  " << outputDir << endl;
    // cout << "Compare: " << compareDir << endl;
    cout << "Params:  k=" << k << " beta=" << beta << " block=" << blockSize
         << " gfR=" << gfRadius << " lambda=" << lambda
         << " saliency=" << saliency << endl;
    cout << "Processing..." << endl;

    double sumSrc = 0, sumDst = 0;
    double totalTime = 0;
    int count = 0;

    for (size_t i = 0; i < filenames.size(); i++) {
        int64 t0 = cv::getTickCount();

        cv::Mat img = cv::imread(filenames[i], cv::IMREAD_COLOR);
        if (img.empty()) { cerr << "  skip: " << filenames[i] << endl; continue; }

        // 原图 sharpness (从灰度算)
        cv::Mat graySrc;
        cv::cvtColor(img, graySrc, cv::COLOR_BGR2GRAY);
        double sharpSrc = ComputeSharpness(graySrc);

        // 增强
        cv::Mat result;
        EnhanceImage(img, result, blockSize, k, beta, lambda, gfRadius, saliency);

        // 输出 sharpness
        cv::Mat grayDst;
        cv::cvtColor(result, grayDst, cv::COLOR_BGR2GRAY);
        double sharpDst = ComputeSharpness(grayDst);

        sumSrc += sharpSrc;
        sumDst += sharpDst;
        count++;

        string fname = filenames[i].substr(filenames[i].find_last_of("/\\") + 1);
        cv::imwrite(outputDir + "/" + fname, result);

        // 拼接对比图: 原图 | 增强图 (方便肉眼对比)
        // {
        //     cv::Mat left = img.clone(), right = result.clone();
        //     int txtH = 30;
        //     cv::copyMakeBorder(left,  left,  txtH, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(40,40,40));
        //     cv::copyMakeBorder(right, right, txtH, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(40,40,40));
        //     cv::putText(left,  "Original",  cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 2);
        //     cv::putText(right, "Enhanced",  cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 2);

        //     // sharpness 标注在右下角
        //     char buf[64];
        //     snprintf(buf, sizeof(buf), "S=%.0f", sharpSrc);
        //     cv::putText(left,  buf, cv::Point(left.cols-120, left.rows-12),  cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255,255,255), 1);
        //     snprintf(buf, sizeof(buf), "S=%.0f", sharpDst);
        //     cv::putText(right, buf, cv::Point(right.cols-120, right.rows-12), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0,255,0), 1);

        //     cv::Mat cmp;
        //     cv::hconcat(left, right, cmp);
        //     cv::imwrite(compareDir + "/" + fname, cmp);
        // }

        int64 t1 = cv::getTickCount();
        double t = (t1 - t0) / cv::getTickFrequency() * 1000.0;  // ms
        totalTime += t;

        if (count % 50 == 0 || i == filenames.size() - 1) {
            cout << "  [" << count << "/" << filenames.size() << "] "
                 << "sharpness: " << sumSrc/count << " -> " << sumDst/count
                 << " (+" << (sumDst/sumSrc - 1.0)*100 << "%)"
                 << "  t=" << totalTime/count << "ms/frame" << endl;
        }
    }

    cout << "\n===== Done =====" << endl;
    cout << "Frames:   " << count << endl;
    cout << "Avg sharpness gain: +" << (sumDst/sumSrc - 1.0)*100 << "%" << endl;
    cout << "Avg time: " << totalTime/count << " ms/frame  "
         << "(" << count*1000.0/totalTime << " FPS)" << endl;
    cout << "Output:   " << outputDir << endl;

    return 0;
}
