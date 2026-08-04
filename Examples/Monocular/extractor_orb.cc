#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <ORBextractor.h>   // ORB-SLAM3 的特征提取器头文件

using namespace std;

// 分布均匀性：计算 10x10 网格中被特征点覆盖的比例
float ComputeDistribution(const std::vector<cv::KeyPoint>& vKeys, int rows, int cols) {
    const int grid_rows = 10;
    const int grid_cols = 10;
    std::vector<bool> covered(grid_rows * grid_cols, false);

    for (const auto& kp : vKeys) {
        int r = static_cast<int>(kp.pt.y * grid_rows / rows);
        int c = static_cast<int>(kp.pt.x * grid_cols / cols);
        if (r >= 0 && r < grid_rows && c >= 0 && c < grid_cols)
            covered[r * grid_cols + c] = true;
    }

    int covered_count = 0;
    for (bool flag : covered)
        if (flag) covered_count++;

    return static_cast<float>(covered_count) / (grid_rows * grid_cols);
}

// 图像清晰度：拉普拉斯方差（方差越大图像越清晰）
double ComputeSharpness(const cv::Mat& gray) {
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar mu, sigma;
    cv::meanStdDev(lap, mu, sigma);
    return sigma.val[0] * sigma.val[0];  // variance
}

// 响应值统计：均值、中位数、标准差
struct ResponseStats {
    float mean, median, stddev, min_val, max_val;
};

ResponseStats ComputeResponseStats(const std::vector<cv::KeyPoint>& vKeys) {
    std::vector<float> responses;
    responses.reserve(vKeys.size());
    for (const auto& kp : vKeys)
        responses.push_back(kp.response);

    std::sort(responses.begin(), responses.end());

    size_t n = responses.size();
    float sum = std::accumulate(responses.begin(), responses.end(), 0.0f);
    float mean = sum / n;
    float median = (n % 2 == 0) ?
        (responses[n/2 - 1] + responses[n/2]) / 2.0f : responses[n/2];

    float sq_sum = 0;
    for (float r : responses)
        sq_sum += (r - mean) * (r - mean);
    float stddev = std::sqrt(sq_sum / n);

    return {mean, median, stddev, responses.front(), responses.back()};
}

// 金字塔层级分布：统计每层的特征点数量及比例
std::vector<int> ComputeOctaveDistribution(const std::vector<cv::KeyPoint>& vKeys, int nLevels) {
    std::vector<int> dist(nLevels, 0);
    for (const auto& kp : vKeys) {
        int oct = kp.octave;
        if (oct >= 0 && oct < nLevels)
            dist[oct]++;
    }
    return dist;
}

// 响应值分档统计：统计各响应区间的点数
void PrintResponseHistogram(const std::vector<cv::KeyPoint>& vKeys) {
    int bins[6] = {0};  // <10, 10-20, 20-50, 50-100, 100-200, >=200
    for (const auto& kp : vKeys) {
        float r = kp.response;
        if (r < 10)       bins[0]++;
        else if (r < 20)  bins[1]++;
        else if (r < 50)  bins[2]++;
        else if (r < 100) bins[3]++;
        else if (r < 200) bins[4]++;
        else              bins[5]++;
    }
    int n = static_cast<int>(vKeys.size());
    cout << "  Response histogram:" << endl;
    cout << "    < 10:  " << bins[0] << " (" << 100.0*bins[0]/n << "%)" << endl;
    cout << "   10-20: " << bins[1] << " (" << 100.0*bins[1]/n << "%)" << endl;
    cout << "   20-50: " << bins[2] << " (" << 100.0*bins[2]/n << "%)" << endl;
    cout << "  50-100: " << bins[3] << " (" << 100.0*bins[3]/n << "%)" << endl;
    cout << " 100-200: " << bins[4] << " (" << 100.0*bins[4]/n << "%)" << endl;
    cout << "   >=200: " << bins[5] << " (" << 100.0*bins[5]/n << "%)" << endl;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: ./extract_orb <image_path>" << endl;
        return -1;
    }

    // 1. 加载灰度图
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Failed to load image: " << argv[1] << endl;
        return -1;
    }
    cout << "Image size: " << img.cols << " x " << img.rows << endl;

    // 2. 设置 ORB 提取参数（应与你的 SLAM 系统保持一致）
    int nFeatures = 2500;       // 目标特征点数
    float scaleFactor = 1.15f;   // 金字塔缩放系数
    int nLevels = 10;            // 金字塔层数
    int iniThFAST = 15;         // FAST 角点初始阈值
    int minThFAST = 6;          // FAST 角点最低阈值

    // 3. 创建 ORB 提取器
    ORB_SLAM3::ORBextractor orbExtractor(nFeatures, scaleFactor, nLevels, iniThFAST, minThFAST);

    // 4. 提取特征点与描述子
    std::vector<cv::KeyPoint> vKeys;
    cv::Mat descriptors;
    cv::Mat mask;                // 创建一个具名的空 mask（即全图提取）
    std::vector<int> vLappingArea = {-1, -1};  // 单目：设无效范围，所有点归入 monoIndex
    orbExtractor(img, mask, vKeys, descriptors, vLappingArea);

    int actualNum = vKeys.size();
    cout << "Extracted features: " << actualNum << endl;

    if (actualNum == 0) {
        cerr << "No feature extracted. Exiting." << endl;
        return 0;
    }

    // 5. 图像清晰度（拉普拉斯方差）
    double sharpness = ComputeSharpness(img);
    cout << "\n=== Image Quality Metrics ===" << endl;
    cout << "Sharpness (Laplacian variance): " << sharpness << endl;

    // 6. 响应值统计
    ResponseStats rs = ComputeResponseStats(vKeys);
    cout << "Response stats:" << endl;
    cout << "  Min:    " << rs.min_val << endl;
    cout << "  Max:    " << rs.max_val << endl;
    cout << "  Mean:   " << rs.mean << endl;
    cout << "  Median: " << rs.median << endl;
    cout << "  StdDev: " << rs.stddev << endl;
    PrintResponseHistogram(vKeys);

    // 7. 金字塔层级分布
    cout << "Octave distribution (level: count, ratio):" << endl;
    std::vector<int> octDist = ComputeOctaveDistribution(vKeys, nLevels);
    for (int lv = 0; lv < nLevels; lv++) {
        if (octDist[lv] > 0)
            cout << "  L" << lv << ": " << octDist[lv]
                 << " (" << 100.0 * octDist[lv] / actualNum << "%)" << endl;
    }

    // 8. 分布均匀性
    float distribution = ComputeDistribution(vKeys, img.rows, img.cols);
    cout << "Distribution uniformity (grid coverage): " << distribution * 100.0 << " %" << endl;

    // 9. 综合评价
    cout << "\n=== Summary ===" << endl;
    cout << "Extracted features: " << actualNum << endl;

    // 清晰度评价
    if (sharpness < 50)
        cout << "[WARN] Image appears blurry (sharpness=" << sharpness << ")" << endl;
    else
        cout << "[OK] Image sharpness is good" << endl;

    // 响应值评价：中位数过低说明角点质量差
    if (rs.median < 15)
        cout << "[WARN] Median response is low (" << rs.median << "), features may be unstable" << endl;
    else
        cout << "[OK] Response quality is acceptable" << endl;

    // 分布评价
    if (distribution < 0.5)
        cout << "[WARN] Feature distribution is poor (" << distribution*100 << "%)" << endl;
    else
        cout << "[OK] Feature distribution is good" << endl;

    // 10. 可视化：在图像上绘制特征点并保存
    cv::Mat imgColor;
    cv::cvtColor(img, imgColor, cv::COLOR_GRAY2BGR);
    for (const auto& kp : vKeys) {
        cv::circle(imgColor, kp.pt, 2, cv::Scalar(0, 255, 0), -1);  // 绿色实心圆
    }
    string outName = "orb_features.png";
    cv::imwrite(outName, imgColor);
    cout << "Saved feature image to " << outName << endl;

    // 11. 将指标写入文件，方便批量处理
    ofstream log("feature_stats.txt", ios::app);
    log << argv[1]
        << " features=" << actualNum
        << " sharpness=" << sharpness
        << " resp_mean=" << rs.mean
        << " resp_median=" << rs.median
        << " resp_stddev=" << rs.stddev
        << " resp_min=" << rs.min_val
        << " resp_max=" << rs.max_val
        << " distribution=" << distribution
        << endl;
    log.close();

    return 0;
}