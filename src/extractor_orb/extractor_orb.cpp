#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <include/ORBextractor.h>   // ORB-SLAM3 的特征提取器头文件

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
    orbExtractor(img, cv::Mat(), vKeys, descriptors);  // 第二个参数为 mask（空表示全图）

    int actualNum = vKeys.size();
    cout << "Extracted features: " << actualNum << endl;

    if (actualNum == 0) {
        cerr << "No feature extracted. Exiting." << endl;
        return 0;
    }

    // 5. 计算最小响应值（特征点按响应排序后，最后一个的响应即是最小值）
    //    ORBextractor 内部已经按响应降序排列，但为安全起见我们显式查找最小值
    float minResponse = vKeys[0].response;
    for (const auto& kp : vKeys) {
        if (kp.response < minResponse)
            minResponse = kp.response;
    }
    cout << "Minimum Harris response among kept points: " << minResponse << endl;

    // 6. 计算分布均匀性
    float distribution = ComputeDistribution(vKeys, img.rows, img.cols);
    cout << "Distribution uniformity (grid coverage): " << distribution * 100.0 << " %" << endl;

    // 7. 可视化：在图像上绘制特征点并保存
    cv::Mat imgColor;
    cv::cvtColor(img, imgColor, cv::COLOR_GRAY2BGR);
    for (const auto& kp : vKeys) {
        cv::circle(imgColor, kp.pt, 2, cv::Scalar(0, 255, 0), -1);  // 绿色实心圆
    }
    string outName = "orb_features.png";
    cv::imwrite(outName, imgColor);
    cout << "Saved feature image to " << outName << endl;

    // (可选) 将指标写入文件，方便批量处理
    ofstream log("feature_stats.txt", ios::app);
    log << argv[1] << " " << actualNum << " " << minResponse << " " << distribution << endl;
    log.close();

    return 0;
}