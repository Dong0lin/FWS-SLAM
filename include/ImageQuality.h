#ifndef IMAGEQUALITY_H
#define IMAGEQUALITY_H

#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <string>
#include <opencv2/opencv.hpp>

namespace ORB_SLAM3
{

struct ImageQualityMetrics
{
    int nFeatures;
    double sharpness;
    float respMean, respMedian, respStddev, respMin, respMax;
    std::vector<int> octaveDist;
    float distribution;
    int histogram[6];  // <10, 10-20, 20-50, 50-100, 100-200, >=200
};

// 匹配相关指标（在 Track() 后填充）
struct MatchingMetrics
{
    int totalKeypoints;       // 当前帧总特征点数
    int matchedPoints;        // 匹配到地图点的特征点数
    int inlierPoints;         // 内点数（优化后非 outlier）
    int outlierPoints;        // 外点数
    float matchRatio;         // 匹配率 = matchedPoints / totalKeypoints
    float inlierRatio;        // 内点率 = inlierPoints / matchedPoints
    float matchedRespMean;    // 匹配成功的特征点响应均值
    float matchedRespMedian;  // 匹配成功的特征点响应中位数
    float matchedRespStddev;  // 匹配成功的特征点响应标准差
};

// 图像清晰度：拉普拉斯方差
inline double ComputeSharpness(const cv::Mat& gray)
{
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar mu, sigma;
    cv::meanStdDev(lap, mu, sigma);
    return sigma.val[0] * sigma.val[0];
}

// 分区间统计响应值
inline void ComputeResponseHistogram(const std::vector<cv::KeyPoint>& vKeys,
                                      int histogram[6])
{
    std::fill(histogram, histogram + 6, 0);
    for (const auto& kp : vKeys) {
        float r = kp.response;
        if      (r < 10)  histogram[0]++;
        else if (r < 20)  histogram[1]++;
        else if (r < 50)  histogram[2]++;
        else if (r < 100) histogram[3]++;
        else if (r < 200) histogram[4]++;
        else              histogram[5]++;
    }
}

// 金字塔层级分布
inline void ComputeOctaveDistribution(const std::vector<cv::KeyPoint>& vKeys,
                                       int nLevels, std::vector<int>& dist)
{
    dist.assign(nLevels, 0);
    for (const auto& kp : vKeys) {
        int oct = kp.octave;
        if (oct >= 0 && oct < nLevels)
            dist[oct]++;
    }
}

// 分布均匀性：10x10 网格覆盖率
inline float ComputeDistribution(const std::vector<cv::KeyPoint>& vKeys,
                                  int rows, int cols)
{
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

// 一站式：计算所有图像质量指标
inline ImageQualityMetrics ComputeAllMetrics(const std::vector<cv::KeyPoint>& vKeys,
                                              const cv::Mat& img, int nLevels)
{
    ImageQualityMetrics m;
    m.nFeatures = static_cast<int>(vKeys.size());

    if (m.nFeatures == 0) {
        m.sharpness = 0;
        m.respMean = m.respMedian = m.respStddev = m.respMin = m.respMax = 0;
        m.octaveDist.assign(nLevels, 0);
        m.distribution = 0;
        std::fill(m.histogram, m.histogram + 6, 0);
        return m;
    }

    // 清晰度
    m.sharpness = ComputeSharpness(img);

    // 响应值统计
    std::vector<float> responses;
    responses.reserve(vKeys.size());
    for (const auto& kp : vKeys)
        responses.push_back(kp.response);
    std::sort(responses.begin(), responses.end());

    size_t n = responses.size();
    float sum = std::accumulate(responses.begin(), responses.end(), 0.0f);
    m.respMean = sum / n;
    m.respMedian = (n % 2 == 0) ?
        (responses[n/2 - 1] + responses[n/2]) / 2.0f : responses[n/2];
    m.respMin = responses.front();
    m.respMax = responses.back();

    float sq_sum = 0;
    for (float r : responses)
        sq_sum += (r - m.respMean) * (r - m.respMean);
    m.respStddev = std::sqrt(sq_sum / n);

    // 响应值分档
    ComputeResponseHistogram(vKeys, m.histogram);

    // 金字塔层级分布
    ComputeOctaveDistribution(vKeys, nLevels, m.octaveDist);

    // 分布均匀性
    m.distribution = ComputeDistribution(vKeys, img.rows, img.cols);

    return m;
}

// 将指标格式化为一行可解析的字符串（key=value 格式）
inline std::string FormatMetricsHeader()
{
    return "frame_id sharpness resp_mean resp_median resp_stddev "
           "resp_min resp_max nFeatures distribution "
           "hist_lt10 hist_10_20 hist_20_50 hist_50_100 hist_100_200 hist_gt200 "
           "total_kp matched_kp inlier_kp outlier_kp "
           "match_ratio inlier_ratio "
           "matched_resp_mean matched_resp_median matched_resp_stddev";
}

inline std::string FormatMetricsLine(int frameId, const ImageQualityMetrics& m,
                                     const MatchingMetrics& mm)
{
    char buf[768];
    snprintf(buf, sizeof(buf),
             "%d %.3f %.3f %.3f %.3f %.3f %.3f %d %.3f "
             "%d %d %d %d %d %d "
             "%d %d %d %d "
             "%.4f %.4f "
             "%.3f %.3f %.3f",
             frameId,
             m.sharpness,
             m.respMean, m.respMedian, m.respStddev,
             m.respMin, m.respMax,
             m.nFeatures,
             m.distribution,
             m.histogram[0], m.histogram[1], m.histogram[2],
             m.histogram[3], m.histogram[4], m.histogram[5],
             mm.totalKeypoints, mm.matchedPoints,
             mm.inlierPoints, mm.outlierPoints,
             mm.matchRatio, mm.inlierRatio,
             mm.matchedRespMean, mm.matchedRespMedian, mm.matchedRespStddev);
    return std::string(buf);
}

} // namespace ORB_SLAM3

#endif // IMAGEQUALITY_H
