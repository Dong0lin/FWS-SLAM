#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// 辅助：从文本文件读取矩阵（每行空格分隔，空行/注释行以 # 开头跳过）
// ---------------------------------------------------------------------------
cv::Mat readMatrixFromFile(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "Error: Cannot open " << path << std::endl;
        return cv::Mat();
    }
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        // 跳过注释和空行
        if (line.empty() || line[0] == '#' || line[0] == '%') continue;
        std::istringstream iss(line);
        std::vector<double> vals;
        double v;
        while (iss >> v) vals.push_back(v);
        if (!vals.empty()) rows.push_back(vals);
    }
    if (rows.empty()) return cv::Mat();
    int ncols = static_cast<int>(rows[0].size());
    cv::Mat M(static_cast<int>(rows.size()), ncols, CV_64FC1);
    for (size_t i = 0; i < rows.size(); ++i)
        for (int j = 0; j < ncols; ++j)
            M.at<double>(static_cast<int>(i), j) = rows[i][j];
    return M;
}

// ---------------------------------------------------------------------------
// 使用标定参数进行"投影式"融合
// 通过将长焦图像每个像素反投影到世界平面 (Z=0) 再投影到短焦图像，
// 得到长焦→短焦的单应性矩阵，从而精确对齐融合区域。
// ---------------------------------------------------------------------------

/**
 * 长焦像素 → 世界平面 (Z=0) → 短焦像素 的单应性矩阵
 *
 * @param K_wide  短焦 3x3 内参矩阵
 * @param K_tele  长焦 3x3 内参矩阵
 * @param R       长焦→短焦的 3x3 旋转矩阵
 * @param T       长焦→短焦的 3x1 平移向量
 * @param plane_Z 假设的物平面 Z 坐标（在短焦坐标系中，正值表示前方）
 * @return        3x3 单应性矩阵 H，使 p_wide = H * p_tele（齐次坐标）
 */
cv::Mat computeHomographyFromPlane(const cv::Mat& K_wide,
                                   const cv::Mat& K_tele,
                                   const cv::Mat& R,
                                   const cv::Mat& T,
                                   double plane_Z = 1.0) {
    // 平面法向量 n = (0, 0, 1)，平面距离 d = plane_Z
    // H = K_wide * (R - T * n^T / d) * K_tele^{-1}
    cv::Mat n = (cv::Mat_<double>(3, 1) << 0, 0, 1);
    cv::Mat H = K_wide * (R - T * n.t() / plane_Z) * K_tele.inv();
    H /= H.at<double>(2, 2);   // 归一化使 H[2,2]=1
    return H;
}


// ---------------------------------------------------------------------------
// 羽化权重矩阵
// ---------------------------------------------------------------------------

/**
 * 为单应性变换后的长焦图像区域生成椭圆形羽化权重
 *
 * 这里不依赖"中心对称"假设，而是根据实际变换后的 mask 轮廓来生成：
 * 先对 mask 做距离变换得到每个像素到边界的距离，再用距离生成渐变权重。
 *
 * @param mask    单应性变换后的长焦有效区域 (0/255)
 * @param feather 羽化带宽（像素）
 * @return        0~1 的浮点权重图
 */
cv::Mat createFeatherFromMask(const cv::Mat& mask, int feather = 20) {
    cv::Mat dist;
    cv::distanceTransform(mask, dist, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    cv::Mat weight;
    // 权重 = min(dist / feather, 1.0)，内部为1，边界处渐变到0
    cv::threshold(dist, weight, feather, 1.0, cv::THRESH_TRUNC);
    weight = weight / static_cast<double>(feather);
    weight.setTo(1.0, mask == 0); // 容错：mask 外的区域（如果有赋值）权重清零
    return weight;
}


// ---------------------------------------------------------------------------
// 核心融合函数
// ---------------------------------------------------------------------------

/**
 * 基于双目标定参数的双焦图像融合
 *
 * @param wide      短焦（广角）图像
 * @param tele      长焦图像
 * @param K_wide    短焦 3x3 内参矩阵
 * @param K_tele    长焦 3x3 内参矩阵
 * @param R         长焦→短焦 3x3 旋转矩阵
 * @param T         长焦→短焦 3x1 平移向量
 * @param plane_Z   假设的场景平面深度（米），默认 500m 用于航拍远景
 * @param feather   羽化带宽（像素）
 * @return          融合后的图像
 */
cv::Mat fuseStereo(const cv::Mat& wide,
                   const cv::Mat& tele,
                   const cv::Mat& K_wide,
                   const cv::Mat& K_tele,
                   const cv::Mat& R,
                   const cv::Mat& T,
                   double plane_Z = 500.0,
                   int feather = 20) {
    int out_rows = wide.rows;
    int out_cols = wide.cols;

    // ---- 1. 计算单应性 H:  tele → wide ----
    cv::Mat H = computeHomographyFromPlane(K_wide, K_tele, R, T, plane_Z);

    // ---- 2. 用单应性将长焦图像 warp 到短焦坐标系 ----
    cv::Mat tele_warped;
    cv::warpPerspective(tele, tele_warped, H, cv::Size(out_cols, out_rows),
                        cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, cv::Scalar(0));

    // ---- 3. 生成长焦在短焦中的有效区域 mask（同样 warp 一张全白图） ----
    cv::Mat tele_mask;
    cv::Mat white(tele.size(), CV_8UC1, cv::Scalar(255));
    cv::warpPerspective(white, tele_mask, H, cv::Size(out_cols, out_rows),
                        cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

    // ---- 4. 根据 mask 生成羽化权重 ----
    cv::Mat weight = createFeatherFromMask(tele_mask, feather);

    // ---- 5. 加权融合 ----
    cv::Mat weight_3ch;
    if (wide.channels() == 3) {
        std::vector<cv::Mat> chs(3);
        for (int i = 0; i < 3; ++i) chs[i] = weight;
        cv::merge(chs, weight_3ch);
    } else {
        weight_3ch = weight;
    }

    cv::Mat weight_wide;
    cv::subtract(cv::Scalar::all(1.0), weight_3ch, weight_wide);

    cv::Mat wide_f, tele_f;
    wide.convertTo(wide_f, CV_32F);
    tele_warped.convertTo(tele_f, CV_32F);

    cv::multiply(wide_f, weight_wide, wide_f);
    cv::multiply(tele_f, weight_3ch, tele_f);

    cv::Mat blended;
    cv::add(wide_f, tele_f, blended);

    cv::Mat result;
    blended.convertTo(result, wide.type());
    return result;
}


// ---------------------------------------------------------------------------
// 无标定的简化融合（仅依赖焦距比，居中假设）
// ---------------------------------------------------------------------------
cv::Mat fuseSimple(const cv::Mat& wide, const cv::Mat& tele,
                   float ratio = 2.0f / 3.0f) {
    int out_rows = wide.rows;
    int out_cols = wide.cols;

    // 降采样
    int t_rows = static_cast<int>(out_rows * ratio);
    int t_cols = static_cast<int>(out_cols * ratio);
    cv::Mat tele_small;
    cv::resize(tele, tele_small, cv::Size(t_cols, t_rows), 0, 0, cv::INTER_LANCZOS4);

    // 居中放置
    cv::Mat tele_full = cv::Mat::zeros(out_rows, out_cols, tele.type());
    int ox = (out_cols - t_cols) / 2;
    int oy = (out_rows - t_rows) / 2;
    tele_small.copyTo(tele_full(cv::Rect(ox, oy, t_cols, t_rows)));

    // 矩形羽化权重
    cv::Mat weight(out_rows, out_cols, CV_32FC1, cv::Scalar(0.0f));
    const int fb = 20;
    int cx = out_cols / 2, cy = out_rows / 2;
    int hw = t_cols / 2, hh = t_rows / 2;
    for (int y = 0; y < out_rows; ++y) {
        float* row = weight.ptr<float>(y);
        for (int x = 0; x < out_cols; ++x) {
            int dx = std::abs(x - cx), dy = std::abs(y - cy);
            float wx = 1.f, wy = 1.f;
            if (dx > hw) { int d = dx - hw; wx = d >= fb ? 0.f : 1.f - (float)d / fb; }
            if (dy > hh) { int d = dy - hh; wy = d >= fb ? 0.f : 1.f - (float)d / fb; }
            row[x] = wx * wy;
        }
    }

    cv::Mat w3;
    if (wide.channels() == 3) {
        std::vector<cv::Mat> chs(3, weight);
        cv::merge(chs, w3);
    } else { w3 = weight; }

    cv::Mat w_wide;
    cv::subtract(cv::Scalar::all(1.0), w3, w_wide);

    cv::Mat wf, tf;
    wide.convertTo(wf, CV_32F);
    tele_full.convertTo(tf, CV_32F);
    cv::multiply(wf, w_wide, wf);
    cv::multiply(tf, w3, tf);
    cv::Mat b;
    cv::add(wf, tf, b);

    cv::Mat r;
    b.convertTo(r, wide.type());
    return r;
}


// ---------------------------------------------------------------------------
// 命令行入口
// ---------------------------------------------------------------------------
static void printUsage(const char* prog) {
    std::cout << "Usage:\n"
              << "  (a) 无标定简单模式:\n"
              << "      " << prog << " <wide> <tele> [output] [ratio]\n"
              << "\n"
              << "  (b) 双目标定模式:\n"
              << "      " << prog << " --calib <K_wide> <K_tele> <R> <T> "
              <<                  "<wide> <tele> [output] [Z]\n"
              << "\n"
              << "Arguments:\n"
              << "  wide     短焦图像路径\n"
              << "  tele     长焦图像路径\n"
              << "  output   输出图像路径 (默认 fused.jpg)\n"
              << "  ratio    焦距比 (默认 0.6667, 仅简单模式)\n"
              << "  K_wide   短焦 3x3 内参矩阵文件\n"
              << "  K_tele   长焦 3x3 内参矩阵文件\n"
              << "  R        3x3 旋转矩阵文件 (长焦→短焦)\n"
              << "  T        3x1 平移向量文件 (长焦→短焦)\n"
              << "  Z        场景平面深度 (米, 默认500, 仅标定模式)\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printUsage(argv[0]);
        return -1;
    }

    std::string mode = argv[1];

    if (mode == "--calib") {
        // ---------- 标定模式 ----------
        if (argc < 9) { printUsage(argv[0]); return -1; }

        std::string k_wide_path = argv[2];
        std::string k_tele_path = argv[3];
        std::string r_path      = argv[4];
        std::string t_path      = argv[5];
        std::string wide_path   = argv[6];
        std::string tele_path   = argv[7];
        std::string out_path    = (argc >= 9)  ? argv[8] : "fused.jpg";
        double      plane_Z     = (argc >= 10) ? std::stod(argv[9]) : 500.0;

        cv::Mat K_wide = readMatrixFromFile(k_wide_path);
        cv::Mat K_tele = readMatrixFromFile(k_tele_path);
        cv::Mat R      = readMatrixFromFile(r_path);
        cv::Mat T      = readMatrixFromFile(t_path);

        if (K_wide.empty() || K_tele.empty() || R.empty() || T.empty()) {
            std::cerr << "Error reading calibration files." << std::endl;
            return -1;
        }
        // 确保尺寸正确
        if (K_wide.rows != 3 || K_wide.cols != 3 ||
            K_tele.rows != 3 || K_tele.cols != 3 ||
            R.rows != 3 || R.cols != 3 ||
            T.rows != 3 || T.cols != 1) {
            std::cerr << "Error: matrix dimensions incorrect.\n"
                      << "  K_wide/K_tele should be 3x3, R 3x3, T 3x1" << std::endl;
            return -1;
        }

        // T 必须是 double 类型（computeHomographyFromPlane 中参与矩阵运算）
        if (T.type() != CV_64FC1) T.convertTo(T, CV_64FC1);
        if (K_wide.type() != CV_64FC1) K_wide.convertTo(K_wide, CV_64FC1);
        if (K_tele.type() != CV_64FC1) K_tele.convertTo(K_tele, CV_64FC1);
        if (R.type() != CV_64FC1) R.convertTo(R, CV_64FC1);

        cv::Mat wide = cv::imread(wide_path, cv::IMREAD_COLOR);
        cv::Mat tele = cv::imread(tele_path, cv::IMREAD_COLOR);
        if (wide.empty()) {
            std::cerr << "Error reading " << wide_path << std::endl;
            return -1;
        }
        if (tele.empty()) {
            std::cerr << "Error reading " << tele_path << std::endl;
            return -1;
        }

        std::cout << "Stereo fusion with depth Z = " << plane_Z << "m\n";
        cv::Mat result = fuseStereo(wide, tele, K_wide, K_tele, R, T, plane_Z);
        cv::imwrite(out_path, result);
        std::cout << "Saved: " << out_path << std::endl;

    } else {
        // ---------- 简单模式 ----------
        std::string wide_path = argv[1];
        std::string tele_path = argv[2];
        std::string out_path  = (argc >= 4) ? argv[3] : "fused.jpg";
        float       ratio     = (argc >= 5) ? std::stof(argv[4]) : (4.0f / 6.0f);

        cv::Mat wide = cv::imread(wide_path, cv::IMREAD_COLOR);
        cv::Mat tele = cv::imread(tele_path, cv::IMREAD_COLOR);
        if (wide.empty()) { std::cerr << "Error reading " << wide_path << std::endl; return -1; }
        if (tele.empty()) { std::cerr << "Error reading " << tele_path << std::endl; return -1; }

        if (wide.size() != tele.size()) {
            std::cout << "Resizing tele to match wide..." << std::endl;
            cv::resize(tele, tele, wide.size(), 0, 0, cv::INTER_LANCZOS4);
        }
        std::cout << "Simple fusion, ratio = " << ratio << std::endl;
        cv::Mat result = fuseSimple(wide, tele, ratio);
        cv::imwrite(out_path, result);
        std::cout << "Saved: " << out_path << std::endl;
    }

    return 0;
}
