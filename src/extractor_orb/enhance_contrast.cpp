#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <dirent.h>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>

using namespace std;

void EnhanceContrast(const cv::Mat& src, cv::Mat& dst, float alpha, int clipLimit, int tileSize) {
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, cv::Size(tileSize, tileSize));

    if (src.channels() >= 3) {
        // 彩色: 转 Lab, 只对 L 通道做 CLAHE+对比度, 保持色彩
        cv::Mat lab;
        cv::cvtColor(src, lab, cv::COLOR_BGR2Lab);

        vector<cv::Mat> channels;
        cv::split(lab, channels);

        clahe->apply(channels[0], channels[0]);
        channels[0].convertTo(channels[0], -1, alpha, 0);

        cv::merge(channels, lab);
        cv::cvtColor(lab, dst, cv::COLOR_Lab2BGR);
    } else {
        // 灰度: 直接处理
        clahe->apply(src, dst);
        dst.convertTo(dst, -1, alpha, 0);
    }
}

// 计算拉普拉斯清晰度
double ComputeSharpness(const cv::Mat& gray) {
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar mu, sigma;
    cv::meanStdDev(lap, mu, sigma);
    return sigma.val[0] * sigma.val[0];
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <input_folder> [alpha=1.2] [clipLimit=2.0] [tileSize=8]" << endl;
        cerr << "  alpha:      线性对比度系数, 1.0=不变, 1.3=增强30% (默认1.2)" << endl;
        cerr << "  clipLimit:  CLAHE对比度限制阈值 (默认2.0, 越大对比度越强)" << endl;
        cerr << "  tileSize:   CLAHE分块大小 (默认8, 越小局部对比度越强)" << endl;
        cerr << "  输出文件夹: <input_folder>_enhanced/" << endl;
        return -1;
    }

    string inputDir = argv[1];
    float alpha = (argc >= 3) ? stof(argv[2]) : 1.2f;
    int clipLimit = (argc >= 4) ? stoi(argv[3]) : 2;
    int tileSize = (argc >= 5) ? stoi(argv[4]) : 8;

    // 移除末尾的 /
    while (!inputDir.empty() && inputDir.back() == '/')
        inputDir.pop_back();

    string outputDir = inputDir + "_enhanced";
    if (mkdir(outputDir.c_str(), 0755) != 0 && errno != EEXIST) {
        cerr << "Failed to create output directory: " << outputDir << endl;
        return -1;
    }

    vector<string> filenames;
    {
        DIR* dir = opendir(inputDir.c_str());
        if (!dir) {
            cerr << "Cannot open directory: " << inputDir << endl;
            return -1;
        }
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            string name(entry->d_name);
            if (name == "." || name == "..") continue;
            size_t dot = name.find_last_of('.');
            if (dot == string::npos) continue;
            string ext = name.substr(dot);
            if (ext == ".jpg" || ext == ".JPG" || ext == ".jpeg" || ext == ".JPEG" ||
                ext == ".png" || ext == ".PNG" || ext == ".bmp" || ext == ".BMP") {
                filenames.push_back(inputDir + "/" + name);
            }
        }
        closedir(dir);
    }
    sort(filenames.begin(), filenames.end());

    if (filenames.empty()) {
        cerr << "No image files found in " << inputDir << endl;
        return -1;
    }

    cout << "Input:  " << inputDir << " (" << filenames.size() << " images)" << endl;
    cout << "Output: " << outputDir << endl;
    cout << "Params: alpha=" << alpha << " clipLimit=" << clipLimit << " tileSize=" << tileSize << endl;
    cout << "Processing..." << endl;

    double totalSharpnessSrc = 0, totalSharpnessDst = 0;
    int count = 0;

    for (size_t i = 0; i < filenames.size(); i++) {
        cv::Mat img = cv::imread(filenames[i], cv::IMREAD_COLOR);
        if (img.empty()) {
            cerr << "  skip (empty): " << filenames[i] << endl;
            continue;
        }

        // 清晰度用灰度算
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        double sharpSrc = ComputeSharpness(gray);

        cv::Mat enhanced;
        EnhanceContrast(img, enhanced, alpha, clipLimit, tileSize);

        double sharpDst = ComputeSharpness(enhanced);
        totalSharpnessSrc += sharpSrc;
        totalSharpnessDst += sharpDst;
        count++;

        // 提取文件名
        string fname = filenames[i].substr(filenames[i].find_last_of("/\\") + 1);
        cv::imwrite(outputDir + "/" + fname, enhanced);

        if (count % 100 == 0 || i == filenames.size() - 1) {
            cout << "  [" << count << "/" << filenames.size() << "] "
                 << "avg_sharpness: " << totalSharpnessSrc / count << " -> "
                 << totalSharpnessDst / count
                 << " (+" << (totalSharpnessDst / totalSharpnessSrc - 1.0) * 100 << "%)" <<endl;
        }
    }

    cout << "\n===== Summary =====" << endl;
    cout << "Frames:    " << count << endl;
    cout << "Params:    alpha=" << alpha << " clipLimit=" << clipLimit << " tileSize=" << tileSize << endl;
    cout << "Sharpness: " << totalSharpnessSrc / count << " -> " << totalSharpnessDst / count
         << " (+" << (totalSharpnessDst / totalSharpnessSrc - 1.0) * 100 << "%)" << endl;
    cout << "Output:    " << outputDir << endl;

    return 0;
}
