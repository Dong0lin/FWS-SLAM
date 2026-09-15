/**
* This file is part of FWS-SLAM (ORB-SLAM3 based).
*
* 长短焦（Multi-focal）双目离线示例：
*   用法1（左右目分开）: ./stereo_multifocal vocab settings path_to_left path_to_right timestamps
*   用法2（单 mav0 根路径）: ./stereo_multifocal vocab settings path_to_mav0_root timestamps
*     ——自动找 <root>/cam0/data（左目/短焦）与 <root>/cam1/data（右目/长焦）
*   左右图像按文件名交集配对，时间戳优先取文件名（如 100.016639.jpg -> 100.016639 s），
*   也可用可选的第五个参数提供时间戳文件（每行一个时间戳，行数需与配对帧数一致）。
*/

#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>
#include<vector>
#include<string>
#include<set>
#include<cstdlib>
#include<execinfo.h>
#include<signal.h>
#include<unistd.h>
#include<sys/stat.h>
#include<dirent.h>
#include<cctype>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

// 崩溃诊断：段错误时打印调用栈（便于排查）
static void SegvHandler(int sig)
{
    void* buf[64];
    int n = backtrace(buf, 64);
    fprintf(stderr, "\n=== SIGSEGV (pid %d) ===\n", getpid());
    backtrace_symbols_fd(buf, n, 2);
    _exit(1);
}

void LoadImages(const string &strPathLeft, const string &strPathRight,
                vector<string> &vstrImageLeft, vector<string> &vstrImageRight,
                vector<double> &vTimestamps, const string &strTimestampFile);

// 图像目录解析：兼容 mav0 根（<base>/<side>/data）、mav0 上一级（<base>/mav0/<side>/data）、扁平目录
static string ResolveCamDir(const string& base, const string& side)
{
    struct stat st;
    const string p1 = base + "/" + side + "/data";
    if (stat(p1.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
        return p1;
    const string p2 = base + "/mav0/" + side + "/data";
    if (stat(p2.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
        return p2;
    return base;
}

static bool HasCamDir(const string& base, const string& side)
{
    struct stat st;
    return (stat((base + "/" + side + "/data").c_str(), &st) == 0 && S_ISDIR(st.st_mode)) ||
           (stat((base + "/mav0/" + side + "/data").c_str(), &st) == 0 && S_ISDIR(st.st_mode));
}

int main(int argc, char **argv)
{
    signal(SIGSEGV, SegvHandler);

    if(argc < 5)
    {
        cerr << endl
             << "Usage: ./stereo_multifocal path_to_vocabulary path_to_settings "
                "(path_to_left_images path_to_right_images | path_to_mav0_root) path_to_timestamps "
                "[max_frames] [viewer:0=none,1=pangolin,2=qt]" << endl;
        cerr << "Example:" << endl;
        cerr << "  ./stereo_multifocal Vocabulary/ORBvoc.txt settings.yaml left/ right/ time.txt" << endl;
        cerr << "  ./stereo_multifocal Vocabulary/ORBvoc.txt settings.yaml /path/to/mav0 time.txt  (自动找 cam0/cam1)" << endl;
        return 1;
    }

    vector<string> vstrImageLeft, vstrImageRight;
    vector<double> vTimestamps;
    string strLeft, strRight, strTimestampFile;

    // 第4参是普通文件 → 单 mav0 根路径格式（第3参=mav0根, 第4参=时间戳）
    // 否则为旧格式（第3参=左目, 第4参=右目, 第5参=时间戳）
    struct stat stArg4;
    if (stat(argv[4], &stArg4) == 0 && S_ISREG(stArg4.st_mode))
    {
        if (!HasCamDir(argv[3], "cam0"))
        {
            cerr << "ERROR: 未在 " << argv[3] << " 下找到 cam0/data（单路径模式需要 mav0 根目录）" << endl;
            return 1;
        }
        strLeft  = ResolveCamDir(argv[3], "cam0");
        strRight = ResolveCamDir(argv[3], "cam1");
        strTimestampFile = argv[4];
        cout << "[stereo_multifocal] 单路径模式: cam0(左)=" << strLeft
             << " cam1(右)=" << strRight << endl;
    }
    else
    {
        if (argc < 6)
        {
            cerr << "ERROR: 左右目分开模式需要 5 个参数（vocab settings left right timestamps）" << endl;
            return 1;
        }
        strLeft  = argv[3];
        strRight = argv[4];
        strTimestampFile = argv[5];
    }

    LoadImages(strLeft, strRight, vstrImageLeft, vstrImageRight, vTimestamps, strTimestampFile);

    const int nImages = (int)vstrImageLeft.size();
    if(nImages == 0)
    {
        cerr << "ERROR: no matched image pairs found!" << endl;
        return 1;
    }
    // 时间戳在 argv[tsIdx]：单路径格式 tsIdx=4，左右目分开格式 tsIdx=5；
    // 可选参数 [max_frames] [viewer] 紧跟其后，避免两种格式下错位
    const int tsIdx = (argc >= 5 && stat(argv[4], &stArg4) == 0 && S_ISREG(stArg4.st_mode)) ? 4 : 5;
    int nMaxFrames = nImages;
    if(argc >= tsIdx + 2)
        nMaxFrames = std::min(nImages, atoi(argv[tsIdx + 1]));
    ORB_SLAM3::System::eViewerType viewerType = ORB_SLAM3::System::VIEWER_PANGOLIN;
    if(argc >= tsIdx + 3)
        viewerType = (ORB_SLAM3::System::eViewerType)atoi(argv[tsIdx + 2]);
    cout << "Matched stereo image pairs: " << nImages << endl;
    cout << "Processing frames: " << nMaxFrames << endl;
    cout << "Viewer type: " << (int)viewerType << endl;

    // Create SLAM system
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::STEREO, viewerType);

    // 质量日志（image_quality.txt）：默认关闭；加 --save-quality 参数开启（每30帧采样）
    for(int i = 0; i < argc; i++)
        if(std::string(argv[i]) == "--save-quality")
            SLAM.GetTracker()->SetSaveQuality(true);

    // 分阶段耗时统计：设置环境变量 SLAM_TIMING=1 开启，结束时打印 ORB/等检测/Track 拆分
    if (const char* envTiming = std::getenv("SLAM_TIMING"))
        ORB_SLAM3::gEnableTimingStats = (std::string(envTiming) == "1" || std::string(envTiming) == "true");
    if (ORB_SLAM3::gEnableTimingStats)
        cout << "[Timing] 耗时统计已开启 (SLAM_TIMING)" << endl;

    vector<float> vTimesTrack;
    vTimesTrack.resize(nMaxFrames);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;

    double t_resize = 0.f;
    for(int ni = 0; ni < nMaxFrames; ni++)
    {
        cv::Mat imLeft = cv::imread(vstrImageLeft[ni], cv::IMREAD_UNCHANGED);
        cv::Mat imRight = cv::imread(vstrImageRight[ni], cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imLeft.empty() || imRight.empty())
        {
            cerr << "Failed to load image pair at index " << ni << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Pass the images to the SLAM system
        SLAM.TrackStereo(imLeft, imRight, tframe);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

        // Wait to load the next frame
        double T = 0;
        if(ni < nImages - 1)
            T = vTimestamps[ni+1] - tframe;
        else if(ni > 0)
            T = tframe - vTimestamps[ni-1];

        if(ttrack < T)
            usleep((T - ttrack) * 1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for(int ni = 0; ni < nMaxFrames; ni++)
        totaltime += vTimesTrack[ni];

    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nMaxFrames/2] << endl;
    cout << "mean tracking time: " << totaltime / nMaxFrames << endl;

    // Save camera trajectory
    // CameraTrajectory.txt 存 TUM 格式（时间戳+位姿，8列），与单目一致，evo_ape tum 可直接使用；
    // KITTI 格式另存一份，方便与 KITTI 工具链对接
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveTrajectoryKITTI("CameraTrajectoryKITTI.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

void LoadImages(const string &strPathLeft, const string &strPathRight,
                vector<string> &vstrImageLeft, vector<string> &vstrImageRight,
                vector<double> &vTimestamps, const string &strTimestampFile)
{
    // 时间戳文件每行一个时间戳（如 85.001631），对应两个文件夹中同名 .jpg
    // （left/85.001631.jpg、right/85.001631.jpg）
    ifstream fTs(strTimestampFile.c_str());
    if(!fTs.is_open())
    {
        cerr << "ERROR: cannot open timestamp file: " << strTimestampFile << endl;
        exit(-1);
    }

    string line;
    int nSkipped = 0;
    while(getline(fTs, line))
    {
        // 去掉首尾空白
        size_t b = line.find_first_not_of(" \t\r\n");
        if(b == string::npos)
            continue;
        size_t e = line.find_last_not_of(" \t\r\n");
        string ts = line.substr(b, e - b + 1);

        // 跳过空行/表头等非数字行
        if(ts.empty() || !isdigit((unsigned char)ts[0]))
            continue;

        string name = ts + ".jpg";
        string pL = strPathLeft + "/" + name;
        string pR = strPathRight + "/" + name;

        ifstream fL(pL.c_str()), fR(pR.c_str());
        if(!fL.good() || !fR.good())
        {
            nSkipped++;
            continue;
        }

        vstrImageLeft.push_back(pL);
        vstrImageRight.push_back(pR);
        vTimestamps.push_back(stod(ts));
    }

    if(vstrImageLeft.empty())
    {
        cerr << "ERROR: no matched image pairs found for timestamps in " << strTimestampFile << endl;
        exit(-1);
    }

    cout << "Loaded " << vstrImageLeft.size() << " stereo pairs from " << strTimestampFile
         << " (" << nSkipped << " timestamps skipped: no matching .jpg in both folders)" << endl;
}
