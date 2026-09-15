/**
 * SLAM Runner（双目/长短焦） — 独立进程，读取双目图像运行 SLAM，
 * 结果写入共享内存供 Qt 界面显示（配套 Qt 的"语义SLAM-双目/语义SLAM-长短焦"模型）
 *
 * 用法:
 *   ./slam_runner_stereo <vocab> <settings> <mav0_root> <timestamp_file> [trajectory_name]
 *   （单路径：mav0 根目录，自动找 cam0=左目/短焦、cam1=右目/长焦；
 *     也兼容 <left_dir>,<right_dir> 逗号分隔）
 *
 * 时间戳文件格式 (与 ORB-SLAM3 mono_euroc 兼容):
 *   每行一个时间戳（FWS 秒 ~86 或 EuRoC 纳秒 ~1e17），同时是图片文件名（不含扩展名）
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <signal.h>
#include <cstring>
#include <cstdlib>

// POSIX 共享内存 & 信号量
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "SlamInterface.h"
#include "ShmData.h"

using namespace std;

// 全局指针用于信号处理
static ShmCtrl*   gShmCtrl   = nullptr;
static ShmFrame*  gShmFrame  = nullptr;
static ShmMap*    gShmMap    = nullptr;
static sem_t*     gSemFrame  = nullptr;
static bool       gRunning   = true;
static bool       gStepMode  = false;  // 步进模式是否激活

static void SignalHandler(int) {
    gRunning = false;
    // 唤醒可能阻塞在 sem_wait 的 Qt 进程
    if (gSemFrame) sem_post(gSemFrame);
}

// ─── 图像目录解析：兼容三种布局 ───
//   1) <base>/<side>/data          (mav0 根，side=cam0 左目 / cam1 右目)
//   2) <base>/mav0/<side>/data     (mav0 上一级)
//   3) <base>                      (扁平目录，直接放 xxx.jpg)
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

// ─── 双目/长短焦：时间戳文件每行一个时间戳（FWS 秒或 EuRoC 纳秒），
//     同时为左右目构造路径：<leftDir>/xxx.jpg、<rightDir>/xxx.jpg ───
static bool LoadStereoImages(const string& leftDir, const string& rightDir,
                             const string& timesPath,
                             vector<string>& vLeft, vector<string>& vRight,
                             vector<double>& vTimestamps)
{
    ifstream f(timesPath);
    if (!f.is_open()) {
        cerr << "[SlamRunner] 无法打开时间戳文件: " << timesPath << endl;
        return false;
    }

    string line;
    while (getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;

        stringstream ss;
        ss << line;
        string tok1, tok2;
        ss >> tok1;
        ss >> tok2;
        if (tok1.empty()) continue;

        // 两种时间戳格式：
        //   EuRoC: "<ns时间戳> <图像名>"        → 图像名取 tok2，时间戳取 tok1(纳秒)
        //   FWS:   "<秒时间戳>"（即图像名，如 86.002631）→ 图像名=时间戳=tok1(秒)
        string imgName;
        double t;
        if (!tok2.empty()) {
            imgName = tok2;
            t = atof(tok1.c_str());
        } else {
            imgName = tok1;
            t = atof(tok1.c_str());
        }
        vLeft.push_back(leftDir + "/" + imgName + ".jpg");
        vRight.push_back(rightDir + "/" + imgName + ".jpg");
        vTimestamps.push_back(t > 1e6 ? t * 1e-9 : t);
    }
    f.close();

    cout << "[SlamRunner] 加载 " << vLeft.size() << " 组双目图像 (左: "
         << leftDir << ", 右: " << rightDir << ")" << endl;
    if (vLeft.empty()) return false;

    ifstream testL(vLeft[0]), testR(vRight[0]);
    if (!testL.good() || !testR.good())
        cerr << "[SlamRunner] 警告: 首组图像不存在: " << vLeft[0] << " / " << vRight[0] << endl;

    return true;
}

// ─── 处理来自 Qt 的命令 ───
static void ProcessCommands(ORB_SLAM3::SlamInterface& slam)
{
    ShmCtrl& ctrl = *gShmCtrl;

    // 仅定位模式
    if (ctrl.cmd_localization != 0) {
        bool enable = (ctrl.cmd_localization > 0);
        slam.SetLocalizationMode(enable);
        ctrl.ack_localization = ctrl.cmd_localization;
        ctrl.cmd_localization = 0;
    }

    // 步进模式 (仅更新主循环中的 gStepMode 标志)
    if (ctrl.cmd_step_by_step != 0) {
        gStepMode = (ctrl.cmd_step_by_step > 0);
        ctrl.ack_step_by_step = ctrl.cmd_step_by_step;
        ctrl.cmd_step_by_step = 0;
        cout << "[SlamRunner] 步进模式: " << (gStepMode ? "开启" : "关闭") << endl;
    }

    // 重置
    if (ctrl.cmd_reset) {
        slam.ResetActiveMap();
        ctrl.cmd_reset = 0;
    }

    // 可视化模式切换
    if (ctrl.cmd_vis_mode != 0) {
        // Qt 写入: -1=原图(短焦), 1=动态一致性, 2=长短焦(右目长焦)
        int mode = (ctrl.cmd_vis_mode == 2) ? 2 : (ctrl.cmd_vis_mode > 0 ? 1 : 0);
        slam.SetVisualizationMode(mode);
        ctrl.cmd_vis_mode = 0;  // 确认已处理
    }

    // 3D 框检测开关
    static bool lastDraw3dBox = false;
    bool draw3d = (ctrl.draw_3dbox != 0);
    if (draw3d != lastDraw3dBox) {
        slam.SetEnable3DBoxDetection(draw3d);
        lastDraw3dBox = draw3d;
        cout << "[SlamRunner] 3D框检测: " << (draw3d ? "开启" : "关闭") << endl;
    }
}

// ─── 将 SLAM 状态写入共享内存 ───
static void WriteShm(ORB_SLAM3::SlamInterface& slam, bool newFrame)
{
    ShmCtrl& ctrl = *gShmCtrl;
    (void)newFrame;

    // 相机位姿
    Eigen::Matrix4f Twc = slam.GetCameraPose();
    for (int i = 0; i < 16; i++)
        ctrl.camera_pose[i] = Twc(i % 4, i / 4);  // 列主序

    // 状态
    ctrl.tracking_state = slam.GetTrackingState();
    ctrl.map_points_cnt  = slam.GetMapPointsCount();
    ctrl.keyframes_cnt   = slam.GetKeyFramesCount();

    // 图像帧
    int w = 0, h = 0;
    const unsigned char* img = slam.GetCurrentFrame(w, h);
    if (img && w > 0 && h > 0) {
        ctrl.image_width  = w;
        ctrl.image_height = h;
        size_t bytes = (size_t)w * h * 3;
        if (bytes <= SHM_MAX_FRAME_BYTES) {
            memcpy(gShmFrame->data, img, bytes);
            gShmFrame->frame_id = ctrl.frame_counter;

            // 通知 Qt：新帧就绪
            sem_post(gSemFrame);
        }
    }

    // ── 写入 3D 地图数据 (ShmMap) ──
    if (gShmMap) {
        ShmMap& map = *gShmMap;
        map.map_frame_id = ctrl.frame_counter;

        // 追加相机位置到轨迹环形缓冲
        Eigen::Vector3f pos = Twc.block<3,1>(0,3);
        int writeIdx = map.traj_start + map.traj_count;
        int idx = writeIdx % MAX_TRAJ_POINTS;
        map.traj[idx][0] = pos.x();
        map.traj[idx][1] = pos.y();
        map.traj[idx][2] = pos.z();

        if (map.traj_count < MAX_TRAJ_POINTS)
            map.traj_count++;
        else
            map.traj_start = (map.traj_start + 1) % MAX_TRAJ_POINTS;

        // 更新关键帧位姿 (每10帧或数量变化时)
        int kfCount = slam.GetKeyFramesCount();
        static int lastKfWriteFrame = -100;
        if (kfCount != map.kf_count || ctrl.frame_counter - lastKfWriteFrame > 10) {
            map.kf_count = kfCount;
            int n = slam.GetAllKeyFramePoses(&map.kf_poses[0][0], map.kf_status, MAX_KF_POSES);
            if (n < map.kf_count) map.kf_count = n;
            lastKfWriteFrame = ctrl.frame_counter;
        }

        // 更新地图点 (每30帧更新一次)
        static int lastMpWriteFrame = -100;
        if (ctrl.frame_counter - lastMpWriteFrame > 30) {
            int n = slam.GetAllMapPoints(&map.map_points[0][0], &map.mp_colors[0][0], MAX_MAP_POINTS);
            map.mp_count = n;
            lastMpWriteFrame = ctrl.frame_counter;
        }

        // 更新平面参数 (每30帧，不依赖3D框开关)
        static int lastPlaneWriteFrame = -100;
        if (ctrl.frame_counter - lastPlaneWriteFrame > 30) {
            // 通过 GetPersistentBoxes 获取平面参数 (即使没有框)
            slam.GetPersistentBoxes(nullptr, map.plane_normal, &map.plane_offset, 0);
            lastPlaneWriteFrame = ctrl.frame_counter;
        }

        // 更新 3D 框 (仅在开启时写入，独立计数器)
        static int lastBoxWriteFrame = -100;
        if (ctrl.draw_3dbox && ctrl.frame_counter - lastBoxWriteFrame > 30) {
            int n = slam.GetPersistentBoxes(&map.boxes[0][0], nullptr, nullptr, MAX_3D_BOXES);
            map.box_count = n;
            lastBoxWriteFrame = ctrl.frame_counter;
            if (n > 0) cout << "[SlamRunner] 写入 " << n << " 个3D框到ShmMap" << endl;
        }
    }
}

// ═══════════════════════════════════════════════════════════════
int main(int argc, char** argv)
{
    if (argc < 5) {
        cerr << "用法: " << argv[0]
             << " <vocab> <settings> <mav0_root> <timestamp_file> [trajectory_name]" << endl;
        cerr << "  mav0_root：自动找 cam0=左目/短焦、cam1=右目/长焦；也兼容 <left>,<right> 逗号分隔" << endl;
        return 1;
    }

    string vocabPath    = argv[1];
    string settingsPath = argv[2];
    string imagePath    = argv[3];
    string timesPath    = argv[4];

    // 解析左右目目录：单一路径（mav0 根，自动找 cam0/cam1）或逗号分隔（显式左右目录）
    string leftDir, rightDir;
    {
        size_t comma = imagePath.find(',');
        if (comma != string::npos) {
            leftDir  = ResolveCamDir(imagePath.substr(0, comma), "cam0");
            rightDir = ResolveCamDir(imagePath.substr(comma + 1), "cam1");
        } else if (HasCamDir(imagePath, "cam0")) {
            leftDir  = ResolveCamDir(imagePath, "cam0");
            rightDir = ResolveCamDir(imagePath, "cam1");
            cout << "[SlamRunner] 检测到双目/长短焦目录: cam0(左)=" << leftDir
                 << " cam1(右)=" << rightDir << endl;
        } else {
            cerr << "[SlamRunner] 错误: 未在 " << imagePath
                 << " 下找到 cam0/data（需要 mav0 根目录，或 <left>,<right>）" << endl;
            return 1;
        }
    }

    // 轨迹保存到 slam_runner 所在目录，同时切换工作目录使 image_quality.txt 等也落在此处
    string trajDir;
    {
        string exePath = argv[0];
        size_t slash = exePath.find_last_of('/');
        trajDir = (slash != string::npos) ? exePath.substr(0, slash) : ".";
    }
    chdir(trajDir.c_str());
    cout << "[SlamRunner] 工作目录 & 轨迹输出: " << trajDir << endl;

    // ── 加载图像列表 ──
    vector<string> vImagesL, vImagesR;
    vector<double> vTimestamps;
    if (!LoadStereoImages(leftDir, rightDir, timesPath, vImagesL, vImagesR, vTimestamps))
        return 1;

    // ── 信号处理 ──
    signal(SIGINT,  SignalHandler);
    signal(SIGTERM, SignalHandler);

    // ── 创建/打开 共享内存 ──
    int fdCtrl = shm_open(SHM_NAME_CTRL, O_CREAT | O_RDWR, 0666);
    if (fdCtrl < 0) {
        perror("[SlamRunner] shm_open ctrl");
        return 1;
    }
    ftruncate(fdCtrl, sizeof(ShmCtrl));
    gShmCtrl = (ShmCtrl*)mmap(nullptr, sizeof(ShmCtrl),
                              PROT_READ | PROT_WRITE, MAP_SHARED, fdCtrl, 0);
    if (gShmCtrl == MAP_FAILED) {
        perror("[SlamRunner] mmap ctrl");
        return 1;
    }
    close(fdCtrl);
    memset(gShmCtrl, 0, sizeof(ShmCtrl));

    int fdFrame = shm_open(SHM_NAME_FRAME, O_CREAT | O_RDWR, 0666);
    if (fdFrame < 0) {
        perror("[SlamRunner] shm_open frame");
        return 1;
    }
    ftruncate(fdFrame, sizeof(ShmFrame));
    gShmFrame = (ShmFrame*)mmap(nullptr, sizeof(ShmFrame),
                                PROT_READ | PROT_WRITE, MAP_SHARED, fdFrame, 0);
    if (gShmFrame == MAP_FAILED) {
        perror("[SlamRunner] mmap frame");
        return 1;
    }
    close(fdFrame);
    memset(gShmFrame, 0, sizeof(ShmFrame));

    // ── 创建信号量 ──
    // 先 unlink 再 open 避免残留
    sem_unlink(SEM_NAME_FRAME);
    gSemFrame = sem_open(SEM_NAME_FRAME, O_CREAT | O_EXCL, 0666, 0);
    if (gSemFrame == SEM_FAILED) {
        perror("[SlamRunner] sem_open");
        return 1;
    }

    // ── 创建 3D 地图共享内存 ──
    int fdMap = shm_open(SHM_NAME_MAP, O_CREAT | O_RDWR, 0666);
    if (fdMap < 0) {
        perror("[SlamRunner] shm_open map");
        return 1;
    }
    ftruncate(fdMap, sizeof(ShmMap));
    gShmMap = (ShmMap*)mmap(nullptr, sizeof(ShmMap),
                            PROT_READ | PROT_WRITE, MAP_SHARED, fdMap, 0);
    close(fdMap);
    if (gShmMap == MAP_FAILED) {
        perror("[SlamRunner] mmap map");
        return 1;
    }
    memset(gShmMap, 0, sizeof(ShmMap));

    // ── 初始化 SLAM ──
    cout << "[SlamRunner] 初始化 SLAM..." << endl;
    ORB_SLAM3::SlamInterface slam;
    if (!slam.Init(vocabPath, settingsPath, ORB_SLAM3::SlamInterface::STEREO)) {
        cerr << "[SlamRunner] SLAM 初始化失败" << endl;
        return 1;
    }
    gShmCtrl->slam_ready = 1;
    float imageScale = slam.GetImageScale();
    cout << "[SlamRunner] SLAM 就绪, 图像缩放: " << imageScale << endl;

    // ── 主循环 ──
    vector<float> vTimesTrack;
    size_t totalImages = vImagesL.size();

    for (size_t i = 0; i < totalImages && gRunning; i++) {
        // 检查关闭命令
        if (gShmCtrl->cmd_shutdown) {
            gRunning = false;
            break;
        }

        // 处理 Qt 命令
        ProcessCommands(slam);

        // 步进模式：等待 Qt 发送 step 命令
        if (gStepMode) {
            // 先清除可能残留的旧 step 命令
            gShmCtrl->cmd_step = 0;
            // cout << "[SlamRunner] 步进等待中 (帧 " << i << ")..." << endl;

            while (gStepMode && gRunning && !gShmCtrl->cmd_shutdown) {
                ProcessCommands(slam);
                if (gShmCtrl->cmd_step) {
                    gShmCtrl->cmd_step = 0;
                    // cout << "[SlamRunner] 步进一帧" << endl;
                    break;
                }
                if (!gStepMode) {
                    cout << "[SlamRunner] 步进模式已关闭，恢复连续运行" << endl;
                    break;
                }
                usleep(10000);
            }
            if (!gRunning || gShmCtrl->cmd_shutdown) break;
        }

        // 读取左右目图像
        cv::Mat imL = cv::imread(vImagesL[i], cv::IMREAD_UNCHANGED);
        cv::Mat imR = cv::imread(vImagesR[i], cv::IMREAD_UNCHANGED);
        if (imL.empty() || imR.empty()) {
            cerr << "[SlamRunner] 读取图像失败: " << vImagesL[i]
                 << " / " << vImagesR[i] << endl;
            continue;
        }

        // 缩放
        if (imageScale != 1.f) {
            int w = imL.cols * imageScale;
            int h = imL.rows * imageScale;
            cv::resize(imL, imL, cv::Size(w, h));
            cv::resize(imR, imR, cv::Size(w, h));
        }

        // 跟踪
        auto t1 = chrono::steady_clock::now();
        slam.TrackStereo(imL.data, imR.data, imL.cols, imL.rows, vTimestamps[i]);
        auto t2 = chrono::steady_clock::now();

        double ttrack = chrono::duration<double>(t2 - t1).count();
        vTimesTrack.push_back(ttrack);

        // 写共享内存
        gShmCtrl->frame_counter++;
        WriteShm(slam, true);

        // 帧率控制 (如果处理比实时快)
        if (i + 1 < totalImages) {
            double T = vTimestamps[i+1] - vTimestamps[i];
            if (ttrack < T)
                usleep((T - ttrack) * 1e6);
        }
    }

    // ── 清理 ──
    cout << "[SlamRunner] 正在关闭..." << endl;
    gShmCtrl->slam_ready = 0;
    gShmCtrl->tracking_state = -1;

    // 通知 Qt 进程结束
    sem_post(gSemFrame);

    slam.SaveTrajectory(trajDir);  // 必须在 Shutdown 前保存
    slam.Shutdown();

    // 计算平均帧率
    if (!vTimesTrack.empty()) {
        double totalTrack = 0;
        for (double t : vTimesTrack) totalTrack += t;
        double meanTrack = totalTrack / vTimesTrack.size() * 1000;
        cout << "[SlamRunner] 平均跟踪时间: " << meanTrack << " ms" << endl;
    }
    cout << "[SlamRunner] 轨迹已保存: " << trajDir << endl;

    // 关闭共享资源
    sem_close(gSemFrame);
    sem_unlink(SEM_NAME_FRAME);
    munmap(gShmCtrl, sizeof(ShmCtrl));
    munmap(gShmFrame, sizeof(ShmFrame));
    munmap(gShmMap, sizeof(ShmMap));
    shm_unlink(SHM_NAME_CTRL);
    shm_unlink(SHM_NAME_FRAME);
    shm_unlink(SHM_NAME_MAP);

    cout << "[SlamRunner] 退出" << endl;
    return 0;
}
