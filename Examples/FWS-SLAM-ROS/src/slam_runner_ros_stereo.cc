/**
 * SLAM Runner ROS — 长短焦双目语义 SLAM 版
 *
 * ROS 订阅长短焦拼接相机图像（2560×720：左半=短焦、右半=长焦），
 * 裁剪左右两半运行双目长短焦 SLAM，写入共享内存供 Qt 界面显示。
 * 支持 3D 框检测、动态一致性可视化、平面显示 等语义功能。
 *
 * 用法:
 *   rosrun FWS-SLAM-ROS slam_runner_ros_stereo <vocab> <settings>
 *
 * 参数:
 *   ~image_topic (string, default: "/usb_cam/image_raw")
 */

#include <iostream>
#include <chrono>
#include <thread>
#include <signal.h>
#include <cstring>
#include <cstdlib>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>

#include "SlamInterface.h"
#include "ShmData.h"

using namespace std;

// 共享内存全局变量
static ShmCtrl*   gShmCtrl   = nullptr;
static ShmFrame*  gShmFrame  = nullptr;
static ShmMap*    gShmMap    = nullptr;
static sem_t*     gSemFrame  = nullptr;
static bool       gRunning   = true;
static bool       gStepMode  = false;
static ORB_SLAM3::SlamInterface* gSlam = nullptr;

static void SignalHandler(int) {
    gRunning = false;
    if (gSemFrame) sem_post(gSemFrame);
}

static void ProcessCommands()
{
    if (!gShmCtrl || !gSlam) return;
    ShmCtrl& ctrl = *gShmCtrl;

    // 仅定位模式
    if (ctrl.cmd_localization != 0) {
        bool enable = (ctrl.cmd_localization > 0);
        gSlam->SetLocalizationMode(enable);
        ctrl.ack_localization = ctrl.cmd_localization;
        ctrl.cmd_localization = 0;
    }

    // 步进模式
    if (ctrl.cmd_step_by_step != 0) {
        gStepMode = (ctrl.cmd_step_by_step > 0);
        ctrl.ack_step_by_step = ctrl.cmd_step_by_step;
        ctrl.cmd_step_by_step = 0;
        cout << "[SlamRunnerROS] 步进模式: " << (gStepMode ? "开启" : "关闭") << endl;
    }

    // 重置
    if (ctrl.cmd_reset) {
        gSlam->ResetActiveMap();
        ctrl.cmd_reset = 0;
    }

    // 可视化模式切换 (原图/短焦 / 动态一致性 / 长短焦-右目长焦)
    if (ctrl.cmd_vis_mode != 0) {
        int mode = (ctrl.cmd_vis_mode == 2) ? 2 : (ctrl.cmd_vis_mode > 0 ? 1 : 0);
        gSlam->SetVisualizationMode(mode);
        ctrl.cmd_vis_mode = 0;
    }

    // 3D 框检测开关
    static bool lastDraw3dBox = false;
    bool draw3d = (ctrl.draw_3dbox != 0);
    if (draw3d != lastDraw3dBox) {
        gSlam->SetEnable3DBoxDetection(draw3d);
        lastDraw3dBox = draw3d;
        cout << "[SlamRunnerROS] 3D框检测: " << (draw3d ? "开启" : "关闭") << endl;
    }
}

static void WriteShm(ORB_SLAM3::SlamInterface& slam)
{
    ShmCtrl& ctrl = *gShmCtrl;

    // 相机位姿
    Eigen::Matrix4f Twc = slam.GetCameraPose();
    for (int i = 0; i < 16; i++)
        ctrl.camera_pose[i] = Twc(i % 4, i / 4);

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
            sem_post(gSemFrame);
        }
    }

    // ── 3D 地图数据 (ShmMap) ──
    if (gShmMap) {
        ShmMap& map = *gShmMap;
        map.map_frame_id = ctrl.frame_counter;

        // 轨迹环形缓冲
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

        // 关键帧位姿 (每10帧或数量变化时)
        int kfCount = slam.GetKeyFramesCount();
        static int lastKfWriteFrame = -100;
        if (kfCount != map.kf_count || ctrl.frame_counter - lastKfWriteFrame > 10) {
            map.kf_count = kfCount;
            int n = slam.GetAllKeyFramePoses(&map.kf_poses[0][0], map.kf_status, MAX_KF_POSES);
            if (n < map.kf_count) map.kf_count = n;
            lastKfWriteFrame = ctrl.frame_counter;
        }

        // 地图点 (每30帧，含语义颜色)
        static int lastMpWriteFrame = -100;
        if (ctrl.frame_counter - lastMpWriteFrame > 30) {
            int n = slam.GetAllMapPoints(&map.map_points[0][0], &map.mp_colors[0][0], MAX_MAP_POINTS);
            map.mp_count = n;
            lastMpWriteFrame = ctrl.frame_counter;
        }

        // 平面参数 (每30帧)
        static int lastPlaneWriteFrame = -100;
        if (ctrl.frame_counter - lastPlaneWriteFrame > 30) {
            slam.GetPersistentBoxes(nullptr, map.plane_normal, &map.plane_offset, 0);
            lastPlaneWriteFrame = ctrl.frame_counter;
        }

        // 3D 框 (仅在开启时写入)
        static int lastBoxWriteFrame = -100;
        if (ctrl.draw_3dbox && ctrl.frame_counter - lastBoxWriteFrame > 30) {
            int n = slam.GetPersistentBoxes(&map.boxes[0][0], nullptr, nullptr, MAX_3D_BOXES);
            map.box_count = n;
            lastBoxWriteFrame = ctrl.frame_counter;
        }
    }
}

// ─── 图像回调 ───
class ImageGrabber {
public:
    ImageGrabber(ORB_SLAM3::SlamInterface* pSlam)
        : mpSlam(pSlam) {}

    void GrabImage(const sensor_msgs::ImageConstPtr& msg) {
        if (!gRunning || !mpSlam) return;

        ProcessCommands();

        // 步进模式：丢弃帧（ROS 回调不能阻塞，步进由主线程处理）
        if (gStepMode) {
            return;
        }

        // 强制转成 BGR8（usb_cam mjpeg 常发布 rgb8，直接按 BGR 处理会导致红蓝互换）
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat imgFull = cv_ptr->image;

        // 长短焦拼接图像 2560×720 → 左半=短焦 1280×720、右半=长焦 1280×720；
        // 若输入本身是单眼图（宽高比<1.5）则左右各用同一张（退化保护）
        cv::Mat imL, imR;
        if (imgFull.cols > imgFull.rows * 1.5f) {
            int halfW = imgFull.cols / 2;
            imL = imgFull(cv::Rect(0, 0, halfW, imgFull.rows)).clone();
            imR = imgFull(cv::Rect(halfW, 0, imgFull.cols - halfW, imgFull.rows)).clone();
        } else {
            imL = imgFull.clone();
            imR = imgFull.clone();
        }

        // 临时补偿：MF_VSHIFT=<px> 把右半图整体上移（实机拼接固定垂直偏移，实测约 32px）。
        // 默认关闭，正式使用请重新标定。
        if (const char* ev = getenv("MF_VSHIFT")) {
            int vy = atoi(ev);
            if (vy > 0 && vy < imR.rows) {
                cv::Mat shifted(imR.rows, imR.cols, imR.type(), cv::Scalar(0, 0, 0));
                imR(cv::Rect(0, vy, imR.cols, imR.rows - vy))
                   .copyTo(shifted(cv::Rect(0, 0, imR.cols, imR.rows - vy)));
                imR = shifted;
            }
        }

        // 通道适配（SlamInterface 期望 CV_8UC3）
        auto toBGR = [](cv::Mat& m) {
            if (m.channels() == 1)      cv::cvtColor(m, m, cv::COLOR_GRAY2BGR);
            else if (m.channels() == 4) cv::cvtColor(m, m, cv::COLOR_BGRA2BGR);
        };
        toBGR(imL);
        toBGR(imR);

        mpSlam->TrackStereo(imL.data, imR.data, imL.cols, imL.rows, msg->header.stamp.toSec());

        gShmCtrl->frame_counter++;
        WriteShm(*mpSlam);
    }

private:
    ORB_SLAM3::SlamInterface* mpSlam;
};

// ═══════════════════════════════════════════════════════════════
int main(int argc, char** argv)
{
    if (argc < 3) {
        cerr << "用法: rosrun FWS-SLAM-ROS slam_runner_ros_stereo <vocab> <settings>" << endl;
        return 1;
    }

    ros::init(argc, argv, "slam_runner_ros_stereo");
    ros::start();

    string vocabPath    = argv[1];
    string settingsPath = argv[2];

    // 轨迹目录：可执行文件所在目录
    string trajDir;
    {
        string exePath = argv[0];
        size_t slash = exePath.find_last_of('/');
        trajDir = (slash != string::npos) ? exePath.substr(0, slash) : ".";
    }
    chdir(trajDir.c_str());
    cout << "[SlamRunnerROS] 工作目录: " << trajDir << endl;

    signal(SIGINT,  SignalHandler);
    signal(SIGTERM, SignalHandler);

    // ── 创建共享内存 ──
    int fdCtrl = shm_open(SHM_NAME_CTRL, O_CREAT | O_RDWR, 0666);
    if (fdCtrl < 0) { perror("[SlamRunnerROS] shm_open ctrl"); return 1; }
    ftruncate(fdCtrl, sizeof(ShmCtrl));
    gShmCtrl = (ShmCtrl*)mmap(nullptr, sizeof(ShmCtrl), PROT_READ | PROT_WRITE, MAP_SHARED, fdCtrl, 0);
    if (gShmCtrl == MAP_FAILED) { perror("[SlamRunnerROS] mmap ctrl"); return 1; }
    close(fdCtrl);
    memset(gShmCtrl, 0, sizeof(ShmCtrl));

    int fdFrame = shm_open(SHM_NAME_FRAME, O_CREAT | O_RDWR, 0666);
    if (fdFrame < 0) { perror("[SlamRunnerROS] shm_open frame"); return 1; }
    ftruncate(fdFrame, sizeof(ShmFrame));
    gShmFrame = (ShmFrame*)mmap(nullptr, sizeof(ShmFrame), PROT_READ | PROT_WRITE, MAP_SHARED, fdFrame, 0);
    if (gShmFrame == MAP_FAILED) { perror("[SlamRunnerROS] mmap frame"); return 1; }
    close(fdFrame);
    memset(gShmFrame, 0, sizeof(ShmFrame));

    sem_unlink(SEM_NAME_FRAME);
    gSemFrame = sem_open(SEM_NAME_FRAME, O_CREAT | O_EXCL, 0666, 0);
    if (gSemFrame == SEM_FAILED) { perror("[SlamRunnerROS] sem_open"); return 1; }

    int fdMap = shm_open(SHM_NAME_MAP, O_CREAT | O_RDWR, 0666);
    if (fdMap < 0) { perror("[SlamRunnerROS] shm_open map"); return 1; }
    ftruncate(fdMap, sizeof(ShmMap));
    gShmMap = (ShmMap*)mmap(nullptr, sizeof(ShmMap), PROT_READ | PROT_WRITE, MAP_SHARED, fdMap, 0);
    close(fdMap);
    if (gShmMap == MAP_FAILED) { perror("[SlamRunnerROS] mmap map"); return 1; }
    memset(gShmMap, 0, sizeof(ShmMap));

    // ── 初始化长短焦双目语义 SLAM ──
    cout << "[SlamRunnerROS] 初始化长短焦双目语义 SLAM..." << endl;
    ORB_SLAM3::SlamInterface slam;
    if (!slam.Init(vocabPath, settingsPath, ORB_SLAM3::SlamInterface::STEREO)) {
        cerr << "[SlamRunnerROS] SLAM 初始化失败" << endl;
        return 1;
    }
    gSlam = &slam;
    gShmCtrl->slam_ready = 1;
    cout << "[SlamRunnerROS] 语义 SLAM 就绪" << endl;

    // ── ROS 参数 ──
    ros::NodeHandle nh("~");
    string imageTopic;
    nh.param<string>("image_topic", imageTopic, "/usb_cam/image_raw");

    // ── 订阅相机话题 ──
    ImageGrabber igb(&slam);
    ros::Subscriber sub = nh.subscribe(imageTopic, 1, &ImageGrabber::GrabImage, &igb);

    cout << "[SlamRunnerROS] 已订阅: " << imageTopic
         << "（长短焦拼接图，自动裁左右半）" << endl;

    // ── ROS 异步旋转 ──
    ros::AsyncSpinner spinner(2);
    spinner.start();

    // ── 主线程：处理 Qt 命令 ──
    while (gRunning && ros::ok()) {
        if (gShmCtrl->cmd_shutdown) {
            gRunning = false;
            break;
        }
        ProcessCommands();
        usleep(50000);  // 50ms 轮询
    }

    // ── 清理 ──
    cout << "[SlamRunnerROS] 正在关闭..." << endl;
    spinner.stop();
    gShmCtrl->slam_ready = 0;
    gShmCtrl->tracking_state = -1;
    sem_post(gSemFrame);

    slam.SaveTrajectory(trajDir);
    slam.Shutdown();

    sem_close(gSemFrame);
    sem_unlink(SEM_NAME_FRAME);
    munmap(gShmCtrl, sizeof(ShmCtrl));
    munmap(gShmFrame, sizeof(ShmFrame));
    munmap(gShmMap, sizeof(ShmMap));
    shm_unlink(SHM_NAME_CTRL);
    shm_unlink(SHM_NAME_FRAME);
    shm_unlink(SHM_NAME_MAP);

    ros::shutdown();
    cout << "[SlamRunnerROS] 退出" << endl;
    return 0;
}
