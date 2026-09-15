/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * ros_multifocal_stereo.cc — 长短焦双目语义 SLAM 版 (ROS 发布)
 *
 * 针对单 USB 长短焦双目相机（输出 2560×720 拼接图像：左半=短焦 1280×720、
 * 右半=长焦 1280×720）的语义 SLAM 双目节点。
 * 从拼接图像裁剪左右两半，分别作为左目(短焦)与右目(长焦)，运行双目长短焦 SLAM。
 * 使用 Pangolin 可视化（非 Qt），用于调试和独立测试。
 *
 * 发布的 ROS 话题:
 *   ~camera_pose     (geometry_msgs/PoseStamped)  相机位姿
 *   ~tracking_state  (std_msgs/Int32)             跟踪状态
 *   ~trajectory      (nav_msgs/Path)              相机轨迹
 *   ~map_points      (sensor_msgs/PointCloud2)    语义地图点云
 *   ~keyframe_poses  (geometry_msgs/PoseArray)    关键帧位姿
 *   ~plane           (visualization_msgs/Marker)  地面平面
 *   ~boxes_3d        (visualization_msgs/MarkerArray) 3D检测框
 *   tf               (tf/tfMessage)               相机坐标系变换
 *
 * 用法:
 *   rosrun FWS-SLAM-ROS MultiFocal_Stereo <vocab> <settings>
 *
 * 参数:
 *   ~image_topic (string, default: "/usb_cam/image_raw")
 *       相机图像话题
 */

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<csignal>
#include<cstdlib>

#include<ros/ros.h>
#include<ros/package.h>
#include <cv_bridge/cv_bridge.h>

#include<opencv2/core/core.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include"../../../include/System.h"

// ROS message types
#include<geometry_msgs/PoseStamped.h>
#include<geometry_msgs/PoseArray.h>
#include<geometry_msgs/TransformStamped.h>
#include<nav_msgs/Path.h>
#include<visualization_msgs/MarkerArray.h>
#include<visualization_msgs/Marker.h>
#include<std_msgs/Int32.h>
#include<sensor_msgs/PointCloud2.h>
#include<sensor_msgs/PointField.h>
#include<tf/transform_broadcaster.h>

#include"../../../include/Map.h"
#include"../../../include/common.h"

using namespace std;

// ─── 全局指针用于信号处理 ───
static ros::Publisher* gPubCameraPose  = nullptr;
static ros::Publisher* gPubTrajectory  = nullptr;
static bool gRunning = true;

static void SignalHandler(int) {
    gRunning = false;
    ros::shutdown();
}

// ─── 转换 Eigen::Matrix4f → geometry_msgs::Pose ───
static inline geometry_msgs::Pose Matrix4fToPose(const Eigen::Matrix4f& mat)
{
    geometry_msgs::Pose pose;
    pose.position.x = mat(0, 3);
    pose.position.y = mat(1, 3);
    pose.position.z = mat(2, 3);
    Eigen::Quaternionf q(mat.block<3,3>(0,0));
    pose.orientation.x = q.x();
    pose.orientation.y = q.y();
    pose.orientation.z = q.z();
    pose.orientation.w = q.w();
    return pose;
}

class ImageGrabber
{
public:
    ImageGrabber(ORB_SLAM3::System* pSLAM, ros::NodeHandle& nh)
        : mpSLAM(pSLAM)
    {
        // ── 发布器 ──
        pubCameraPose    = nh.advertise<geometry_msgs::PoseStamped>("camera_pose", 100);
        pubTrackingState = nh.advertise<std_msgs::Int32>("tracking_state", 10);
        pubTrajectory    = nh.advertise<nav_msgs::Path>("trajectory", 10, true);  // latched
        pubMapPoints     = nh.advertise<sensor_msgs::PointCloud2>("map_points", 10);
        pubKeyFrames     = nh.advertise<geometry_msgs::PoseArray>("keyframe_poses", 10);
        pubPlane         = nh.advertise<visualization_msgs::Marker>("plane", 10);
        pubBoxes3D       = nh.advertise<visualization_msgs::MarkerArray>("boxes_3d", 10);

        // 全局指针用于信号处理
        gPubCameraPose = &pubCameraPose;
        gPubTrajectory = &pubTrajectory;

        trajectory.header.frame_id = "world";

        // ── 定时器：每 100ms 发布一次语义地图数据 ──
        timer = nh.createTimer(ros::Duration(0.1), &ImageGrabber::PublishSemanticData, this);
    }

    void GrabImage(const sensor_msgs::ImageConstPtr& msg);

    void PublishSemanticData(const ros::TimerEvent& event);

    ORB_SLAM3::System* mpSLAM;

    // 发布器
    ros::Publisher pubCameraPose;
    ros::Publisher pubTrackingState;
    ros::Publisher pubTrajectory;
    ros::Publisher pubMapPoints;
    ros::Publisher pubKeyFrames;
    ros::Publisher pubPlane;
    ros::Publisher pubBoxes3D;
    ros::Timer timer;

    // 累积轨迹
    nav_msgs::Path trajectory;
    int frameCounter = 0;
};

void ImageGrabber::GrabImage(const sensor_msgs::ImageConstPtr& msg)
{
    // Copy the ros image message to cv::Mat.
    // 强制转成 BGR8（usb_cam mjpeg 常发布 rgb8，直接按 BGR 处理会导致红蓝互换）
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv::Mat imgFull = cv_ptr->image;

    // 长短焦拼接图像 2560×720 → 左半=短焦 1280×720、右半=长焦 1280×720
    int fullW = imgFull.cols;
    int halfW = fullW / 2;
    cv::Mat imgLeft  = imgFull(cv::Rect(0, 0, halfW, imgFull.rows)).clone();
    cv::Mat imgRight = imgFull(cv::Rect(halfW, 0, fullW - halfW, imgFull.rows)).clone();

    // 临时补偿：MF_VSHIFT=<px> 把右半图整体上移（实机拼接图常见固定垂直偏移，
    // 实测本机右目相对左目低约 32px → MF_VSHIFT=32）。默认关闭，正式使用请重新标定。
    if (const char* ev = getenv("MF_VSHIFT")) {
        int vy = atoi(ev);
        if (vy > 0 && vy < imgRight.rows) {
            cv::Mat shifted(imgRight.rows, imgRight.cols, imgRight.type(), cv::Scalar(0, 0, 0));
            imgRight(cv::Rect(0, vy, imgRight.cols, imgRight.rows - vy))
                    .copyTo(shifted(cv::Rect(0, 0, imgRight.cols, imgRight.rows - vy)));
            imgRight = shifted;
            ROS_WARN("MF_VSHIFT=%d：右半图上移 %d 像素（临时补偿，正式使用请重新标定）", vy, vy);
        }
    }

    // 诊断：MF_DUMP_IMGS=1 保存前 10 帧左右半图，核对拼接布局/右目是否黑屏
    static int dumpCount = 0;
    if (getenv("MF_DUMP_IMGS") && dumpCount < 10) {
        cv::imwrite("/tmp/mf_ros_left_" + std::to_string(dumpCount) + ".jpg", imgLeft);
        cv::imwrite("/tmp/mf_ros_right_" + std::to_string(dumpCount) + ".jpg", imgRight);
        ROS_INFO("已保存诊断图 /tmp/mf_ros_left_%d.jpg 与 /tmp/mf_ros_right_%d.jpg",
                 dumpCount, dumpCount);
        dumpCount++;
    }

    mpSLAM->TrackStereo(imgLeft, imgRight, cv_ptr->header.stamp.toSec());

    // ── 每帧发布相机位姿和轨迹 ──
    ORB_SLAM3::MapDrawer* pMapDrawer = mpSLAM->GetMapDrawer();
    if (!pMapDrawer) return;

    Eigen::Matrix4f Twc = pMapDrawer->GetCurrentCameraPose();

    // 发布相机位姿
    geometry_msgs::PoseStamped poseMsg;
    poseMsg.header.stamp    = ros::Time::now();
    poseMsg.header.frame_id = "world";
    poseMsg.pose = Matrix4fToPose(Twc);
    pubCameraPose.publish(poseMsg);

    // 发布跟踪状态
    std_msgs::Int32 stateMsg;
    stateMsg.data = mpSLAM->GetTrackingState();
    pubTrackingState.publish(stateMsg);

    // 累积轨迹
    trajectory.header.stamp = poseMsg.header.stamp;
    trajectory.poses.push_back(poseMsg);
    pubTrajectory.publish(trajectory);

    // ── 广播 TF: world → camera ──
    static tf::TransformBroadcaster br;
    geometry_msgs::TransformStamped tfMsg;
    tfMsg.header.stamp    = poseMsg.header.stamp;
    tfMsg.header.frame_id = "world";
    tfMsg.child_frame_id  = "camera";
    tfMsg.transform.translation.x = Twc(0, 3);
    tfMsg.transform.translation.y = Twc(1, 3);
    tfMsg.transform.translation.z = Twc(2, 3);
    Eigen::Quaternionf q(Twc.block<3,3>(0,0));
    tfMsg.transform.rotation.x = q.x();
    tfMsg.transform.rotation.y = q.y();
    tfMsg.transform.rotation.z = q.z();
    tfMsg.transform.rotation.w = q.w();
    br.sendTransform(tfMsg);

    frameCounter++;
}

void ImageGrabber::PublishSemanticData(const ros::TimerEvent&)
{
    ORB_SLAM3::MapDrawer* pMapDrawer = mpSLAM->GetMapDrawer();
    if (!pMapDrawer || !pMapDrawer->mpAtlas) return;

    ORB_SLAM3::Map* pMap = pMapDrawer->mpAtlas->GetCurrentMap();
    if (!pMap) return;

    ros::Time now = ros::Time::now();

    // ── 1. 发布地图点 (PointCloud2) ──
    {
        auto vpMPs = pMap->GetAllMapPoints();
        if (!vpMPs.empty()) {
            sensor_msgs::PointCloud2 cloud;
            cloud.header.frame_id = "world";
            cloud.header.stamp    = now;
            cloud.width  = 0;
            cloud.height = 1;
            cloud.is_bigendian = false;
            cloud.is_dense = true;

            // fields: x, y, z, rgb (uint32)
            cloud.fields.resize(4);
            cloud.fields[0].name = "x";      cloud.fields[0].datatype = sensor_msgs::PointField::FLOAT32; cloud.fields[0].count = 1; cloud.fields[0].offset = 0;
            cloud.fields[1].name = "y";      cloud.fields[1].datatype = sensor_msgs::PointField::FLOAT32; cloud.fields[1].count = 1; cloud.fields[1].offset = 4;
            cloud.fields[2].name = "z";      cloud.fields[2].datatype = sensor_msgs::PointField::FLOAT32; cloud.fields[2].count = 1; cloud.fields[2].offset = 8;
            cloud.fields[3].name = "rgb";    cloud.fields[3].datatype = sensor_msgs::PointField::FLOAT32; cloud.fields[3].count = 1; cloud.fields[3].offset = 12;

            cloud.point_step = 16;
            cloud.row_step   = 0;

            std::vector<uint8_t> data;
            data.reserve(vpMPs.size() * 16);

            for (auto* pMP : vpMPs) {
                if (!pMP || pMP->isBad()) continue;
                Eigen::Vector3f pos = pMP->GetWorldPos();

                // 语义着色 (同 SlamInterface::GetAllMapPoints)
                unsigned int r=0, g=0, b=255; // 默认蓝色(非语义)
                int semClass = pMP->mnSemanticClass;
                if (semClass >= 0 && !pMP->IsDynamicMapPoint()) {
                    const auto& c = COLORS[semClass % COLORS.size()];
                    r = c[0]; g = c[1]; b = c[2];
                } else if (semClass >= 0 && pMP->IsDynamicMapPoint()) {
                    r = 255; g = 0; b = 0; // 红色=动态
                }
                unsigned int rgb = (r << 16) | (g << 8) | b;

                const size_t off = data.size();
                data.resize(off + 16);
                memcpy(&data[off],      &pos.x(), 4);
                memcpy(&data[off + 4],  &pos.y(), 4);
                memcpy(&data[off + 8],  &pos.z(), 4);
                memcpy(&data[off + 12], &rgb,     4);
            }

            cloud.data = std::move(data);
            cloud.width = cloud.data.size() / cloud.point_step;
            cloud.row_step = cloud.data.size();

            pubMapPoints.publish(cloud);
        }
    }

    // ── 2. 发布关键帧位姿 (PoseArray) ──
    {
        auto vpKFs = pMap->GetAllKeyFrames();
        geometry_msgs::PoseArray arr;
        arr.header.frame_id = "world";
        arr.header.stamp    = now;
        arr.poses.reserve(vpKFs.size());
        for (auto* pKF : vpKFs) {
            if (!pKF || pKF->isBad()) continue;
            Eigen::Matrix4f Twc = pKF->GetPoseInverse().matrix();
            arr.poses.push_back(Matrix4fToPose(Twc));
        }
        pubKeyFrames.publish(arr);
    }

    // ── 3. 发布地面平面 (Marker::CYLINDER 半透明面片) ──
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp    = now;
        marker.ns = "plane";
        marker.id = 0;
        marker.action = visualization_msgs::Marker::DELETEALL;
        pubPlane.publish(marker);

        if (pMap->IsPlaneEstimated()) {
            marker.action = visualization_msgs::Marker::ADD;
            marker.type   = visualization_msgs::Marker::CYLINDER;

            Eigen::Vector3f N = pMap->GetPlaneNormal();
            float d = pMap->GetPlaneOffsets()[0];

            // 平面方程: N·X + d = 0
            // 用一个大的圆柱体表示地面（中心在平面上，法线朝上）
            // 取平面与垂直轴的交点作为中心
            float denom = N.z();
            if (fabs(denom) > 1e-6f) {
                float z0 = -d / denom;
                marker.pose.position.x = 0;
                marker.pose.position.y = 0;
                marker.pose.position.z = z0;

                // 朝向：从平面法线旋转到 (0,0,1)
                Eigen::Vector3f up(0, 0, 1);
                Eigen::Quaternionf q;
                q.setFromTwoVectors(N, up);
                marker.pose.orientation.x = q.x();
                marker.pose.orientation.y = q.y();
                marker.pose.orientation.z = q.z();
                marker.pose.orientation.w = q.w();
            }

            marker.scale.x = 10.0;
            marker.scale.y = 10.0;
            marker.scale.z = 0.05;

            marker.color.a = 0.3;
            marker.color.r = 0.0;
            marker.color.g = 1.0;
            marker.color.b = 0.0;

            marker.lifetime = ros::Duration(30);
            pubPlane.publish(marker);
        }
    }

    // ── 4. 发布 3D 检测框 (MarkerArray: CUBE + TEXT) ──
    {
        visualization_msgs::MarkerArray markers;

        const auto& boxes = pMap->GetPersistentBoxes();
        int idx = 0;
        for (const auto& box : boxes) {
            if (!box.bValid) continue;

            // 框体
            visualization_msgs::Marker m;
            m.header.frame_id = "world";
            m.header.stamp    = now;
            m.ns = "box3d";
            m.id = idx++;
            m.type = visualization_msgs::Marker::CUBE;
            m.action = visualization_msgs::Marker::ADD;

            m.pose.position.x = box.center.x();
            m.pose.position.y = box.center.y();
            m.pose.position.z = box.center.z() + box.height * 0.5f; // 中心在底面+半高

            m.scale.x = box.width;
            m.scale.y = box.depth;
            m.scale.z = box.height;

            // 朝向：CUBE 本地坐标 X=宽、Y=深(车长)、Z=高 →
            // 世界 U=车宽方向、V=车长方向(heading)、N=平面法向
            Eigen::Vector3f N = pMap->GetPlaneNormal().normalized();
            Eigen::Vector3f U, V;
            if (box.heading.squaredNorm() > 0.5f) {
                V = box.heading.normalized();
                U = N.cross(V).normalized();
            } else {
                Eigen::Vector3f ref = (std::abs(N.x()) < 0.9f)
                                      ? Eigen::Vector3f::UnitX() : Eigen::Vector3f::UnitZ();
                U = N.cross(ref).normalized();
                V = N.cross(U).normalized();
            }
            Eigen::Matrix3f R;
            R.col(0) = U; R.col(1) = V; R.col(2) = N;
            Eigen::Quaternionf q(R);
            m.pose.orientation.x = q.x();
            m.pose.orientation.y = q.y();
            m.pose.orientation.z = q.z();
            m.pose.orientation.w = q.w();

            // 类别颜色
            const auto& c = COLORS[box.class_id % COLORS.size()];
            m.color.a = 0.5;
            m.color.r = c[0] / 255.0f;
            m.color.g = c[1] / 255.0f;
            m.color.b = c[2] / 255.0f;

            m.lifetime = ros::Duration(30);
            markers.markers.push_back(m);

            // 类别标签
            visualization_msgs::Marker label;
            label.header.frame_id = "world";
            label.header.stamp    = now;
            label.ns = "box3d_label";
            label.id = idx++;
            label.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            label.action = visualization_msgs::Marker::ADD;

            label.pose.position.x = box.center.x();
            label.pose.position.y = box.center.y();
            label.pose.position.z = box.center.z() + box.height + 0.3f;
            label.pose.orientation.w = 1.0;

            label.scale.z = 0.5;

            const char* name = (box.class_id >= 0 && box.class_id < (int)CLASS_NAMES.size())
                               ? CLASS_NAMES[box.class_id].c_str() : "unknown";
            char txt[64];
            snprintf(txt, sizeof(txt), "%s (obs:%d)", name, box.nObservations);
            label.text = txt;

            label.color.a = 1.0;
            label.color.r = 1.0;
            label.color.g = 1.0;
            label.color.b = 1.0;

            label.lifetime = ros::Duration(30);
            markers.markers.push_back(label);
        }

        // 先清除旧框
        visualization_msgs::Marker clear;
        clear.action = visualization_msgs::Marker::DELETEALL;
        clear.ns = "box3d";
        markers.markers.push_back(clear);
        clear.ns = "box3d_label";
        markers.markers.push_back(clear);

        pubBoxes3D.publish(markers);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "MultiFocal_Stereo");
    ros::start();

    signal(SIGINT,  SignalHandler);
    signal(SIGTERM, SignalHandler);

    if(argc != 3)
    {
        cerr << endl << "用法: rosrun FWS-SLAM-ROS MultiFocal_Stereo path_to_vocabulary path_to_settings" << endl;
        ros::shutdown();
        return 1;
    }

    // 使用 VIEWER_PANGOLIN 模式（有 Pangolin 可视化窗口）
    // 注意：这是语义 SLAM 版本，包含目标检测、动态一致性等增强功能
    // 长短焦双目：拼接图左右两半分别作为短焦/长焦，走双目 SLAM
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::STEREO, ORB_SLAM3::System::VIEWER_PANGOLIN);

    ros::NodeHandle nh("~");

    ImageGrabber igb(&SLAM, nh);

    // 获取相机话题名称（可通过私有参数修改）
    string imageTopic;
    nh.param<string>("image_topic", imageTopic, "/usb_cam/image_raw");

    ros::Subscriber sub = nh.subscribe(imageTopic, 1, &ImageGrabber::GrabImage, &igb);

    ros::spin();

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    ros::shutdown();

    return 0;
}
