#ifndef DETECTOR_H
#define DETECTOR_H

#include <string>
#include <thread>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <dirent.h>
#include <iomanip>

#include "Tracking.h"  // 移除这个包含，避免循环引用（使用第32行的前向声明代替）
#include <opencv2/opencv.hpp>	
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "logging.h"
#include "NvInfer.h"
#include "cuda_runtime_api.h"

using namespace nvinfer1;
using namespace cv;

namespace ORB_SLAM3
{
// 定义了对象跟踪类，调用了orb-slam中的tracking
class Tracking;

// 检测结果结构体
struct Detection
{
    float conf;         ///< 检测置信度，范围0-1
    int class_id;       ///< 类别ID，对应VisDrone数据集的8个类别
    Rect bbox;          ///< 边界框坐标和尺寸 (x, y, width, height)
    float moving_prob = 0.4f;    ///< 物体动态概率
    bool is_dynamic = false;    ///< 是否为动态物体
};

// 定义了目标检测器类
class Detector
{
public:
    // 初始化目标检测器类
    Detector();
	
    ~Detector();

    // 创建对象跟踪类的指针对象
    Tracking *mpTracker;

    // std::string engine_path;
    std::string engine_path = "/home/dl/FWS-SLAM/engine/visdrone_11m.engine";

    std::vector<Detection> dynamic_boxes;

    std::vector<Detection> objects;

    // 预定义Run方法
    void Run();
    // 预定义对象跟踪设置方法
    void SetTracker(Tracking *pTracker);
    // 预定义是否是新的图像方法
    bool isNewImgArrived();
    // 预定义目标检测方法
    void Detect();

    // 预定义检测是否完成的方法
    bool isFinished();

    // 预定义完成请求的方法
    void RequestFinish();

    // 图像预处理方法
    void preprocess(Mat& image);

    // 模型推理方法
    void infer();

    // 后处理方法
    void postprocess(std::vector<Detection>& output);

    // 绘制检测结果方法
    void draw(const cv::Mat& image, const std::vector<Detection>& output);
    
    // TensorRT引擎构建方法
    void build(std::string onnxPath, nvinfer1::ILogger& logger);
    
    // 保存TensorRT引擎方法
    bool saveEngine(const std::string& filename);

    // 初始化TensorRT引擎方法
    void init(std::string engine_path, nvinfer1::ILogger& logger);

    // 这个表示异步获取新的图像的操作
    std::mutex mMutexGetNewImg;
    
    // 定义属性来标记这个图像是不是新来的
    bool mbNewImgFlag;

    // 条件变量：用于检测线程高效等待新图像（替代 usleep 忙等待）
    std::condition_variable mCvNewImg;
    std::mutex mMutexCvNewImg;

    // 条件变量：用于通知 Tracking 线程检测完成（替代 usleep 忙等待）
    std::condition_variable mCvDetDone;
    std::mutex mMutexCvDetDone;

    // 定义属性来标记这个请求是否已经完成（是不是目标检测请求已经完成？）
    bool mbFinishRequested;
    // 定义异步操作，完成整个语义分割
    std::mutex mMutexFinish;

    // 模型参数
    int input_w;                            ///< 模型输入宽度
    int input_h;                            ///< 模型输入高度
    int num_detections;                     ///< 最大检测数量
    int detection_attribute_size;           ///< 每个检测的属性数量 (坐标+类别概率)
    int num_classes = 8;                    ///< 类别数量，默认VisDrone数据集的8个类别
    const int MAX_IMAGE_SIZE = 4096 * 4096; ///< 最大支持的图像尺寸
    float conf_threshold = 0.5f;            ///< 置信度阈值，低于此值的检测将被过滤
    float conf_threshold_people = 0.5f;     ///< 人体检测置信度阈值
    float nms_threshold = 0.5f; 
    
    // 转换参数缓存
    float ratio_h = 0.0f;                       ///< 高度缩放比例
    float ratio_w = 0.0f;                       ///< 宽度缩放比例
    float offset_x = 0.0f;                      ///< x坐标偏移量
    float offset_y = 0.0f;                      ///< y坐标偏移量
    bool isPrecomputed = false;                  ///< 是否已预计算转换参数

    // TensorRT相关成员变量
    IRuntime* runtime;                   ///< TensorRT运行时
    ICudaEngine* engine;                 ///< TensorRT引擎
    IExecutionContext* context;         ///< 执行上下文
    cudaStream_t stream;                 ///< CUDA流
    float* gpu_buffers[2];               ///< GPU缓冲区数组
    float* cpu_output_buffer;            ///< CPU输出缓冲区
    bool warmup = true;                  ///< 模型预热标志

    // 检测框容器
    // cv::Rect dynamic_boxes;  ///< 动态物体检测框
    // cv::Rect half_boxes;     ///< 人体分割检测框
    // std::vector<cv::Rect> boxes1;         ///< 临时检测框容器
    // int label[80] = {0};                  ///< 类别标签数组

    // 视频捕获对象
    // cv::VideoCapture cap;                ///< 视频捕获对象

	cv::Mat mImg;	

	// 新图像检测标志的互斥锁
    std::mutex mMutexNewImgDetection;     ///< 新图像检测标志的互斥锁

    // 设置目标检测标志量方法声明
    void SetDetectionFlag();

    // 类别特定的处理函数声明
    void ProcessPedestrian(Detection& detection, const cv::Mat& image);
    void ProcessPeople(Detection& detection, const cv::Mat& image);
    void ProcessBicycle(Detection& detection, const cv::Mat& image);
    void ProcessCar(Detection& detection, const cv::Mat& image);
    void ProcessUnknown(Detection& detection, const cv::Mat& image);
    
    // 检测框坐标转换函数
    void PrecomputeTransformParams(const cv::Mat& image);
    void TransformBoxes(std::vector<Detection>& output);

    // ===== 检测耗时/目标数量统计（用于区分“模型变慢”还是“目标变多变慢”）=====
    long long mnDetectCount   = 0;      ///< 累计检测帧数
    double mdTotalInferTime   = 0.0;    ///< 累计纯推理耗时(ms)，只与模型有关，与目标数无关
    double mdTotalPostTime    = 0.0;    ///< 累计后处理(NMS)+坐标转换耗时(ms)，随目标数增长
    double mdTotalDetectTime  = 0.0;    ///< 累计整帧检测耗时(ms)
    long long mnTotalObjects  = 0;      ///< 累计检测到的目标框总数（用于算每帧平均目标数）
    double mdMaxDetectTime    = 0.0;    ///< 单帧最大整帧耗时(ms)
    void PrintDetectionStatistics();    ///< 运行结束后输出一次统计
};

}// namespace ORB_SLAM3

#endif // DETECTOR_H