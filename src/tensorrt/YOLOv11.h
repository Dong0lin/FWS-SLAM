#pragma once

#include "NvInfer.h"
#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace std;
using namespace cv;

/**
 * @brief 检测结果结构体，包含目标检测的详细信息
 */
struct Detection
{
    float conf;         ///< 检测置信度，范围0-1
    int class_id;       ///< 类别ID，对应COCO数据集的80个类别
    Rect bbox;          ///< 边界框坐标和尺寸 (x, y, width, height)
};

/**
 * @brief YOLOv11目标检测类，基于TensorRT进行加速推理
 * 
 * 该类实现了YOLOv11模型的加载、预处理、推理和后处理功能，
 * 支持ONNX模型和TensorRT引擎文件的加载和推理。
 */
class YOLOv11
{

public:
    /**
     * @brief 构造函数，根据模型路径初始化YOLOv11检测器
     * @param model_path 模型文件路径，支持.engine或.onnx格式
     * @param logger TensorRT日志记录器
     */
    YOLOv11(string model_path, nvinfer1::ILogger& logger);
    
    /**
     * @brief 析构函数，释放所有分配的资源
     */
    ~YOLOv11();

    /**
     * @brief 图像预处理，将输入图像转换为模型需要的格式
     * @param image 输入图像，将被调整为模型输入尺寸并进行归一化
     */
    void preprocess(Mat& image);
    
    /**
     * @brief 执行模型推理
     */
    void infer();
    
    /**
     * @brief 后处理，解析模型输出并生成检测结果
     * @param output 输出参数，存储检测结果向量
     */
    void postprocess(vector<Detection>& output);
    
    /**
     * @brief 在图像上绘制检测结果
     * @param image 输入图像，将在该图像上绘制检测框和标签
     * @param output 检测结果向量
     */
    void draw(Mat& image, const vector<Detection>& output);

private:
    /**
     * @brief 初始化TensorRT引擎
     * @param engine_path TensorRT引擎文件路径
     * @param logger TensorRT日志记录器
     */
    void init(std::string engine_path, nvinfer1::ILogger& logger);

    float* gpu_buffers[2];               ///< GPU缓冲区数组：[0]输入缓冲区，[1]输出缓冲区
    float* cpu_output_buffer;            ///< CPU端输出缓冲区，用于存储推理结果

    cudaStream_t stream;                 ///< CUDA流，用于异步操作
    IRuntime* runtime;                   ///< TensorRT运行时，用于反序列化引擎
    ICudaEngine* engine;                 ///< TensorRT引擎，包含优化的网络
    IExecutionContext* context;          ///< 执行上下文，用于执行推理

    // 模型参数
    int input_w;                         ///< 模型输入宽度
    int input_h;                         ///< 模型输入高度
    int num_detections;                  ///< 最大检测数量
    int detection_attribute_size;        ///< 每个检测的属性数量 (坐标+类别概率)
    int num_classes = 80;                ///< 类别数量，默认COCO数据集的80个类别
    const int MAX_IMAGE_SIZE = 4096 * 4096; ///< 最大支持的图像尺寸
    float conf_threshold = 0.2f;         ///< 置信度阈值，低于此值的检测将被过滤
    float nms_threshold = 0.4f;          ///< 非极大值抑制阈值

    vector<Scalar> colors;               ///< 不同类别对应的颜色，用于绘制检测框

    /**
     * @brief 从ONNX模型构建TensorRT引擎
     * @param onnxPath ONNX模型文件路径
     * @param logger TensorRT日志记录器
     */
    void build(std::string onnxPath, nvinfer1::ILogger& logger);
    
    /**
     * @brief 保存构建的TensorRT引擎到文件
     * @param filename 保存的文件名
     * @return 保存是否成功
     */
    bool saveEngine(const std::string& filename);
};