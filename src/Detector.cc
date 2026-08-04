#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include <NvOnnxParser.h>

#include "cuda_runtime_api.h"
#include "logging.h"
#include <thread>
#include <iomanip>
#include "Detector.h"
// #include "Tracking.h"  // 添加这个包含，解决 incomplete type 问题
#include "cuda_utils.h"
#include "preprocess.h"
#include "common.h"
#include "macros.h"
#include <vector>
#include "NvOnnxParser.h"

/*
1. 宏定义与全局变量
    DEVICE：指定使用的 GPU 设备编号（默认为 0）。
    NMS_THRESH：非极大值抑制（NMS）的阈值（0.5），用于过滤重叠度高的目标框。
    BBOX_CONF_THRESH：目标框置信度阈值（0.5），低于该值的目标框会被过滤。
    INPUT_W/INPUT_H：模型输入图像的尺寸（640x640），需与训练时的输入尺寸一致。
    INPUT_BLOB_NAME/OUTPUT_BLOB_NAME：TensorRT 引擎中输入 / 输出张量的名称（需与模型导出时一致）。
    gLogger：TensorRT 的日志对象，用于输出推理过程中的日志信息。
*/

#define isFP16 true
#define warmup true
#define DEVICE 0  // GPU id GPU设备的编号，需要在终端中查看

using namespace nvinfer1;
using namespace nvonnxparser;

static Logger gLogger;


namespace ORB_SLAM3
{
// Detector类的构造函数
// 初始化目标检测器的基本状态和配置参数
Detector::Detector()
{
    // 初始化新图像到达标志为false
    // 表示当前没有新的图像需要处理
    mbNewImgFlag = false;

    // 初始化完成请求标志为false
    // 表示检测器尚未收到停止运行的请求
    mbFinishRequested = false;
    
    // 设置TensorRT引擎文件路径
    // 使用预训练的目标检测模型进行推理
    // std::string engine_path = "/home/dl/FWS-SLAM/engine/yolo11s_fp16.engine";

    // 检查模型文件类型：如果是.engine文件则直接反序列化，否则从ONNX构建
    if (engine_path.find(".onnx") == std::string::npos)
    {
        // 加载预编译的TensorRT引擎文件
        init(engine_path, gLogger);
    }
    // 从ONNX模型文件构建TensorRT引擎
    else
    {
        // 构建TensorRT引擎
        build(engine_path, gLogger);
        // 保存构建的引擎以便下次直接使用
        saveEngine(engine_path);
    }

    #if NV_TENSORRT_MAJOR < 10
        // TensorRT 9.x及以下版本：使用getBindingDimensions方法
        auto input_dims = engine->getBindingDimensions(0);
        input_h = input_dims.d[2];  // 获取输入高度（通常是第3个维度）
        input_w = input_dims.d[3];  // 获取输入宽度（通常是第4个维度）
    #else
        // TensorRT 10.x及以上版本：使用getTensorShape方法
        auto input_dims = engine->getTensorShape(engine->getIOTensorName(0));
        input_h = input_dims.d[2];  // 获取输入高度
        input_w = input_dims.d[3];  // 获取输入宽度
    #endif

}

/**
 * @brief YOLOv11类的析构函数，负责清理所有分配的资源
 * 
 * 按照正确的顺序释放所有GPU和CPU资源，避免内存泄漏：
 * 1. 同步并销毁CUDA流
 * 2. 释放GPU内存缓冲区
 * 3. 释放CPU内存缓冲区
 * 4. 销毁预处理模块
 * 5. 销毁TensorRT相关对象
 * 
 * 注意：释放顺序很重要，必须先等待所有GPU操作完成再释放资源
 */
Detector::~Detector()
{
    // 步骤1：释放CUDA流和相关缓冲区
    // 等待流中所有操作完成，确保没有未完成的GPU操作
    CUDA_CHECK(cudaStreamSynchronize(stream));
    // 销毁CUDA流
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    // 步骤2：释放GPU内存缓冲区
    // 循环释放两个GPU缓冲区（输入和输出缓冲区）
    for (int i = 0; i < 2; i++)
        CUDA_CHECK(cudaFree(gpu_buffers[i]));
    
    // 步骤3：释放CPU内存缓冲区（锁页内存）
    // 释放存储推理结果的CPU端缓冲区
    CUDA_CHECK(cudaFreeHost(cpu_output_buffer));

    // 步骤4：销毁预处理模块
    // 释放预处理模块分配的所有GPU内存
    cuda_preprocess_destroy();

    // 步骤5：销毁TensorRT引擎和相关对象
    // 按照创建的反顺序销毁：上下文 → 引擎 → 运行时
    delete context;   // 销毁执行上下文
    delete engine;    // 销毁CUDA引擎
    delete runtime;   // 销毁运行时环境
}


/**
 * @brief 初始化TensorRT引擎，加载预编译的引擎文件并设置推理环境
 * 
 * 该函数执行以下主要步骤：
 * 1. 读取并反序列化TensorRT引擎文件
 * 2. 获取模型的输入输出维度信息
 * 3. 分配GPU和CPU内存缓冲区
 * 4. 初始化CUDA预处理模块
 * 5. 创建CUDA流用于异步操作
 * 6. 执行模型预热（warmup）优化首次推理性能
 * 
 * @param engine_path TensorRT引擎文件路径（.engine格式）
 * @param logger TensorRT日志记录器
 */
void Detector::init(std::string engine_path, nvinfer1::ILogger& logger)
{
    // 步骤1：读取TensorRT引擎文件
    // 以二进制模式打开引擎文件
    ifstream engineStream(engine_path, ios::binary);
    
    // 获取文件大小：移动到文件末尾获取位置
    engineStream.seekg(0, ios::end);
    const size_t modelSize = engineStream.tellg();  // 获取文件大小（字节数）
    engineStream.seekg(0, ios::beg);                // 重新移动到文件开头
    
    // 分配内存缓冲区存储引擎数据
    unique_ptr<char[]> engineData(new char[modelSize]);
    
    // 读取整个引擎文件到内存缓冲区
    engineStream.read(engineData.get(), modelSize);
    engineStream.close();  // 关闭文件流

    // 步骤2：反序列化TensorRT引擎
    // 创建TensorRT运行时环境
    runtime = createInferRuntime(logger);
    
    // 从内存数据反序列化CUDA引擎
    engine = runtime->deserializeCudaEngine(engineData.get(), modelSize);
    
    // 创建执行上下文，用于实际推理操作
    context = engine->createExecutionContext();

    // 步骤3：获取模型的输入输出维度信息
    // 获取输入张量维度：绑定索引0为输入
    input_h = engine->getBindingDimensions(0).d[2];  // 输入高度（通常是第3维）
    input_w = engine->getBindingDimensions(0).d[3];  // 输入宽度（通常是第4维）
    
    // 获取输出张量维度：绑定索引1为输出
    detection_attribute_size = engine->getBindingDimensions(1).d[1];  // 每个检测的属性数量
    num_detections = engine->getBindingDimensions(1).d[2];            // 最大检测数量
    
    // 计算类别数量：总属性数减去4个坐标值（cx, cy, w, h）
    num_classes = detection_attribute_size - 4;

    // 步骤4：初始化内存缓冲区
    // 分配CPU端输出缓冲区：使用锁页内存（pinned memory）实现真正的异步 GPU→CPU 传输
    CUDA_CHECK(cudaMallocHost((void**)&cpu_output_buffer, detection_attribute_size * num_detections * sizeof(float)));
    
    // 分配GPU端输入缓冲区：存储预处理后的图像数据
    CUDA_CHECK(cudaMalloc(&gpu_buffers[0], 3 * input_w * input_h * sizeof(float)));
    
    // 分配GPU端输出缓冲区：存储模型推理结果
    CUDA_CHECK(cudaMalloc(&gpu_buffers[1], detection_attribute_size * num_detections * sizeof(float)));

    // 步骤5：初始化CUDA预处理模块
    // 传入最大支持的图像尺寸，分配预处理所需的内存
    cuda_preprocess_init(MAX_IMAGE_SIZE);

    // 步骤6：创建CUDA流用于异步操作
    // CUDA流允许并行执行多个GPU操作，提高效率
    CUDA_CHECK(cudaStreamCreate(&stream));

    // 设置 GPU 设备（只需初始化时设置一次，避免每帧调用 cudaSetDevice 的开销）
    cudaSetDevice(DEVICE);

    // 步骤7：模型预热（warmup）
    // 执行10次空推理，优化首次推理性能（缓存预热、内核编译等）
    if (warmup) {
        for (int i = 0; i < 10; i++) {
            this->infer();  // 执行推理（无实际数据）
        }
        printf("model warmup 10 times\n");  // 输出预热完成信息
        printf("input_h: %d, input_w: %d\n", input_h, input_w);
    }
}


void ORB_SLAM3::Detector::Run()
{    
    while(1)
    {
        // 使用条件变量高效等待新图像（零CPU开销，唤醒延迟仅几微秒）
        {
            std::unique_lock<std::mutex> lock(mMutexCvNewImg);
            mCvNewImg.wait(lock, [this]{ return mbNewImgFlag || mbFinishRequested; });
        }

        //收到结束请求就退出循环
        if(isFinished())
        {
            break;
        }

        // 消费新图像标志
        {
            std::unique_lock<std::mutex> lock(mMutexGetNewImg);
            mbNewImgFlag = false;
        }

        dynamic_boxes.clear();        
		
		// perform YOLO detection
		Detect();

	}

    // 检测线程结束，输出一次耗时/目标数量统计（仅在开启统计时）
    if (gEnableTimingStats)
        PrintDetectionStatistics();

}


// 运行结束后输出检测耗时统计，用于区分“模型推理变慢”还是“目标变多导致后处理/下游变慢”
void Detector::PrintDetectionStatistics()
{
    if (mnDetectCount == 0)
    {
        std::cout << "[Detector] 未执行任何检测，无统计数据。" << std::endl;
        return;
    }

    double avgInfer  = mdTotalInferTime  / mnDetectCount;
    double avgPost   = mdTotalPostTime   / mnDetectCount;
    double avgDetect = mdTotalDetectTime / mnDetectCount;
    double avgObjs   = static_cast<double>(mnTotalObjects) / mnDetectCount;

    std::cout << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "        OBJECT DETECTION STATS           " << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Total detected frames:   " << mnDetectCount << std::endl;
    std::cout << "Avg objects per frame:   " << avgObjs << std::endl;
    std::cout << "Avg infer time (GPU):    " << avgInfer  << " ms   <- 只与模型有关" << std::endl;
    std::cout << "Avg postproc+transform:  " << avgPost   << " ms   <- 随目标数增长" << std::endl;
    std::cout << "Avg total detect time:   " << avgDetect << " ms" << std::endl;
    std::cout << "Max total detect time:   " << mdMaxDetectTime << " ms" << std::endl;
    std::cout << "Avg detection FPS:       " << (avgDetect > 0.0 ? 1000.0 / avgDetect : 0.0) << " fps" << std::endl;
    std::cout << "=========================================" << std::endl;
}


// 查询是否目标检测过程已经完成
bool Detector::isFinished()
{
    // 这是独占锁，这个进程生命周期结束后，其他的进程才能运行
    unique_lock<mutex> lock(mMutexFinish);
    
    // 返回一个标志变量，这个变量将在过程完成时被标记为true
    return mbFinishRequested;
}

// 过程已经完成时候将标记变量赋值为true  
void Detector::RequestFinish()
{
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinishRequested=true;
    }
    // 唤醒检测线程，使其能检测到 mbFinishRequested 并退出
    mCvNewImg.notify_one();
}

// 给mp跟踪器赋值
void Detector::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

/**
 * @brief 图像预处理函数，将输入图像转换为模型所需格式
 * 
 * 该函数执行以下操作：
 * 1. 检查图像格式，如果是灰度图像则转换为BGR格式
 * 2. 调用CUDA预处理函数，在GPU上并行执行图像预处理
 * 3. 等待预处理操作完成，确保数据就绪
 * 
 * @param image 输入图像（cv::Mat格式）
 */
void Detector::preprocess(Mat& image) {
    // 检查图像格式，如果是灰度图像则转换为BGR
    if (image.channels() == 1) {
        // 灰度图像：转换为BGR格式
        cv::Mat processed_image;
        cv::cvtColor(image, processed_image, cv::COLOR_GRAY2BGR);
        cuda_preprocess(processed_image.ptr(), processed_image.cols, processed_image.rows, 
            gpu_buffers[0], input_w, input_h, stream);
    } else {
        // 彩色图像：直接使用
        cuda_preprocess(image.ptr(), image.cols, image.rows, gpu_buffers[0], input_w, input_h, stream);
    }
    
    // 注意：不需要 cudaStreamSynchronize！
    // preprocess 和 infer 在同一个 stream 上，CUDA 流内操作天然有序，
    // 推理会自动等待预处理完成，无需 CPU 端同步阻塞。
}


/**
 * @brief 在图像上绘制检测结果
 * 
 * 该函数将YOLOv11模型输出的检测结果绘制到原始图像上，包括：
 * 1. 检测框坐标从模型输入尺寸反变换到原始图像尺寸
 * 2. 绘制彩色检测框
 * 3. 在检测框上方绘制类别名称和置信度标签
 * 
 * @param image 输入/输出图像，绘制结果将直接修改此图像
 * @param output 检测结果向量，包含每个检测目标的置信度、类别ID和边界框信息
 */
void Detector::draw(const cv::Mat& image, const vector<Detection>& output)
{
    // 遍历所有检测结果
    for (int i = 0; i < output.size(); i++)
    {
        // 获取当前检测结果
        auto detection = output[i];
        auto box = detection.bbox;        // 边界框坐标（已转换为原始图像坐标）
        auto class_id = detection.class_id; // 类别ID
        auto conf = detection.conf;        // 置信度
        auto moving_prob = detection.moving_prob; // 物体动态概率
        auto is_dynamic = detection.is_dynamic; // 是否动态目标
        

        cv::Scalar color = cv::Scalar(COLORS[class_id][2], COLORS[class_id][1], COLORS[class_id][0]);
        if (is_dynamic) {
            // 动态目标：使用红色框
            // 非动态目标：使用对应类别颜色框
            color = cv::Scalar(0, 0, 255);
            rectangle(image, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), color, 4);
        }
        else {
            rectangle(image, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), color, 3);
        }


        // 绘制运动概率文本标签（人不绘制，固定为1）
        if(class_id != 0 && class_id != 1) {
            std::stringstream ss;
            ss << fixed << std::setprecision(3) << detection.moving_prob;
            std::string prob_string = ss.str();
            
            Size text_size = getTextSize(prob_string, FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
            
            Scalar text_color = (detection.moving_prob > 0.5) ? Scalar(0, 0, 255) : color;
            
            putText(image, prob_string, Point(box.x + 5, box.y - 5), 
                    FONT_HERSHEY_SIMPLEX, 0.8, text_color, 1.5, 0);
        }
    }
}


void Detector::infer()
{
    // TensorRT版本兼容性处理
#if NV_TENSORRT_MAJOR < 10
    // TensorRT 9.x及以下版本：使用enqueueV2接口
    // (void**)gpu_buffers: GPU缓冲区数组指针
    // stream: 执行推理的CUDA流
    // nullptr: 可选的事件对象（此处不使用）
    context->enqueueV2((void**)gpu_buffers, stream, nullptr);
#else
    // TensorRT 10.x及以上版本：使用enqueueV3接口
    // 新版本API简化了参数传递
    this->context->enqueueV3(this->stream);
#endif
}

/**
 * @brief 后处理函数，解析模型输出并生成检测结果
 * 
 * 该函数执行以下主要步骤：
 * 1. 将GPU推理结果异步复制到CPU内存
 * 2. 解析每个检测框的坐标和类别概率
 * 3. 应用置信度阈值过滤低置信度检测
 * 4. 执行非极大值抑制(NMS)去除重复检测框
 * 5. 生成最终的检测结果向量
 * 
 * @param output 输出参数，存储处理后的检测结果向量
 */
void Detector::PrecomputeTransformParams(const cv::Mat& image) {
    ratio_h = input_h / (float)image.rows;
    ratio_w = input_w / (float)image.cols;
    
    if (ratio_h > ratio_w) {
        offset_y = (input_h - ratio_w * image.rows) / 2;
        offset_x = 0;
    } else {
        offset_x = (input_w - ratio_h * image.cols) / 2;
        offset_y = 0;
    }
    
    isPrecomputed = true;
}

void Detector::TransformBoxes(std::vector<Detection>& output) {
    // 遍历所有检测结果
    for (int i = 0; i < output.size(); i++) {
        auto& detection = output[i];
        auto& box = detection.bbox;

        // 根据不同的缩放策略反变换边界框坐标
        if (ratio_h > ratio_w) {
            box.x = box.x / ratio_w;
            box.y = (box.y - offset_y) / ratio_w;
            box.width = box.width / ratio_w;
            box.height = box.height / ratio_w;
        } else {
            box.x = (box.x - offset_x) / ratio_h;
            box.y = box.y / ratio_h;
            box.width = box.width / ratio_h;
            box.height = box.height / ratio_h;
        }
}
}


void Detector::postprocess(std::vector<Detection>& output)
{
    // 步骤1：将GPU推理结果异步复制到CPU内存
    // 从GPU输出缓冲区(gpu_buffers[1])复制到CPU输出缓冲区(cpu_output_buffer)
    // 数据大小：检测数量 × 每个检测的属性数 × float大小
    CUDA_CHECK(cudaMemcpyAsync(cpu_output_buffer, gpu_buffers[1], 
        num_detections * detection_attribute_size * sizeof(float), 
        cudaMemcpyDeviceToHost, stream));
    
    // 等待数据复制完成，确保CPU端数据就绪
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // 步骤2：初始化临时存储容器
    vector<Rect> boxes;           // 存储检测框坐标
    vector<int> class_ids;        // 存储类别ID
    vector<float> confidences;    // 存储置信度分数
    vector<float> moving_probs;   // 存储动态概率

    // 将CPU输出缓冲区包装为OpenCV矩阵，便于处理
    // 矩阵尺寸：属性数 × 检测数，数据类型：CV_32F（32位浮点数）
    const Mat det_output(detection_attribute_size, num_detections, CV_32F, cpu_output_buffer);

    // 步骤3：遍历所有检测框，解析坐标和类别信息
    for (int i = 0; i < det_output.cols; ++i) {
        // 提取当前检测框的类别概率部分（跳过前4个坐标值）
        // 行范围：4 到 4+num_classes-1（包含80个类别的概率）
        const Mat classes_scores = det_output.col(i).rowRange(4, 4 + num_classes);
        
        Point class_id_point;  // 存储最大概率类别的索引位置
        double score;          // 存储最大概率值
        
        // 找到当前检测框中概率最高的类别及其分数
        minMaxLoc(classes_scores, nullptr, &score, nullptr, &class_id_point);


        // 步骤4：应用置信度阈值过滤（人对小尺寸降低阈值）
        float threshold = (class_id_point.y == 0 || class_id_point.y == 1)
                          ? conf_threshold_people : conf_threshold;
        if (score > threshold) {
            // 提取检测框的坐标信息（YOLO格式：中心点坐标 + 宽高）
            const float cx = det_output.at<float>(0, i);  // 中心点x坐标
            const float cy = det_output.at<float>(1, i);  // 中心点y坐标
            const float ow = det_output.at<float>(2, i);  // 检测框宽度
            const float oh = det_output.at<float>(3, i);  // 检测框高度
            
            // 将YOLO格式坐标转换为OpenCV的Rect格式（左上角坐标 + 宽高）
            Rect box;
            box.x = static_cast<int>((cx - 0.5 * ow));  // 左上角x坐标 = 中心x - 宽度/2
            box.y = static_cast<int>((cy - 0.5 * oh));  // 左上角y坐标 = 中心y - 高度/2
            box.width = static_cast<int>(ow);           // 检测框宽度
            box.height = static_cast<int>(oh);          // 检测框高度

            // 将有效检测框信息存储到临时容器中
            boxes.push_back(box);
            class_ids.push_back(class_id_point.y);  // 类别ID（y坐标即为索引）
            confidences.push_back(score);           // 置信度分数
            moving_probs.push_back(MOTION_PROBABILITIES[class_id_point.y]);  // 动态概率
        }
    }

    // 步骤5：执行非极大值抑制(NMS)去除重复检测框
    vector<int> nms_result;  // 存储NMS处理后保留的检测框索引
    
    // 使用OpenCV的NMSBoxes函数进行非极大值抑制
    // 参数说明：
    // - boxes: 检测框坐标向量
    // - confidences: 置信度向量
    // - conf_threshold: 置信度阈值（已在前一步过滤，此处作为参考）
    // - nms_threshold: NMS重叠阈值（IoU阈值），高于此值的框会被抑制
    // - nms_result: 输出保留的检测框索引
    dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, nms_result);

    // 步骤6：生成最终的检测结果
    for (int i = 0; i < nms_result.size(); i++)
    {
        Detection result;              // 创建单个检测结果结构体
        int idx = nms_result[i];       // 获取NMS处理后保留的索引
        
        // 填充检测结果信息
        result.class_id = class_ids[idx];  // 类别ID
        result.conf = confidences[idx];    // 置信度
        result.bbox = boxes[idx];          // 边界框坐标
        result.moving_prob = moving_probs[idx];  // 动态概率
        
        // 将结果添加到输出向量中
        output.push_back(result);
    }
}


void Detector::Detect()
{
    cv::Mat image = mImg;

    // 如果帧为空（视频结束），退出循环
    if (image.empty()) return;

    // vector<Detection> objects;  // 检测结果容器

    objects.clear();  // 清空上一次的检测结果，准备存储新的结果

    // 整帧检测计时起点（预处理 + 推理 + 后处理 + 坐标转换）
    auto detectStart = std::chrono::steady_clock::now();

    // 执行目标检测流程
    preprocess(image);  // 图像预处理

    // 记录推理开始时间（纯 GPU 推理：只与模型有关，与检测到的目标数量无关）
    auto start = std::chrono::steady_clock::now();
    infer();  // 执行模型推理（enqueue 为异步，仅入队）
    // 仅在开启统计时同步等待 GPU 推理真正完成，以获得准确的推理耗时；
    // 否则保持原异步行为（同步点在 postprocess 内部），不影响正常性能。
    if (gEnableTimingStats)
        cudaStreamSynchronize(stream);
    auto end = std::chrono::steady_clock::now();  // 记录推理结束时间

    postprocess(objects);  // 后处理，解析检测结果（NMS 随候选框数量增长）

    // 预计算转换参数并转换检测框坐标
    if (!isPrecomputed) {
        PrecomputeTransformParams(image);
    }

    TransformBoxes(objects);

    // ===== 累计耗时与目标数量统计（运行结束后统一输出一次）=====
    if (gEnableTimingStats)
    {
        auto detectEnd = std::chrono::steady_clock::now();
        double inferMs  = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
        double detectMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(detectEnd - detectStart).count();
        double postMs   = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(detectEnd - end).count();  // 后处理+坐标转换

        mnDetectCount++;
        mdTotalInferTime  += inferMs;
        mdTotalPostTime   += postMs;
        mdTotalDetectTime += detectMs;
        mnTotalObjects    += static_cast<long long>(objects.size());
        if (detectMs > mdMaxDetectTime)
            mdMaxDetectTime = detectMs;
    }

    // 计算并输出推理耗时（毫秒）
    // auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
    // printf("cost %2.4lf ms\n", tc);
    // 显示检测结果,用于调试
    // draw(image, objects);  // 在图像上绘制检测框
    // // 将绘制后的图像保存回成员变量mImg，供FrameDrawer使用
    // {
    //     unique_lock<mutex> lock(mMutexGetNewImg);
    //     image.copyTo(mImg);  // 深拷贝，确保mImg包含检测框
    // }
    // int dynamic_counter {0};
    // 对每个检测到的物体进行类别特定的处理
    // for (auto& obj : objects) {
    //     // 根据类别ID调用相应的处理函数
    //     switch (obj.class_id) {
    //         case 0: // pedestrian - 行人（单独处理）
    //             ProcessPedestrian(obj, image);
    //             break;
    //         case 1: // people - 人群（单独处理）
    //             ProcessPeople(obj, image);
    //             break;
    //         case 2: // bicycle - 自行车
    //         case 6: // tricycle - 三轮车
    //         case 9: // motor - 摩托车
    //             // 驾驶员可见
    //             ProcessBicycle(obj, image);
    //             break;
    //         case 3: // car - 汽车
    //         case 4: // van - 货车
    //         case 5: // truck - 卡车
    //         case 7: // awning-tricycle - 带篷三轮车
    //         case 8: // bus - 公交车
    //             // 四轮车/大型车辆统一处理：驾驶员不可见或影响较小
    //             ProcessCar(obj, image);
    //             break;
    //         default:
    //             // 处理未知类别
    //             ProcessUnknown(obj, image);
    //             break;
    //     }
    // } 

    // 设置检测完成标志，通知跟踪器（参考原版代码）
    SetDetectionFlag();

}

//判断图像是否是新来的，是新来的就改为false，返回true。线程独占锁。
bool Detector::isNewImgArrived()
{
    unique_lock<mutex> lock(mMutexGetNewImg);
    if(mbNewImgFlag)
    {
        mbNewImgFlag=false;
        return true;
    }
    else
    	return false;
}


//设置目标检测标志量。通过条件变量通知 Tracking 线程。
void Detector::SetDetectionFlag()
{
    {
        // 使用条件变量的同一把锁保护共享标志，确保内存可见性
        std::unique_lock<std::mutex> lock(mMutexCvDetDone);
        mpTracker->mbNewDetImgFlag = true;
    }
    // 通知 Tracking 线程检测已完成（零延迟唤醒）
    mCvDetDone.notify_one();
}


/**
 * @brief 从ONNX模型文件构建TensorRT引擎
 * 
 * 该函数执行以下主要步骤：
 * 1. 创建TensorRT构建器和配置
 * 2. 解析ONNX模型文件
 * 3. 构建优化的TensorRT引擎
 * 4. 反序列化引擎并创建执行上下文
 * 5. 清理临时资源
 * 
 * @param onnxPath ONNX模型文件路径
 * @param logger TensorRT日志记录器
 */
void Detector::build(std::string onnxPath, nvinfer1::ILogger& logger)
{
    // 步骤1：创建TensorRT构建器
    // 构建器负责管理整个引擎构建过程
    auto builder = createInferBuilder(logger);
    
    // 设置网络定义标志：显式批处理模式
    // kEXPLICIT_BATCH标志表示网络支持动态批处理
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    
    // 创建网络定义对象，用于描述神经网络结构
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    
    // 创建构建配置对象，设置优化参数
    IBuilderConfig* config = builder->createBuilderConfig();
    // 步骤2：配置FP16精度模式（如果启用）
    // 检查全局标志isFP16，如果为true则启用FP16推理
    if (isFP16)
    {
        // 设置FP16标志，启用16位浮点数推理
        // 优点：减少内存使用，提高推理速度
        // 缺点：可能损失少量精度
        config->setFlag(BuilderFlag::kFP16);
    }

    // 步骤3：创建ONNX解析器并解析模型文件
    // 创建ONNX解析器，用于将ONNX模型转换为TensorRT网络
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
    
    // 解析ONNX模型文件
    // onnxPath.c_str(): ONNX文件路径
    // static_cast<int>(nvinfer1::ILogger::Severity::kINFO): 日志级别为INFO
    bool parsed = parser->parseFromFile(onnxPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));

    // 步骤4：构建序列化的TensorRT引擎
    // 根据网络定义和配置构建优化的TensorRT引擎
    // 返回IHostMemory对象，包含序列化的引擎数据
    IHostMemory* plan{ builder->buildSerializedNetwork(*network, *config) };

    // 步骤5：创建运行时环境并反序列化引擎
    // 创建TensorRT运行时，用于加载和执行引擎
    runtime = createInferRuntime(logger);

    // 从序列化数据反序列化CUDA引擎
    // plan->data(): 序列化引擎数据的指针
    // plan->size(): 数据大小
    engine = runtime->deserializeCudaEngine(plan->data(), plan->size());

    // 步骤6：创建执行上下文
    // 执行上下文用于实际执行推理操作
    // 一个引擎可以有多个上下文（支持多流推理）
    context = engine->createExecutionContext();

    // 步骤7：清理临时资源
    // 释放构建过程中分配的临时对象
    delete network;  // 释放网络定义
    delete config;   // 释放构建配置
    delete parser;   // 释放ONNX解析器
    delete plan;     // 释放序列化引擎数据
}


/**
 * @brief 保存TensorRT引擎到文件
 * 
 * 该函数执行以下主要步骤：
 * 1. 根据ONNX文件路径生成引擎文件路径
 * 2. 序列化TensorRT引擎为二进制数据
 * 3. 将序列化数据写入到.engine文件
 * 4. 清理临时资源并返回保存结果
 * 
 * 保存的引擎文件可以在下次使用时直接加载，避免重复构建，显著提高启动速度。
 * 
 * @param onnxpath ONNX模型文件路径，用于生成对应的引擎文件路径
 * @return bool 保存是否成功：true-成功，false-失败
 */
bool Detector::saveEngine(const std::string& onnxpath)
{
    // 步骤1：根据ONNX文件路径生成引擎文件路径
    std::string make_engine_path;  // 存储生成的引擎文件路径
    
    // 查找ONNX文件路径中最后一个点号的位置（文件扩展名分隔符）
    size_t dotIndex = onnxpath.find_last_of(".");
    
    // 检查是否成功找到点号（确保是有效的文件路径）
    if (dotIndex != std::string::npos) {
        // 生成引擎文件路径：去掉.onnx扩展名，添加.engine扩展名
        // 例如：yolo11s.onnx → yolo11s.engine
        make_engine_path = onnxpath.substr(0, dotIndex) + ".engine";
    }
    else
    {
        // 无效的文件路径（没有找到点号），返回失败
        return false;
    }

    // 步骤2：保存引擎到文件
    // 检查引擎对象是否有效（确保已经成功构建）
    if (engine)
    {
        // 序列化TensorRT引擎为二进制数据
        // 序列化后的数据可以保存到文件或通过网络传输
        nvinfer1::IHostMemory* data = engine->serialize();
        
        // 创建文件输出流，以二进制写入模式打开文件
        std::ofstream file;
        file.open(make_engine_path, std::ios::binary | std::ios::out);
        
        // 检查文件是否成功打开
        if (!file.is_open())
        {
            // 文件打开失败，输出错误信息
            std::cout << "Create engine file" << make_engine_path << " failed" << std::endl;
            return 0;  // 返回失败
        }
        
        // 将序列化的引擎数据写入文件
        // (const char*)data->data(): 序列化数据的指针（转换为char*）
        // data->size(): 数据大小（字节数）
        file.write((const char*)data->data(), data->size());
        
        // 关闭文件流，确保数据完全写入磁盘
        file.close();

        // 步骤3：清理临时资源
        // 释放序列化数据占用的内存
        delete data;
    }
    
    // 返回保存成功
    return true;
}


// 类别特定的处理函数实现

/**
 * @brief 处理行人检测结果
 * @param detection 检测到的行人信息
 * @param image 原始图像（用于可视化或进一步处理）
 */
void Detector::ProcessPedestrian(Detection& detection, const cv::Mat& image) {
    // 高运动概率物体，直接计入动态物体列表
    dynamic_boxes.push_back(detection);
    
}

/**
 * @brief 处理人群检测结果
 * @param detection 检测到的人群信息
 * @param image 原始图像
 */
void Detector::ProcessPeople(Detection& detection, const cv::Mat& image) {
    // std::cout << "[PEOPLE] 检测到人群: 置信度=" << detection.conf 
    //           << ", 原始运动概率=" << detection.moving_prob 
    //           << ", 位置=(" << detection.bbox.x << "," << detection.bbox.y 
    //           << "," << detection.bbox.width << "," << detection.bbox.height << ")" << std::endl;
    
    // 计算检测框的长宽比例
    float aspect_ratio = 0.0f;
    if (detection.bbox.width > 0) {
        aspect_ratio = static_cast<float>(detection.bbox.height) / static_cast<float>(detection.bbox.width);
    }
    
    // std::cout << "[PEOPLE] 检测框长宽比例: " << aspect_ratio << " (高/宽)" << std::endl;
    
    // 根据长宽比例调整运动概率
    float adjusted_moving_prob = detection.moving_prob;
    
    // 长宽比例较小的可能是坐着的人，降低运动概率
    if (aspect_ratio <= 1.6f) {
        // 比例小于等于1.6:1，很可能是坐着的人
        // std::cout << "[PEOPLE] 检测到可能坐着的人，降低运动概率 " << std::endl;
        adjusted_moving_prob *= 0.3f; // 降低70%的运动概率
        detection.moving_prob = adjusted_moving_prob; // 更新检测结果中的运动概率
        // std::cout << "[PEOPLE] 检测到可能坐着的人，降低运动概率: " 
        //           << detection.moving_prob << std::endl;
    } else {
        // 比例大于3:1，很可能是站立的人，保持较高运动概率
        // std::cout << "[PEOPLE] 检测到站立的人，保持运动概率: " << adjusted_moving_prob << std::endl;
        dynamic_boxes.push_back(detection);
    }
}

/**
 * @brief 处理自行车检测结果
 * @param detection 检测到的自行车信息
 * @param image 原始图像
 */
void Detector::ProcessBicycle(Detection& detection, const cv::Mat& image) {
    // std::cout << "[BICYCLE] 检测到自行车: 置信度=" << detection.conf 
    //           << ", 原始运动概率=" << detection.moving_prob 
    //           << ", 位置=(" << detection.bbox.x << "," << detection.bbox.y 
    //           << "," << detection.bbox.width << "," << detection.bbox.height << ")" << std::endl;
    
    // 自行车特定的处理逻辑
    // 如果检测框内有行人或人群，且长宽比例小于2，可能是骑自行车的人
    // 这是一个简单的假设，实际情况可能更复杂（如考虑多个行人）
    
    float adjusted_moving_prob = detection.moving_prob;
        
    // 检查是否有行人或人群位于自行车检测框内（可能是骑自行车的人）
    for (const auto& obj : objects) {
        // 只检查行人(pedestrian)和人群(people)
        if (obj.class_id == 0 || obj.class_id == 1) { // pedestrian=0, people=1
            // 计算行人检测框的中心点
            cv::Point person_center(
                obj.bbox.x + obj.bbox.width / 2,
                obj.bbox.y + obj.bbox.height / 2
            );
            
            // 检查中心点是否在自行车检测框内
            if (detection.bbox.contains(person_center)) {
                // 计算行人的长宽比例
                float person_aspect_ratio = 0.0f;
                if (obj.bbox.width > 0) {
                    person_aspect_ratio = static_cast<float>(obj.bbox.height) / static_cast<float>(obj.bbox.width);
                }
                
                // std::cout << "[BICYCLE] 发现行人位于自行车框内: 类别=" << CLASS_NAMES[obj.class_id]
                //           << ", 长宽比例=" << person_aspect_ratio 
                //           << ", 位置=(" << obj.bbox.x << "," << obj.bbox.y 
                //           << "," << obj.bbox.width << "," << obj.bbox.height << ")" << std::endl;
                
                // 如果长宽比例小于2，很可能是骑自行车的姿势（坐着）
                if (person_aspect_ratio < 2.0f) {
                    adjusted_moving_prob = 1.0f; // 有人骑自行车，运动概率设为最高
                    detection.moving_prob = adjusted_moving_prob; // 更新检测结果中的运动概率
                    dynamic_boxes.push_back(detection);
                    // std::cout << "[BICYCLE] 检测到骑自行车的人！运动概率提升至: " << adjusted_moving_prob << std::endl;
                    break; // 找到一个骑手就足够了，不需要继续检查其他行人
                } 
            }
        }
    }
}



/**
 * @brief 处理汽车检测结果
 * @param detection 检测到的汽车信息
 * @param image 原始图像
 */
void Detector::ProcessCar(Detection& detection, const cv::Mat& image) {
    // std::cout << "[CAR] 检测到四轮/大型车辆: " << CLASS_NAMES[detection.class_id] 
    //           << ", 置信度=" << detection.conf 
    //           << ", 原始运动概率=" << detection.moving_prob 
    //           << ", 位置=(" << detection.bbox.x << "," << detection.bbox.y 
    //           << "," << detection.bbox.width << "," << detection.bbox.height << ")" << std::endl;
    
    float adjusted_moving_prob = detection.moving_prob;
    
    // 3. 检查是否有行人位于车辆附近（可能是上下车）
    bool has_nearby_pedestrian = false;
    for (const auto& obj : objects) {
        if (obj.class_id == 0 || obj.class_id == 1) { // pedestrian或people
            // 计算两个检测框的中心距离
            cv::Point vehicle_center(
                detection.bbox.x + detection.bbox.width / 2,
                detection.bbox.y + detection.bbox.height / 2
            );
            cv::Point person_center(
                obj.bbox.x + obj.bbox.width / 2,
                obj.bbox.y + obj.bbox.height / 2
            );
            
            float distance = cv::norm(vehicle_center - person_center);
            
            // 如果行人在车辆附近（距离小于车辆宽度的一半）
            if (distance < detection.bbox.width / 2) {
                has_nearby_pedestrian = true;
                // std::cout << "[CAR] 发现附近有行人，距离=" << distance 
                //           << ", 可能上下车，降低运动概率" << std::endl;
                adjusted_moving_prob *= 0.3f; // 大幅降低运动概率
                detection.moving_prob = adjusted_moving_prob; // 更新检测结果中的运动概率
                break;
            }
        }
    }
    
    
}



/**
 * @brief 处理未知类别检测结果
 * @param detection 检测到的未知物体信息
 * @param image 原始图像
 */
void Detector::ProcessUnknown(Detection& detection, const cv::Mat& image) {    
    // 未知类别的处理逻辑
    // 1. 采用保守策略，默认认为是动态物体
    // 2. 可以记录未知类别用于后续分析
    // 3. 根据置信度决定是否添加到动态列表
    
    if (detection.conf > 0.5f) {
        dynamic_boxes.push_back(detection);
        // std::cout << "[UNKNOWN] 添加到动态物体列表（保守策略）" << std::endl;
    }

}

} //namespace ORB_SLAM3