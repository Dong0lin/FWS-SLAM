#include "YOLOv11.h"
#include "logging.h"
#include "cuda_utils.h"
#include "macros.h"
#include "preprocess.h"
#include <NvOnnxParser.h>
#include "common.h"
#include <fstream>
#include <iostream>


static Logger logger;
#define isFP16 true
#define warmup true


/**
 * @brief YOLOv11类的构造函数，根据模型路径初始化检测器
 * 
 * 该构造函数支持两种模型加载方式：
 * 1. 直接加载预编译的TensorRT引擎文件(.engine)
 * 2. 从ONNX模型文件(.onnx)构建并保存TensorRT引擎
 * 
 * 初始化完成后，会自动进行模型预热（warmup）以提高首次推理速度。
 * 
 * @param model_path 模型文件路径，支持.engine或.onnx格式
 * @param logger TensorRT日志记录器，用于记录构建和推理过程
 */
YOLOv11::YOLOv11(string model_path, nvinfer1::ILogger& logger)
{
    // 检查模型文件类型：如果是.engine文件则直接反序列化，否则从ONNX构建
    if (model_path.find(".onnx") == std::string::npos)
    {
        // 加载预编译的TensorRT引擎文件
        init(model_path, logger);
    }
    // 从ONNX模型文件构建TensorRT引擎
    else
    {
        // 构建TensorRT引擎
        build(model_path, logger);
        // 保存构建的引擎以便下次直接使用
        saveEngine(model_path);
    }

    // TensorRT版本兼容性处理：不同版本API获取输入维度的方法不同
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
void YOLOv11::init(std::string engine_path, nvinfer1::ILogger& logger)
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
    // 分配CPU端输出缓冲区：存储推理结果
    cpu_output_buffer = new float[detection_attribute_size * num_detections];
    
    // 分配GPU端输入缓冲区：存储预处理后的图像数据
    // 大小：3通道 × 宽度 × 高度 × float大小
    CUDA_CHECK(cudaMalloc(&gpu_buffers[0], 3 * input_w * input_h * sizeof(float)));
    
    // 分配GPU端输出缓冲区：存储模型推理结果
    CUDA_CHECK(cudaMalloc(&gpu_buffers[1], detection_attribute_size * num_detections * sizeof(float)));



    // 步骤5：初始化CUDA预处理模块
    // 传入最大支持的图像尺寸，分配预处理所需的内存
    cuda_preprocess_init(MAX_IMAGE_SIZE);



    // 步骤6：创建CUDA流用于异步操作
    // CUDA流允许并行执行多个GPU操作，提高效率
    CUDA_CHECK(cudaStreamCreate(&stream));



    // 步骤7：模型预热（warmup）
    // 执行10次空推理，优化首次推理性能（缓存预热、内核编译等）
    if (warmup) {
        for (int i = 0; i < 10; i++) {
            this->infer();  // 执行推理（无实际数据）
        }
        printf("model warmup 10 times\n");  // 输出预热完成信息
    }
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
YOLOv11::~YOLOv11()
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
    
    // 步骤3：释放CPU内存缓冲区
    // 释放存储推理结果的CPU端缓冲区
    delete[] cpu_output_buffer;

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
 * @brief 图像预处理函数，在GPU上执行图像预处理操作
 * 
 * 将输入的OpenCV图像转换为模型需要的格式：
 * 1. 调整图像尺寸到模型输入大小（保持宽高比）
 * 2. 执行颜色空间转换（BGR→RGB）
 * 3. 像素值归一化（0-255 → 0-1）
 * 4. 通道重排（interleaved → planar格式）
 * 
 * 处理结果直接存储在GPU输入缓冲区中，供推理使用。
 * 
 * @param image 输入的OpenCV图像（BGR格式）
 */
void YOLOv11::preprocess(Mat& image) {
    // 调用CUDA预处理函数，在GPU上并行执行图像预处理
    // image.ptr(): 获取图像数据指针
    // image.cols, image.rows: 源图像尺寸
    // gpu_buffers[0]: GPU输入缓冲区（存储预处理结果）
    // input_w, input_h: 模型输入尺寸
    // stream: CUDA流用于异步操作
    cuda_preprocess(image.ptr(), image.cols, image.rows, gpu_buffers[0], input_w, input_h, stream);
    
    // 等待预处理操作完成，确保数据就绪
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

/**
 * @brief 执行模型推理函数
 * 
 * 使用TensorRT引擎执行实际的推理操作：
 * - 将预处理后的数据从GPU输入缓冲区传递给模型
 * - 在GPU上执行神经网络前向传播
 * - 结果存储在GPU输出缓冲区中
 * 
 * 支持TensorRT不同版本的API：
 * - TensorRT 9.x及以下：使用enqueueV2接口
 * - TensorRT 10.x及以上：使用enqueueV3接口
 * 
 * 注意：此函数是异步的，推理操作在指定的CUDA流中执行
 */
void YOLOv11::infer()
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
void YOLOv11::postprocess(vector<Detection>& output)
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




        // 步骤4：应用置信度阈值过滤
        if (score > conf_threshold) {
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
        
        // 将结果添加到输出向量中
        output.push_back(result);
    }
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
void YOLOv11::build(std::string onnxPath, nvinfer1::ILogger& logger)
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
bool YOLOv11::saveEngine(const std::string& onnxpath)
{
    // 步骤1：根据ONNX文件路径生成引擎文件路径
    std::string engine_path;  // 存储生成的引擎文件路径
    
    // 查找ONNX文件路径中最后一个点号的位置（文件扩展名分隔符）
    size_t dotIndex = onnxpath.find_last_of(".");
    
    // 检查是否成功找到点号（确保是有效的文件路径）
    if (dotIndex != std::string::npos) {
        // 生成引擎文件路径：去掉.onnx扩展名，添加.engine扩展名
        // 例如：yolo11s.onnx → yolo11s.engine
        engine_path = onnxpath.substr(0, dotIndex) + ".engine";
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
        file.open(engine_path, std::ios::binary | std::ios::out);
        
        // 检查文件是否成功打开
        if (!file.is_open())
        {
            // 文件打开失败，输出错误信息
            std::cout << "Create engine file" << engine_path << " failed" << std::endl;
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
void YOLOv11::draw(Mat& image, const vector<Detection>& output)
{
    // 计算原始图像尺寸与模型输入尺寸的比例
    // 用于将模型输出的归一化坐标反变换到原始图像坐标
    const float ratio_h = input_h / (float)image.rows;
    const float ratio_w = input_w / (float)image.cols;

    // 遍历所有检测结果
    for (int i = 0; i < output.size(); i++)
    {
        // 获取当前检测结果
        auto detection = output[i];
        auto box = detection.bbox;        // 边界框坐标
        auto class_id = detection.class_id; // 类别ID
        auto conf = detection.conf;        // 置信度
        
        // 根据类别ID获取对应的颜色（从预定义的COLORS数组中）
        cv::Scalar color = cv::Scalar(COLORS[class_id][2], COLORS[class_id][1], COLORS[class_id][0]);

        // 根据不同的缩放策略反变换边界框坐标
        // 处理letterbox缩放时可能出现的黑边填充情况
        if (ratio_h > ratio_w)
        {
            // 高度方向缩放比例更大，宽度方向有黑边填充
            box.x = box.x / ratio_w;  // x坐标反缩放
            // y坐标需要减去黑边高度的一半后再反缩放
            box.y = (box.y - (input_h - ratio_w * image.rows) / 2) / ratio_w;
            box.width = box.width / ratio_w;   // 宽度反缩放
            box.height = box.height / ratio_w; // 高度反缩放
        }
        else
        {
            // 宽度方向缩放比例更大，高度方向有黑边填充
            // x坐标需要减去黑边宽度的一半后再反缩放
            box.x = (box.x - (input_w - ratio_h * image.cols) / 2) / ratio_h;
            box.y = box.y / ratio_h;           // y坐标反缩放
            box.width = box.width / ratio_h;   // 宽度反缩放
            box.height = box.height / ratio_h; // 高度反缩放
        }

        // 在图像上绘制检测框
        // 使用对应类别的颜色，线宽为3像素
        rectangle(image, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), color, 3);

        // 绘制检测框文本标签
        // 格式：类别名称 + 置信度（保留4位小数）
        string class_string = CLASS_NAMES[class_id] + ' ' + to_string(conf).substr(0, 4);
        
        // 计算文本尺寸，用于确定标签背景框大小
        Size text_size = getTextSize(class_string, FONT_HERSHEY_DUPLEX, 1, 2, 0);
        
        // 创建文本标签的背景矩形框（位于检测框上方）
        Rect text_rect(box.x, box.y - 40, text_size.width + 10, text_size.height + 20);
        
        // 绘制文本背景框（填充模式）
        rectangle(image, text_rect, color, FILLED);
        
        // 在背景框上绘制文本（黑色字体）
        putText(image, class_string, Point(box.x + 5, box.y - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0), 2, 0);
    }
}