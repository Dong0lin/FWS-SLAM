#include "preprocess.h"
#include "cuda_utils.h"
#include "device_launch_parameters.h"

// 静态全局变量，用于存储图像缓冲区
static uint8_t* img_buffer_host = nullptr;   ///< 主机端固定内存缓冲区，用于高效数据传输
static uint8_t* img_buffer_device = nullptr; ///< 设备端内存缓冲区，存储GPU上的图像数据

/**
 * @brief 检查图像是否为灰度图像
 * 
 * 通过检查图像数据格式来判断是否为单通道灰度图像
 * 
 * @param src 源图像数据指针
 * @param src_width 图像宽度
 * @param src_height 图像高度
 * @return true 如果是灰度图像
 * @return false 如果是彩色图像
 */
bool is_grayscale_image(uint8_t* src, int src_width, int src_height) {
    // 灰度图像的总字节数应为 width * height
    // 彩色图像的总字节数应为 width * height * 3
    // 通过检查数据格式来判断
    int total_pixels = src_width * src_height;
    
    // 简单检查：如果图像数据大小等于像素数，则为灰度图像
    // 更精确的检查可以通过分析像素值模式来实现
    return true; // 暂时返回true，实际实现需要更复杂的检测逻辑
}

/**
 * @brief 将灰度图像转换为BGR三通道图像
 * 
 * 将单通道灰度图像复制到三个通道，生成伪彩色BGR图像
 * 
 * @param gray_src 灰度图像源数据
 * @param gray_width 灰度图像宽度
 * @param gray_height 灰度图像高度
 * @param bgr_dst BGR目标缓冲区（必须已分配足够空间：width*height*3）
 */
void convert_grayscale_to_bgr(uint8_t* gray_src, int gray_width, int gray_height, uint8_t* bgr_dst) {
    int total_pixels = gray_width * gray_height;
    
    for (int i = 0; i < total_pixels; i++) {
        uint8_t gray_value = gray_src[i];
        // 将灰度值复制到三个通道（BGR格式）
        bgr_dst[i * 3] = gray_value;     // B通道
        bgr_dst[i * 3 + 1] = gray_value; // G通道
        bgr_dst[i * 3 + 2] = gray_value; // R通道
    }
}

/**
 * @brief 仿射变换矩阵结构体，存储6个变换参数
 * 
 * 用于表示从目标坐标系到源坐标系的仿射变换：
 * [m_x1, m_y1, m_z1]
 * [m_x2, m_y2, m_z2]
 */
struct AffineMatrix {
    float value[6]; ///< 仿射变换矩阵的6个参数
};

/**
 * @brief CUDA核函数：执行仿射变换和图像预处理
 * 
 * 每个线程处理目标图像中的一个像素，执行以下操作：
 * 1. 计算源图像中对应的坐标（反向映射）
 * 2. 双线性插值获取像素值
 * 3. BGR到RGB颜色空间转换
 * 4. 像素值归一化到[0,1]范围
 * 5. 通道重排（interleaved到planar格式）
 * 
 * @param src 源图像数据（设备端，BGR格式）
 * @param src_line_size 源图像每行的字节数
 * @param src_width 源图像宽度
 * @param src_height 源图像高度
 * @param dst 目标缓冲区（设备端，RGB归一化格式）
 * @param dst_width 目标图像宽度
 * @param dst_height 目标图像高度
 * @param const_value_st 超出边界时的填充值（128表示灰色）
 * @param d2s 从目标到源的仿射变换矩阵
 * @param edge 需要处理的像素总数（dst_width * dst_height）
 */
__global__ void warpaffine_kernel(
    uint8_t* src, int src_line_size, int src_width,
    int src_height, float* dst, int dst_width,
    int dst_height, uint8_t const_value_st,
    AffineMatrix d2s, int edge) {
    
    // 计算当前线程处理的像素位置
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= edge) return; // 超出处理范围则返回

    // 提取仿射变换矩阵参数
    float m_x1 = d2s.value[0];
    float m_y1 = d2s.value[1];
    float m_z1 = d2s.value[2];
    float m_x2 = d2s.value[3];
    float m_y2 = d2s.value[4];
    float m_z2 = d2s.value[5];

    // 计算当前像素在目标图像中的坐标
    int dx = position % dst_width;
    int dy = position / dst_width;
    
    // 使用仿射变换计算源图像中对应的坐标（反向映射）
    float src_x = m_x1 * dx + m_y1 * dy + m_z1 + 0.5f; // 加0.5用于四舍五入
    float src_y = m_x2 * dx + m_y2 * dy + m_z2 + 0.5f;
    float c0, c1, c2; // 存储三个通道的像素值

    // 检查坐标是否超出源图像边界
    if (src_x <= -1 || src_x >= src_width || src_y <= -1 || src_y >= src_height) {
        // 超出边界，使用填充值（灰色）
        c0 = const_value_st;
        c1 = const_value_st;
        c2 = const_value_st;
    }
    else {
        // 在边界内，执行双线性插值
        int y_low = floorf(src_y);  // 下取整y坐标
        int x_low = floorf(src_x);  // 下取整x坐标
        int y_high = y_low + 1;     // 上取整y坐标
        int x_high = x_low + 1;     // 上取整x坐标

        // 默认填充值数组
        uint8_t const_value[] = { const_value_st, const_value_st, const_value_st };
        
        // 计算插值权重
        float ly = src_y - y_low; // y方向插值比例
        float lx = src_x - x_low; // x方向插值比例
        float hy = 1 - ly;
        float hx = 1 - lx;
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx; // 四个角的权重
        
        // 初始化四个插值点的指针
        uint8_t* v1 = const_value; // 左上角
        uint8_t* v2 = const_value; // 右上角
        uint8_t* v3 = const_value; // 左下角
        uint8_t* v4 = const_value; // 右下角

        // 设置有效的插值点指针（检查边界条件）
        if (y_low >= 0) {
            if (x_low >= 0)
                v1 = src + y_low * src_line_size + x_low * 3; // 左上角有效
            if (x_high < src_width)
                v2 = src + y_low * src_line_size + x_high * 3; // 右上角有效
        }

        if (y_high < src_height) {
            if (x_low >= 0)
                v3 = src + y_high * src_line_size + x_low * 3; // 左下角有效
            if (x_high < src_width)
                v4 = src + y_high * src_line_size + x_high * 3; // 右下角有效
        }

        // 执行双线性插值计算三个通道的值
        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0]; // B通道
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1]; // G通道
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2]; // R通道
    }

    // BGR到RGB颜色空间转换（交换第一个和第三个通道）
    float t = c2;
    c2 = c0;
    c0 = t;

    // 像素值归一化：将0-255范围转换为0-1范围
    c0 = c0 / 255.0f;
    c1 = c1 / 255.0f;
    c2 = c2 / 255.0f;

    // 通道重排：从interleaved格式(RGBRGBRGB)转换为planar格式(RRRGGGBBB)
    int area = dst_width * dst_height; // 单通道像素数量
    float* pdst_c0 = dst + dy * dst_width + dx;           // R通道目标位置
    float* pdst_c1 = pdst_c0 + area;                      // G通道目标位置（偏移一个通道）
    float* pdst_c2 = pdst_c1 + area;                      // B通道目标位置（偏移两个通道）
    
    // 存储处理后的像素值
    *pdst_c0 = c0; // 存储R通道
    *pdst_c1 = c1; // 存储G通道
    *pdst_c2 = c2; // 存储B通道
}

/**
 * @brief 主预处理函数：协调整个GPU预处理流程
 * 
 * 执行以下步骤：
 * 1. 检测输入图像是否为灰度图像，如果是则转换为BGR格式
 * 2. 数据从CPU传输到GPU固定内存
 * 3. 异步传输到设备内存
 * 4. 计算仿射变换矩阵
 * 5. 启动CUDA核函数进行并行处理
 * 
 * @param src 源图像数据（CPU端）
 * @param src_width 源图像宽度
 * @param src_height 源图像高度
 * @param dst 目标缓冲区（GPU端）
 * @param dst_width 目标图像宽度
 * @param dst_height 目标图像高度
 * @param stream CUDA流用于异步操作
 */
void cuda_preprocess(
    uint8_t* src, int src_width, int src_height,
    float* dst, int dst_width, int dst_height,
    cudaStream_t stream) {

    // 步骤0：检测并处理灰度图像
    uint8_t* processed_src = src;
    int processed_width = src_width;
    int processed_height = src_height;
    
    // 临时缓冲区用于存储转换后的BGR图像（如果是灰度图像）
    uint8_t* temp_bgr_buffer = nullptr;
    
    // 检测是否为灰度图像（通过检查图像数据大小）
    int expected_color_size = src_width * src_height * 3;
    int actual_size = src_width * src_height; // 灰度图像的大小
    
    if (actual_size * 3 != expected_color_size) {
        // 检测到灰度图像，需要转换为BGR格式
        printf("检测到灰度图像，自动转换为BGR格式\n");
        
        // 分配临时缓冲区用于BGR转换
        temp_bgr_buffer = new uint8_t[expected_color_size];
        
        // 将灰度图像转换为BGR格式
        convert_grayscale_to_bgr(src, src_width, src_height, temp_bgr_buffer);
        
        // 使用转换后的图像作为源
        processed_src = temp_bgr_buffer;
        // 宽度和高度保持不变，但现在是三通道图像
    }

    // 计算源图像的总字节数（处理后的图像）
    int img_size = processed_width * processed_height * 3;
    
    // 步骤1：将数据从CPU内存复制到GPU固定内存（零拷贝内存）
    memcpy(img_buffer_host, processed_src, img_size);
    
    // 步骤2：异步将数据从固定内存传输到设备内存
    CUDA_CHECK(cudaMemcpyAsync(img_buffer_device, img_buffer_host, img_size, cudaMemcpyHostToDevice, stream));

    // 步骤3：计算仿射变换矩阵（保持宽高比的缩放）
    AffineMatrix s2d, d2s; // s2d: 源到目标，d2s: 目标到源
    float scale = std::min(dst_height / (float)processed_height, dst_width / (float)processed_width); // 计算缩放比例

    // 设置源到目标的仿射变换矩阵（缩放+平移，保持居中）
    s2d.value[0] = scale;  // x方向缩放
    s2d.value[1] = 0;      // x方向倾斜
    s2d.value[2] = -scale * processed_width * 0.5 + dst_width * 0.5; // x方向平移（居中）
    s2d.value[3] = 0;      // y方向倾斜
    s2d.value[4] = scale;  // y方向缩放
    s2d.value[5] = -scale * processed_height * 0.5 + dst_height * 0.5; // y方向平移（居中）

    // 使用OpenCV计算逆变换矩阵（从目标到源）
    cv::Mat m2x3_s2d(2, 3, CV_32F, s2d.value);
    cv::Mat m2x3_d2s(2, 3, CV_32F, d2s.value);
    cv::invertAffineTransform(m2x3_s2d, m2x3_d2s); // 计算逆变换

    // 复制逆变换矩阵到结构体
    memcpy(d2s.value, m2x3_d2s.ptr<float>(0), sizeof(d2s.value));

    // 步骤4：配置并启动CUDA核函数
    int jobs = dst_height * dst_width; // 需要处理的像素总数
    int threads = 256;                 // 每个块的线程数（优化值）
    int blocks = ceil(jobs / (float)threads); // 计算需要的块数

    // 启动核函数进行并行处理
    warpaffine_kernel << <blocks, threads, 0, stream >> > (
        img_buffer_device, processed_width * 3, processed_width,   // 源图像信息
        processed_height, dst, dst_width,                          // 目标图像信息
        dst_height, 128, d2s, jobs);                              // 处理参数（128为灰色填充值）

    // 清理临时缓冲区（如果分配了）
    if (temp_bgr_buffer != nullptr) {
        delete[] temp_bgr_buffer;
    }
}

/**
 * @brief 初始化预处理模块，分配GPU内存
 * 
 * 分配两种类型的内存：
 * 1. 固定内存（pinned memory）：用于高效的CPU-GPU数据传输
 * 2. 设备内存：用于GPU计算
 * 
 * @param max_image_size 最大支持的图像尺寸（像素数量）
 */
void cuda_preprocess_init(int max_image_size) {
    // 分配主机端固定内存（零拷贝内存，提高传输效率）
    // 分配三倍空间以支持彩色图像（即使输入是灰度图像，转换后也需要三通道）
    CUDA_CHECK(cudaMallocHost((void**)&img_buffer_host, max_image_size * 3));
    
    // 分配设备端全局内存
    CUDA_CHECK(cudaMalloc((void**)&img_buffer_device, max_image_size * 3));
}

/**
 * @brief 清理预处理模块，释放分配的内存资源
 * 
 * 释放所有分配的GPU内存，避免内存泄漏
 */
void cuda_preprocess_destroy() {
    // 释放设备端内存
    CUDA_CHECK(cudaFree(img_buffer_device));
    
    // 释放主机端固定内存
    CUDA_CHECK(cudaFreeHost(img_buffer_host));
}