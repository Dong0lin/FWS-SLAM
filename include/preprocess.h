#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <opencv2/opencv.hpp>

/**
 * @brief 初始化CUDA预处理模块，分配GPU内存
 * @param max_image_size 最大支持的图像尺寸（像素数量）
 * 
 * 该函数在GPU上分配固定内存(pinned memory)和设备内存，
 * 用于存储输入图像数据，提高数据传输效率。
 */
void cuda_preprocess_init(int max_image_size);

/**
 * @brief 销毁CUDA预处理模块，释放分配的内存资源
 * 
 * 清理所有分配的GPU内存，包括固定内存和设备内存。
 * 应在程序退出前调用此函数以避免内存泄漏。
 */
void cuda_preprocess_destroy();

/**
 * @brief 在GPU上执行图像预处理操作
 * 
 * 该函数将输入图像从CPU内存传输到GPU，执行仿射变换、颜色空间转换、
 * 归一化等预处理操作，并将结果存储在指定的GPU缓冲区中。
 * 
 * @param src 源图像数据指针（CPU端，BGR格式）
 * @param src_width 源图像宽度
 * @param src_height 源图像高度
 * @param dst 目标缓冲区指针（GPU端，RGB归一化格式）
 * @param dst_width 目标图像宽度（模型输入宽度）
 * @param dst_height 目标图像高度（模型输入高度）
 * @param stream CUDA流，用于异步操作
 * 
 * 预处理流程：
 * 1. 保持宽高比的缩放（letterbox方式）
 * 2. BGR到RGB颜色空间转换
 * 3. 像素值归一化到[0,1]范围
 * 4. 通道重排：从RGBRGBRGB到RRRGGGBBB格式
 */
void cuda_preprocess(uint8_t* src, int src_width, int src_height,
    float* dst, int dst_width, int dst_height,
    cudaStream_t stream);