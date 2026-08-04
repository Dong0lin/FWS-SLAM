#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <iostream>
#include <string>
#include "YOLOv11.h"


/**
 * @brief 检查指定路径是否存在（跨平台实现）
 * 
 * 该函数使用平台特定的API来检查文件或目录路径是否存在
 * 
 * @param path 要检查的路径字符串
 * @return true 路径存在
 * @return false 路径不存在
 */
bool IsPathExist(const string& path) {
#ifdef _WIN32
    // Windows平台：使用GetFileAttributesA获取文件属性
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    // 如果属性不是无效值，说明路径存在
    return (fileAttributes != INVALID_FILE_ATTRIBUTES);
#else
    // Linux/Unix平台：使用access函数检查文件可访问性
    // F_OK标志用于检查文件是否存在
    return (access(path.c_str(), F_OK) == 0);
#endif
}



/**
 * @brief 检查指定路径是否为普通文件（跨平台实现）
 * 
 * 该函数首先检查路径是否存在，然后判断是否为普通文件而非目录
 * 
 * @param path 要检查的路径字符串
 * @return true 路径存在且为普通文件
 * @return false 路径不存在或不是普通文件
 */
bool IsFile(const string& path) {
    // 首先检查路径是否存在
    if (!IsPathExist(path)) {
        // 路径不存在时输出错误信息
        printf("%s:%d %s not exist\n", __FILE__, __LINE__, path.c_str());
        return false;
    }

#ifdef _WIN32
    // Windows平台：获取文件属性并检查是否为目录
    DWORD fileAttributes = GetFileAttributesA(path.c_str());
    // 路径存在且不是目录（FILE_ATTRIBUTE_DIRECTORY标志为0）
    return ((fileAttributes != INVALID_FILE_ATTRIBUTES) && ((fileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0));
#else
    // Linux/Unix平台：使用stat获取文件状态信息
    struct stat buffer;
    // 路径存在且是普通文件（S_ISREG检查是否为常规文件）
    return (stat(path.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
#endif
}

/**
 * @brief TensorRT日志记录器类
 * 
 * 该类继承自nvinfer1::ILogger接口，用于处理TensorRT引擎的日志输出
 * 只输出警告级别及以上的日志信息，避免过多的调试信息干扰
 */
class Logger : public nvinfer1::ILogger {
    /**
     * @brief 日志记录方法（重写ILogger接口）
     * 
     * 根据日志严重性级别决定是否输出日志信息
     * 只输出警告（kWARNING）、错误（kERROR）和内部错误（kINTERNAL_ERROR）级别的日志
     * 
     * @param severity 日志严重性级别
     * @param msg 日志消息内容
     */
    void log(Severity severity, const char* msg) noexcept override {
        // 只输出严重性级别大于等于警告的日志
        // 忽略信息（kINFO）和详细（kVERBOSE）级别的日志
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;  // 全局日志记录器实例


/**
 * @brief YOLOv11-TensorRT目标检测程序主函数
 * 
 * 该函数是程序的入口点，负责处理命令行参数、加载模型、执行目标检测并显示结果
 * 支持对单张图像、图像文件夹和视频文件进行目标检测
 * 
 * @param argc 命令行参数个数，必须为3（程序名、引擎文件路径、检测目标路径）
 * @param argv 命令行参数数组
 *             argv[0]: 程序名称
 *             argv[1]: TensorRT引擎文件路径（.engine文件）
 *             argv[2]: 检测目标路径（图像文件、视频文件或图像文件夹）
 * @return int 程序退出状态码（0表示成功）
 */
int main(int argc, char** argv)
{
    // 解析命令行参数
    const string engine_file_path{ argv[1] };  // TensorRT引擎文件路径
    const string path{ argv[2] };              // 检测目标路径（图像/视频/文件夹）
    vector<string> imagePathList;              // 图像路径列表
    bool isVideo{ false };                     // 标记是否为视频文件
    
    // 验证命令行参数数量，必须为3个
    assert(argc == 3);

    // 判断输入路径类型并处理
    if (IsFile(path))
    {
        // 如果是文件，提取文件扩展名
        string suffix = path.substr(path.find_last_of('.') + 1);
        
        // 检查是否为支持的图像格式
        if (suffix == "jpg" || suffix == "jpeg" || suffix == "png")
        {
            imagePathList.push_back(path);  // 添加到图像路径列表
        }
        // 检查是否为支持的视频格式
        else if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || 
                 suffix == "mpeg" || suffix == "mov" || suffix == "mkv" || suffix == "webm")
        {
            isVideo = true;  // 标记为视频文件
        }
        else {
            // 不支持的格式，输出错误信息并终止程序
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            abort();
        }
    }
    else if (IsPathExist(path))
    {
        // 如果是文件夹，使用glob函数获取文件夹中的所有jpg图像
        glob(path + "/*.jpg", imagePathList);
    }

    // 初始化YOLOv11模型
    YOLOv11 model(engine_file_path, logger);

    // 视频处理分支
    if (isVideo) {
        // 打开视频文件
        cv::VideoCapture cap(path);

        // 循环处理视频的每一帧
        while (1)
        {
            Mat image;
            cap >> image;  // 读取下一帧

            // 如果帧为空（视频结束），退出循环
            if (image.empty()) break;

            vector<Detection> objects;  // 检测结果容器
            
            // 执行目标检测流程
            model.preprocess(image);  // 图像预处理

            // 记录推理开始时间
            auto start = std::chrono::system_clock::now();
            model.infer();  // 执行模型推理
            auto end = std::chrono::system_clock::now();  // 记录推理结束时间

            model.postprocess(objects);  // 后处理，解析检测结果
            model.draw(image, objects);  // 在图像上绘制检测框

            // 计算并输出推理耗时（毫秒）
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);

            // 显示检测结果
            imshow("prediction", image);
            waitKey(1);  // 等待1毫秒，保持视频播放流畅
        }

        // 释放资源
        destroyAllWindows();  // 关闭所有OpenCV窗口
        cap.release();        // 释放视频捕获对象
    }
    else {
        // 图像处理分支（处理单张图像或图像文件夹）
        for (const auto& imagePath : imagePathList)
        {
            // 打开图像文件
            Mat image = imread(imagePath);
            if (image.empty())
            {
                // 图像读取失败，输出错误信息并跳过
                cerr << "Error reading image: " << imagePath << endl;
                continue;
            }

            vector<Detection> objects;  // 检测结果容器
            
            // 执行目标检测流程
            model.preprocess(image);  // 图像预处理

            // 记录推理开始时间
            auto start = std::chrono::system_clock::now();
            model.infer();  // 执行模型推理
            auto end = std::chrono::system_clock::now();  // 记录推理结束时间

            model.postprocess(objects);  // 后处理，解析检测结果
            model.draw(image, objects);  // 在图像上绘制检测框

            // 计算并输出推理耗时（毫秒）
            auto tc = (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.;
            printf("cost %2.4lf ms\n", tc);

            // 再次绘制检测结果（确保显示效果）
            model.draw(image, objects);
            imshow("Result", image);  // 显示结果图像

            waitKey(0);  // 等待用户按键，单张图像模式下保持显示
        }
    }

    return 0;  // 程序正常退出
}