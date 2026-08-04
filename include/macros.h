#ifndef __MACROS_H
#define __MACROS_H

#ifdef API_EXPORTS
#if defined(_MSC_VER)
#define API __declspec(dllexport)  // Windows平台：导出符号
#else
#define API __attribute__((visibility("default")))  // Linux/macOS：导出符号
#endif
#else
#if defined(_MSC_VER)
#define API __declspec(dllimport)  // Windows平台：导入符号
#else
#define API  // Linux/macOS：默认可见性
#endif
#endif  // API_EXPORTS




#if NV_TENSORRT_MAJOR >= 8
#define TRT_NOEXCEPT noexcept      // TensorRT 8+：使用noexcept
#define TRT_CONST_ENQUEUE const    // TensorRT 8+：enqueue函数为const
#else
#define TRT_NOEXCEPT               // TensorRT 7及以下：不使用noexcept
#define TRT_CONST_ENQUEUE          // TensorRT 7及以下：enqueue函数非const
#endif

#endif  // __MACROS_H