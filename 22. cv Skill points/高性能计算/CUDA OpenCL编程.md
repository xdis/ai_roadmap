# CUDA 与 OpenCL 编程入门

## 简介

CUDA 和 OpenCL 是两种主流的并行计算框架，它们允许开发者利用 GPU 的强大计算能力来加速计算密集型任务。

- **CUDA (Compute Unified Device Architecture)**: 由 NVIDIA 开发的并行计算平台和编程模型，专用于 NVIDIA GPU。
- **OpenCL (Open Computing Language)**: 由 Khronos 集团开发的开放标准，可在各种硬件上运行，包括 CPU、GPU、DSP 和 FPGA 等。

## 基本概念

### 1. 并行计算架构

GPU 与 CPU 的区别在于：
- CPU: 少量强大的核心，适合串行任务
- GPU: 大量简单的核心，适合并行任务

![CPU vs GPU](https://mermaid.ink/img/pako:eNptkU1PwzAMhv9KlBOIScCh14rCgAlxQBw4hB6sxm2jJXGVOBuT-O-4_Vgn7ZbXfvw4fh3tSFVoaLzjJ4_OW_xIgY0jzBJrkhHg0wlZx9qEUFkUucbsJXiCCxeCPd0cnR1Y2oRnQsNQ8YwK7ZG3-o2-UJ7eS8OdxM7Rgnt0UgfTJ9l2IwEOe4iM5lZAE6P0fS6k8CywZV5eXt_V2Qal2dJwCCkJJ_nfzUaZ01CKLZhKbsmO6L6u1wjEMn0Ou1AhfB0YtugC2_3v9lftlM-5lq6g8VNKYsxmNJmUoqIJkc-x14_UDT4N2_P-J7NolHM_DHU0fO9GpcM3NPo35w?type=png)

### 2. 核心编程模型

两者的编程模型相似，包括：
- **主机端代码**（Host Code）：在 CPU 上运行
- **设备端代码**（Device Code）：在 GPU 上运行的内核函数

## CUDA 编程示例

以下是一个简单的向量加法 CUDA 程序：

```c
// 核函数定义 - 在 GPU 上运行
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // 计算当前线程处理的元素索引
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    // 确保不越界
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    // 向量大小
    int n = 10000;
    size_t size = n * sizeof(float);
    
    // 分配主机内存
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);
    
    // 初始化向量
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // 分配设备内存
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // 拷贝数据到设备
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // 定义块和网格大小
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // 启动核函数
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    
    // 拷贝结果回主机
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // 验证结果
    for (int i = 0; i < n; i++) {
        if (fabs(h_c[i] - 3.0f) > 1e-5) {
            printf("验证失败！\n");
            break;
        }
    }
    
    printf("向量加法成功完成！\n");
    
    // 释放内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}
```

### 关键概念解释

1. **`__global__`**: 修饰符，表示这是一个核函数，可以从 CPU 调用，在 GPU 上执行
2. **线程层次结构**:
   - **线程(Thread)**: 最基本的执行单元
   - **线程块(Block)**: 多个线程组成一个块
   - **网格(Grid)**: 多个块组成一个网格

3. **内存管理**:
   - `cudaMalloc`: 在 GPU 上分配内存
   - `cudaMemcpy`: 在主机和设备之间复制数据
   - `cudaFree`: 释放 GPU 内存

4. **核函数启动语法**: `kernelName<<<blocksPerGrid, threadsPerBlock>>>(参数列表);`

## OpenCL 编程示例

OpenCL 实现同样的向量加法:

```c
#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

// 向量加法的 OpenCL 核心代码
const char *kernelSource = 
"__kernel void vectorAdd(__global const float *a, __global const float *b, __global float *c, int n) { \n"
"    int i = get_global_id(0);                                                  \n"
"    if (i < n) {                                                               \n"
"        c[i] = a[i] + b[i];                                                    \n"
"    }                                                                          \n"
"}                                                                              \n";

int main() {
    // 向量大小
    int n = 10000;
    size_t datasize = n * sizeof(float);
    
    // 分配主机内存并初始化
    float *h_a = (float*)malloc(datasize);
    float *h_b = (float*)malloc(datasize);
    float *h_c = (float*)malloc(datasize);
    
    for(int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // 获取平台和设备
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    
    // 创建上下文和命令队列
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
    
    // 创建内存对象
    cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, NULL);
    cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, NULL);
    cl_mem d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, NULL);
    
    // 写入数据到设备
    clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, datasize, h_a, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, datasize, h_b, 0, NULL, NULL);
    
    // 创建并编译程序
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    
    // 创建核函数
    cl_kernel kernel = clCreateKernel(program, "vectorAdd", NULL);
    
    // 设置核函数参数
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    clSetKernelArg(kernel, 3, sizeof(int), &n);
    
    // 定义工作项数量
    size_t global_size = n;
    size_t local_size = 256;
    
    // 执行核函数
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
    
    // 读取结果
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, datasize, h_c, 0, NULL, NULL);
    
    // 验证结果
    for(int i = 0; i < n; i++) {
        if(fabs(h_c[i] - 3.0f) > 1e-5) {
            printf("验证失败！\n");
            break;
        }
    }
    
    printf("向量加法成功完成！\n");
    
    // 清理资源
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}
```

### OpenCL 关键概念

1. **平台模型**:
   - **Platform**: 代表一个 OpenCL 实现，如 NVIDIA 或 AMD
   - **Device**: 平台中的计算设备，如 GPU 或 CPU
   - **Context**: 执行环境
   - **Command Queue**: 命令执行队列

2. **内存模型**:
   - 使用 `clCreateBuffer` 创建内存对象
   - 使用 `clEnqueueWriteBuffer` 和 `clEnqueueReadBuffer` 传输数据

3. **执行模型**:
   - 使用 `clEnqueueNDRangeKernel` 执行核函数
   - **Work-Item**: 类似 CUDA 的线程
   - **Work-Group**: 类似 CUDA 的线程块

## CUDA 与 OpenCL 比较

| 特性 | CUDA | OpenCL |
|------|------|--------|
| 兼容性 | 仅 NVIDIA GPU | 跨平台（CPU, GPU, FPGA 等） |
| 性能 | 在 NVIDIA 硬件上通常更快 | 可能略低于 CUDA |
| 学习曲线 | 相对平缓 | 较陡峭 |
| 开发工具 | 完善 | 相对有限 |
| 社区支持 | 强大 | 有但不如 CUDA |

## 实际应用场景

1. **图像处理**: 卷积、滤波、特征提取
2. **深度学习**: 矩阵乘法加速、神经网络训练
3. **物理模拟**: 流体动力学、分子动力学
4. **金融分析**: 蒙特卡洛模拟、风险计算
5. **密码学**: 密码破解、哈希计算

## 入门建议

1. 确定你的硬件(NVIDIA GPU 选 CUDA，其他情况考虑 OpenCL)
2. 安装必要的开发工具(CUDA Toolkit 或 OpenCL SDK)
3. 从简单例子开始学习
4. 理解并行思维方式
5. 优化代码性能(避免线程分歧、优化内存访问等)

## 总结

- CUDA 和 OpenCL 是利用 GPU 进行高性能计算的主要框架
- 基本原理相似，但 API 和具体实现不同
- CUDA 专注于 NVIDIA 硬件，提供更好的性能和开发体验
- OpenCL 提供跨平台能力，适用于多种硬件架构

通过合理使用这些技术，可以将计算密集型任务的性能提升 10-100 倍，甚至更多。