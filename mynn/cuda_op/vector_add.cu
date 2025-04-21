#include <cuda_runtime.h>
#include <stdio.h>

// CUDA 核函数：每个线程处理一个元素
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// 封装函数，供外部调用
extern "C" __declspec(dllexport) void vector_add_cuda(float *a, float *b, float *c, int n) {
    float *d_a, *d_b, *d_c;  // 设备端的指针
    size_t size = n * sizeof(float);

    // 在 GPU 上分配内存
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // 将数据从主机复制到设备
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // 启动核函数
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // 将结果从设备复制回主机
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // 释放 GPU 内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}