#include <cuda_runtime.h>
#include <iostream>

int main(){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // 获取 GPU 0 的设备属性
    int sm_count = prop.multiProcessorCount;  // 获取 GPU 上的 SM 数量
    std::cout << "GPU has " << sm_count << " SMs." << std::endl;
}