#include <cuda_runtime.h>
#include <stdio.h>


__global__ void conv2d_forward(float *input, float *kernel, float *bias, float *output,
    int batchsize, int in_channels, int out_channels,
    int H, int W, int kernel_size, int stride, int padding) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // W
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // H
    int b = blockIdx.z / out_channels; // batch_size
    int o = blockIdx.z % out_channels; // channel


    int new_H = (H - kernel_size + 2 * padding) / stride + 1;
    int new_W = (W - kernel_size + 2 * padding) / stride + 1;

    if (i < new_H && j < new_W && b < batchsize && o < out_channels) {
        float sum = 0.0f;
        for (int c = 0; c < in_channels; ++c) {
            for (int ki = 0; ki < kernel_size; ++ki) {
                for (int kj = 0; kj < kernel_size; ++kj) {
                    int input_i = i * stride + ki - padding;
                    int input_j = j * stride + kj - padding;
                    if (input_i >= 0 && input_i < H && input_j >= 0 && input_j < W) {
                        sum += input[b * in_channels * H * W + c * H * W + input_i * W + input_j] *
                        kernel[o * in_channels * kernel_size * kernel_size + c * kernel_size * kernel_size + ki * kernel_size + kj];
                    }
                }
            }
        }
        output[b * out_channels * new_H * new_W + o * new_H * new_W + i * new_W + j] = sum + bias[o];
    }
}

