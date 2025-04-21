#include <stdio.h>
#define THREADS_PER_BLOCK 16  // 16x16 block size
#define TILE_WIDTH 16

__global__ void matrixMulKernel(float* A, float* B, float* C, int M, int N) {
    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;
    
    float value = 0.0f;
    
    for (int t = 0; t < (N + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        if (row < M && t * TILE_WIDTH + tx < N) {
            sharedA[ty][tx] = A[row * N + t * TILE_WIDTH + tx];
        } else {
            sharedA[ty][tx] = 0.0f;
        }
        
        if (col < N && t * TILE_WIDTH + ty < N) {
            sharedB[ty][tx] = B[(t * TILE_WIDTH + ty) * N + col];
        } else {
            sharedB[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_WIDTH; k++) {
            value += sharedA[ty][k] * sharedB[k][tx];
        }
        
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

void printM(float* A, size_t N, size_t M) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f\t", A[i * N + j]);
        }
        printf("\n");
    }
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    int M = 10240, N = 1024;
    size_t size = M * N * sizeof(float);
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    for (int i = 0; i < M * N; i++) {
        h_A[i] = 1.2;
        h_B[i] = 2.2;
    }

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printM(h_C, M, N);

    return 0;
}
