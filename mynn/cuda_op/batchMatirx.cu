#include <stdio.h>
#define TILE_WIDTH 16  // Define a block size

// Kernel for batch matrix multiplication
__global__ void batchMatrixMulKernel(float* A, float* B, float* C, int M, int N, int K, int batch_size) {
    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Compute the row and column of C for this thread
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float value = 0.0f;

    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        // Load A and B sub-matrices into shared memory
        if (row < M && t * TILE_WIDTH + tx < K) {
            sharedA[ty][tx] = A[blockIdx.z * M * K + row * K + t * TILE_WIDTH + tx];
        } else {
            sharedA[ty][tx] = 0.0f;
        }

        if (col < N && t * TILE_WIDTH + ty < K) {
            sharedB[ty][tx] = B[blockIdx.z * K * N + (t * TILE_WIDTH + ty) * N + col];
        } else {
            sharedB[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Multiply the sub-matrices
        for (int k = 0; k < TILE_WIDTH; k++) {
            value += sharedA[ty][k] * sharedB[k][tx];
        }

        __syncthreads();
    }

    // Store the result into C if the row and column are valid
    if (row < M && col < N) {
        C[blockIdx.z * M * N + row * N + col] = value;
    }
}

// Function to print matrix (for debugging)
void printM(float* A, size_t M, size_t N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f\t", A[i * N + j]);
        }
        printf("\n");
    }
}

// The entry point for external Python code (e.g., PyTorch or custom wrapper)
extern "C" void batchMatrixMultiplication(float* A, float* B, float* C, int M, int N, int K, int batch_size) {
    float *d_A, *d_B, *d_C;
    size_t sizeA = batch_size * M * K * sizeof(float);
    size_t sizeB = batch_size * K * N * sizeof(float);
    size_t sizeC = batch_size * M * N * sizeof(float);

    // Allocate memory on the device
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    // Copy data from host to device
    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH, batch_size);

    batchMatrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K, batch_size);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result from device to host
    cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

