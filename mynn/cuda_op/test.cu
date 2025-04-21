#include <stdio.h>
#define THREADS_PER_BLOCK 1024


__global__ void printBlockIdx(float* d_A, float* d_B, float* d_C, int size) {
    __shared__ float sharedMem[THREADS_PER_BLOCK];

    if (threadIdx.x<size){
        sharedMem[threadIdx.x] = d_A[threadIdx.x+blockIdx.x*size]*d_B[threadIdx.x*blockDim.x+blockIdx.y]; // per element mult
        // printf("%d, %d\n", threadIdx.x, sharedMem[threadIdx.x]);
    }else{
        printf("is zero!\n");
        sharedMem[threadIdx.x] = 0;
    }
    __syncthreads();
    // add them up 
    for (int stride = THREADS_PER_BLOCK/2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sharedMem[threadIdx.x] += sharedMem[threadIdx.x + stride];
        }
        __syncthreads();  
    }
    if (threadIdx.x == 0) {
        d_C[blockIdx.x+blockIdx.y*blockDim.x] = sharedMem[0];
    }
}

void printM(float* A, size_t N, size_t M){
    for(int i=0; i<N; i++){
        for(int j=0; j<M; j++){
            printf("%f\t",A[i*M+j]);
        }
        printf("\n");
    }
}

// extern "C" __declspec(dllexport) void matrix_mult();

int main() {
    // specify the size of the matrix, and allocate the space in cpu
    float *h_A, *h_B, *h_C; // the matirx in the cpu side
    float *d_A, *d_B, *d_C; // the matrix in the gpu side
    int M = 10240;
    int N = 1024;
    size_t size = M * N * sizeof(float);
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // initialize A and B
    for (int i = 0; i < M * N; i++) {
        h_A[i] = 1.2;
        h_B[i] = 2.2;
    }

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // copy to the device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = N;   // every block have N threads
    dim3 blockPerGrid(M, N);  // every grid have (N, N) blocks 

    // 运行核函数
    printBlockIdx<<<blockPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 等待 GPU 运行完成
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost); // copy to the host

    printM(h_C, M, N);

    return 0;
}
