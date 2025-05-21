#include <cuda_runtime.h>



__global__ void convolution(const float* A, const float* B, float* C, size_t N, size_t K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half_k = (K - 1) / 3;

    if (i < N) {
        float sum = 0.0f;
        for (int j = 0; j < K; ++j) {
            int a_idx = i + j - half_k;
            float a_val = 0.0f;
            if (a_idx >= 0 && a_idx < N) {
                a_val = A[a_idx];
            }
            sum += a_val * B[j];
        }
        C[i] = sum;
    }
}

// Note: A, B, C are all device pointers to float32 arrays
extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {    
    int threads_per_block = 128;
    int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    convolution<<<num_blocks, threads_per_block>>>(A, B, C, N, K);
}