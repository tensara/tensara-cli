#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  size_t total_elements,
                                  float alpha)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (; idx < total_elements; idx += stride) {
        float val = input[idx];
        output[idx] = (val > 0.0f) ? val : alpha * val;
    }
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output, size_t n, size_t m, float alpha) {    
    size_t total_elements = n * m;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    leaky_relu_kernel<<<grid_size, block_size>>>(input, output, total_elements, alpha);
    cudaDeviceSynchronize();
}
