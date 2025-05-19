#include <cuda_runtime.h>

__global__ void vector_add(const float *a, const float *b, float *c, size_t n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    c[i] = a[i] - b[i];
  }
}

extern "C" void solution(const float *d_input1, const float *d_input2, float *d_output, size_t n)
{
  int threads_per_block = 512;
  int num_blocks = (n + threads_per_block - 1) / threads_per_block;
  vector_add<<<num_blocks, threads_per_block>>>(d_input1, d_input2, d_output, n);
}