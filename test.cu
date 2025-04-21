#include <cuda_runtime.h>
#include <iostream>

__global__ void addVectors(const float* a, const float* b, float* c, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

int main() {
  const int N = 1000;
  const int SIZE = N * sizeof(float);

  // Host memory allocation
  float *h_a = new float[N];
  float *h_b = new float[N];
  float *h_c = new float[N];

  // Initialize input vectors
  for (int i = 0; i < N; i++) {
    h_a[i] = static_cast<float>(i);
    h_b[i] = static_cast<float>(i * 2);
  }

  // Device memory allocation
  float *d_a, *d_b, *d_c;
  cudaMalloc((void**)&d_a, SIZE);
  cudaMalloc((void**)&d_b, SIZE);
  cudaMalloc((void**)&d_c, SIZE);

  // Copy data from host to device
  cudaMemcpy(d_a, h_a, SIZE, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, SIZE, cudaMemcpyHostToDevice);

  // Launch kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

  // Copy result back to host
  cudaMemcpy(h_c, d_c, SIZE, cudaMemcpyDeviceToHost);

  // Print a few results
  for (int i = 0; i < 10; i++) {
    std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;
  }

  // Free memory
  delete[] h_a;
  delete[] h_b;
  delete[] h_c;
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}