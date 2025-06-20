#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

__global__ __launch_bounds__(1024) void
mysgemm_v1(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

    int gx = blockIdx.x * blockDim.x + threadIdx.x; // global x
    int gy = blockIdx.y * blockDim.y + threadIdx.y; // global y

    float tmp = 0.;
    for (int i = 0; i < K; i++) {
        tmp += A[gy * K + i] * B[i * N + gx]; // Two global memory accesses and one FMA (accumulate multiply)
    }
    C[gy * N + gx] = alpha * tmp + beta * C[gy * N + gx];
}