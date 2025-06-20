#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

/*
=====================================
CUDA operations
=====================================
*/
void cudaCheck(cudaError_t error, const char *file, int line); // CUDA error checking
void CudaDeviceInfo();                                         // Print CUDA info

/*
=====================================
Matrix operations
=====================================
*/
void randomize_matrix(float *mat, int N);            // Randomly initialize matrix
void copy_matrix(float *src, float *dest, int N);    // Copy matrix
void print_matrix(const float *A, int M, int N);     // Print matrix
bool verify_matrix(float *mat1, float *mat2, int N); // Verify matrix

/*
=====================================
Timing operations
=====================================
*/
float get_current_sec();                        // Get current time
float cpu_elapsed_time(float &beg, float &end); // Calculate time difference

/*
=====================================
Kernel operations
=====================================
*/
// Call the specified kernel function to compute matrix multiplication
void test_kernel(int kernel_num, int m, int n, int k, float alpha, float *A, float *B, float beta, float *C, cublasHandle_t handle);