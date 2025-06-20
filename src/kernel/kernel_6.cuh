#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

#define OFFSET(row, col, ld) ((row)*(ld)+(col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template<const int BM,
        const int BN,
        const int BK,
        const int TM,
        const int TN>
__global__ void mysgemm_v6(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int block_row_thread = BN / TN;
    const int block_col_thread = BM / TM;
    const int thread_num = block_row_thread * block_col_thread; // One thread is responsible for TM*TN elements in the block

    // The top-left element of the thread tile in the block for the current thread
    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;

    __shared__ float As[BK * BM];
    __shared__ float Bs[BK * BN];


    const int ldg_a_num = BK * BM / thread_num / 4; // Each thread moves 4 floats, all threads together need ldg_a_num rounds to move As
    const int ldg_b_num = BK * BN / thread_num / 4; // Each thread moves 4 floats, all threads together need ldg_b_num rounds to move Bs

    int a_tile_row = threadIdx.x / (BK / 4); // Each 4 bytes per row as a memory block, current thread moves the a_tile_col-th block in a_tile_row-th row
    int a_tile_col = threadIdx.x % (BK / 4) * 4;
    int a_tile_stride = BM / ldg_a_num; // BM rows in total, ldg_a_num rounds, each round moves a_tile_stride rows

    int b_tile_row = threadIdx.x / (BN / 4); // Each 4 bytes per row as a memory block, current thread moves the b_tile_col-th block in b_tile_row-th row
    int b_tile_col = threadIdx.x % (BN / 4) * 4;
    int b_tile_stride = BK / ldg_b_num; // BK rows in total, ldg_b_num rounds, each round moves b_tile_stride rows

    float accum[TM][TN] = {0.}; // Each thread is responsible for TM*TN elements, so TM*TN registers are needed to store accumulations, one extra for cache

    // All parameters for ldg_a_num must be const, otherwise cannot be used for array size declaration
    float ldg_a_reg[4 * ldg_a_num] = {0.}; // Each thread moves ldg_a_num rounds, register caches ldg_a_num float4 elements, used for transposing As

    float a_frag[TM];  // Cache for As shared memory
    float b_frag[TN];  // Cache for Bs shared memory

    // Move to current block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

#pragma unroll
    for (int k = 0; k < K; k += BK) {
#pragma unroll
        for (int i = 0; i < BM; i += a_tile_stride) {
            int ldg_index = i / a_tile_stride * 4;  // ldg_index-th round
            FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
                    FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
            // Transpose As, ldg_a_reg as intermediate cache, to allow reading as FLOAT4
            As[OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
            As[OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
            As[OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
            As[OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
        }
#pragma unroll
        for (int i = 0; i < BK; i += b_tile_stride) {
            FETCH_FLOAT4(Bs[OFFSET(b_tile_row + i, b_tile_col, BN)]) =
                    FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]); // No need to transpose
        }
        __syncthreads();
        A += BK;
        B += BK * N;
#pragma unroll
        for (int i = 0; i < BK; i++) {
#pragma unroll
            for (int m = 0; m < TM; m += 4) {
                FETCH_FLOAT4(a_frag[m]) = FETCH_FLOAT4(As[OFFSET(i, ty + m, BM)]); // Offset to current thread tile
            }
#pragma unroll
            for (int n = 0; n < TN; n += 4) {
                FETCH_FLOAT4(b_frag[n]) = FETCH_FLOAT4(Bs[OFFSET(i, tx + n, BN)]); // Offset to current thread tile
            }
#pragma unroll
            for (int m = 0; m < TM; m++) {
#pragma unroll
                for (int n = 0; n < TN; n++) {
                    accum[m][n] += a_frag[m] * b_frag[n];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            float4 ctmp = FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]);
            //float4 atmp = FETCH_FLOAT4(accum[m][n]);
            ctmp.x = alpha * accum[m][n] + beta * ctmp.x;
            ctmp.y = alpha * accum[m][n + 1] + beta * ctmp.y;
            ctmp.z = alpha * accum[m][n + 2] + beta * ctmp.z;
            ctmp.w = alpha * accum[m][n + 3] + beta * ctmp.w;
            FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]) = ctmp;
        }
    }
}