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
__global__ void mysgemm_v7(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int block_row_thread = BN / TN;
    const int block_col_thread = BM / TM;
    const int thread_num = block_row_thread * block_col_thread; // One thread is responsible for calculating TM*TN elements in the block

    // The position of the top-left element of the thread tile corresponding to the current thread in the block
    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;

    __shared__ float As[2][BK * BM]; // Double the shared memory size for caching
    __shared__ float Bs[2][BK * BN];


    const int ldg_a_num = BK * BM / thread_num / 4; // Each thread moves 4 floats, all threads need ldg_a_num rounds to move to As
    const int ldg_b_num = BK * BN / thread_num / 4; // Each thread moves 4 floats, all threads need ldg_b_num rounds to move to Bs

    int a_tile_row = threadIdx.x / (BK / 4); // Each row is a 4-byte memory block, current thread is responsible for the a_tile_col block in a_tile_row row
    int a_tile_col = threadIdx.x % (BK / 4) * 4;
    int a_tile_stride = BM / ldg_a_num; // There are BM rows in total, ldg_a_num rounds, each round moves a_tile_stride rows

    int b_tile_row = threadIdx.x / (BN / 4); // Each row is a 4-byte memory block, current thread is responsible for the b_tile_col block in b_tile_row row
    int b_tile_col = threadIdx.x % (BN / 4) * 4;
    int b_tile_stride = BK / ldg_b_num; // There are BK rows in total, ldg_b_num rounds, each round moves b_tile_stride rows

    float accum[TM][TN] = {0.}; // Each thread is responsible for TM*TN elements, so TM*TN registers are needed to store the accumulated values, with an extra register for caching

    // All parameters for ldg_a_num must be const, otherwise cannot be used to declare array size
    float ldg_a_reg[4 * ldg_a_num] = {0.}; // Each thread moves ldg_a_num rounds, register caches ldg_a_num float4 elements, used for transposing As matrix
    float ldg_b_reg[4 * ldg_b_num] = {0.}; // Each thread moves ldg_a_num rounds, register caches ldg_a_num float4 elements, used for transposing As matrix

    float a_frag[2][TM];  // Cache As shared memory, double the register size for caching
    float b_frag[2][TN];  // Cache Bs shared memory, double the register size for caching

    // Move to current block
    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    // first global to shared
#pragma unroll
    for (int i = 0; i < BM; i += a_tile_stride) {
        int ldg_index = i / a_tile_stride * 4;  // ldg_index-th round
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
                FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
        // As is stored transposed, ldg_a_reg is used as an intermediate cache, so that reading can be done as FLOAT4
        As[0][OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
        As[0][OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
        As[0][OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
        As[0][OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
    }
#pragma unroll
    for (int i = 0; i < BK; i += b_tile_stride) {
        FETCH_FLOAT4(Bs[0][OFFSET(b_tile_row + i, b_tile_col, BN)]) =
                FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]); // No need to transpose
    }
    __syncthreads();

    // first shared to frag
#pragma unroll
    for (int m = 0; m < TM; m += 4) {
        FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(As[0][OFFSET(0, ty + m, BM)]); // Offset to current thread tile
    }
#pragma unroll
    for (int n = 0; n < TN; n += 4) {
        FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(Bs[0][OFFSET(0, tx + n, BN)]); // Offset to current thread tile
    }


    int write_index = 1;
    int load_index;
    int k = 0;
    do {
        k += BK;
        // load global to reg
        if (k < K) {
#pragma unroll
            for (int i = 0; i < BM; i += a_tile_stride) {
                int ldg_index = i / a_tile_stride * 4;  // ldg_index-th round
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
                        FETCH_FLOAT4(A[OFFSET(a_tile_row + i, k + a_tile_col, K)]);
            }
#pragma unroll
            for (int i = 0; i < BK; i += b_tile_stride) {
                int ldg_index = i / b_tile_stride * 4;  // ldg_index-th round
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) =
                        FETCH_FLOAT4(B[OFFSET(k + b_tile_row + i, b_tile_col, N)]);
            }
        }

        load_index = write_index ^ 1;
#pragma unroll
        for (int bk = 0; bk < BK - 1; bk++) {
            for (int m = 0; m < TM; m += 4) {
                FETCH_FLOAT4(a_frag[(bk + 1) % 2][m]) = FETCH_FLOAT4(
                        As[load_index][OFFSET(bk + 1, ty + m, BM)]); // Offset to current thread tile
            }
#pragma unroll
            for (int n = 0; n < TN; n += 4) {
                FETCH_FLOAT4(b_frag[(bk + 1) % 2][n]) = FETCH_FLOAT4(
                        Bs[load_index][OFFSET(bk + 1, tx + n, BN)]); // Offset to current thread tile
            }
#pragma unroll
            for (int m = 0; m < TM; m++) {
                for (int n = 0; n < TN; n++) {
                    accum[m][n] += a_frag[bk % 2][m] * b_frag[bk % 2][n];
                }
            }
        }
        if (k < K) {
#pragma unroll
            for (int i = 0; i < BM; i += a_tile_stride) {
                int ldg_index = i / a_tile_stride * 4;
                As[write_index][OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
                As[write_index][OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
                As[write_index][OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
                As[write_index][OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
            }
#pragma unroll
            for (int i = 0; i < BK; i += b_tile_stride) {
                int ldg_index = i / b_tile_stride * 4;
                FETCH_FLOAT4(Bs[write_index][OFFSET(b_tile_row + i, b_tile_col, BN)]) =
                        FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }
            __syncthreads();
#pragma unroll
            for (int m = 0; m < TM; m += 4) {
                FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(
                        As[write_index][OFFSET(0, ty + m, BM)]); // Offset to current thread tile
            }
#pragma unroll
            for (int n = 0; n < TN; n += 4) {
                FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(
                        Bs[write_index][OFFSET(0, tx + n, BN)]); // Offset to current thread tile
            }

            write_index ^= 1;
        }
#pragma unroll
        for (int m = 0; m < TM; m++) {
#pragma unroll
            for (int n = 0; n < TN; n++) {
                accum[m][n] += a_frag[(BK - 1) % 2][m] * b_frag[(BK - 1) % 2][n];
            }
        }


    } while (k < K);
    
    // C = alpha*AB+C
#pragma unroll
    for (int m = 0; m < TM; m++) {
#pragma unroll
        for (int n = 0; n < TN; n += 4) {
            float4 ctmp = FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]);
            ctmp.x = alpha * accum[m][n] + beta * ctmp.x;
            ctmp.y = alpha * accum[m][n + 1] + beta * ctmp.y;
            ctmp.z = alpha * accum[m][n + 2] + beta * ctmp.z;
            ctmp.w = alpha * accum[m][n + 3] + beta * ctmp.w;
            FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]) = ctmp;
        }
    }
}