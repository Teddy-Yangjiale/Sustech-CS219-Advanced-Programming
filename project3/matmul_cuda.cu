#include <cuda_runtime.h>
#include <stdio.h>
#include "matrix.h"

// ========================================================
// 核心密码：CUDA 版本的 "GOLD_MC, GOLD_NC, GOLD_KC"
// ========================================================
#define BM 128 // 每个 Block 处理 C 矩阵的行数 (类似于 MC)
#define BN 128 // 每个 Block 处理 C 矩阵的列数 (类似于 NC)
#define BK 8   // 每次内层循环加载进共享内存的 K 维度深度 (类似于 KC)

#define TM 8   // 每个线程处理的微内核行数 (类似于 6x16 里的 6)
#define TN 8   // 每个线程处理的微内核列数 (类似于 6x16 里的 16)

// 极致优化的 CUDA 内核
__global__ void matmul_cuda_optimized(float *A, float *B, float *C, int M, int N, int K) {
    // 1. 申请共享内存 (相当于你的 pack_A 和 pack_B 分配在极速 L1 缓存)
    __shared__ float sA[BM * BK];
    __shared__ float sB[BK * BN];

    // 2. 线程局部寄存器 (相当于 c00, c01... 等 ymm 寄存器)
    float rC[TM][TN] = {0.0f}; // 存储结果
    float rA[TM];              // 缓存 A 
    float rB[TN];              // 缓存 B

    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;

    // 当前线程负责的 C 矩阵的起始全局坐标
    int rowC = bidy * BM + tidy * TM;
    int colC = bidx * BN + tidx * TN;

    // 当前线程在 Block 内的一维 ID (用于协同搬运内存)
    // 一个 Block 有 (128/8)*(128/8) = 256 个线程
    int tid = tidy * blockDim.x + tidx;
    
    // 计算当前线程该去搬运 A 和 B 矩阵的哪个元素进共享内存
    int load_a_row = tid / BK;
    int load_a_col = tid % BK;
    int load_b_row = tid / BN;
    int load_b_col = tid % BN;

    // ========================================================
    // 外层 K 维度循环：分块加载 (Blocking / Tiling)
    // ========================================================
    for (int k = 0; k < K; k += BK) {
        
        // --- 步骤 A：所有人一起去内存把 A 和 B 搬进共享内存 (协同加载 + 边界保护) ---
        if (bidy * BM + load_a_row < M && k + load_a_col < K)
            sA[load_a_row * BK + load_a_col] = A[(bidy * BM + load_a_row) * K + (k + load_a_col)];
        else
            sA[load_a_row * BK + load_a_col] = 0.0f;

        if (k + load_b_row < K && bidx * BN + load_b_col < N)
            sB[load_b_row * BN + load_b_col] = B[(k + load_b_row) * N + (bidx * BN + load_b_col)];
        else
            sB[load_b_row * BN + load_b_col] = 0.0f;

        // 隐式屏障：等所有 256 个线程都搬完了，再开始计算
        __syncthreads();

        // --- 步骤 B：使用寄存器执行微内核计算 (相当于你的 kernel_6x16_intrinsic) ---
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // 将当前计算需要的数，从共享内存提取到最快的寄存器里
            for (int i = 0; i < TM; ++i) rA[i] = sA[(tidy * TM + i) * BK + dotIdx];
            for (int i = 0; i < TN; ++i) rB[i] = sB[dotIdx * BN + (tidx * TN + i)];

            // FMA (融合乘加) 核心指令执行区
            for (int i = 0; i < TM; ++i) {
                for (int j = 0; j < TN; ++j) {
                    rC[i][j] += rA[i] * rB[j];
                }
            }
        }
        
        // 等所有人算完这一个块，再进行下一次 K 循环的内存覆盖
        __syncthreads();
    }

    // 3. 将寄存器里的结果写回全局内存的 C 矩阵
    for (int i = 0; i < TM; ++i) {
        for (int j = 0; j < TN; ++j) {
            int global_row = rowC + i;
            int global_col = colC + j;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = rC[i][j];
            }
        }
    }
}

// 供 C 语言调用的包装接口
extern "C" void matmul_cuda(const Matrix* A, const Matrix* B, Matrix* C) {
    int M = A->rows;
    int N = B->cols;
    int K = A->cols;

    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // 申请显存
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // 拷贝数据到 GPU
    cudaMemcpy(d_A, A->data, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B->data, size_B, cudaMemcpyHostToDevice);

    // 设定 Block 大小。256个线程完美映射到 GPU 的 Warp 调度机制
    dim3 blockSize(BN / TN, BM / TM); 
    dim3 gridSize((N + BN - 1) / BN, (M + BM - 1) / BM);

    // 启动狂飙
    matmul_cuda_optimized<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);

    // 结果拷贝回主机
    cudaMemcpy(C->data, d_C, size_C, cudaMemcpyDeviceToHost);

    // 释放显存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}