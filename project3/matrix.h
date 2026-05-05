#ifndef MATRIX_H
#define MATRIX_H
#include <stddef.h>

typedef struct {
    size_t rows;
    size_t cols;
    float* data;
} Matrix;

Matrix create_matrix(size_t rows, size_t cols);
void free_matrix(Matrix* m);

//云服务器最佳性能的矩阵乘法实现
void matmul_plain(const Matrix* A, const Matrix* B, Matrix* C);
void matmul_improved(const Matrix* A, const Matrix* B, Matrix* C);
void matmul_jit(const Matrix* A, const Matrix* B, Matrix* C);

//本地运行的矩阵乘法实现(v1-v10)
// 阶段 1: 朴素实现 (Benchmark) [cite: 6]
int matmul_v1_plain(const Matrix *A, const Matrix *B, Matrix *C);
// 阶段 2: 访存顺序优化 (i-k-j) + 多线程并行 (OpenMP) [cite: 7]
int matmul_v2_omp(const Matrix *A, const Matrix *B, Matrix *C);
// 阶段 3: 进一步结合硬件加速 (AVX2/FMA SIMD) [cite: 7]
int matmul_v3_simd(const Matrix *A, const Matrix *B, Matrix *C);
// 阶段 4: 基于块的矩阵乘法 (Blocked Matrix Multiplication) + SIMD
int matmul_v4_advanced(const Matrix *A, const Matrix *B, Matrix *C);
// 阶段 5：微内核设计 (Microkernel) + 寄存器阻塞 (Register Blocking)
int matmul_v5_micro(const Matrix *A, const Matrix *B, Matrix *C);
// 阶段 6：数据打包 (Data Packing) + 微内核 + 寄存器阻塞
int matmul_v6_packed(const Matrix *A, const Matrix *B, Matrix *C);
// 阶段 7：工业级双重打包 (Double Packing) + 极致微内核
int matmul_v7_extreme(const Matrix *A, const Matrix *B, Matrix *C);
// 阶段 8：针对 Lunar Lake 架构的 P-core 优化 + K 步长展开 + 预取
int matmul_v8_lunar(const Matrix *A, const Matrix *B, Matrix *C);
// 阶段 9：特定分块参数优化 + Lunar Lake P-core + 预取指令
int matmul_v9_lunar_max(const Matrix *A, const Matrix *B, Matrix *C);
// 阶段 10：本地运行最终版本，综合前面所有优化
int matmul_v10_final(const Matrix *A, const Matrix *B, Matrix *C);
#endif