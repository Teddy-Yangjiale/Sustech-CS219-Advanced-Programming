#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>
#include "matrix.h"
#include <string.h>

#define MC 120
#define NC 256
#define KC 512

void matmul_plain(const Matrix* A, const Matrix* B, Matrix* C) {
    if (A == NULL || B == NULL || C == NULL || A->data == NULL || B->data == NULL || C->data == NULL) {
        fprintf(stderr, "Error: Null pointer passed to matmul_plain.\n");
        return;
    }
    if (A->cols != B->rows || A->rows != C->rows || B->cols != C->cols) {
        fprintf(stderr, "Error: Matrix dimensions do not match for multiplication.\n");
        return;
    }

    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < B->cols; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < A->cols; k++) {
                sum += A->data[i * A->cols + k] * B->data[k * B->cols + j];
            }
            C->data[i * C->cols + j] = sum;
        }
    }
}


static inline void simd_pack_A(int k_len, const float *src, int ld_src, float *dest) {
    for (int k = 0; k < k_len; k++) {
        dest[0] = src[0 * ld_src + k];
        dest[1] = src[1 * ld_src + k];
        dest[2] = src[2 * ld_src + k];
        dest[3] = src[3 * ld_src + k];
        dest[4] = src[4 * ld_src + k];
        dest[5] = src[5 * ld_src + k];
        dest += 6;
    }
}

// 高速 SIMD 打包 B (16列一组)
static inline void simd_pack_B(int k_len, const float *src, int ld_src, float *dest) {
    for (int k = 0; k < k_len; k++) {
        _mm256_store_ps(dest,     _mm256_loadu_ps(src + k * ld_src));
        _mm256_store_ps(dest + 8, _mm256_loadu_ps(src + k * ld_src + 8));
        dest += 16;
    }
}

void matmul_improved(const Matrix *A, const Matrix *B, Matrix *C) {
    if (!A || !B || !C || A->cols != B->rows) return;
    
    const size_t M = A->rows, N = B->cols, K = A->cols;
    const size_t M_padded = ((M + 5) / 6) * 6;
    const size_t N_padded = ((N + 15) / 16) * 16;
    
    // 1. 预分配内存，同时保证64字节对齐 
    float *A_packed = (float*)aligned_alloc(64, M_padded * K * sizeof(float));
    float *B_packed = (float*)aligned_alloc(64, K * N_padded * sizeof(float));

    #pragma omp parallel
    {
        // 2. 并行全局打包 A (每个 A 块只打一次！)
        #pragma omp for schedule(static)
        for (size_t i = 0; i < M; i += 6) {
            if (i + 5 < M) {
                simd_pack_A(K, &A->data[i * K], K, &A_packed[i * K]);
            } else {
                // 仅对最后不足 6 行的块进行填充
                for (size_t k = 0; k < K; ++k) {
                    for (size_t r = 0; r < 6; ++r) {
                        A_packed[i * K + k * 6 + r] = (i + r < M) ? A->data[(i + r) * K + k] : 0.0f;
                    }
                }
            }
        }

        // 3. 并行全局打包 B (每个 B 块只打一次！)
        #pragma omp for schedule(static)
        for (size_t j = 0; j < N; j += 16) {
            if (j + 15 < N) {
                simd_pack_B(K, &B->data[j], N, &B_packed[j * K]);
            } else {
                for (size_t k = 0; k < K; ++k) {
                    for (size_t c = 0; c < 16; ++c) {
                        B_packed[j * K + k * 16 + c] = (j + c < N) ? B->data[k * N + j + c] : 0.0f;
                    }
                }
            }
        }

        // 4. 核心计算
        #pragma omp for collapse(2) schedule(dynamic)
        for (size_t ic = 0; ic < M; ic += MC) {
            for (size_t jc = 0; jc < N; jc += NC) {
                size_t m_tile = (ic + MC > M) ? M - ic : MC;
                size_t n_tile = (jc + NC > N) ? N - jc : NC;

                for (size_t kc = 0; kc < K; kc += KC) {
                    size_t k_len = (kc + KC > K) ? K - kc : KC;

                    for (size_t i = 0; i < m_tile; i += 6) {
                        float *pa_base = A_packed + (ic + i) * K + kc * 6;
                        for (size_t j = 0; j < n_tile; j += 16) {
                            float *pb = B_packed + (jc + j) * K + kc * 16;
                            float *pa = pa_base;

                            __m256 c00=_mm256_setzero_ps(), c01=_mm256_setzero_ps();
                            __m256 c10=_mm256_setzero_ps(), c11=_mm256_setzero_ps();
                            __m256 c20=_mm256_setzero_ps(), c21=_mm256_setzero_ps();
                            __m256 c30=_mm256_setzero_ps(), c31=_mm256_setzero_ps();
                            __m256 c40=_mm256_setzero_ps(), c41=_mm256_setzero_ps();
                            __m256 c50=_mm256_setzero_ps(), c51=_mm256_setzero_ps();

                            // 预取距离调优
                            for (size_t k = 0; k < k_len; ++k) {
                                _mm_prefetch((const char*)(pb + 320), _MM_HINT_T0);
                                __m256 b0 = _mm256_load_ps(pb);
                                __m256 b1 = _mm256_load_ps(pb + 8);
                                __m256 a;
                                a = _mm256_broadcast_ss(pa++); c00=_mm256_fmadd_ps(a, b0, c00); c01=_mm256_fmadd_ps(a, b1, c01);
                                a = _mm256_broadcast_ss(pa++); c10=_mm256_fmadd_ps(a, b0, c10); c11=_mm256_fmadd_ps(a, b1, c11);
                                a = _mm256_broadcast_ss(pa++); c20=_mm256_fmadd_ps(a, b0, c20); c21=_mm256_fmadd_ps(a, b1, c21);
                                a = _mm256_broadcast_ss(pa++); c30=_mm256_fmadd_ps(a, b0, c30); c31=_mm256_fmadd_ps(a, b1, c31);
                                a = _mm256_broadcast_ss(pa++); c40=_mm256_fmadd_ps(a, b0, c40); c41=_mm256_fmadd_ps(a, b1, c41);
                                a = _mm256_broadcast_ss(pa++); c50=_mm256_fmadd_ps(a, b0, c50); c51=_mm256_fmadd_ps(a, b1, c51);
                                pb += 16;
                            }

                            // 结果写回
                            for (size_t r = 0; r < 6; ++r) {
                                if (ic + i + r < M) {
                                    float *c_ptr = &C->data[(ic + i + r) * N + (jc + j)];
                                    __m256 r0, r1;
                                    if(r==0){r0=c00; r1=c01;} else if(r==1){r0=c10; r1=c11;}
                                    else if(r==2){r0=c20; r1=c21;} else if(r==3){r0=c30; r1=c31;}
                                    else if(r==4){r0=c40; r1=c41;} else {r0=c50; r1=c51;}

                                    if (jc + j + 15 < N) {
                                        _mm256_storeu_ps(c_ptr, _mm256_add_ps(r0, _mm256_loadu_ps(c_ptr)));
                                        _mm256_storeu_ps(c_ptr+8, _mm256_add_ps(r1, _mm256_loadu_ps(c_ptr+8)));
                                    } else {
                                        float res[16]; _mm256_storeu_ps(res, r0); _mm256_storeu_ps(res+8, r1);
                                        for(size_t c_idx=0; c_idx<16 && jc+j+c_idx<N; ++c_idx) c_ptr[c_idx] += res[c_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    free(A_packed); free(B_packed);
}












// ==================== 优化过程记录 ====================
// 记录优化过程，从v1到v10
#define BLOCK_SIZE 64 // 针对典型 CPU L2 缓存大小设置的分块大小
#define BK 256
#define BI 256

#define MC_V9 512
#define KC_V9 512
#define NC_V9 1024
#define MC_V10 256
#define KC_V10 256
#define NC_V10 512

int matmul_v1_plain(const Matrix *A, const Matrix *B, Matrix *C)
{
    if (A->cols != B->rows)
        return -1;
    for (size_t i = 0; i < A->rows; i++)
    {
        for (size_t j = 0; j < B->cols; j++)
        {
            float sum = 0.0f;
            for (size_t k = 0; k < A->cols; k++)
            {
                sum += A->data[i * A->cols + k] * B->data[k * B->cols + j];
            }
            C->data[i * C->cols + j] = sum;
        }
    }
    return 0;
}

// 阶段 2: 改变循环顺序为 i-k-j 并加入 OpenMP
int matmul_v2_omp(const Matrix *A, const Matrix *B, Matrix *C)
{
    if (A->cols != B->rows)
        return -1;
    memset(C->data, 0, C->rows * C->cols * sizeof(float));

#pragma omp parallel for
    for (size_t i = 0; i < A->rows; i++)
    {
        for (size_t k = 0; k < A->cols; k++)
        {
            float temp = A->data[i * A->cols + k];
            for (size_t j = 0; j < B->cols; j++)
            {
                C->data[i * C->cols + j] += temp * B->data[k * B->cols + j];
            }
        }
    }
    return 0;
}

// 阶段 3: i-k-j + OpenMP + AVX2/FMA SIMD
int matmul_v3_simd(const Matrix *A, const Matrix *B, Matrix *C)
{
    if (A->cols != B->rows)
        return -1;
    memset(C->data, 0, C->rows * C->cols * sizeof(float));
#pragma omp parallel for
for (size_t i = 0; i < A->rows; i++)
{
    for (size_t k = 0; k < A->cols; k++)
    {
        float a_val = A->data[i * A->cols + k];
        __m256 va = _mm256_set1_ps(a_val);

        // 【优化点】：提前计算出能够被 8 整除的最大边界
        // 比如 B->cols 是 100，j_limit 就是 96。
        // 这比每次循环判断 j + 7 < cols 可读性好得多，且能减轻编译器推导循环边界的压力。
        size_t j_limit = (B->cols / 8) * 8; 
        
        // 也可以使用极其硬核的位运算写法，效果完全等价，常用于数学库底层：
        // size_t j_limit = B->cols & ~7; 

        size_t j = 0;
        
        // 核心 SIMD 循环：条件判断现在变得极其清爽
        for (; j < j_limit; j += 8)
        {
            __m256 vb = _mm256_loadu_ps(&B->data[k * B->cols + j]);
            __m256 vc = _mm256_loadu_ps(&C->data[i * C->cols + j]);
            vc = _mm256_fmadd_ps(va, vb, vc); // FMA 指令加速
            _mm256_storeu_ps(&C->data[i * C->cols + j], vc);
        }

        // 处理尾部不满足 8 个的部分 (j 从 j_limit 继续往后走)
        for (; j < B->cols; j++)
        {
            C->data[i * C->cols + j] += a_val * B->data[k * B->cols + j];
        }
    }
}
    return 0;
}

int matmul_v4_advanced(const Matrix *A, const Matrix *B, Matrix *C)
{
    if (!A || !B || !C || A->cols != B->rows)
        return -1;
    memset(C->data, 0, C->rows * C->cols * sizeof(float));

// 1. 分块 (Tiling) 循环
#pragma omp parallel for collapse(2)
    for (size_t bi = 0; bi < A->rows; bi += BLOCK_SIZE)
    {
        for (size_t bk = 0; bk < A->cols; bk += BLOCK_SIZE)
        {
            // 计算当前块的边界
            size_t i_end = (bi + BLOCK_SIZE > A->rows) ? A->rows : bi + BLOCK_SIZE;
            size_t k_end = (bk + BLOCK_SIZE > A->cols) ? A->cols : bk + BLOCK_SIZE;

            for (size_t i = bi; i < i_end; ++i)
            {
                for (size_t k = bk; k < k_end; ++k)
                {
                    float a_val = A->data[i * A->cols + k];
                    __m256 va = _mm256_set1_ps(a_val);

                    size_t j = 0;
                    // 2. 手动循环展开 (Unrolling)：一次处理 4 个向量 (32 个 float)
                    // 这能显著提高发射频率，减少循环开销
                    for (; j + 31 < B->cols; j += 32)
                    {
                        __m256 vb0 = _mm256_loadu_ps(&B->data[k * B->cols + j]);
                        __m256 vb1 = _mm256_loadu_ps(&B->data[k * B->cols + j + 8]);
                        __m256 vb2 = _mm256_loadu_ps(&B->data[k * B->cols + j + 16]);
                        __m256 vb3 = _mm256_loadu_ps(&B->data[k * B->cols + j + 24]);

                        __m256 vc0 = _mm256_loadu_ps(&C->data[i * C->cols + j]);
                        __m256 vc1 = _mm256_loadu_ps(&C->data[i * C->cols + j + 8]);
                        __m256 vc2 = _mm256_loadu_ps(&C->data[i * C->cols + j + 16]);
                        __m256 vc3 = _mm256_loadu_ps(&C->data[i * C->cols + j + 24]);

                        vc0 = _mm256_fmadd_ps(va, vb0, vc0);
                        vc1 = _mm256_fmadd_ps(va, vb1, vc1);
                        vc2 = _mm256_fmadd_ps(va, vb2, vc2);
                        vc3 = _mm256_fmadd_ps(va, vb3, vc3);

                        _mm256_storeu_ps(&C->data[i * C->cols + j], vc0);
                        _mm256_storeu_ps(&C->data[i * C->cols + j + 8], vc1);
                        _mm256_storeu_ps(&C->data[i * C->cols + j + 16], vc2);
                        _mm256_storeu_ps(&C->data[i * C->cols + j + 24], vc3);
                    }

                    // 3. 处理剩余的小块 (8 的倍数)
                    for (; j + 7 < B->cols; j += 8)
                    {
                        __m256 vb = _mm256_loadu_ps(&B->data[k * B->cols + j]);
                        __m256 vc = _mm256_loadu_ps(&C->data[i * C->cols + j]);
                        vc = _mm256_fmadd_ps(va, vb, vc);
                        _mm256_storeu_ps(&C->data[i * C->cols + j], vc);
                    }

                    // 4. 处理最终剩余元素
                    for (; j < B->cols; ++j)
                    {
                        C->data[i * C->cols + j] += a_val * B->data[k * B->cols + j];
                    }
                }
            }
        }
    }
    return 0;
}
int matmul_v5_micro(const Matrix *A, const Matrix *B, Matrix *C)
{

    if (!A || !B || !C || A->cols != B->rows)
        return -1;
    memset(C->data, 0, C->rows * C->cols * sizeof(float));

    size_t rows = A->rows;
    size_t cols = B->cols;
    size_t K = A->cols;

#pragma omp parallel for collapse(2)
    for (size_t i_tile = 0; i_tile < rows; i_tile += BI)
    {
        for (size_t k_tile = 0; k_tile < K; k_tile += BK)
        {

            size_t i_limit = (i_tile + BI > rows) ? rows : i_tile + BI;
            size_t k_limit = (k_tile + BK > K) ? K : k_tile + BK;

            // 微内核：每次处理 6 行 (Register Blocking)
            for (size_t i = i_tile; i < i_limit; i += 6)
            {
                // 如果剩余行数不足 6 行，回退到普通处理
                if (i + 5 >= i_limit)
                {
                    for (size_t i_remain = i; i_remain < i_limit; ++i_remain)
                    {
                        for (size_t k = k_tile; k < k_limit; ++k)
                        {
                            float a_val = A->data[i_remain * K + k];
                            __m256 va = _mm256_set1_ps(a_val);
                            for (size_t j = 0; j + 7 < cols; j += 8)
                            {
                                __m256 vb = _mm256_loadu_ps(&B->data[k * cols + j]);
                                __m256 vc = _mm256_loadu_ps(&C->data[i_remain * cols + j]);
                                _mm256_storeu_ps(&C->data[i_remain * cols + j], _mm256_fmadd_ps(va, vb, vc));
                            }
                        }
                    }
                    break;
                }

                // 核心 J 循环：每次处理 16 列 (2 个 AVX 向量)
                for (size_t j = 0; j + 15 < cols; j += 16)
                {
                    // 使用 12 个寄存器存放 6行x2列向量 的中间结果
                    __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
                    __m256 c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
                    __m256 c4 = _mm256_setzero_ps(), c5 = _mm256_setzero_ps();
                    __m256 c6 = _mm256_setzero_ps(), c7 = _mm256_setzero_ps();
                    __m256 c8 = _mm256_setzero_ps(), c9 = _mm256_setzero_ps();
                    __m256 ca = _mm256_setzero_ps(), cb = _mm256_setzero_ps();

                    for (size_t k = k_tile; k < k_limit; ++k)
                    {
                        __m256 b0 = _mm256_loadu_ps(&B->data[k * cols + j]);
                        __m256 b1 = _mm256_loadu_ps(&B->data[k * cols + j + 8]);

                        c0 = _mm256_fmadd_ps(_mm256_set1_ps(A->data[(i + 0) * K + k]), b0, c0);
                        c1 = _mm256_fmadd_ps(_mm256_set1_ps(A->data[(i + 0) * K + k]), b1, c1);
                        c2 = _mm256_fmadd_ps(_mm256_set1_ps(A->data[(i + 1) * K + k]), b0, c2);
                        c3 = _mm256_fmadd_ps(_mm256_set1_ps(A->data[(i + 1) * K + k]), b1, c3);
                        c4 = _mm256_fmadd_ps(_mm256_set1_ps(A->data[(i + 2) * K + k]), b0, c4);
                        c5 = _mm256_fmadd_ps(_mm256_set1_ps(A->data[(i + 2) * K + k]), b1, c5);
                        c6 = _mm256_fmadd_ps(_mm256_set1_ps(A->data[(i + 3) * K + k]), b0, c6);
                        c7 = _mm256_fmadd_ps(_mm256_set1_ps(A->data[(i + 3) * K + k]), b1, c7);
                        c8 = _mm256_fmadd_ps(_mm256_set1_ps(A->data[(i + 4) * K + k]), b0, c8);
                        c9 = _mm256_fmadd_ps(_mm256_set1_ps(A->data[(i + 4) * K + k]), b1, c9);
                        ca = _mm256_fmadd_ps(_mm256_set1_ps(A->data[(i + 5) * K + k]), b0, ca);
                        cb = _mm256_fmadd_ps(_mm256_set1_ps(A->data[(i + 5) * K + k]), b1, cb);
                    }

                    // 写回内存时加上原有值
                    _mm256_storeu_ps(&C->data[(i + 0) * cols + j], _mm256_add_ps(c0, _mm256_loadu_ps(&C->data[(i + 0) * cols + j])));
                    _mm256_storeu_ps(&C->data[(i + 0) * cols + j + 8], _mm256_add_ps(c1, _mm256_loadu_ps(&C->data[(i + 0) * cols + j + 8])));
                    _mm256_storeu_ps(&C->data[(i + 1) * cols + j], _mm256_add_ps(c2, _mm256_loadu_ps(&C->data[(i + 1) * cols + j])));
                    _mm256_storeu_ps(&C->data[(i + 1) * cols + j + 8], _mm256_add_ps(c3, _mm256_loadu_ps(&C->data[(i + 1) * cols + j + 8])));
                    _mm256_storeu_ps(&C->data[(i + 2) * cols + j], _mm256_add_ps(c4, _mm256_loadu_ps(&C->data[(i + 2) * cols + j])));
                    _mm256_storeu_ps(&C->data[(i + 2) * cols + j + 8], _mm256_add_ps(c5, _mm256_loadu_ps(&C->data[(i + 2) * cols + j + 8])));
                    _mm256_storeu_ps(&C->data[(i + 3) * cols + j], _mm256_add_ps(c6, _mm256_loadu_ps(&C->data[(i + 3) * cols + j])));
                    _mm256_storeu_ps(&C->data[(i + 3) * cols + j + 8], _mm256_add_ps(c7, _mm256_loadu_ps(&C->data[(i + 3) * cols + j + 8])));
                    _mm256_storeu_ps(&C->data[(i + 4) * cols + j], _mm256_add_ps(c8, _mm256_loadu_ps(&C->data[(i + 4) * cols + j])));
                    _mm256_storeu_ps(&C->data[(i + 4) * cols + j + 8], _mm256_add_ps(c9, _mm256_loadu_ps(&C->data[(i + 4) * cols + j + 8])));
                    _mm256_storeu_ps(&C->data[(i + 5) * cols + j], _mm256_add_ps(ca, _mm256_loadu_ps(&C->data[(i + 5) * cols + j])));
                    _mm256_storeu_ps(&C->data[(i + 5) * cols + j + 8], _mm256_add_ps(cb, _mm256_loadu_ps(&C->data[(i + 5) * cols + j + 8])));
                }
            }
        }
    }
    return 0;
}
// 将 B 的子块打包到连续内存中，以减少 TLB Miss
static inline void pack_B_panel(size_t k_len, size_t n_len, const float *src, size_t ld_src, float *dest)
{
    for (size_t k = 0; k < k_len; ++k)
    {
        memcpy(dest + k * n_len, src + k * ld_src, n_len * sizeof(float));
    }
}
int matmul_v6_packed(const Matrix *A, const Matrix *B, Matrix *C)
{
    if (!A || !B || !C || A->cols != B->rows || A->rows != C->rows || B->cols != C->cols)
    {
        return -1;
    }

    memset(C->data, 0, C->rows * C->cols * sizeof(float)); // 初始化结果矩阵

    size_t M = A->rows;
    size_t N = B->cols;
    size_t K = A->cols;

#pragma omp parallel
    {
        // 为每个线程分配对齐的打包缓冲区
        float *b_pack = (float *)aligned_alloc(32, KC * NC * sizeof(float));

#pragma omp for collapse(2)
        for (size_t jc = 0; jc < N; jc += NC)
        {
            for (size_t kc = 0; kc < K; kc += KC)
            {
                size_t n_len = (jc + NC > N) ? N - jc : NC;
                size_t k_len = (kc + KC > K) ? K - kc : KC;

                // 打包 B 矩阵块以提高访存局部性
                pack_B_panel(k_len, n_len, &B->data[kc * N + jc], N, b_pack);

                for (size_t ic = 0; ic < M; ic += MC)
                {
                    size_t m_len = (ic + MC > M) ? M - ic : MC;

                    for (size_t i = ic; i < ic + m_len; i += 6)
                    {
                        // 边界处理逻辑
                        if (i + 5 >= ic + m_len)
                        {
                            for (size_t ir = i; ir < ic + m_len; ++ir)
                            {
                                for (size_t k = 0; k < k_len; ++k)
                                {
                                    __m256 va = _mm256_set1_ps(A->data[ir * K + (kc + k)]);
                                    for (size_t j = 0; j + 7 < n_len; j += 8)
                                    {
                                        __m256 vb = _mm256_loadu_ps(&b_pack[k * n_len + j]);
                                        __m256 vc = _mm256_loadu_ps(&C->data[ir * N + (jc + j)]);
                                        _mm256_storeu_ps(&C->data[ir * N + (jc + j)], _mm256_fmadd_ps(va, vb, vc));
                                    }
                                    for (size_t j = (n_len & ~7); j < n_len; ++j)
                                        C->data[ir * N + (jc + j)] += A->data[ir * K + (kc + k)] * b_pack[k * n_len + j];
                                }
                            }
                            continue;
                        }

                        // 核心 6x16 计算循环 (寄存器阻塞)
                        for (size_t j = 0; j + 15 < n_len; j += 16)
                        {
                            __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
                            __m256 c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
                            __m256 c4 = _mm256_setzero_ps(), c5 = _mm256_setzero_ps();
                            __m256 c6 = _mm256_setzero_ps(), c7 = _mm256_setzero_ps();
                            __m256 c8 = _mm256_setzero_ps(), c9 = _mm256_setzero_ps();
                            __m256 ca = _mm256_setzero_ps(), cb = _mm256_setzero_ps();

                            for (size_t k = 0; k < k_len; ++k)
                            {
                                __m256 b0 = _mm256_loadu_ps(&b_pack[k * n_len + j]);
                                __m256 b1 = _mm256_loadu_ps(&b_pack[k * n_len + j + 8]);

                                __m256 va0 = _mm256_set1_ps(A->data[(i + 0) * K + (kc + k)]);
                                c0 = _mm256_fmadd_ps(va0, b0, c0);
                                c1 = _mm256_fmadd_ps(va0, b1, c1);

                                __m256 va1 = _mm256_set1_ps(A->data[(i + 1) * K + (kc + k)]);
                                c2 = _mm256_fmadd_ps(va1, b0, c2);
                                c3 = _mm256_fmadd_ps(va1, b1, c3);

                                __m256 va2 = _mm256_set1_ps(A->data[(i + 2) * K + (kc + k)]);
                                c4 = _mm256_fmadd_ps(va2, b0, c4);
                                c5 = _mm256_fmadd_ps(va2, b1, c5);

                                __m256 va3 = _mm256_set1_ps(A->data[(i + 3) * K + (kc + k)]);
                                c6 = _mm256_fmadd_ps(va3, b0, c6);
                                c7 = _mm256_fmadd_ps(va3, b1, c7);

                                __m256 va4 = _mm256_set1_ps(A->data[(i + 4) * K + (kc + k)]);
                                c8 = _mm256_fmadd_ps(va4, b0, c8);
                                c9 = _mm256_fmadd_ps(va4, b1, c9);

                                __m256 va5 = _mm256_set1_ps(A->data[(i + 5) * K + (kc + k)]);
                                ca = _mm256_fmadd_ps(va5, b0, ca);
                                cb = _mm256_fmadd_ps(va5, b1, cb);
                            }

// 修复部分：替换 Lambda，直接使用标准写回逻辑
#define STORE_RESULT(row_idx, col_off, reg)                    \
    _mm256_storeu_ps(&C->data[(row_idx) * N + (jc + col_off)], \
                     _mm256_add_ps(reg, _mm256_loadu_ps(&C->data[(row_idx) * N + (jc + col_off)])))

                            STORE_RESULT(i + 0, j, c0);
                            STORE_RESULT(i + 0, j + 8, c1);
                            STORE_RESULT(i + 1, j, c2);
                            STORE_RESULT(i + 1, j + 8, c3);
                            STORE_RESULT(i + 2, j, c4);
                            STORE_RESULT(i + 2, j + 8, c5);
                            STORE_RESULT(i + 3, j, c6);
                            STORE_RESULT(i + 3, j + 8, c7);
                            STORE_RESULT(i + 4, j, c8);
                            STORE_RESULT(i + 4, j + 8, c9);
                            STORE_RESULT(i + 5, j, ca);
                            STORE_RESULT(i + 5, j + 8, cb);
#undef STORE_RESULT
                        }
                    }
                }
            }
        }
        free(b_pack);
    }
    return 0;
}

static inline void pack_A_panel(int k_len, const float *src, int ld_src, float *dest)
{
    for (int k = 0; k < k_len; ++k)
    {
        for (int i = 0; i < 6; ++i)
        {
            *dest++ = src[i * ld_src + k];
        }
    }
}

// 打包 B 矩阵面板：保持行主序连续

int matmul_v7_extreme(const Matrix *A, const Matrix *B, Matrix *C)
{
    if (!A || !B || !C || A->cols != B->rows)
        return -1; // 参数检查

    memset(C->data, 0, C->rows * C->cols * sizeof(float));
    size_t M = A->rows, N = B->cols, K = A->cols;

#pragma omp parallel
    {
        // 预分配线程私有缓冲区，消除内存申请开销
        float *a_pack = (float *)aligned_alloc(32, MC * KC * sizeof(float));
        float *b_pack = (float *)aligned_alloc(32, KC * NC * sizeof(float));

#pragma omp for collapse(1)
        for (size_t jc = 0; jc < N; jc += NC)
        {
            size_t n_len = (jc + NC > N) ? N - jc : NC;
            for (size_t kc = 0; kc < K; kc += KC)
            {
                size_t k_len = (kc + KC > K) ? K - kc : KC;

                // 打包 B 面板
                pack_B_panel(k_len, n_len, &B->data[kc * N + jc], N, b_pack);

                for (size_t ic = 0; ic < M; ic += MC)
                {
                    size_t m_len = (ic + MC > M) ? M - ic : MC;

                    for (size_t i = ic; i < ic + m_len; i += 6)
                    {
                        // 边界处理：剩余不足 6 行时
                        if (i + 5 >= ic + m_len)
                        {
                            for (size_t ir = i; ir < ic + m_len; ++ir)
                            {
                                for (size_t k = 0; k < k_len; ++k)
                                {
                                    __m256 va = _mm256_set1_ps(A->data[ir * K + (kc + k)]);
                                    for (size_t j = 0; j + 7 < n_len; j += 8)
                                    {
                                        __m256 vb = _mm256_loadu_ps(&b_pack[k * n_len + j]);
                                        __m256 vc = _mm256_loadu_ps(&C->data[ir * N + (jc + j)]);
                                        _mm256_storeu_ps(&C->data[ir * N + (jc + j)], _mm256_fmadd_ps(va, vb, vc));
                                    }
                                }
                            }
                            continue;
                        }

                        // 打包 A 矩阵的 6 行面板
                        pack_A_panel(k_len, &A->data[i * K + kc], K, a_pack);

                        // 极致微内核：此时 A 和 B 都是绝对连续访问
                        for (size_t j = 0; j + 15 < n_len; j += 16)
                        {
                            // 正确的初始化方式：使用内联函数置零
                            __m256 c0 = _mm256_setzero_ps();
                            __m256 c1 = _mm256_setzero_ps();
                            __m256 c2 = _mm256_setzero_ps();
                            __m256 c3 = _mm256_setzero_ps();
                            __m256 c4 = _mm256_setzero_ps();
                            __m256 c5 = _mm256_setzero_ps();
                            __m256 c6 = _mm256_setzero_ps();
                            __m256 c7 = _mm256_setzero_ps();
                            __m256 c8 = _mm256_setzero_ps();
                            __m256 c9 = _mm256_setzero_ps();
                            __m256 ca = _mm256_setzero_ps();
                            __m256 cb = _mm256_setzero_ps();

                            for (size_t k = 0; k < k_len; ++k)
                            {
                                // B 矩阵连续读取
                                __m256 b0 = _mm256_loadu_ps(&b_pack[k * n_len + j]);
                                __m256 b1 = _mm256_loadu_ps(&b_pack[k * n_len + j + 8]);

                                // A 矩阵面板连续读取
                                const float *ap = &a_pack[k * 6];
                                c0 = _mm256_fmadd_ps(_mm256_set1_ps(ap[0]), b0, c0);
                                c1 = _mm256_fmadd_ps(_mm256_set1_ps(ap[0]), b1, c1);
                                c2 = _mm256_fmadd_ps(_mm256_set1_ps(ap[1]), b0, c2);
                                c3 = _mm256_fmadd_ps(_mm256_set1_ps(ap[1]), b1, c3);
                                c4 = _mm256_fmadd_ps(_mm256_set1_ps(ap[2]), b0, c4);
                                c5 = _mm256_fmadd_ps(_mm256_set1_ps(ap[2]), b1, c5);
                                c6 = _mm256_fmadd_ps(_mm256_set1_ps(ap[3]), b0, c6);
                                c7 = _mm256_fmadd_ps(_mm256_set1_ps(ap[3]), b1, c7);
                                c8 = _mm256_fmadd_ps(_mm256_set1_ps(ap[4]), b0, c8);
                                c9 = _mm256_fmadd_ps(_mm256_set1_ps(ap[4]), b1, c9);
                                ca = _mm256_fmadd_ps(_mm256_set1_ps(ap[5]), b0, ca);
                                cb = _mm256_fmadd_ps(_mm256_set1_ps(ap[5]), b1, cb);
                            }

// 结果写回并累加到 C 矩阵
#define ADD_STORE(r, off, reg)                           \
    _mm256_storeu_ps(&C->data[(i + r) * N + (jc + off)], \
                     _mm256_add_ps(reg, _mm256_loadu_ps(&C->data[(i + r) * N + (jc + off)])))

                            ADD_STORE(0, j, c0);
                            ADD_STORE(0, j + 8, c1);
                            ADD_STORE(1, j, c2);
                            ADD_STORE(1, j + 8, c3);
                            ADD_STORE(2, j, c4);
                            ADD_STORE(2, j + 8, c5);
                            ADD_STORE(3, j, c6);
                            ADD_STORE(3, j + 8, c7);
                            ADD_STORE(4, j, c8);
                            ADD_STORE(4, j + 8, c9);
                            ADD_STORE(5, j, ca);
                            ADD_STORE(5, j + 8, cb);
#undef ADD_STORE
                        }
                    }
                }
            }
        }
        free(a_pack);
        free(b_pack);
    }
    return 0;
}




static inline void pack_A_v8(int k_len, const float *src, int ld_src, float *dest) {
    for (int k = 0; k < k_len; ++k) {
        for (int i = 0; i < 6; ++i) {
            dest[k * 6 + i] = src[i * ld_src + k];
        }
    }
}

static inline void pack_B_v8(int k_len, int n_len, const float *src, int ld_src, float *dest) {
    for (int k = 0; k < k_len; ++k) {
        memcpy(dest + k * n_len, src + k * ld_src, n_len * sizeof(float));
    }
}

int matmul_v8_lunar(const Matrix *A, const Matrix *B, Matrix *C) {
    if (!A || !B || !C || A->cols != B->rows) return -1;
    memset(C->data, 0, C->rows * C->cols * sizeof(float));

    size_t M = A->rows, N = B->cols, K = A->cols;

    #pragma omp parallel
    {
        float *a_pack = (float*)aligned_alloc(32, MC * KC * sizeof(float));
        float *b_pack = (float*)aligned_alloc(32, KC * NC * sizeof(float));

        #pragma omp for collapse(1) schedule(static)
        for (size_t jc = 0; jc < N; jc += NC) {
            size_t n_len = (jc + NC > N) ? N - jc : NC;
            for (size_t kc = 0; kc < K; kc += KC) {
                size_t k_len = (kc + KC > K) ? K - kc : KC;
                pack_B_v8(k_len, n_len, &B->data[kc * N + jc], N, b_pack);

                for (size_t ic = 0; ic < M; ic += MC) {
                    size_t m_len = (ic + MC > M) ? M - ic : MC;
                    for (size_t i = ic; i < ic + m_len; i += 6) {
                        if (i + 5 >= ic + m_len) { /* 边界处理同 V7 */ continue; }

                        pack_A_v8(k_len, &A->data[i * K + kc], K, a_pack);

                        for (size_t j = 0; j + 15 < n_len; j += 16) {
                            __m256 c0=_mm256_setzero_ps(), c1=_mm256_setzero_ps(), c2=_mm256_setzero_ps(), c3=_mm256_setzero_ps();
                            __m256 c4=_mm256_setzero_ps(), c5=_mm256_setzero_ps(), c6=_mm256_setzero_ps(), c7=_mm256_setzero_ps();
                            __m256 c8=_mm256_setzero_ps(), c9=_mm256_setzero_ps(), ca=_mm256_setzero_ps(), cb=_mm256_setzero_ps();

                            // K 维度 4 倍展开，压榨 Lion Cove 的发射宽度
                            size_t k = 0;
                            for (; k + 3 < k_len; k += 4) {
                                // 软件预取：提前拉取后续 B 块数据
                                _mm_prefetch((const char*)&b_pack[(k + 8) * n_len + j], _MM_HINT_T0);
                                
                                for(int kk=0; kk<4; ++kk) {
                                    __m256 b0 = _mm256_loadu_ps(&b_pack[(k+kk) * n_len + j]);
                                    __m256 b1 = _mm256_loadu_ps(&b_pack[(k+kk) * n_len + j + 8]);
                                    const float *ap = &a_pack[(k+kk) * 6];
                                    
                                    c0 = _mm256_fmadd_ps(_mm256_set1_ps(ap[0]), b0, c0);
                                    c1 = _mm256_fmadd_ps(_mm256_set1_ps(ap[0]), b1, c1);
                                    c2 = _mm256_fmadd_ps(_mm256_set1_ps(ap[1]), b0, c2);
                                    c3 = _mm256_fmadd_ps(_mm256_set1_ps(ap[1]), b1, c3);
                                    c4 = _mm256_fmadd_ps(_mm256_set1_ps(ap[2]), b0, c4);
                                    c5 = _mm256_fmadd_ps(_mm256_set1_ps(ap[2]), b1, c5);
                                    c6 = _mm256_fmadd_ps(_mm256_set1_ps(ap[3]), b0, c6);
                                    c7 = _mm256_fmadd_ps(_mm256_set1_ps(ap[3]), b1, c7);
                                    c8 = _mm256_fmadd_ps(_mm256_set1_ps(ap[4]), b0, c8);
                                    c9 = _mm256_fmadd_ps(_mm256_set1_ps(ap[4]), b1, c9);
                                    ca = _mm256_fmadd_ps(_mm256_set1_ps(ap[5]), b0, ca);
                                    cb = _mm256_fmadd_ps(_mm256_set1_ps(ap[5]), b1, cb);
                                }
                            }
                            // 剩余 K 处理...
                            for (; k < k_len; ++k) {
                                __m256 b0 = _mm256_loadu_ps(&b_pack[k * n_len + j]);
                                __m256 b1 = _mm256_loadu_ps(&b_pack[k * n_len + j + 8]);
                                const float *ap = &a_pack[k * 6];
                                c0 = _mm256_fmadd_ps(_mm256_set1_ps(ap[0]), b0, c0);
                                c1 = _mm256_fmadd_ps(_mm256_set1_ps(ap[0]), b1, c1);
                                c2 = _mm256_fmadd_ps(_mm256_set1_ps(ap[1]), b0, c2);
                                c3 = _mm256_fmadd_ps(_mm256_set1_ps(ap[1]), b1, c3);
                                c4 = _mm256_fmadd_ps(_mm256_set1_ps(ap[2]), b0, c4);
                                c5 = _mm256_fmadd_ps(_mm256_set1_ps(ap[2]), b1, c5);
                                c6 = _mm256_fmadd_ps(_mm256_set1_ps(ap[3]), b0, c6);
                                c7 = _mm256_fmadd_ps(_mm256_set1_ps(ap[3]), b1, c7);
                                c8 = _mm256_fmadd_ps(_mm256_set1_ps(ap[4]), b0, c8);
                                c9 = _mm256_fmadd_ps(_mm256_set1_ps(ap[4]), b1, c9);
                                ca = _mm256_fmadd_ps(_mm256_set1_ps(ap[5]), b0, ca);
                                cb = _mm256_fmadd_ps(_mm256_set1_ps(ap[5]), b1, cb);

                            }

                            // 写回结果并累加至 C
                            #define ADD_STORE_V8(r, off, reg) \
                                _mm256_storeu_ps(&C->data[(i+r)*N + (jc+off)], \
                                _mm256_add_ps(reg, _mm256_loadu_ps(&C->data[(i+r)*N + (jc+off)])))
                            ADD_STORE_V8(0, j, c0); ADD_STORE_V8(0, j+8, c1);
                            ADD_STORE_V8(1, j, c2); ADD_STORE_V8(1, j+8, c3);
                            ADD_STORE_V8(2, j, c4); ADD_STORE_V8(2, j+8, c5);
                            ADD_STORE_V8(3, j, c6); ADD_STORE_V8(3, j+8, c7);
                            ADD_STORE_V8(4, j, c8); ADD_STORE_V8(4, j+8, c9);
                            ADD_STORE_V8(5, j, ca); ADD_STORE_V8(5, j+8, cb);
                            #undef ADD_STORE_V8
                        }
                    }
                }
            }
        }
        free(a_pack); free(b_pack);
    }
    return 0;
}


static inline void pack_B_v9(int k_len, int n_len, const float *src, int ld_src, float *dest) {
    for (int k = 0; k < k_len; ++k) {
        memcpy(dest + k * n_len, src + k * ld_src, n_len * sizeof(float));
    }
}

int matmul_v9_lunar_max(const Matrix *A, const Matrix *B, Matrix *C) {
    if (!A || !B || !C || A->cols != B->rows) return -1;
    memset(C->data, 0, C->rows * C->cols * sizeof(float));

    size_t M = A->rows, N = B->cols, K = A->cols;

    #pragma omp parallel
    {
        float *a_pack = (float*)aligned_alloc(32, MC_V9 * KC_V9 * sizeof(float));
        float *b_pack = (float*)aligned_alloc(32, KC_V9 * NC_V9 * sizeof(float));

        #pragma omp for collapse(1)
        for (size_t jc = 0; jc < N; jc += NC_V9) {
            size_t n_len = (jc + NC_V9 > N) ? N - jc : NC_V9;
            for (size_t kc = 0; kc < K; kc += KC_V9) {
                size_t k_len = (kc + KC_V9 > K) ? K - kc : KC_V9;
                pack_B_v9(k_len, n_len, &B->data[kc * N + jc], N, b_pack);

                for (size_t ic = 0; ic < M; ic += MC_V9) {
                    size_t m_len = (ic + MC_V9 > M) ? M - ic : MC_V9;
                    for (size_t i = ic; i < ic + m_len; i += 6) {
                        if (i + 5 >= ic + m_len) continue;

                        for (size_t j = 0; j + 15 < n_len; j += 16) {
                            __m256 c0=_mm256_setzero_ps(), c1=_mm256_setzero_ps(), c2=_mm256_setzero_ps();
                            __m256 c3=_mm256_setzero_ps(), c4=_mm256_setzero_ps(), c5=_mm256_setzero_ps();
                            __m256 c6=_mm256_setzero_ps(), c7=_mm256_setzero_ps(), c8=_mm256_setzero_ps();
                            __m256 c9=_mm256_setzero_ps(), ca=_mm256_setzero_ps(), cb=_mm256_setzero_ps();

                            for (size_t k = 0; k < k_len; ++k) {
                                __m256 b0 = _mm256_load_ps(&b_pack[k * n_len + j]);
                                __m256 b1 = _mm256_load_ps(&b_pack[k * n_len + j + 8]);

                                __m256 va0 = _mm256_set1_ps(A->data[(i+0)*K+(kc+k)]);
                                c0 = _mm256_fmadd_ps(va0, b0, c0);
                                c1 = _mm256_fmadd_ps(va0, b1, c1);
                                __m256 va1 = _mm256_set1_ps(A->data[(i+1)*K+(kc+k)]);
                                c2 = _mm256_fmadd_ps(va1, b0, c2);
                                c3 = _mm256_fmadd_ps(va1, b1, c3);
                                __m256 va2 = _mm256_set1_ps(A->data[(i+2)*K+(kc+k)]);
                                c4 = _mm256_fmadd_ps(va2, b0, c4);
                                c5 = _mm256_fmadd_ps(va2, b1, c5);
                                __m256 va3 = _mm256_set1_ps(A->data[(i+3)*K+(kc+k)]);
                                c6 = _mm256_fmadd_ps(va3, b0, c6);
                                c7 = _mm256_fmadd_ps(va3, b1, c7);
                                __m256 va4 = _mm256_set1_ps(A->data[(i+4)*K+(kc+k)]);
                                c8 = _mm256_fmadd_ps(va4, b0, c8);
                                c9 = _mm256_fmadd_ps(va4, b1, c9);
                                __m256 va5 = _mm256_set1_ps(A->data[(i+5)*K+(kc+k)]);
                                ca = _mm256_fmadd_ps(va5, b0, ca);
                                cb = _mm256_fmadd_ps(va5, b1, cb);
                            }

                            // 使用宏代替 Lambda 以保持 C 语言兼容性
                            #define V9_STORE(row, offset, reg) \
                                _mm256_storeu_ps(&C->data[(i+row)*N + (jc+offset)], \
                                    _mm256_add_ps(reg, _mm256_loadu_ps(&C->data[(i+row)*N + (jc+offset)])))

                            V9_STORE(0, 0, c0); V9_STORE(0, 8, c1);
                            V9_STORE(1, 0, c2); V9_STORE(1, 8, c3);
                            V9_STORE(2, 0, c4); V9_STORE(2, 8, c5);
                            V9_STORE(3, 0, c6); V9_STORE(3, 8, c7);
                            V9_STORE(4, 0, c8); V9_STORE(4, 8, c9);
                            V9_STORE(5, 0, ca); V9_STORE(5, 8, cb);
                            #undef V9_STORE
                        }
                    }
                }
            }
        }
        free(a_pack); free(b_pack);
    }
    return 0;
}



// 回归最稳健的分块，确保完全装入 2.5MB L2
static inline void pack_A_v10(int k_len, const float *src, int ld_src, float *dest) {
    for (int k = 0; k < k_len; ++k) {
        dest[k * 6 + 0] = src[0 * ld_src + k];
        dest[k * 6 + 1] = src[1 * ld_src + k];
        dest[k * 6 + 2] = src[2 * ld_src + k];
        dest[k * 6 + 3] = src[3 * ld_src + k];
        dest[k * 6 + 4] = src[4 * ld_src + k];
        dest[k * 6 + 5] = src[5 * ld_src + k];
    }
}

// 打包 B 面板
static inline void pack_B_v10(int k_len, int n_len, const float *src, int ld_src, float *dest) {
    for (int k = 0; k < k_len; ++k) {
        memcpy(dest + k * n_len, src + k * ld_src, n_len * sizeof(float));
    }
}

int matmul_v10_final(const Matrix *A, const Matrix *B, Matrix *C) {
    if (!A || !B || !C || A->cols != B->rows) return -1;
    memset(C->data, 0, C->rows * C->cols * sizeof(float));

    size_t M = A->rows, N = B->cols, K = A->cols;

    #pragma omp parallel
    {
        float *a_pack = (float*)aligned_alloc(32, MC_V10 * KC_V10 * sizeof(float));
        float *b_pack = (float*)aligned_alloc(32, KC_V10 * NC_V10 * sizeof(float));

        #pragma omp for collapse(1) schedule(static)
        for (size_t jc = 0; jc < N; jc += NC_V10) {
            size_t n_len = (jc + NC_V10 > N) ? N - jc : NC_V10;
            for (size_t kc = 0; kc < K; kc += KC_V10) {
                size_t k_len = (kc + KC_V10 > K) ? K - kc : KC_V10;
                pack_B_v10(k_len, n_len, &B->data[kc * N + jc], N, b_pack);

                for (size_t ic = 0; ic < M; ic += MC_V10) {
                    size_t m_len = (ic + MC_V10 > M) ? M - ic : MC_V10;
                    for (size_t i = ic; i < ic + m_len; i += 6) {
                        if (i + 5 >= ic + m_len) continue;

                        pack_A_v10(k_len, &A->data[i * K + kc], K, a_pack);

                        for (size_t j = 0; j + 15 < n_len; j += 16) {
                            // 使用正确的 Intrinsics 置零
                            __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
                            __m256 c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
                            __m256 c4 = _mm256_setzero_ps(), c5 = _mm256_setzero_ps();
                            __m256 c6 = _mm256_setzero_ps(), c7 = _mm256_setzero_ps();
                            __m256 c8 = _mm256_setzero_ps(), c9 = _mm256_setzero_ps();
                            __m256 ca = _mm256_setzero_ps(), cb = _mm256_setzero_ps();

                            for (size_t k = 0; k < k_len; ++k) {
                                __m256 b0 = _mm256_load_ps(&b_pack[k * n_len + j]);
                                __m256 b1 = _mm256_load_ps(&b_pack[k * n_len + j + 8]);

                                const float *ap = &a_pack[k * 6];
                                // 使用 broadcast 指令压榨 Lion Cove 的发射能力
                                c0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&ap[0]), b0, c0);
                                c1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&ap[0]), b1, c1);
                                c2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&ap[1]), b0, c2);
                                c3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&ap[1]), b1, c3);
                                c4 = _mm256_fmadd_ps(_mm256_broadcast_ss(&ap[2]), b0, c4);
                                c5 = _mm256_fmadd_ps(_mm256_broadcast_ss(&ap[2]), b1, c5);
                                c6 = _mm256_fmadd_ps(_mm256_broadcast_ss(&ap[3]), b0, c6);
                                c7 = _mm256_fmadd_ps(_mm256_broadcast_ss(&ap[3]), b1, c7);
                                c8 = _mm256_fmadd_ps(_mm256_broadcast_ss(&ap[4]), b0, c8);
                                c9 = _mm256_fmadd_ps(_mm256_broadcast_ss(&ap[4]), b1, c9);
                                ca = _mm256_fmadd_ps(_mm256_broadcast_ss(&ap[5]), b0, ca);
                                cb = _mm256_fmadd_ps(_mm256_broadcast_ss(&ap[5]), b1, cb);
                            }

                            // 结果累加写回
                            #define V10_STORE(r, off, reg) \
                                _mm256_storeu_ps(&C->data[(i+r)*N + (jc+off)], \
                                _mm256_add_ps(reg, _mm256_loadu_ps(&C->data[(i+r)*N + (jc+off)])))

                            V10_STORE(0, 0, c0); V10_STORE(0, 8, c1);
                            V10_STORE(1, 0, c2); V10_STORE(1, 8, c3);
                            V10_STORE(2, 0, c4); V10_STORE(2, 8, c5);
                            V10_STORE(3, 0, c6); V10_STORE(3, 8, c7);
                            V10_STORE(4, 0, c8); V10_STORE(4, 8, c9);
                            V10_STORE(5, 0, ca); V10_STORE(5, 8, cb);
                            #undef V10_STORE
                        }
                    }
                }
            }
        }
        free(a_pack); free(b_pack);
    }
    return 0;
}