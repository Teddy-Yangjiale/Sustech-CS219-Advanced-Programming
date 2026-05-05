#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arm_neon.h>
#include <omp.h>

// 专为 Neoverse-N2 1MB L2 Cache 优化的分块参数
#define MC 120
#define NC 256
#define KC 512

// ==================== 基础功能 ====================
Matrix create_matrix(size_t rows, size_t cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    size_t size_in_bytes = rows * cols * sizeof(float);
    
    // 保证 64 字节对齐，对齐 Cache Line
    if (posix_memalign((void**)&m.data, 64, size_in_bytes) != 0) {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        exit(EXIT_FAILURE);
    }
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows * cols; i++) {
        m.data[i] = 0.0f;
    }
    return m;
}

void free_matrix(Matrix* m) {
    if (m != NULL && m->data != NULL) {
        free(m->data);
        m->data = NULL;
        m->rows = 0;
        m->cols = 0;
    }
}

void matmul_plain(const Matrix* A, const Matrix* B, Matrix* C) {
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

// ==================== ARM NEON 优化核心 ====================

// 标量打包 A 矩阵 (6行一组)
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

// NEON 向量化打包 B 矩阵 (16列一组)
static inline void simd_pack_B(int k_len, const float *src, int ld_src, float *dest) {
    for (int k = 0; k < k_len; k++) {
        vst1q_f32(dest,      vld1q_f32(src + k * ld_src));
        vst1q_f32(dest + 4,  vld1q_f32(src + k * ld_src + 4));
        vst1q_f32(dest + 8,  vld1q_f32(src + k * ld_src + 8));
        vst1q_f32(dest + 12, vld1q_f32(src + k * ld_src + 12));
        dest += 16;
    }
}

// 极致 6x16 ARM NEON 微内核
static inline void kernel_6x16_neon(size_t k_len, const float *pa, const float *pb, float *C, size_t ldc, size_t m_len, size_t n_len) {
    // 初始化 24 个 128-bit 累加器为 0
    float32x4_t c00 = vdupq_n_f32(0.0f), c01 = vdupq_n_f32(0.0f), c02 = vdupq_n_f32(0.0f), c03 = vdupq_n_f32(0.0f);
    float32x4_t c10 = vdupq_n_f32(0.0f), c11 = vdupq_n_f32(0.0f), c12 = vdupq_n_f32(0.0f), c13 = vdupq_n_f32(0.0f);
    float32x4_t c20 = vdupq_n_f32(0.0f), c21 = vdupq_n_f32(0.0f), c22 = vdupq_n_f32(0.0f), c23 = vdupq_n_f32(0.0f);
    float32x4_t c30 = vdupq_n_f32(0.0f), c31 = vdupq_n_f32(0.0f), c32 = vdupq_n_f32(0.0f), c33 = vdupq_n_f32(0.0f);
    float32x4_t c40 = vdupq_n_f32(0.0f), c41 = vdupq_n_f32(0.0f), c42 = vdupq_n_f32(0.0f), c43 = vdupq_n_f32(0.0f);
    float32x4_t c50 = vdupq_n_f32(0.0f), c51 = vdupq_n_f32(0.0f), c52 = vdupq_n_f32(0.0f), c53 = vdupq_n_f32(0.0f);

    for (size_t k = 0; k < k_len; ++k) {
        // 软件预取
        __builtin_prefetch(pb + 256, 0, 3);
        
        float32x4_t b0 = vld1q_f32(pb);
        float32x4_t b1 = vld1q_f32(pb + 4);
        float32x4_t b2 = vld1q_f32(pb + 8);
        float32x4_t b3 = vld1q_f32(pb + 12);
        float32x4_t a;
        
        a = vld1q_dup_f32(pa++); c00=vfmaq_f32(c00,a,b0); c01=vfmaq_f32(c01,a,b1); c02=vfmaq_f32(c02,a,b2); c03=vfmaq_f32(c03,a,b3);
        a = vld1q_dup_f32(pa++); c10=vfmaq_f32(c10,a,b0); c11=vfmaq_f32(c11,a,b1); c12=vfmaq_f32(c12,a,b2); c13=vfmaq_f32(c13,a,b3);
        a = vld1q_dup_f32(pa++); c20=vfmaq_f32(c20,a,b0); c21=vfmaq_f32(c21,a,b1); c22=vfmaq_f32(c22,a,b2); c23=vfmaq_f32(c23,a,b3);
        a = vld1q_dup_f32(pa++); c30=vfmaq_f32(c30,a,b0); c31=vfmaq_f32(c31,a,b1); c32=vfmaq_f32(c32,a,b2); c33=vfmaq_f32(c33,a,b3);
        a = vld1q_dup_f32(pa++); c40=vfmaq_f32(c40,a,b0); c41=vfmaq_f32(c41,a,b1); c42=vfmaq_f32(c42,a,b2); c43=vfmaq_f32(c43,a,b3);
        a = vld1q_dup_f32(pa++); c50=vfmaq_f32(c50,a,b0); c51=vfmaq_f32(c51,a,b1); c52=vfmaq_f32(c52,a,b2); c53=vfmaq_f32(c53,a,b3);
        
        pb += 16;
    }

    float32x4_t r0[6] = {c00, c10, c20, c30, c40, c50};
    float32x4_t r1[6] = {c01, c11, c21, c31, c41, c51};
    float32x4_t r2[6] = {c02, c12, c22, c32, c42, c52};
    float32x4_t r3[6] = {c03, c13, c23, c33, c43, c53};
    
    // 结果写回
    for (size_t r = 0; r < 6; ++r) {
        if (r < m_len) {
            float *c_ptr = C + r * ldc;
            if (n_len >= 16) {
                vst1q_f32(c_ptr,      vaddq_f32(r0[r], vld1q_f32(c_ptr)));
                vst1q_f32(c_ptr + 4,  vaddq_f32(r1[r], vld1q_f32(c_ptr + 4)));
                vst1q_f32(c_ptr + 8,  vaddq_f32(r2[r], vld1q_f32(c_ptr + 8)));
                vst1q_f32(c_ptr + 12, vaddq_f32(r3[r], vld1q_f32(c_ptr + 12)));
            } else {
                float temp[16];
                vst1q_f32(temp, r0[r]); vst1q_f32(temp + 4, r1[r]);
                vst1q_f32(temp + 8, r2[r]); vst1q_f32(temp + 12, r3[r]);
                for (size_t c = 0; c < n_len; ++c) c_ptr[c] += temp[c];
            }
        }
    }
}

// ==================== 局部打包辅助函数 ====================

// 局部打包单个 A 矩阵分块并处理边界
static inline void pack_A_tile(size_t m_len, size_t k_len, const float *src, size_t ld_src, float *dest) {
    size_t i = 0;
    for (; i + 5 < m_len; i += 6) {
        simd_pack_A((int)k_len, src + i * ld_src, (int)ld_src, dest + i * k_len);
    }
    if (i < m_len) {
        float *dst_ptr = dest + i * k_len;
        for (size_t k = 0; k < k_len; ++k) {
            for (size_t r = 0; r < 6; ++r) {
                dst_ptr[k * 6 + r] = (i + r < m_len) ? src[(i + r) * ld_src + k] : 0.0f;
            }
        }
    }
}

// 局部打包单个 B 矩阵分块并处理边界
static inline void pack_B_tile(size_t k_len, size_t n_len, const float *src, size_t ld_src, float *dest) {
    size_t j = 0;
    for (; j + 15 < n_len; j += 16) {
        simd_pack_B((int)k_len, src + j, (int)ld_src, dest + j * k_len);
    }
    if (j < n_len) {
        float *dst_ptr = dest + j * k_len;
        for (size_t k = 0; k < k_len; ++k) {
            for (size_t c = 0; c < 16; ++c) {
                dst_ptr[k * 16 + c] = (j + c < n_len) ? src[k * ld_src + j + c] : 0.0f;
            }
        }
    }
}

// 主函数接口：局部实时打包 + OpenMP 调度
void matmul_improved(const Matrix *A, const Matrix *B, Matrix *C) {
    if (!A || !B || !C || A->cols != B->rows) return;
    
    const size_t M = A->rows, N = B->cols, K = A->cols;

    #pragma omp parallel
    {
        // 每一个 OpenMP 线程在运行时只申请针对其当前计算分块（Tile）大小的局部缓冲区
        float *local_A = NULL;
        float *local_B = NULL;
        
        if (posix_memalign((void**)&local_A, 64, MC * KC * sizeof(float)) == 0 &&
            posix_memalign((void**)&local_B, 64, KC * NC * sizeof(float)) == 0) {

            // 外两层循环分配矩阵分块任务，线程之间计算的 C 矩阵局部块完全不冲突
            #pragma omp for collapse(2) schedule(dynamic)
            for (size_t ic = 0; ic < M; ic += MC) {
                for (size_t jc = 0; jc < N; jc += NC) {
                    size_t m_tile = (ic + MC > M) ? M - ic : MC;
                    size_t n_tile = (jc + NC > N) ? N - jc : NC;

                    // 沿着计算深度（K 维度）累加乘积
                    for (size_t kc = 0; kc < K; kc += KC) {
                        size_t k_len = (kc + KC > K) ? K - kc : KC;

                        // 将当前计算所需的 A 矩阵块实时打包进线程局部 L2 Cache 缓冲区
                        pack_A_tile(m_tile, k_len, &A->data[ic * K + kc], K, local_A);

                        // 将当前计算所需的 B 矩阵块实时打包进线程局部 L2 Cache 缓冲区
                        pack_B_tile(k_len, n_tile, &B->data[kc * N + jc], N, local_B);

                        // 运行 6x16 NEON 极致微内核
                        for (size_t i = 0; i < m_tile; i += 6) {
                            float *pa = local_A + i * k_len;
                            for (size_t j = 0; j < n_tile; j += 16) {
                                float *pb = local_B + j * k_len;
                                
                                size_t am = (i + 6 > m_tile) ? m_tile - i : 6;
                                size_t an = (j + 16 > n_tile) ? n_tile - j : 16;
                                
                                kernel_6x16_neon(k_len, pa, pb, &C->data[(ic + i) * N + (jc + j)], N, am, an);
                            }
                        }
                    }
                }
            }
            free(local_A);
            free(local_B);
        } else {
            if (local_A) free(local_A);
            if (local_B) free(local_B);
        }
    }
}s