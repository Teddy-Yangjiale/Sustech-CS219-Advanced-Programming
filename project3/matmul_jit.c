#include "matrix.h"
#include "matrix_jit.h"
#include <stdlib.h>
#include <string.h>
#include <omp.h>

//经反复调试，最适合这个机器的分块参数
#define GOLD_MC 120
#define GOLD_NC 512
#define GOLD_KC 256

//为1024大小矩阵准备的分块参数
#define SMALL_MC 120
#define SMALL_NC 128
#define SMALL_KC 256

void matmul_jit(const Matrix *A, const Matrix *B, Matrix *C) {
    if (!A || !B || !C || A->cols != B->rows) return;
    
    const size_t M = A->rows;
    const size_t N = B->cols;
    const size_t K = A->cols;

    // ========================================================
    // 【路径1】极小矩阵 (<= 128) - 单线程 + 全局打包
    // ========================================================
    if (N <= 128) {
        size_t M_pad = ((M + 5) / 6) * 6, N_pad = ((N + 15) / 16) * 16;
        float *pa_tmp = (float*)_mm_malloc(M_pad * K * sizeof(float), 64);
        float *pb_tmp = (float*)_mm_malloc(K * N_pad * sizeof(float), 64);
        
        pack_A_JIT(M, K, A->data, K, pa_tmp);
        pack_B_JIT(K, N, B->data, N, pb_tmp);
        
        for (size_t i = 0; i < M_pad; i += 6) {
            for (size_t j = 0; j < N_pad; j += 16) {
                size_t am = (i + 6 > M) ? M - i : 6;
                size_t an = (j + 16 > N) ? N - j : 16;
                kernel_6x16_intrinsic(K, pa_tmp + i*K, pb_tmp + j*K, &C->data[i*N+j], N, am, an);
            }
        }
        _mm_free(pa_tmp); _mm_free(pb_tmp);
        return;
    }

    // ========================================================
    // 【路径2】1024  - 全局打包 + 宏展开
    // ========================================================
    // 1024 只有 12MB，完全放得下 35MB L3，全局打包一次收益最高
    if (N <= 1024) {
        size_t M_pad = ((M + 5) / 6) * 6, N_pad = ((N + 15) / 16) * 16;
        float *A_pack = (float*)_mm_malloc(M_pad * K * sizeof(float), 64);
        float *B_pack = (float*)_mm_malloc(K * N_pad * sizeof(float), 64);
        
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < M; i += 6) pack_A_JIT((i+5<M?6:M-i), K, &A->data[i*K], K, &A_pack[i*K]);
            #pragma omp for schedule(static)
            for (size_t j = 0; j < N; j += 16) pack_B_JIT(K, (j+15<N?16:N-j), &B->data[j], N, &B_pack[j*K]);
            
            // SMALL_NC=128 会产生 72 个任务，28核完美负载均衡
            #pragma omp for collapse(2) schedule(dynamic)
            for (size_t ic = 0; ic < M; ic += SMALL_MC) {
                for (size_t jc = 0; jc < N; jc += SMALL_NC) {
                    size_t actual_mc = (ic + SMALL_MC > M) ? M - ic : SMALL_MC;
                    size_t actual_nc = (jc + SMALL_NC > N) ? N - jc : SMALL_NC;
                    
                    for (size_t i = 0; i < actual_mc; ++i) 
                        memset(&C->data[(ic + i) * N + jc], 0, actual_nc * sizeof(float));

                    for (size_t kc = 0; kc < K; kc += SMALL_KC) {
                        size_t actual_kc = (kc + SMALL_KC > K) ? K - kc : SMALL_KC;
                        
                        if (actual_mc == SMALL_MC && actual_nc == SMALL_NC && actual_kc == SMALL_KC) {
                            for (size_t i = 0; i < SMALL_MC; i += 6) {
                                for (size_t j = 0; j < SMALL_NC; j += 16) {
                                    kernel_6x16_intrinsic(SMALL_KC, A_pack + (ic+i)*K + kc*6, B_pack + (jc+j)*K + kc*16, &C->data[(ic+i)*N+(jc+j)], N, 6, 16);
                                }
                            }
                        } else {
                            size_t m_pad = ((actual_mc+5)/6)*6, n_pad = ((actual_nc+15)/16)*16;
                            for (size_t i = 0; i < m_pad; i += 6) {
                                for (size_t j = 0; j < n_pad; j += 16) {
                                    size_t am = (i+6>actual_mc)?actual_mc-i:6, an = (j+16>actual_nc)?actual_nc-j:16;
                                    kernel_6x16_intrinsic(actual_kc, A_pack + (ic+i)*K + kc*6, B_pack + (jc+j)*K + kc*16, &C->data[(ic+i)*N+(jc+j)], N, am, an);
                                }
                            }
                        }
                    }
                }
            }
        }
        _mm_free(A_pack); _mm_free(B_pack);
        return;
    }

    // ========================================================
    // 【路径3】(8192, 16384)
    // ========================================================
    
    if (N <= 16384) {
        #pragma omp parallel
        {
            float *local_A = (float*)_mm_malloc(GOLD_MC * GOLD_KC * sizeof(float), 64);
            float *local_B = (float*)_mm_malloc(GOLD_KC * GOLD_NC * sizeof(float), 64);

            #pragma omp for collapse(2) schedule(dynamic)
            for (size_t ic = 0; ic < M; ic += GOLD_MC) {
                for (size_t jc = 0; jc < N; jc += GOLD_NC) {
                    size_t actual_mc = (ic + GOLD_MC > M) ? M - ic : GOLD_MC;
                    size_t actual_nc = (jc + GOLD_NC > N) ? N - jc : GOLD_NC;
                    
                    for (size_t i = 0; i < actual_mc; ++i) 
                        memset(&C->data[(ic + i) * N + jc], 0, actual_nc * sizeof(float));

                    for (size_t kc = 0; kc < K; kc += GOLD_KC) {
                        size_t actual_kc = (kc + GOLD_KC > K) ? K - kc : GOLD_KC;

                        pack_A_JIT(actual_mc, actual_kc, &A->data[ic*K + kc], K, local_A);
                        pack_B_JIT(actual_kc, actual_nc, &B->data[kc*N + jc], N, local_B);

                        if (actual_mc == GOLD_MC && actual_nc == GOLD_NC && actual_kc == GOLD_KC) {
                            for (size_t i = 0; i < GOLD_MC; i += 6) {
                                for (size_t j = 0; j < GOLD_NC; j += 16) {
                                    kernel_6x16_intrinsic(GOLD_KC, local_A + i*GOLD_KC, local_B + j*GOLD_KC, &C->data[(ic+i)*N + (jc+j)], N, 6, 16);
                                }
                            }
                        } else {
                            size_t m_pad = ((actual_mc+5)/6)*6, n_pad = ((actual_nc+15)/16)*16;
                            for (size_t i = 0; i < m_pad; i += 6) {
                                for (size_t j = 0; j < n_pad; j += 16) {
                                    size_t am = (i+6>actual_mc)?actual_mc-i:6, an = (j+16>actual_nc)?actual_nc-j:16;
                                    kernel_6x16_intrinsic(actual_kc, local_A + i*actual_kc, local_B + j*actual_kc, &C->data[(ic+i)*N + (jc+j)], N, am, an);
                                }
                            }
                        }
                    }
                }
            }
            _mm_free(local_A); _mm_free(local_B);
        }
        return;
    }

    // ========================================================
    // 【路径4】(>= 32768) - Control-Tree JIT
    // ========================================================
    // 一次打包，多核共享
    float *shared_B = (float*)_mm_malloc(GOLD_KC * GOLD_NC * sizeof(float), 64);
    
    #pragma omp parallel
    {
        float *local_A = (float*)_mm_malloc(GOLD_MC * GOLD_KC * sizeof(float), 64);

        // 外两层循环所有线程共同执行（不使用 omp for 分发），保证在同一时刻处理同一个大块
        for (size_t jc = 0; jc < N; jc += GOLD_NC) {
            size_t actual_nc = (jc + GOLD_NC > N) ? N - jc : GOLD_NC;
            
            for (size_t kc = 0; kc < K; kc += GOLD_KC) {
                size_t actual_kc = (kc + GOLD_KC > K) ? K - kc : GOLD_KC;

                // 隐式屏障：只派一个线程去内存里拿 512KB 的 B 矩阵，放入 L3 缓存
                #pragma omp single
                {
                    pack_B_JIT(actual_kc, actual_nc, &B->data[kc*N + jc], N, shared_B);
                }

                // 28 线程运行 M 维度的任务，共享 L3 中的 B 矩阵
                #pragma omp for schedule(dynamic)
                for (size_t ic = 0; ic < M; ic += GOLD_MC) {
                    size_t actual_mc = (ic + GOLD_MC > M) ? M - ic : GOLD_MC;
                    
                    if (kc == 0) {
                        for (size_t i = 0; i < actual_mc; ++i) 
                            memset(&C->data[(ic + i) * N + jc], 0, actual_nc * sizeof(float));
                    }

                    // A 矩阵留在各自核心的 L2 缓存中
                    pack_A_JIT(actual_mc, actual_kc, &A->data[ic*K + kc], K, local_A);

                    if (actual_mc == GOLD_MC && actual_nc == GOLD_NC && actual_kc == GOLD_KC) {
                        for (size_t i = 0; i < GOLD_MC; i += 6) {
                            for (size_t j = 0; j < GOLD_NC; j += 16) {
                                kernel_6x16_intrinsic(GOLD_KC, local_A + i*GOLD_KC, shared_B + j*GOLD_KC, &C->data[(ic+i)*N + (jc+j)], N, 6, 16);
                            }
                        }
                    } else {
                        size_t m_pad = ((actual_mc+5)/6)*6, n_pad = ((actual_nc+15)/16)*16;
                        for (size_t i = 0; i < m_pad; i += 6) {
                            for (size_t j = 0; j < n_pad; j += 16) {
                                size_t am = (i+6>actual_mc)?actual_mc-i:6, an = (j+16>actual_nc)?actual_nc-j:16;
                                kernel_6x16_intrinsic(actual_kc, local_A + i*actual_kc, shared_B + j*actual_kc, &C->data[(ic+i)*N + (jc+j)], N, am, an);
                            }
                        }
                    }
                }
            }
        }
        _mm_free(local_A);
    }
    _mm_free(shared_B);
}

