#ifndef MATRIX_JIT_H
#define MATRIX_JIT_H
#include <immintrin.h>
#include <string.h>

// 高速 SIMD 实时打包 A 矩阵 (6 行一组)
static inline void pack_A_JIT(size_t m_len, size_t k_len, const float *src, size_t ld, float *dest) {
    size_t i = 0;
    for (; i + 5 < m_len; i += 6) {
        for (size_t k = 0; k < k_len; ++k) {
            dest[0] = src[(i + 0) * ld + k];
            dest[1] = src[(i + 1) * ld + k];
            dest[2] = src[(i + 2) * ld + k];
            dest[3] = src[(i + 3) * ld + k];
            dest[4] = src[(i + 4) * ld + k];
            dest[5] = src[(i + 5) * ld + k];
            dest += 6;
        }
    }
    if (i < m_len) {
        for (size_t k = 0; k < k_len; ++k) {
            for (size_t r = 0; r < 6; ++r) {
                dest[r] = (i + r < m_len) ? src[(i + r) * ld + k] : 0.0f;
            }
            dest += 6;
        }
    }
}

// 高速 SIMD 实时打包 B 矩阵 (16 列一组)
static inline void pack_B_JIT(size_t k_len, size_t n_len, const float *src, size_t ld, float *dest) {
    size_t j = 0;
    for (; j + 15 < n_len; j += 16) {
        for (size_t k = 0; k < k_len; ++k) {
            _mm256_storeu_ps(dest,     _mm256_loadu_ps(&src[k * ld + j]));
            _mm256_storeu_ps(dest + 8, _mm256_loadu_ps(&src[k * ld + j + 8]));
            dest += 16;
        }
    }
    if (j < n_len) {
        for (size_t k = 0; k < k_len; ++k) {
            for (size_t c = 0; c < 16; ++c) {
                dest[c] = (j + c < n_len) ? src[k * ld + j + c] : 0.0f;
            }
            dest += 16;
        }
    }
}

// 6x16 极致微内核：寄存器 100% 满载
static inline void kernel_6x16_intrinsic(size_t k_len, const float *pa, const float *pb, float *C, size_t ldc, size_t m_len, size_t n_len) {
    __m256 c00=_mm256_setzero_ps(), c01=_mm256_setzero_ps();
    __m256 c10=_mm256_setzero_ps(), c11=_mm256_setzero_ps();
    __m256 c20=_mm256_setzero_ps(), c21=_mm256_setzero_ps();
    __m256 c30=_mm256_setzero_ps(), c31=_mm256_setzero_ps();
    __m256 c40=_mm256_setzero_ps(), c41=_mm256_setzero_ps();
    __m256 c50=_mm256_setzero_ps(), c51=_mm256_setzero_ps();

    for (size_t k = 0; k < k_len; ++k) {
        _mm_prefetch((const char*)(pb + 256), _MM_HINT_T0); // 增加预取深度适应大矩阵
        
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

    __m256 r0[6] = {c00, c10, c20, c30, c40, c50};
    __m256 r1[6] = {c01, c11, c21, c31, c41, c51};
    
    for (size_t r = 0; r < 6; ++r) {
        if (r < m_len) {
            float *c_ptr = C + r * ldc;
            if (n_len >= 16) {
                _mm256_storeu_ps(c_ptr,     _mm256_add_ps(r0[r], _mm256_loadu_ps(c_ptr)));
                _mm256_storeu_ps(c_ptr + 8, _mm256_add_ps(r1[r], _mm256_loadu_ps(c_ptr + 8)));
            } else {
                float temp[16];
                _mm256_storeu_ps(temp, r0[r]);
                _mm256_storeu_ps(temp + 8, r1[r]);
                for (size_t c = 0; c < n_len; ++c) c_ptr[c] += temp[c];
            }
        }
    }
}
#endif