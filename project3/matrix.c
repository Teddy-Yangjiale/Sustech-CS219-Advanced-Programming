#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <immintrin.h>
#include <omp.h>
#include <sys/mman.h>

Matrix create_matrix(size_t rows, size_t cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    size_t size_in_bytes = rows * cols * sizeof(float);

    // 针对32K/64K 矩阵，尝试调用内核大页映射
    if (size_in_bytes >= 1024ULL * 1024 * 1024) {
        m.data = (float*)mmap(NULL, size_in_bytes, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        if (m.data == MAP_FAILED) {
            m.data = (float*)_mm_malloc(size_in_bytes, 64);
        }
    } else {
        m.data = (float*)_mm_malloc(size_in_bytes, 64);
    }
    
    if (m.data == NULL || m.data == MAP_FAILED) {
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
    if (m != NULL && m->data != NULL && m->data != MAP_FAILED) {
        size_t size_in_bytes = m->rows * m->cols * sizeof(float);
        if (size_in_bytes >= 1024ULL * 1024 * 1024) {
            munmap(m->data, size_in_bytes);
        } else {
            _mm_free(m->data);
        }
        m->data = NULL;
        m->rows = 0;
        m->cols = 0;
    }
}