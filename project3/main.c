#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cblas.h>
#include <time.h>

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

double calculate_gflops(size_t n, double time_spent) {
    if (time_spent <= 0) return 0.0;
    double ops = 2.0 * (double)n * (double)n * (double)n;
    return ops / (time_spent * 1e9);
}

void fill_random(Matrix* m) {
    for (size_t i = 0; i < m->rows * m->cols; i++) {
        m->data[i] = (float)rand() / RAND_MAX;
    }
}

void test_matmul(size_t size) {
    printf("=========================================\n");
    printf("Testing Matrix Size: %zu x %zu\n", size, size);

    Matrix A = create_matrix(size, size);
    Matrix B = create_matrix(size, size);
    if (A.data == NULL || B.data == NULL) {
        printf("Memory allocation failed for A or B!\n");
        return;
    }
    fill_random(&A);
    fill_random(&B);

    double start, end;
    double t_plain = 0, t_improved = 0, t_openblas = 0, t_jit = 0;

    if (size <= 1024) {
        Matrix C = create_matrix(size, size);
        start = get_time();
        matmul_plain(&A, &B, &C);
        end = get_time();
        t_plain = end - start;
        printf("Plain    | Time: %10.6f s | GFLOPS: %10.2f\n", t_plain, calculate_gflops(size, t_plain));
        free_matrix(&C);
    } else {
        printf("Plain    | Time:   Skipped    | GFLOPS:   Skipped\n");
    }

    {
        Matrix C = create_matrix(size, size);
        start = get_time();
        matmul_improved(&A, &B, &C);
        end = get_time();
        t_improved = end - start;
        printf("Improved | Time: %10.6f s | GFLOPS: %10.2f\n", t_improved, calculate_gflops(size, t_improved));
        free_matrix(&C);
    }

    {
        Matrix C = create_matrix(size, size);
        start = get_time();
        matmul_jit(&A, &B, &C);
        end = get_time();
        t_jit = end - start;
        printf("JIT      | Time: %10.6f s | GFLOPS: %10.2f\n", t_jit, calculate_gflops(size, t_jit));
        free_matrix(&C);
    }

    {
        Matrix C = create_matrix(size, size);
        start = get_time();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    size, size, size, 1.0f, 
                    A.data, size, B.data, size, 
                    0.0f, C.data, size);
        end = get_time();
        t_openblas = end - start;
        printf("OpenBLAS | Time: %10.6f s | GFLOPS: %10.2f\n", t_openblas, calculate_gflops(size, t_openblas));
        free_matrix(&C);
    }

    printf("-----------------------------------------\n");
    printf("Speedup (OpenBLAS vs Improved): %.2f x\n", t_improved / t_openblas);
    printf("Speedup (OpenBLAS vs JIT): %.2f x\n", t_jit / t_openblas);

    free_matrix(&A);
    free_matrix(&B);
}

int main() {
    srand(42); 

    size_t sizes[] = {16, 128, 1024, 8192, 16384, 32768, 65536};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        test_matmul(sizes[i]);
    }

    return 0;
}