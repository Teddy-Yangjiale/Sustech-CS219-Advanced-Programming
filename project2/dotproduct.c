#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// 计算纳秒时间差
long long get_elapsed_ns(struct timespec start, struct timespec end) {
    return (long long)(end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);
}

// ==========================================
// 1. int 类型
// ==========================================
void test_int(size_t length) {
    int *v1 = (int *)malloc(length * sizeof(int));
    int *v2 = (int *)malloc(length * sizeof(int));
    if (v1 == NULL || v2 == NULL) exit(1);

    for (size_t k = 0; k < length; k++) {
        v1[k] = (int)(rand() % 10);
        v2[k] = (int)(rand() % 10);
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int dot_product = 0;
    for (size_t k = 0; k < length; k++) {
        dot_product += v1[k] * v2[k];
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    fprintf(stderr, "Result int: %d\n", dot_product);
    printf("%lld\n", get_elapsed_ns(start, end));

    free(v1);
    free(v2);
}

// ==========================================
// 2. short 类型
// ==========================================
void test_short(size_t length) {
    short *v1 = (short *)malloc(length * sizeof(short));
    short *v2 = (short *)malloc(length * sizeof(short));
    if (v1 == NULL || v2 == NULL) exit(1);

    for (size_t k = 0; k < length; k++) {
        v1[k] = (short)(rand() % 10);
        v2[k] = (short)(rand() % 10);
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    short dot_product = 0;
    for (size_t k = 0; k < length; k++) {
        dot_product += v1[k] * v2[k];
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    fprintf(stderr, "Result short: %d\n", (int)dot_product);
    printf("%lld\n", get_elapsed_ns(start, end));

    free(v1);
    free(v2);
}

// ==========================================
// 3. signed char 类型
// ==========================================
void test_char(size_t length) {
    signed char *v1 = (signed char *)malloc(length * sizeof(signed char));
    signed char *v2 = (signed char *)malloc(length * sizeof(signed char));
    if (v1 == NULL || v2 == NULL) exit(1);

    for (size_t k = 0; k < length; k++) {
        v1[k] = (signed char)(rand() % 10);
        v2[k] = (signed char)(rand() % 10);
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    signed char dot_product = 0;
    for (size_t k = 0; k < length; k++) {
        dot_product += v1[k] * v2[k];
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    fprintf(stderr, "Result char: %d\n", (int)dot_product);
    printf("%lld\n", get_elapsed_ns(start, end));

    free(v1);
    free(v2);
}

// ==========================================
// 4. float 类型
// ==========================================
void test_float(size_t length) {
    float *v1 = (float *)malloc(length * sizeof(float));
    float *v2 = (float *)malloc(length * sizeof(float));
    if (v1 == NULL || v2 == NULL) exit(1);

    for (size_t k = 0; k < length; k++) {
        v1[k] = (float)(rand() % 10) / 2.0f;
        v2[k] = (float)(rand() % 10) / 2.0f;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    float dot_product = 0.0f;
    for (size_t k = 0; k < length; k++) {
        dot_product += v1[k] * v2[k];
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    fprintf(stderr, "Result float: %f\n", dot_product);
    printf("%lld\n", get_elapsed_ns(start, end));

    free(v1);
    free(v2);
}

// ==========================================
// 5. double 类型
// ==========================================
void test_double(size_t length) {
    double *v1 = (double *)malloc(length * sizeof(double));
    double *v2 = (double *)malloc(length * sizeof(double));
    if (v1 == NULL || v2 == NULL) exit(1);

    for (size_t k = 0; k < length; k++) {
        v1[k] = (double)(rand() % 10) / 2.0;
        v2[k] = (double)(rand() % 10) / 2.0;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    double dot_product = 0.0;
    for (size_t k = 0; k < length; k++) {
        dot_product += v1[k] * v2[k];
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    fprintf(stderr, "Result double: %lf\n", dot_product);
    printf("%lld\n", get_elapsed_ns(start, end));

    free(v1);
    free(v2);
}

// ==========================================
// 主函数
// ==========================================
int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: ./dotproduct <data_type> <vector_length>\n");
        return 1;
    }

    char *data_type = argv[1];
    
    size_t length = 0;
    if (sscanf(argv[2], "%zu", &length) != 1) {
        fprintf(stderr, "Invalid length\n");
        return 1;
    }

    srand(42); 

    if (strcmp(data_type, "int") == 0) test_int(length);
    else if (strcmp(data_type, "short") == 0) test_short(length);
    else if (strcmp(data_type, "char") == 0) test_char(length);
    else if (strcmp(data_type, "float") == 0) test_float(length);
    else if (strcmp(data_type, "double") == 0) test_double(length);
    else {
        fprintf(stderr, "Unknown type\n");
        return 1;
    }

    return 0;
}