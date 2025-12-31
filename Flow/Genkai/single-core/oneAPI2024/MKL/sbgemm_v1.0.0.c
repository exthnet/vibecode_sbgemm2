// sbgemm_v1.0.0.c - Intel MKL BF16 GEMM implementation
// PG1.7: oneAPI2024/MKL
// Using cblas_gemm_bf16bf16f32 from Intel MKL

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mkl.h>

typedef uint16_t bf16;

/* --- BF16 <-> FP32 conversion --- */

// FP32 -> BF16 (round-to-nearest)
static inline bf16 float_to_bf16_round(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    u += 0x00008000u;
    return (bf16)(u >> 16);
}

// BF16 -> FP32
static inline float bf16_to_float(bf16 x) {
    uint32_t u = ((uint32_t)x) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/* MKL-based sbgemm implementation
 * C = alpha * op(A) * op(B) + beta * C
 * - A, B: BF16 arrays
 * - C: FP32 array
 * - Row-Major layout
 */
void sbgemm_mkl(CBLAS_LAYOUT layout,
                CBLAS_TRANSPOSE transA,
                CBLAS_TRANSPOSE transB,
                int M, int N, int K,
                float alpha,
                const bf16 *A, int lda,
                const bf16 *B, int ldb,
                float beta,
                float *C, int ldc)
{
    // MKL_BF16 is compatible with uint16_t (bf16)
    cblas_gemm_bf16bf16f32(layout, transA, transB,
                           M, N, K,
                           alpha,
                           (const MKL_BF16*)A, lda,
                           (const MKL_BF16*)B, ldb,
                           beta,
                           C, ldc);
}

// Initialize matrix with random values
void init_matrix_bf16(bf16 *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        float val = (float)(rand() % 100) / 100.0f;
        mat[i] = float_to_bf16_round(val);
    }
}

// Initialize matrix with zeros
void init_matrix_f32_zero(float *mat, int rows, int cols) {
    memset(mat, 0, rows * cols * sizeof(float));
}

// Performance measurement
double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char *argv[]) {
    // Test matrix sizes
    int sizes[] = {128, 256, 512, 1024, 2048, 4096, 8192, 10000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    // Set MKL to single-threaded mode (1 core constraint)
    mkl_set_num_threads(1);

    printf("Matrix Size,Time(sec),GFLOPS\n");

    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];
        int M = N, K = N;

        // Allocate matrices
        bf16 *A = (bf16*)mkl_malloc(M * K * sizeof(bf16), 64);
        bf16 *B = (bf16*)mkl_malloc(K * N * sizeof(bf16), 64);
        float *C = (float*)mkl_malloc(M * N * sizeof(float), 64);

        if (!A || !B || !C) {
            fprintf(stderr, "Memory allocation failed for size %d\n", N);
            continue;
        }

        // Initialize matrices
        srand(42);
        init_matrix_bf16(A, M, K);
        init_matrix_bf16(B, K, N);
        init_matrix_f32_zero(C, M, N);

        // Warm-up run
        sbgemm_mkl(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                   M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

        // Benchmark
        int num_runs = (N <= 1024) ? 10 : 3;
        double start = get_time_sec();

        for (int r = 0; r < num_runs; r++) {
            sbgemm_mkl(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                       M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
        }

        double end = get_time_sec();
        double elapsed = (end - start) / num_runs;

        // Calculate GFLOPS: 2*M*N*K operations
        double flops = 2.0 * M * N * K;
        double gflops = flops / elapsed / 1e9;

        printf("%d,%.6f,%.2f\n", N, elapsed, gflops);

        // Free memory
        mkl_free(A);
        mkl_free(B);
        mkl_free(C);
    }

    return 0;
}
