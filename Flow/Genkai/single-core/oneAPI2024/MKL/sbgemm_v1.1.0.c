// sbgemm_v1.1.0.c - Intel MKL BF16 GEMM with verification
// PG1.7: oneAPI2024/MKL
// Added: Correctness verification, improved memory alignment

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <mkl.h>

typedef uint16_t bf16;

/* --- BF16 <-> FP32 conversion --- */

static inline bf16 float_to_bf16_round(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    u += 0x00008000u;
    return (bf16)(u >> 16);
}

static inline float bf16_to_float(bf16 x) {
    uint32_t u = ((uint32_t)x) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/* MKL-based sbgemm */
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
    cblas_gemm_bf16bf16f32(layout, transA, transB,
                           M, N, K,
                           alpha,
                           (const MKL_BF16*)A, lda,
                           (const MKL_BF16*)B, ldb,
                           beta,
                           C, ldc);
}

/* Reference implementation for verification */
void sbgemm_ref(int M, int N, int K,
                float alpha,
                const bf16 *A, int lda,
                const bf16 *B, int ldb,
                float beta,
                float *C, int ldc)
{
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float a_ik = bf16_to_float(A[i * lda + k]);
                float b_kj = bf16_to_float(B[k * ldb + j]);
                sum += a_ik * b_kj;
            }
            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}

void init_matrix_bf16(bf16 *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        float val = (float)(rand() % 100) / 100.0f;
        mat[i] = float_to_bf16_round(val);
    }
}

void init_matrix_f32_zero(float *mat, int rows, int cols) {
    memset(mat, 0, rows * cols * sizeof(float));
}

double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Verify correctness
int verify_result(float *C_mkl, float *C_ref, int M, int N, float tol) {
    int errors = 0;
    float max_diff = 0.0f;

    for (int i = 0; i < M * N; i++) {
        float diff = fabsf(C_mkl[i] - C_ref[i]);
        float rel_diff = diff / (fabsf(C_ref[i]) + 1e-6f);
        if (rel_diff > tol) {
            errors++;
            if (diff > max_diff) max_diff = diff;
        }
    }

    if (errors > 0) {
        printf("Verification: %d errors, max_diff=%.6f\n", errors, max_diff);
    }
    return errors;
}

int main(int argc, char *argv[]) {
    int sizes[] = {128, 256, 512, 1024, 2048, 4096, 8192, 10000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int do_verify = 1; // Enable verification for small sizes

    mkl_set_num_threads(1);

    printf("# MKL BF16 GEMM Benchmark (v1.1.0)\n");
    printf("Matrix_Size,Time_sec,GFLOPS,Verified\n");

    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];
        int M = N, K = N;

        bf16 *A = (bf16*)mkl_malloc(M * K * sizeof(bf16), 64);
        bf16 *B = (bf16*)mkl_malloc(K * N * sizeof(bf16), 64);
        float *C = (float*)mkl_malloc(M * N * sizeof(float), 64);
        float *C_ref = NULL;

        if (!A || !B || !C) {
            fprintf(stderr, "Memory allocation failed for size %d\n", N);
            continue;
        }

        srand(42);
        init_matrix_bf16(A, M, K);
        init_matrix_bf16(B, K, N);
        init_matrix_f32_zero(C, M, N);

        // Verification for small sizes only
        int verified = -1;
        if (do_verify && N <= 512) {
            C_ref = (float*)mkl_malloc(M * N * sizeof(float), 64);
            if (C_ref) {
                init_matrix_f32_zero(C_ref, M, N);
                sbgemm_ref(M, N, K, 1.0f, A, K, B, N, 0.0f, C_ref, N);
            }
        }

        // Warm-up
        sbgemm_mkl(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                   M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

        // Verify
        if (C_ref) {
            verified = verify_result(C, C_ref, M, N, 0.01f);
            mkl_free(C_ref);
        }

        // Benchmark
        int num_runs = (N <= 1024) ? 10 : 3;
        double start = get_time_sec();

        for (int r = 0; r < num_runs; r++) {
            sbgemm_mkl(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                       M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
        }

        double end = get_time_sec();
        double elapsed = (end - start) / num_runs;
        double flops = 2.0 * M * N * K;
        double gflops = flops / elapsed / 1e9;

        const char *verify_str = (verified == 0) ? "PASS" :
                                 (verified > 0) ? "FAIL" : "SKIP";
        printf("%d,%.6f,%.2f,%s\n", N, elapsed, gflops, verify_str);

        mkl_free(A);
        mkl_free(B);
        mkl_free(C);
    }

    return 0;
}
