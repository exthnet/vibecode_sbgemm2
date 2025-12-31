// sbgemm_v1.0.0.c - Baseline code for performance measurement
// PG1.3: GCC 13.3.1 SIMD optimization
// Copy from BaseCode/sbgemm.c with timing and larger matrix

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef uint16_t bf16;

typedef enum {
    CblasRowMajor = 101,
    CblasColMajor = 102
} CBLAS_LAYOUT;

typedef enum {
    CblasNoTrans  = 111,
    CblasTrans    = 112,
    CblasConjTrans= 113
} CBLAS_TRANSPOSE;

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

/* --- Matrix element access (Row-Major) --- */
static inline float get_elem_bf16(const bf16 *A, int lda, int row, int col) {
    return bf16_to_float(A[row * lda + col]);
}

static inline void set_elem_f32(float *C, int ldc, int row, int col, float v) {
    C[row * ldc + col] = v;
}

/* C = alpha * op(A) * op(B) + beta * C */
void sbgemm_nolib(CBLAS_LAYOUT layout,
                  CBLAS_TRANSPOSE transA,
                  CBLAS_TRANSPOSE transB,
                  int M, int N, int K,
                  float alpha,
                  const bf16 *A, int lda,
                  const bf16 *B, int ldb,
                  float beta,
                  float *C, int ldc)
{
    if (layout != CblasRowMajor) {
        fprintf(stderr, "Only CblasRowMajor is supported.\n");
        exit(1);
    }

    // Pre-apply beta*C
    if (beta == 0.0f) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                set_elem_f32(C, ldc, i, j, 0.0f);
            }
        }
    } else if (beta != 1.0f) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float cij = C[i * ldc + j];
                set_elem_f32(C, ldc, i, j, beta * cij);
            }
        }
    }

    // Core computation
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < K; ++p) {
                float a_ip;
                if (transA == CblasNoTrans) {
                    a_ip = get_elem_bf16(A, lda, i, p);
                } else {
                    a_ip = get_elem_bf16(A, lda, p, i);
                }

                float b_pj;
                if (transB == CblasNoTrans) {
                    b_pj = get_elem_bf16(B, ldb, p, j);
                } else {
                    b_pj = get_elem_bf16(B, ldb, j, p);
                }

                sum += a_ip * b_pj;
            }
            C[i * ldc + j] += alpha * sum;
        }
    }
}

double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    // Test with larger matrix for performance measurement
    const int M = 512, K = 512, N = 512;
    const int num_trials = 5;

    printf("Matrix size: M=%d, K=%d, N=%d\n", M, K, N);
    printf("Number of trials: %d\n", num_trials);

    // Allocate matrices
    bf16 *A_bf16 = (bf16*)aligned_alloc(64, sizeof(bf16) * M * K);
    bf16 *B_bf16 = (bf16*)aligned_alloc(64, sizeof(bf16) * K * N);
    float *C_f32 = (float*)aligned_alloc(64, sizeof(float) * M * N);

    if (!A_bf16 || !B_bf16 || !C_f32) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize with random values
    srand(42);
    for (int i = 0; i < M * K; ++i) {
        float val = (float)(rand() % 100) / 100.0f;
        A_bf16[i] = float_to_bf16_round(val);
    }
    for (int i = 0; i < K * N; ++i) {
        float val = (float)(rand() % 100) / 100.0f;
        B_bf16[i] = float_to_bf16_round(val);
    }

    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    // Warm-up run
    memset(C_f32, 0, sizeof(float) * M * N);
    sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 M, N, K, 1.0f, A_bf16, lda, B_bf16, ldb, 0.0f, C_f32, ldc);

    // Benchmark
    double total_time = 0.0;
    double min_time = 1e30;
    double max_time = 0.0;

    for (int t = 0; t < num_trials; ++t) {
        memset(C_f32, 0, sizeof(float) * M * N);

        double start = get_time_sec();
        sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                     M, N, K, 1.0f, A_bf16, lda, B_bf16, ldb, 0.0f, C_f32, ldc);
        double end = get_time_sec();

        double elapsed = end - start;
        total_time += elapsed;
        if (elapsed < min_time) min_time = elapsed;
        if (elapsed > max_time) max_time = elapsed;

        printf("Trial %d: %.6f sec\n", t + 1, elapsed);
    }

    double avg_time = total_time / num_trials;

    // FLOPS calculation: 2*M*N*K (multiply-add = 2 ops)
    double flops = 2.0 * M * N * K;
    double gflops = flops / avg_time / 1e9;

    printf("\n=== Results ===\n");
    printf("Average time: %.6f sec\n", avg_time);
    printf("Min time: %.6f sec\n", min_time);
    printf("Max time: %.6f sec\n", max_time);
    printf("Performance: %.2f GFLOPS\n", gflops);

    // Verification: check a few elements
    printf("\nSample C values: C[0][0]=%.4f, C[M/2][N/2]=%.4f\n",
           C_f32[0], C_f32[(M/2)*ldc + N/2]);

    free(A_bf16);
    free(B_bf16);
    free(C_f32);
    return 0;
}
