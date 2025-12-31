// sbgemm_v1.1.0.c - AVX-512 SIMD optimized version
// PG1.4: Intel oneAPI 2023 SIMD optimization
// Optimization: Loop reorder (i-k-j), AVX-512 vectorization
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

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

/* Convert 16 BF16 values to 16 FP32 values using AVX-512 */
static inline __m512 bf16x16_to_fp32(__m256i bf16_vec) {
    // Expand bf16 (16-bit) to fp32 (32-bit) by shifting left by 16
    __m512i extended = _mm512_cvtepu16_epi32(bf16_vec);
    __m512i shifted = _mm512_slli_epi32(extended, 16);
    return _mm512_castsi512_ps(shifted);
}

/* C = alpha * op(A) * op(B) + beta * C
 * Optimized for NoTrans case with AVX-512
 */
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

    // Currently only support NoTrans for optimized path
    if (transA != CblasNoTrans || transB != CblasNoTrans) {
        fprintf(stderr, "Only NoTrans is supported in optimized version.\n");
        exit(1);
    }

    // Apply beta*C using SIMD
    if (beta == 0.0f) {
        for (int i = 0; i < M; ++i) {
            int j = 0;
            // Process 16 elements at a time with AVX-512
            for (; j + 16 <= N; j += 16) {
                _mm512_storeu_ps(&C[i * ldc + j], _mm512_setzero_ps());
            }
            // Handle remainder
            for (; j < N; ++j) {
                C[i * ldc + j] = 0.0f;
            }
        }
    } else if (beta != 1.0f) {
        __m512 beta_vec = _mm512_set1_ps(beta);
        for (int i = 0; i < M; ++i) {
            int j = 0;
            for (; j + 16 <= N; j += 16) {
                __m512 c_vec = _mm512_loadu_ps(&C[i * ldc + j]);
                c_vec = _mm512_mul_ps(c_vec, beta_vec);
                _mm512_storeu_ps(&C[i * ldc + j], c_vec);
            }
            for (; j < N; ++j) {
                C[i * ldc + j] *= beta;
            }
        }
    }

    __m512 alpha_vec = _mm512_set1_ps(alpha);

    // Core computation with loop reordering (i-k-j) for better cache usage
    // and AVX-512 vectorization along j dimension
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            // Load A[i][k] and broadcast
            float a_ik = bf16_to_float(A[i * lda + k]);
            __m512 a_vec = _mm512_set1_ps(a_ik * alpha);

            int j = 0;
            // Process 16 elements at a time
            for (; j + 16 <= N; j += 16) {
                // Load B[k][j:j+16] as BF16 and convert to FP32
                __m256i b_bf16 = _mm256_loadu_si256((const __m256i*)&B[k * ldb + j]);
                __m512 b_vec = bf16x16_to_fp32(b_bf16);

                // Load C[i][j:j+16]
                __m512 c_vec = _mm512_loadu_ps(&C[i * ldc + j]);

                // C += A[i][k] * B[k][j:j+16]
                c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);

                // Store result
                _mm512_storeu_ps(&C[i * ldc + j], c_vec);
            }

            // Handle remainder (scalar)
            for (; j < N; ++j) {
                float b_kj = bf16_to_float(B[k * ldb + j]);
                C[i * ldc + j] += a_ik * alpha * b_kj;
            }
        }
    }
}

// Get current time in seconds
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char *argv[]) {
    int M = 512, N = 512, K = 512;
    int num_runs = 5;

    if (argc >= 2) M = N = K = atoi(argv[1]);
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    if (argc >= 5) num_runs = atoi(argv[4]);

    printf("=== SBGEMM AVX-512 SIMD v1.1.0 ===\n");
    printf("Matrix size: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Number of runs: %d\n", num_runs);

    // Allocate aligned memory for better SIMD performance
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
    memset(C_f32, 0, M * N * sizeof(float));

    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    // Warm-up run
    sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 M, N, K, 1.0f, A_bf16, lda, B_bf16, ldb, 0.0f, C_f32, ldc);

    // Performance measurement
    double total_time = 0.0;
    double min_time = 1e10;

    for (int run = 0; run < num_runs; ++run) {
        memset(C_f32, 0, M * N * sizeof(float));

        double start = get_time();
        sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                     M, N, K, 1.0f, A_bf16, lda, B_bf16, ldb, 0.0f, C_f32, ldc);
        double end = get_time();

        double elapsed = end - start;
        total_time += elapsed;
        if (elapsed < min_time) min_time = elapsed;

        printf("Run %d: %.6f sec\n", run + 1, elapsed);
    }

    double flops = 2.0 * M * N * K;
    double avg_time = total_time / num_runs;
    double avg_gflops = flops / avg_time / 1e9;
    double peak_gflops = flops / min_time / 1e9;

    printf("\n=== Results ===\n");
    printf("Average time: %.6f sec\n", avg_time);
    printf("Min time: %.6f sec\n", min_time);
    printf("Average GFLOPS: %.2f\n", avg_gflops);
    printf("Peak GFLOPS: %.2f\n", peak_gflops);

    printf("\nSample C[0][0] = %.6f\n", C_f32[0]);

    free(A_bf16);
    free(B_bf16);
    free(C_f32);

    return 0;
}
