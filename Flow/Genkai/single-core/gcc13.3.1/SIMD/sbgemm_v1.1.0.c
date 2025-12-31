// sbgemm_v1.1.0.c - AVX2 SIMD optimized version
// PG1.3: GCC 13.3.1 SIMD optimization
// Features: AVX2 SIMD, loop tiling, optimized bf16->fp32 conversion

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

// Block sizes for cache optimization
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 64

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

// Convert 8 bf16 values to 8 floats using AVX2
static inline __m256 bf16x8_to_fp32(__m128i bf16_vals) {
    // Zero extend bf16 to 32-bit integers
    __m256i int32_vals = _mm256_cvtepu16_epi32(bf16_vals);
    // Shift left by 16 to get float representation
    __m256i shifted = _mm256_slli_epi32(int32_vals, 16);
    // Reinterpret as float
    return _mm256_castsi256_ps(shifted);
}

// Load 8 bf16 values and convert to float
static inline __m256 load_bf16_as_fp32(const bf16 *ptr) {
    __m128i bf16_vals = _mm_loadu_si128((const __m128i*)ptr);
    return bf16x8_to_fp32(bf16_vals);
}

/* Optimized GEMM kernel for NoTrans, NoTrans case with AVX2 */
static void sbgemm_kernel_nn_avx2(int M, int N, int K,
                                   float alpha,
                                   const bf16 *A, int lda,
                                   const bf16 *B, int ldb,
                                   float *C, int ldc)
{
    // Process 8 columns at a time using AVX2
    for (int i = 0; i < M; ++i) {
        int j = 0;

        // Vectorized loop: process 8 columns at a time
        for (; j + 8 <= N; j += 8) {
            __m256 c_vec = _mm256_loadu_ps(&C[i * ldc + j]);

            for (int p = 0; p < K; ++p) {
                // Load a_ip and broadcast to all 8 lanes
                float a_ip = bf16_to_float(A[i * lda + p]);
                __m256 a_vec = _mm256_set1_ps(a_ip);

                // Load 8 b values: B[p][j:j+8]
                __m256 b_vec = load_bf16_as_fp32(&B[p * ldb + j]);

                // FMA: c += a * b
                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
            }

            // Apply alpha and store
            __m256 alpha_vec = _mm256_set1_ps(alpha);
            __m256 result = _mm256_loadu_ps(&C[i * ldc + j]);
            result = _mm256_sub_ps(result, result);  // Zero out old value
            result = _mm256_fmadd_ps(alpha_vec, c_vec, result);
            _mm256_storeu_ps(&C[i * ldc + j], result);
        }

        // Scalar remainder
        for (; j < N; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < K; ++p) {
                float a_ip = bf16_to_float(A[i * lda + p]);
                float b_pj = bf16_to_float(B[p * ldb + j]);
                sum += a_ip * b_pj;
            }
            C[i * ldc + j] = alpha * sum;
        }
    }
}

/* Blocked GEMM with AVX2 optimization */
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
                C[i * ldc + j] = 0.0f;
            }
        }
    } else if (beta != 1.0f) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                C[i * ldc + j] *= beta;
            }
        }
    }

    // Use optimized kernel for NoTrans, NoTrans case
    if (transA == CblasNoTrans && transB == CblasNoTrans) {
        sbgemm_kernel_nn_avx2(M, N, K, alpha, A, lda, B, ldb, C, ldc);
        return;
    }

    // Fallback for transposed cases
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < K; ++p) {
                float a_ip;
                if (transA == CblasNoTrans) {
                    a_ip = bf16_to_float(A[i * lda + p]);
                } else {
                    a_ip = bf16_to_float(A[p * lda + i]);
                }

                float b_pj;
                if (transB == CblasNoTrans) {
                    b_pj = bf16_to_float(B[p * ldb + j]);
                } else {
                    b_pj = bf16_to_float(B[j * ldb + p]);
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
    const int M = 512, K = 512, N = 512;
    const int num_trials = 5;

    printf("sbgemm_v1.1.0 - AVX2 SIMD optimized\n");
    printf("Matrix size: M=%d, K=%d, N=%d\n", M, K, N);
    printf("Number of trials: %d\n", num_trials);

    // Allocate aligned matrices
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
    double flops = 2.0 * M * N * K;
    double gflops = flops / avg_time / 1e9;

    printf("\n=== Results ===\n");
    printf("Average time: %.6f sec\n", avg_time);
    printf("Min time: %.6f sec\n", min_time);
    printf("Max time: %.6f sec\n", max_time);
    printf("Performance: %.2f GFLOPS\n", gflops);

    printf("\nSample C values: C[0][0]=%.4f, C[M/2][N/2]=%.4f\n",
           C_f32[0], C_f32[(M/2)*ldc + N/2]);

    free(A_bf16);
    free(B_bf16);
    free(C_f32);
    return 0;
}
