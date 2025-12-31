// sbgemm_v1.3.0.c - AVX2 SIMD with prefetch and register blocking
// PG1.3: GCC 13.3.1 SIMD optimization
// Features: AVX2 SIMD, FMA, loop tiling, prefetch, K-unrolling

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
#define BLOCK_K 128

// Prefetch distance
#define PREFETCH_DISTANCE 64

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
    __m256i int32_vals = _mm256_cvtepu16_epi32(bf16_vals);
    __m256i shifted = _mm256_slli_epi32(int32_vals, 16);
    return _mm256_castsi256_ps(shifted);
}

static inline __m256 load_bf16_as_fp32(const bf16 *ptr) {
    __m128i bf16_vals = _mm_loadu_si128((const __m128i*)ptr);
    return bf16x8_to_fp32(bf16_vals);
}

/* 6x16 micro-kernel with K-unrolling and prefetch */
static inline void micro_kernel_6x16(
    int K_block,
    const bf16 *A, int lda,
    const bf16 *B, int ldb,
    float *C, int ldc)
{
    // 6 rows x 16 columns = 12 AVX2 registers for accumulators
    __m256 c00 = _mm256_loadu_ps(&C[0 * ldc + 0]);
    __m256 c01 = _mm256_loadu_ps(&C[0 * ldc + 8]);
    __m256 c10 = _mm256_loadu_ps(&C[1 * ldc + 0]);
    __m256 c11 = _mm256_loadu_ps(&C[1 * ldc + 8]);
    __m256 c20 = _mm256_loadu_ps(&C[2 * ldc + 0]);
    __m256 c21 = _mm256_loadu_ps(&C[2 * ldc + 8]);
    __m256 c30 = _mm256_loadu_ps(&C[3 * ldc + 0]);
    __m256 c31 = _mm256_loadu_ps(&C[3 * ldc + 8]);
    __m256 c40 = _mm256_loadu_ps(&C[4 * ldc + 0]);
    __m256 c41 = _mm256_loadu_ps(&C[4 * ldc + 8]);
    __m256 c50 = _mm256_loadu_ps(&C[5 * ldc + 0]);
    __m256 c51 = _mm256_loadu_ps(&C[5 * ldc + 8]);

    int p = 0;
    // Main loop with K-unrolling by 4
    for (; p + 4 <= K_block; p += 4) {
        // Prefetch next B rows
        _mm_prefetch((const char*)&B[(p + PREFETCH_DISTANCE) * ldb], _MM_HINT_T0);

        // Iteration 0
        {
            __m256 b0 = load_bf16_as_fp32(&B[p * ldb + 0]);
            __m256 b1 = load_bf16_as_fp32(&B[p * ldb + 8]);
            __m256 a0 = _mm256_set1_ps(bf16_to_float(A[0 * lda + p]));
            __m256 a1 = _mm256_set1_ps(bf16_to_float(A[1 * lda + p]));
            __m256 a2 = _mm256_set1_ps(bf16_to_float(A[2 * lda + p]));
            __m256 a3 = _mm256_set1_ps(bf16_to_float(A[3 * lda + p]));
            __m256 a4 = _mm256_set1_ps(bf16_to_float(A[4 * lda + p]));
            __m256 a5 = _mm256_set1_ps(bf16_to_float(A[5 * lda + p]));
            c00 = _mm256_fmadd_ps(a0, b0, c00); c01 = _mm256_fmadd_ps(a0, b1, c01);
            c10 = _mm256_fmadd_ps(a1, b0, c10); c11 = _mm256_fmadd_ps(a1, b1, c11);
            c20 = _mm256_fmadd_ps(a2, b0, c20); c21 = _mm256_fmadd_ps(a2, b1, c21);
            c30 = _mm256_fmadd_ps(a3, b0, c30); c31 = _mm256_fmadd_ps(a3, b1, c31);
            c40 = _mm256_fmadd_ps(a4, b0, c40); c41 = _mm256_fmadd_ps(a4, b1, c41);
            c50 = _mm256_fmadd_ps(a5, b0, c50); c51 = _mm256_fmadd_ps(a5, b1, c51);
        }
        // Iteration 1
        {
            __m256 b0 = load_bf16_as_fp32(&B[(p+1) * ldb + 0]);
            __m256 b1 = load_bf16_as_fp32(&B[(p+1) * ldb + 8]);
            __m256 a0 = _mm256_set1_ps(bf16_to_float(A[0 * lda + p+1]));
            __m256 a1 = _mm256_set1_ps(bf16_to_float(A[1 * lda + p+1]));
            __m256 a2 = _mm256_set1_ps(bf16_to_float(A[2 * lda + p+1]));
            __m256 a3 = _mm256_set1_ps(bf16_to_float(A[3 * lda + p+1]));
            __m256 a4 = _mm256_set1_ps(bf16_to_float(A[4 * lda + p+1]));
            __m256 a5 = _mm256_set1_ps(bf16_to_float(A[5 * lda + p+1]));
            c00 = _mm256_fmadd_ps(a0, b0, c00); c01 = _mm256_fmadd_ps(a0, b1, c01);
            c10 = _mm256_fmadd_ps(a1, b0, c10); c11 = _mm256_fmadd_ps(a1, b1, c11);
            c20 = _mm256_fmadd_ps(a2, b0, c20); c21 = _mm256_fmadd_ps(a2, b1, c21);
            c30 = _mm256_fmadd_ps(a3, b0, c30); c31 = _mm256_fmadd_ps(a3, b1, c31);
            c40 = _mm256_fmadd_ps(a4, b0, c40); c41 = _mm256_fmadd_ps(a4, b1, c41);
            c50 = _mm256_fmadd_ps(a5, b0, c50); c51 = _mm256_fmadd_ps(a5, b1, c51);
        }
        // Iteration 2
        {
            __m256 b0 = load_bf16_as_fp32(&B[(p+2) * ldb + 0]);
            __m256 b1 = load_bf16_as_fp32(&B[(p+2) * ldb + 8]);
            __m256 a0 = _mm256_set1_ps(bf16_to_float(A[0 * lda + p+2]));
            __m256 a1 = _mm256_set1_ps(bf16_to_float(A[1 * lda + p+2]));
            __m256 a2 = _mm256_set1_ps(bf16_to_float(A[2 * lda + p+2]));
            __m256 a3 = _mm256_set1_ps(bf16_to_float(A[3 * lda + p+2]));
            __m256 a4 = _mm256_set1_ps(bf16_to_float(A[4 * lda + p+2]));
            __m256 a5 = _mm256_set1_ps(bf16_to_float(A[5 * lda + p+2]));
            c00 = _mm256_fmadd_ps(a0, b0, c00); c01 = _mm256_fmadd_ps(a0, b1, c01);
            c10 = _mm256_fmadd_ps(a1, b0, c10); c11 = _mm256_fmadd_ps(a1, b1, c11);
            c20 = _mm256_fmadd_ps(a2, b0, c20); c21 = _mm256_fmadd_ps(a2, b1, c21);
            c30 = _mm256_fmadd_ps(a3, b0, c30); c31 = _mm256_fmadd_ps(a3, b1, c31);
            c40 = _mm256_fmadd_ps(a4, b0, c40); c41 = _mm256_fmadd_ps(a4, b1, c41);
            c50 = _mm256_fmadd_ps(a5, b0, c50); c51 = _mm256_fmadd_ps(a5, b1, c51);
        }
        // Iteration 3
        {
            __m256 b0 = load_bf16_as_fp32(&B[(p+3) * ldb + 0]);
            __m256 b1 = load_bf16_as_fp32(&B[(p+3) * ldb + 8]);
            __m256 a0 = _mm256_set1_ps(bf16_to_float(A[0 * lda + p+3]));
            __m256 a1 = _mm256_set1_ps(bf16_to_float(A[1 * lda + p+3]));
            __m256 a2 = _mm256_set1_ps(bf16_to_float(A[2 * lda + p+3]));
            __m256 a3 = _mm256_set1_ps(bf16_to_float(A[3 * lda + p+3]));
            __m256 a4 = _mm256_set1_ps(bf16_to_float(A[4 * lda + p+3]));
            __m256 a5 = _mm256_set1_ps(bf16_to_float(A[5 * lda + p+3]));
            c00 = _mm256_fmadd_ps(a0, b0, c00); c01 = _mm256_fmadd_ps(a0, b1, c01);
            c10 = _mm256_fmadd_ps(a1, b0, c10); c11 = _mm256_fmadd_ps(a1, b1, c11);
            c20 = _mm256_fmadd_ps(a2, b0, c20); c21 = _mm256_fmadd_ps(a2, b1, c21);
            c30 = _mm256_fmadd_ps(a3, b0, c30); c31 = _mm256_fmadd_ps(a3, b1, c31);
            c40 = _mm256_fmadd_ps(a4, b0, c40); c41 = _mm256_fmadd_ps(a4, b1, c41);
            c50 = _mm256_fmadd_ps(a5, b0, c50); c51 = _mm256_fmadd_ps(a5, b1, c51);
        }
    }

    // Remainder loop
    for (; p < K_block; ++p) {
        __m256 b0 = load_bf16_as_fp32(&B[p * ldb + 0]);
        __m256 b1 = load_bf16_as_fp32(&B[p * ldb + 8]);
        __m256 a0 = _mm256_set1_ps(bf16_to_float(A[0 * lda + p]));
        __m256 a1 = _mm256_set1_ps(bf16_to_float(A[1 * lda + p]));
        __m256 a2 = _mm256_set1_ps(bf16_to_float(A[2 * lda + p]));
        __m256 a3 = _mm256_set1_ps(bf16_to_float(A[3 * lda + p]));
        __m256 a4 = _mm256_set1_ps(bf16_to_float(A[4 * lda + p]));
        __m256 a5 = _mm256_set1_ps(bf16_to_float(A[5 * lda + p]));
        c00 = _mm256_fmadd_ps(a0, b0, c00); c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10); c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20); c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30); c31 = _mm256_fmadd_ps(a3, b1, c31);
        c40 = _mm256_fmadd_ps(a4, b0, c40); c41 = _mm256_fmadd_ps(a4, b1, c41);
        c50 = _mm256_fmadd_ps(a5, b0, c50); c51 = _mm256_fmadd_ps(a5, b1, c51);
    }

    // Store results
    _mm256_storeu_ps(&C[0 * ldc + 0], c00); _mm256_storeu_ps(&C[0 * ldc + 8], c01);
    _mm256_storeu_ps(&C[1 * ldc + 0], c10); _mm256_storeu_ps(&C[1 * ldc + 8], c11);
    _mm256_storeu_ps(&C[2 * ldc + 0], c20); _mm256_storeu_ps(&C[2 * ldc + 8], c21);
    _mm256_storeu_ps(&C[3 * ldc + 0], c30); _mm256_storeu_ps(&C[3 * ldc + 8], c31);
    _mm256_storeu_ps(&C[4 * ldc + 0], c40); _mm256_storeu_ps(&C[4 * ldc + 8], c41);
    _mm256_storeu_ps(&C[5 * ldc + 0], c50); _mm256_storeu_ps(&C[5 * ldc + 8], c51);
}

/* Tiled GEMM kernel with 6x16 micro-kernel */
static void sbgemm_kernel_tiled_6x16(int M, int N, int K,
                                      float alpha,
                                      const bf16 *A, int lda,
                                      const bf16 *B, int ldb,
                                      float *C, int ldc)
{
    for (int pk = 0; pk < K; pk += BLOCK_K) {
        int K_block = (pk + BLOCK_K <= K) ? BLOCK_K : (K - pk);

        for (int pi = 0; pi < M; pi += BLOCK_M) {
            int M_block = (pi + BLOCK_M <= M) ? BLOCK_M : (M - pi);

            for (int pj = 0; pj < N; pj += BLOCK_N) {
                int N_block = (pj + BLOCK_N <= N) ? BLOCK_N : (N - pj);

                // Process 6x16 micro-blocks
                int i;
                for (i = 0; i + 6 <= M_block; i += 6) {
                    int j;
                    for (j = 0; j + 16 <= N_block; j += 16) {
                        micro_kernel_6x16(
                            K_block,
                            &A[(pi + i) * lda + pk], lda,
                            &B[pk * ldb + (pj + j)], ldb,
                            &C[(pi + i) * ldc + (pj + j)], ldc);
                    }
                    // Handle N remainder (< 16 columns)
                    for (; j < N_block; ++j) {
                        for (int ii = 0; ii < 6; ++ii) {
                            float sum = 0.0f;
                            for (int p = 0; p < K_block; ++p) {
                                sum += bf16_to_float(A[(pi+i+ii)*lda + pk+p]) *
                                       bf16_to_float(B[(pk+p)*ldb + pj+j]);
                            }
                            C[(pi+i+ii)*ldc + pj+j] += sum;
                        }
                    }
                }
                // Handle M remainder (< 6 rows)
                for (; i < M_block; ++i) {
                    for (int j = 0; j < N_block; ++j) {
                        float sum = 0.0f;
                        for (int p = 0; p < K_block; ++p) {
                            sum += bf16_to_float(A[(pi+i)*lda + pk+p]) *
                                   bf16_to_float(B[(pk+p)*ldb + pj+j]);
                        }
                        C[(pi+i)*ldc + pj+j] += sum;
                    }
                }
            }
        }
    }

    // Apply alpha
    if (alpha != 1.0f) {
        __m256 alpha_vec = _mm256_set1_ps(alpha);
        for (int i = 0; i < M; ++i) {
            int j = 0;
            for (; j + 8 <= N; j += 8) {
                __m256 c = _mm256_loadu_ps(&C[i*ldc + j]);
                _mm256_storeu_ps(&C[i*ldc + j], _mm256_mul_ps(c, alpha_vec));
            }
            for (; j < N; ++j) C[i*ldc + j] *= alpha;
        }
    }
}

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
        __m256 zero = _mm256_setzero_ps();
        for (int i = 0; i < M; ++i) {
            int j = 0;
            for (; j + 8 <= N; j += 8) _mm256_storeu_ps(&C[i*ldc + j], zero);
            for (; j < N; ++j) C[i*ldc + j] = 0.0f;
        }
    } else if (beta != 1.0f) {
        __m256 beta_vec = _mm256_set1_ps(beta);
        for (int i = 0; i < M; ++i) {
            int j = 0;
            for (; j + 8 <= N; j += 8) {
                __m256 c = _mm256_loadu_ps(&C[i*ldc + j]);
                _mm256_storeu_ps(&C[i*ldc + j], _mm256_mul_ps(c, beta_vec));
            }
            for (; j < N; ++j) C[i*ldc + j] *= beta;
        }
    }

    if (transA == CblasNoTrans && transB == CblasNoTrans) {
        sbgemm_kernel_tiled_6x16(M, N, K, alpha, A, lda, B, ldb, C, ldc);
        return;
    }

    // Fallback
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < K; ++p) {
                float a = (transA == CblasNoTrans) ?
                    bf16_to_float(A[i*lda + p]) : bf16_to_float(A[p*lda + i]);
                float b = (transB == CblasNoTrans) ?
                    bf16_to_float(B[p*ldb + j]) : bf16_to_float(B[j*ldb + p]);
                sum += a * b;
            }
            C[i*ldc + j] += alpha * sum;
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

    printf("sbgemm_v1.3.0 - AVX2 SIMD with 6x16 micro-kernel and K-unrolling\n");
    printf("Matrix size: M=%d, K=%d, N=%d\n", M, K, N);
    printf("Block sizes: BLOCK_M=%d, BLOCK_N=%d, BLOCK_K=%d\n", BLOCK_M, BLOCK_N, BLOCK_K);
    printf("Number of trials: %d\n", num_trials);

    bf16 *A_bf16 = (bf16*)aligned_alloc(64, sizeof(bf16) * M * K);
    bf16 *B_bf16 = (bf16*)aligned_alloc(64, sizeof(bf16) * K * N);
    float *C_f32 = (float*)aligned_alloc(64, sizeof(float) * M * N);

    if (!A_bf16 || !B_bf16 || !C_f32) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    srand(42);
    for (int i = 0; i < M * K; ++i) {
        A_bf16[i] = float_to_bf16_round((float)(rand() % 100) / 100.0f);
    }
    for (int i = 0; i < K * N; ++i) {
        B_bf16[i] = float_to_bf16_round((float)(rand() % 100) / 100.0f);
    }

    const int lda = K, ldb = N, ldc = N;

    // Warm-up
    memset(C_f32, 0, sizeof(float) * M * N);
    sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 M, N, K, 1.0f, A_bf16, lda, B_bf16, ldb, 0.0f, C_f32, ldc);

    double total_time = 0.0, min_time = 1e30, max_time = 0.0;

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
