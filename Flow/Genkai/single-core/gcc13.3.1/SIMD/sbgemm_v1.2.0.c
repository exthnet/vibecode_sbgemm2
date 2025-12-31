// sbgemm_v1.2.0.c - AVX2 SIMD with loop tiling
// PG1.3: GCC 13.3.1 SIMD optimization
// Features: AVX2 SIMD, FMA, loop tiling for better cache utilization

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
// Tuned for L1/L2 cache (32KB L1D, 512KB L2 typical for EPYC)
#define BLOCK_M 32
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
    __m256i int32_vals = _mm256_cvtepu16_epi32(bf16_vals);
    __m256i shifted = _mm256_slli_epi32(int32_vals, 16);
    return _mm256_castsi256_ps(shifted);
}

static inline __m256 load_bf16_as_fp32(const bf16 *ptr) {
    __m128i bf16_vals = _mm_loadu_si128((const __m128i*)ptr);
    return bf16x8_to_fp32(bf16_vals);
}

/* Micro-kernel: compute a small block of C */
static inline void micro_kernel_4x8(
    int K_block,
    const bf16 *A, int lda,  // A block: 4 x K_block
    const bf16 *B, int ldb,  // B block: K_block x 8
    float *C, int ldc)       // C block: 4 x 8
{
    // 4 accumulators for 4 rows, each 8 wide
    __m256 c0 = _mm256_loadu_ps(&C[0 * ldc]);
    __m256 c1 = _mm256_loadu_ps(&C[1 * ldc]);
    __m256 c2 = _mm256_loadu_ps(&C[2 * ldc]);
    __m256 c3 = _mm256_loadu_ps(&C[3 * ldc]);

    for (int p = 0; p < K_block; ++p) {
        // Load B row (8 elements)
        __m256 b_vec = load_bf16_as_fp32(&B[p * ldb]);

        // Load A elements and broadcast
        __m256 a0 = _mm256_set1_ps(bf16_to_float(A[0 * lda + p]));
        __m256 a1 = _mm256_set1_ps(bf16_to_float(A[1 * lda + p]));
        __m256 a2 = _mm256_set1_ps(bf16_to_float(A[2 * lda + p]));
        __m256 a3 = _mm256_set1_ps(bf16_to_float(A[3 * lda + p]));

        // FMA
        c0 = _mm256_fmadd_ps(a0, b_vec, c0);
        c1 = _mm256_fmadd_ps(a1, b_vec, c1);
        c2 = _mm256_fmadd_ps(a2, b_vec, c2);
        c3 = _mm256_fmadd_ps(a3, b_vec, c3);
    }

    // Store results
    _mm256_storeu_ps(&C[0 * ldc], c0);
    _mm256_storeu_ps(&C[1 * ldc], c1);
    _mm256_storeu_ps(&C[2 * ldc], c2);
    _mm256_storeu_ps(&C[3 * ldc], c3);
}

/* Tiled GEMM kernel */
static void sbgemm_kernel_tiled(int M, int N, int K,
                                 float alpha,
                                 const bf16 *A, int lda,
                                 const bf16 *B, int ldb,
                                 float *C, int ldc)
{
    // Loop over blocks of K (outermost for better cache reuse of B)
    for (int pk = 0; pk < K; pk += BLOCK_K) {
        int K_block = (pk + BLOCK_K <= K) ? BLOCK_K : (K - pk);

        // Loop over blocks of M
        for (int pi = 0; pi < M; pi += BLOCK_M) {
            int M_block = (pi + BLOCK_M <= M) ? BLOCK_M : (M - pi);

            // Loop over blocks of N
            for (int pj = 0; pj < N; pj += BLOCK_N) {
                int N_block = (pj + BLOCK_N <= N) ? BLOCK_N : (N - pj);

                // Process 4x8 micro-blocks
                for (int i = 0; i < M_block - 3; i += 4) {
                    for (int j = 0; j < N_block - 7; j += 8) {
                        micro_kernel_4x8(
                            K_block,
                            &A[(pi + i) * lda + pk], lda,
                            &B[pk * ldb + (pj + j)], ldb,
                            &C[(pi + i) * ldc + (pj + j)], ldc);
                    }

                    // Handle N remainder
                    for (int j = (N_block / 8) * 8; j < N_block; ++j) {
                        for (int ii = 0; ii < 4; ++ii) {
                            float sum = 0.0f;
                            for (int p = 0; p < K_block; ++p) {
                                float a_val = bf16_to_float(A[(pi + i + ii) * lda + (pk + p)]);
                                float b_val = bf16_to_float(B[(pk + p) * ldb + (pj + j)]);
                                sum += a_val * b_val;
                            }
                            C[(pi + i + ii) * ldc + (pj + j)] += sum;
                        }
                    }
                }

                // Handle M remainder
                for (int i = (M_block / 4) * 4; i < M_block; ++i) {
                    for (int j = 0; j < N_block; ++j) {
                        float sum = 0.0f;
                        for (int p = 0; p < K_block; ++p) {
                            float a_val = bf16_to_float(A[(pi + i) * lda + (pk + p)]);
                            float b_val = bf16_to_float(B[(pk + p) * ldb + (pj + j)]);
                            sum += a_val * b_val;
                        }
                        C[(pi + i) * ldc + (pj + j)] += sum;
                    }
                }
            }
        }
    }

    // Apply alpha if not 1.0
    if (alpha != 1.0f) {
        __m256 alpha_vec = _mm256_set1_ps(alpha);
        for (int i = 0; i < M; ++i) {
            int j = 0;
            for (; j + 8 <= N; j += 8) {
                __m256 c_vec = _mm256_loadu_ps(&C[i * ldc + j]);
                c_vec = _mm256_mul_ps(c_vec, alpha_vec);
                _mm256_storeu_ps(&C[i * ldc + j], c_vec);
            }
            for (; j < N; ++j) {
                C[i * ldc + j] *= alpha;
            }
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
        for (int i = 0; i < M; ++i) {
            int j = 0;
            __m256 zero = _mm256_setzero_ps();
            for (; j + 8 <= N; j += 8) {
                _mm256_storeu_ps(&C[i * ldc + j], zero);
            }
            for (; j < N; ++j) {
                C[i * ldc + j] = 0.0f;
            }
        }
    } else if (beta != 1.0f) {
        __m256 beta_vec = _mm256_set1_ps(beta);
        for (int i = 0; i < M; ++i) {
            int j = 0;
            for (; j + 8 <= N; j += 8) {
                __m256 c_vec = _mm256_loadu_ps(&C[i * ldc + j]);
                c_vec = _mm256_mul_ps(c_vec, beta_vec);
                _mm256_storeu_ps(&C[i * ldc + j], c_vec);
            }
            for (; j < N; ++j) {
                C[i * ldc + j] *= beta;
            }
        }
    }

    // Use tiled kernel for NoTrans, NoTrans case
    if (transA == CblasNoTrans && transB == CblasNoTrans) {
        sbgemm_kernel_tiled(M, N, K, alpha, A, lda, B, ldb, C, ldc);
        return;
    }

    // Fallback for transposed cases
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < K; ++p) {
                float a_ip, b_pj;
                if (transA == CblasNoTrans)
                    a_ip = bf16_to_float(A[i * lda + p]);
                else
                    a_ip = bf16_to_float(A[p * lda + i]);

                if (transB == CblasNoTrans)
                    b_pj = bf16_to_float(B[p * ldb + j]);
                else
                    b_pj = bf16_to_float(B[j * ldb + p]);

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

    printf("sbgemm_v1.2.0 - AVX2 SIMD with loop tiling\n");
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
        float val = (float)(rand() % 100) / 100.0f;
        A_bf16[i] = float_to_bf16_round(val);
    }
    for (int i = 0; i < K * N; ++i) {
        float val = (float)(rand() % 100) / 100.0f;
        B_bf16[i] = float_to_bf16_round(val);
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
