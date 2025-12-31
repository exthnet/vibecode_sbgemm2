// sbgemm_v1.1.0.c - AVX2 SIMD + Cache Blocking Optimized BF16 GEMM
// PG1.1 - GCC 8.5.0 + SIMD (AVX2)
//
// Optimizations (v1.0.0):
// - AVX2 vectorization for bf16 to fp32 conversion
// - 8-wide FMA operations
//
// New in v1.1.0:
// - Cache blocking (tile-based computation)
// - Improved data locality
// - Loop reordering for better cache utilization

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
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

// Block sizes for cache optimization
// Tuned for typical L1 (32KB), L2 (256KB-2MB) cache sizes
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 256

/* --- BF16 <-> FP32 Conversion --- */

static inline float bf16_to_float(bf16 x) {
    uint32_t u = ((uint32_t)x) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static inline bf16 float_to_bf16_round(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    u += 0x00008000u;
    return (bf16)(u >> 16);
}

// AVX2: Convert 8 BF16 values to 8 FP32 values
static inline __m256 bf16x8_to_fp32(__m128i bf16_vals) {
    __m256i expanded = _mm256_cvtepu16_epi32(bf16_vals);
    __m256i shifted = _mm256_slli_epi32(expanded, 16);
    return _mm256_castsi256_ps(shifted);
}

static inline __m128i load_bf16x8(const bf16* ptr) {
    return _mm_loadu_si128((const __m128i*)ptr);
}

/* --- Micro-kernel: Compute small block with AVX2 --- */

static void microkernel_avx2(
    int m_size, int n_size, int k_size,
    const bf16 *A, int lda,
    const bf16 *B, int ldb,
    float *C, int ldc,
    float alpha)
{
    // Process rows of C
    for (int i = 0; i < m_size; ++i) {
        // Process columns of C
        for (int j = 0; j < n_size; ++j) {
            __m256 sum_vec = _mm256_setzero_ps();
            float sum_scalar = 0.0f;

            int p = 0;
            // Vectorized loop: process 8 elements at a time
            for (; p + 7 < k_size; p += 8) {
                // Load A[i, p:p+8]
                __m128i a_bf16 = load_bf16x8(&A[i * lda + p]);
                __m256 a_fp32 = bf16x8_to_fp32(a_bf16);

                // Load B[p:p+8, j] (strided for NoTrans B)
                float b_vals[8] __attribute__((aligned(32)));
                for (int k = 0; k < 8; ++k) {
                    b_vals[k] = bf16_to_float(B[(p + k) * ldb + j]);
                }
                __m256 b_fp32 = _mm256_load_ps(b_vals);

                // FMA
                sum_vec = _mm256_fmadd_ps(a_fp32, b_fp32, sum_vec);
            }

            // Horizontal sum
            __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
            __m128 sum_low = _mm256_castps256_ps128(sum_vec);
            __m128 sum128 = _mm_add_ps(sum_high, sum_low);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum_scalar = _mm_cvtss_f32(sum128);

            // Scalar remainder
            for (; p < k_size; ++p) {
                float a_val = bf16_to_float(A[i * lda + p]);
                float b_val = bf16_to_float(B[p * ldb + j]);
                sum_scalar += a_val * b_val;
            }

            C[i * ldc + j] += alpha * sum_scalar;
        }
    }
}

/* --- Micro-kernel optimized for Trans B (contiguous B access) --- */

static void microkernel_avx2_transB(
    int m_size, int n_size, int k_size,
    const bf16 *A, int lda,
    const bf16 *B, int ldb,  // B is transposed, so B[j, :] is contiguous
    float *C, int ldc,
    float alpha)
{
    for (int i = 0; i < m_size; ++i) {
        for (int j = 0; j < n_size; ++j) {
            __m256 sum_vec = _mm256_setzero_ps();
            float sum_scalar = 0.0f;

            int p = 0;
            // Both A[i,:] and B[j,:] are contiguous - ideal for vectorization
            for (; p + 7 < k_size; p += 8) {
                __m128i a_bf16 = load_bf16x8(&A[i * lda + p]);
                __m256 a_fp32 = bf16x8_to_fp32(a_bf16);

                __m128i b_bf16 = load_bf16x8(&B[j * ldb + p]);
                __m256 b_fp32 = bf16x8_to_fp32(b_bf16);

                sum_vec = _mm256_fmadd_ps(a_fp32, b_fp32, sum_vec);
            }

            // Horizontal sum
            __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
            __m128 sum_low = _mm256_castps256_ps128(sum_vec);
            __m128 sum128 = _mm_add_ps(sum_high, sum_low);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum_scalar = _mm_cvtss_f32(sum128);

            // Remainder
            for (; p < k_size; ++p) {
                float a_val = bf16_to_float(A[i * lda + p]);
                float b_val = bf16_to_float(B[j * ldb + p]);
                sum_scalar += a_val * b_val;
            }

            C[i * ldc + j] += alpha * sum_scalar;
        }
    }
}

/* --- Main SBGEMM with Cache Blocking --- */

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

    // Apply beta to C
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

    // Cache blocking: iterate over blocks
    if (transA == CblasNoTrans && transB == CblasNoTrans) {
        // Block over K (outer) for better reuse of C blocks
        for (int kk = 0; kk < K; kk += BLOCK_K) {
            int k_size = (kk + BLOCK_K <= K) ? BLOCK_K : (K - kk);

            // Block over M
            for (int ii = 0; ii < M; ii += BLOCK_M) {
                int m_size = (ii + BLOCK_M <= M) ? BLOCK_M : (M - ii);

                // Block over N
                for (int jj = 0; jj < N; jj += BLOCK_N) {
                    int n_size = (jj + BLOCK_N <= N) ? BLOCK_N : (N - jj);

                    // Compute block: C[ii:ii+m_size, jj:jj+n_size] +=
                    //                A[ii:ii+m_size, kk:kk+k_size] *
                    //                B[kk:kk+k_size, jj:jj+n_size]
                    microkernel_avx2(
                        m_size, n_size, k_size,
                        &A[ii * lda + kk], lda,
                        &B[kk * ldb + jj], ldb,
                        &C[ii * ldc + jj], ldc,
                        alpha);
                }
            }
        }
    }
    else if (transA == CblasNoTrans && transB == CblasTrans) {
        // A: NoTrans, B: Trans -> B[j, :] is contiguous
        for (int kk = 0; kk < K; kk += BLOCK_K) {
            int k_size = (kk + BLOCK_K <= K) ? BLOCK_K : (K - kk);

            for (int ii = 0; ii < M; ii += BLOCK_M) {
                int m_size = (ii + BLOCK_M <= M) ? BLOCK_M : (M - ii);

                for (int jj = 0; jj < N; jj += BLOCK_N) {
                    int n_size = (jj + BLOCK_N <= N) ? BLOCK_N : (N - jj);

                    microkernel_avx2_transB(
                        m_size, n_size, k_size,
                        &A[ii * lda + kk], lda,
                        &B[jj * ldb + kk], ldb,
                        &C[ii * ldc + jj], ldc,
                        alpha);
                }
            }
        }
    }
    else if (transA == CblasTrans && transB == CblasNoTrans) {
        // Fallback to scalar for Trans A cases
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int p = 0; p < K; ++p) {
                    float a_val = bf16_to_float(A[p * lda + i]);
                    float b_val = bf16_to_float(B[p * ldb + j]);
                    sum += a_val * b_val;
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    }
    else {
        // Trans A x Trans B - fallback
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int p = 0; p < K; ++p) {
                    float a_val = bf16_to_float(A[p * lda + i]);
                    float b_val = bf16_to_float(B[j * ldb + p]);
                    sum += a_val * b_val;
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    }
}

/* --- Benchmark Code --- */

double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void init_matrix_bf16(bf16 *M, int rows, int cols, float scale) {
    for (int i = 0; i < rows * cols; ++i) {
        float val = (float)(i % 100) * scale / 100.0f;
        M[i] = float_to_bf16_round(val);
    }
}

void init_matrix_fp32(float *M, int rows, int cols, float val) {
    for (int i = 0; i < rows * cols; ++i) {
        M[i] = val;
    }
}

int main(int argc, char *argv[]) {
    int sizes[] = {128, 256, 512, 1024, 2048, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int num_runs = 3;

    printf("SBGEMM Benchmark - AVX2 SIMD + Cache Blocking (v1.1.0)\n");
    printf("Compiler: GCC 8.5.0\n");
    printf("Block sizes: M=%d, N=%d, K=%d\n", BLOCK_M, BLOCK_N, BLOCK_K);
    printf("====================================================\n");
    printf("%8s %12s %12s\n", "Size", "Time(s)", "GFLOPS");
    printf("----------------------------------------------------\n");

    for (int s = 0; s < num_sizes; ++s) {
        int N = sizes[s];
        int M_size = N, K_size = N;

        bf16 *A = (bf16*)aligned_alloc(32, sizeof(bf16) * M_size * K_size);
        bf16 *B = (bf16*)aligned_alloc(32, sizeof(bf16) * K_size * N);
        float *C = (float*)aligned_alloc(32, sizeof(float) * M_size * N);

        if (!A || !B || !C) {
            fprintf(stderr, "Memory allocation failed for size %d\n", N);
            continue;
        }

        init_matrix_bf16(A, M_size, K_size, 1.0f);
        init_matrix_bf16(B, K_size, N, 1.0f);
        init_matrix_fp32(C, M_size, N, 0.0f);

        // Warmup
        sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                     M_size, N, K_size, 1.0f, A, K_size, B, N, 0.0f, C, N);

        // Benchmark
        double total_time = 0.0;
        for (int r = 0; r < num_runs; ++r) {
            init_matrix_fp32(C, M_size, N, 0.0f);

            double start = get_time_sec();
            sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                         M_size, N, K_size, 1.0f, A, K_size, B, N, 0.0f, C, N);
            double end = get_time_sec();

            total_time += (end - start);
        }

        double avg_time = total_time / num_runs;
        double flops = 2.0 * M_size * N * K_size;
        double gflops = flops / avg_time / 1e9;

        printf("%8d %12.6f %12.2f\n", N, avg_time, gflops);

        free(A);
        free(B);
        free(C);
    }

    printf("====================================================\n");

    return 0;
}
