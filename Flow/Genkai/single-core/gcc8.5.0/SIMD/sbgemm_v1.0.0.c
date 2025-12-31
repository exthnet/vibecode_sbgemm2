// sbgemm_v1.0.0.c - AVX2 SIMD Optimized BF16 GEMM
// PG1.1 - GCC 8.5.0 + SIMD (AVX2)
//
// Optimizations:
// - AVX2 vectorization for bf16 to fp32 conversion
// - 8-wide FMA operations
// - Basic loop unrolling

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

/* --- BF16 <-> FP32 Conversion --- */

// Scalar: BF16 -> FP32
static inline float bf16_to_float(bf16 x) {
    uint32_t u = ((uint32_t)x) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

// Scalar: FP32 -> BF16 (round to nearest)
static inline bf16 float_to_bf16_round(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    u += 0x00008000u;
    return (bf16)(u >> 16);
}

// AVX2: Convert 8 BF16 values to 8 FP32 values
static inline __m256 bf16x8_to_fp32(__m128i bf16_vals) {
    // Zero-extend bf16 to 32-bit and shift left by 16
    __m256i expanded = _mm256_cvtepu16_epi32(bf16_vals);
    __m256i shifted = _mm256_slli_epi32(expanded, 16);
    return _mm256_castsi256_ps(shifted);
}

// Load 8 BF16 values from memory
static inline __m128i load_bf16x8(const bf16* ptr) {
    return _mm_loadu_si128((const __m128i*)ptr);
}

/* --- Optimized SBGEMM --- */

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

    __m256 valpha = _mm256_set1_ps(alpha);

    // Main computation - NoTrans x NoTrans case optimized
    if (transA == CblasNoTrans && transB == CblasNoTrans) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                __m256 sum_vec = _mm256_setzero_ps();
                float sum_scalar = 0.0f;

                int p = 0;
                // Process 8 elements at a time with AVX2
                for (; p + 7 < K; p += 8) {
                    // Load 8 BF16 values from A[i, p:p+8]
                    __m128i a_bf16 = load_bf16x8(&A[i * lda + p]);
                    __m256 a_fp32 = bf16x8_to_fp32(a_bf16);

                    // Load 8 BF16 values from B[p:p+8, j] (strided access)
                    // For NoTrans B, we need B[p][j], B[p+1][j], ...
                    // This requires gather or scalar loads
                    float b_vals[8];
                    for (int k = 0; k < 8; ++k) {
                        b_vals[k] = bf16_to_float(B[(p + k) * ldb + j]);
                    }
                    __m256 b_fp32 = _mm256_loadu_ps(b_vals);

                    // FMA: sum += a * b
                    sum_vec = _mm256_fmadd_ps(a_fp32, b_fp32, sum_vec);
                }

                // Horizontal sum of vector
                __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
                __m128 sum_low = _mm256_castps256_ps128(sum_vec);
                __m128 sum128 = _mm_add_ps(sum_high, sum_low);
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum128 = _mm_hadd_ps(sum128, sum128);
                sum_scalar = _mm_cvtss_f32(sum128);

                // Handle remaining elements
                for (; p < K; ++p) {
                    float a_val = bf16_to_float(A[i * lda + p]);
                    float b_val = bf16_to_float(B[p * ldb + j]);
                    sum_scalar += a_val * b_val;
                }

                C[i * ldc + j] += alpha * sum_scalar;
            }
        }
    }
    // Trans A x NoTrans B
    else if (transA == CblasTrans && transB == CblasNoTrans) {
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
    // NoTrans A x Trans B
    else if (transA == CblasNoTrans && transB == CblasTrans) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                __m256 sum_vec = _mm256_setzero_ps();
                float sum_scalar = 0.0f;

                int p = 0;
                // Both A row and B row are contiguous - ideal for vectorization
                for (; p + 7 < K; p += 8) {
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
                for (; p < K; ++p) {
                    float a_val = bf16_to_float(A[i * lda + p]);
                    float b_val = bf16_to_float(B[j * ldb + p]);
                    sum_scalar += a_val * b_val;
                }

                C[i * ldc + j] += alpha * sum_scalar;
            }
        }
    }
    // Trans A x Trans B
    else {
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
    // Default sizes for benchmarking
    int sizes[] = {128, 256, 512, 1024, 2048, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int num_runs = 3;

    printf("SBGEMM Benchmark - AVX2 SIMD Optimized (v1.0.0)\n");
    printf("Compiler: GCC 8.5.0\n");
    printf("================================================\n");
    printf("%8s %12s %12s %12s\n", "Size", "Time(s)", "GFLOPS", "Efficiency");
    printf("------------------------------------------------\n");

    for (int s = 0; s < num_sizes; ++s) {
        int N = sizes[s];
        int M = N, K = N;

        // Allocate matrices
        bf16 *A = (bf16*)aligned_alloc(32, sizeof(bf16) * M * K);
        bf16 *B = (bf16*)aligned_alloc(32, sizeof(bf16) * K * N);
        float *C = (float*)aligned_alloc(32, sizeof(float) * M * N);

        if (!A || !B || !C) {
            fprintf(stderr, "Memory allocation failed for size %d\n", N);
            continue;
        }

        // Initialize
        init_matrix_bf16(A, M, K, 1.0f);
        init_matrix_bf16(B, K, N, 1.0f);
        init_matrix_fp32(C, M, N, 0.0f);

        // Warmup
        sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                     M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

        // Benchmark
        double total_time = 0.0;
        for (int r = 0; r < num_runs; ++r) {
            init_matrix_fp32(C, M, N, 0.0f);

            double start = get_time_sec();
            sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                         M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
            double end = get_time_sec();

            total_time += (end - start);
        }

        double avg_time = total_time / num_runs;
        double flops = 2.0 * M * N * K;  // multiply-add = 2 ops
        double gflops = flops / avg_time / 1e9;

        printf("%8d %12.6f %12.2f %12s\n", N, avg_time, gflops, "TBD");

        free(A);
        free(B);
        free(C);
    }

    printf("================================================\n");

    return 0;
}
