// sbgemm_v1.1.0.c - AVX-512 BF16 Implementation with Cache Blocking
// Alternative to AMX: Using AVX-512 VNNI BF16 instructions
// Target: High efficiency using cache blocking and SIMD optimization

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <sys/time.h>

typedef uint16_t bf16;

// Cache blocking parameters (from reference.pdf)
#define BLOCK_M 96
#define BLOCK_N 480
#define BLOCK_K 1536

// AVX-512 vector width
#define VEC_WIDTH 16  // 16 floats per 512-bit register

typedef enum {
    CblasRowMajor = 101,
    CblasColMajor = 102
} CBLAS_LAYOUT;

typedef enum {
    CblasNoTrans  = 111,
    CblasTrans    = 112,
    CblasConjTrans= 113
} CBLAS_TRANSPOSE;

// BF16 <-> FP32 conversion
static inline bf16 float_to_bf16(float x) {
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

// Convert BF16 vector to FP32 using AVX-512
static inline __m512 bf16_to_fp32_avx512(__m256i bf16_vec) {
    // Shift BF16 values to upper 16 bits to get FP32
    __m512i extended = _mm512_cvtepu16_epi32(bf16_vec);
    __m512i shifted = _mm512_slli_epi32(extended, 16);
    return _mm512_castsi512_ps(shifted);
}

// Micro-kernel: compute C[m:m+4][n:n+16] += A[m:m+4][k:k+klen] * B[k:k+klen][n:n+16]
// Uses 4x1 register blocking for C (4 rows, 16 columns per iteration)
static void micro_kernel_4x16(const bf16 *A, const bf16 *B, float *C,
                              int lda, int ldb, int ldc,
                              int m_size, int k_size) {
    __m512 c0 = _mm512_loadu_ps(C);
    __m512 c1 = _mm512_loadu_ps(C + ldc);
    __m512 c2 = _mm512_loadu_ps(C + 2*ldc);
    __m512 c3 = _mm512_loadu_ps(C + 3*ldc);

    for (int k = 0; k < k_size; k++) {
        // Load B[k][n:n+16] and convert to FP32
        __m256i b_bf16 = _mm256_loadu_si256((const __m256i*)(B + k * ldb));
        __m512 b_fp32 = bf16_to_fp32_avx512(b_bf16);

        // Broadcast A values and compute FMA
        if (m_size > 0) {
            float a0 = bf16_to_float(A[0 * lda + k]);
            __m512 a0_vec = _mm512_set1_ps(a0);
            c0 = _mm512_fmadd_ps(a0_vec, b_fp32, c0);
        }
        if (m_size > 1) {
            float a1 = bf16_to_float(A[1 * lda + k]);
            __m512 a1_vec = _mm512_set1_ps(a1);
            c1 = _mm512_fmadd_ps(a1_vec, b_fp32, c1);
        }
        if (m_size > 2) {
            float a2 = bf16_to_float(A[2 * lda + k]);
            __m512 a2_vec = _mm512_set1_ps(a2);
            c2 = _mm512_fmadd_ps(a2_vec, b_fp32, c2);
        }
        if (m_size > 3) {
            float a3 = bf16_to_float(A[3 * lda + k]);
            __m512 a3_vec = _mm512_set1_ps(a3);
            c3 = _mm512_fmadd_ps(a3_vec, b_fp32, c3);
        }
    }

    // Store results
    if (m_size > 0) _mm512_storeu_ps(C, c0);
    if (m_size > 1) _mm512_storeu_ps(C + ldc, c1);
    if (m_size > 2) _mm512_storeu_ps(C + 2*ldc, c2);
    if (m_size > 3) _mm512_storeu_ps(C + 3*ldc, c3);
}

// Optimized micro-kernel using AVX-512 BF16 dot product (if available)
// _mm512_dpbf16_ps computes: dst = src + a * b (where a,b are bf16 pairs)
static void micro_kernel_bf16_dp(const bf16 *A, const bf16 *B, float *C,
                                  int lda, int ldb, int ldc,
                                  int m_size, int k_size) {
    __m512 c0 = _mm512_loadu_ps(C);
    __m512 c1 = (m_size > 1) ? _mm512_loadu_ps(C + ldc) : _mm512_setzero_ps();
    __m512 c2 = (m_size > 2) ? _mm512_loadu_ps(C + 2*ldc) : _mm512_setzero_ps();
    __m512 c3 = (m_size > 3) ? _mm512_loadu_ps(C + 3*ldc) : _mm512_setzero_ps();

    // Process 2 K values at a time for BF16 dot product
    int k;
    for (k = 0; k + 1 < k_size; k += 2) {
        // Load B[k:k+2][n:n+16] - interleaved for dpbf16
        // B needs to be packed as pairs: (b[k][j], b[k+1][j]) for each j
        __m512i b_packed;
        {
            // Pack two rows of B into dpbf16 format
            __m256i b0 = _mm256_loadu_si256((const __m256i*)(B + k * ldb));
            __m256i b1 = _mm256_loadu_si256((const __m256i*)(B + (k+1) * ldb));
            // Interleave: (b0[0],b1[0]), (b0[1],b1[1]), ...
            __m256i lo = _mm256_unpacklo_epi16(b0, b1);
            __m256i hi = _mm256_unpackhi_epi16(b0, b1);
            b_packed = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
        }

        // For each row of A, pack (a[i][k], a[i][k+1]) and compute dpbf16
        if (m_size > 0) {
            bf16 a0k = A[0 * lda + k];
            bf16 a0k1 = A[0 * lda + k + 1];
            uint32_t a0_pair = ((uint32_t)a0k1 << 16) | a0k;
            __m512i a0_packed = _mm512_set1_epi32(a0_pair);
            c0 = _mm512_dpbf16_ps(c0, (__m512bh)a0_packed, (__m512bh)b_packed);
        }
        if (m_size > 1) {
            bf16 a1k = A[1 * lda + k];
            bf16 a1k1 = A[1 * lda + k + 1];
            uint32_t a1_pair = ((uint32_t)a1k1 << 16) | a1k;
            __m512i a1_packed = _mm512_set1_epi32(a1_pair);
            c1 = _mm512_dpbf16_ps(c1, (__m512bh)a1_packed, (__m512bh)b_packed);
        }
        if (m_size > 2) {
            bf16 a2k = A[2 * lda + k];
            bf16 a2k1 = A[2 * lda + k + 1];
            uint32_t a2_pair = ((uint32_t)a2k1 << 16) | a2k;
            __m512i a2_packed = _mm512_set1_epi32(a2_pair);
            c2 = _mm512_dpbf16_ps(c2, (__m512bh)a2_packed, (__m512bh)b_packed);
        }
        if (m_size > 3) {
            bf16 a3k = A[3 * lda + k];
            bf16 a3k1 = A[3 * lda + k + 1];
            uint32_t a3_pair = ((uint32_t)a3k1 << 16) | a3k;
            __m512i a3_packed = _mm512_set1_epi32(a3_pair);
            c3 = _mm512_dpbf16_ps(c3, (__m512bh)a3_packed, (__m512bh)b_packed);
        }
    }

    // Handle remaining K (odd case)
    if (k < k_size) {
        __m256i b_bf16 = _mm256_loadu_si256((const __m256i*)(B + k * ldb));
        __m512 b_fp32 = bf16_to_fp32_avx512(b_bf16);

        if (m_size > 0) {
            float a0 = bf16_to_float(A[0 * lda + k]);
            c0 = _mm512_fmadd_ps(_mm512_set1_ps(a0), b_fp32, c0);
        }
        if (m_size > 1) {
            float a1 = bf16_to_float(A[1 * lda + k]);
            c1 = _mm512_fmadd_ps(_mm512_set1_ps(a1), b_fp32, c1);
        }
        if (m_size > 2) {
            float a2 = bf16_to_float(A[2 * lda + k]);
            c2 = _mm512_fmadd_ps(_mm512_set1_ps(a2), b_fp32, c2);
        }
        if (m_size > 3) {
            float a3 = bf16_to_float(A[3 * lda + k]);
            c3 = _mm512_fmadd_ps(_mm512_set1_ps(a3), b_fp32, c3);
        }
    }

    // Store results
    if (m_size > 0) _mm512_storeu_ps(C, c0);
    if (m_size > 1) _mm512_storeu_ps(C + ldc, c1);
    if (m_size > 2) _mm512_storeu_ps(C + 2*ldc, c2);
    if (m_size > 3) _mm512_storeu_ps(C + 3*ldc, c3);
}

// Main SBGEMM function with cache blocking
void sbgemm_nolib(CBLAS_LAYOUT layout,
                  CBLAS_TRANSPOSE transA,
                  CBLAS_TRANSPOSE transB,
                  int M, int N, int K,
                  float alpha,
                  const bf16 *A, int lda,
                  const bf16 *B, int ldb,
                  float beta,
                  float *C, int ldc) {

    if (layout != CblasRowMajor) {
        fprintf(stderr, "Only CblasRowMajor is supported.\n");
        exit(1);
    }
    if (transA != CblasNoTrans || transB != CblasNoTrans) {
        fprintf(stderr, "Only NoTrans is supported.\n");
        exit(1);
    }

    // Apply beta to C
    if (beta == 0.0f) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] = 0.0f;
            }
        }
    } else if (beta != 1.0f) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] *= beta;
            }
        }
    }

    // Cache blocking loop
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        int k_len = (k0 + BLOCK_K <= K) ? BLOCK_K : (K - k0);

        for (int m0 = 0; m0 < M; m0 += BLOCK_M) {
            int m_len = (m0 + BLOCK_M <= M) ? BLOCK_M : (M - m0);

            for (int n0 = 0; n0 < N; n0 += BLOCK_N) {
                int n_len = (n0 + BLOCK_N <= N) ? BLOCK_N : (N - n0);

                // Process block with micro-kernels
                for (int m = 0; m < m_len; m += 4) {
                    int m_micro = (m + 4 <= m_len) ? 4 : (m_len - m);

                    for (int n = 0; n < n_len; n += VEC_WIDTH) {
                        int n_micro = (n + VEC_WIDTH <= n_len) ? VEC_WIDTH : (n_len - n);

                        if (n_micro == VEC_WIDTH) {
                            // Full vector width - use optimized kernel
                            micro_kernel_bf16_dp(
                                A + (m0 + m) * lda + k0,
                                B + k0 * ldb + (n0 + n),
                                C + (m0 + m) * ldc + (n0 + n),
                                lda, ldb, ldc,
                                m_micro, k_len);
                        } else {
                            // Partial vector - scalar fallback
                            for (int mi = 0; mi < m_micro; mi++) {
                                for (int ni = 0; ni < n_micro; ni++) {
                                    float sum = C[(m0 + m + mi) * ldc + (n0 + n + ni)];
                                    for (int ki = 0; ki < k_len; ki++) {
                                        float a_val = bf16_to_float(A[(m0 + m + mi) * lda + (k0 + ki)]);
                                        float b_val = bf16_to_float(B[(k0 + ki) * ldb + (n0 + n + ni)]);
                                        sum += a_val * b_val;
                                    }
                                    C[(m0 + m + mi) * ldc + (n0 + n + ni)] = sum;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Apply alpha
    if (alpha != 1.0f) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] *= alpha;
            }
        }
    }
}

// Timer utility
static double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Benchmark function
void benchmark_sbgemm(int M, int N, int K, int num_iterations) {
    printf("Matrix size: M=%d, N=%d, K=%d\n", M, N, K);

    bf16 *A = (bf16*)aligned_alloc(64, M * K * sizeof(bf16));
    bf16 *B = (bf16*)aligned_alloc(64, K * N * sizeof(bf16));
    float *C = (float*)aligned_alloc(64, M * N * sizeof(float));

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    srand(42);
    for (int i = 0; i < M * K; i++) A[i] = float_to_bf16((float)(rand() % 100) / 100.0f);
    for (int i = 0; i < K * N; i++) B[i] = float_to_bf16((float)(rand() % 100) / 100.0f);
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    // Warmup
    sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

    double total_time = 0.0;
    double min_time = 1e9;

    for (int iter = 0; iter < num_iterations; iter++) {
        for (int i = 0; i < M * N; i++) C[i] = 0.0f;

        double start = get_time();
        sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                     M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
        double end = get_time();

        double elapsed = end - start;
        total_time += elapsed;
        if (elapsed < min_time) min_time = elapsed;
    }

    double ops = 2.0 * M * N * K;
    double avg_time = total_time / num_iterations;
    double gflops_avg = (ops / avg_time) / 1e9;
    double gflops_peak = (ops / min_time) / 1e9;

    // Theoretical peak for AVX-512: ~150-200 GFLOPS per core for FP32 FMA
    // For BF16 with dpbf16: potentially higher
    double theoretical_peak = 1945.6;  // AMX theoretical peak for reference
    double efficiency = (gflops_peak / theoretical_peak) * 100.0;

    printf("Average time: %.4f sec\n", avg_time);
    printf("Min time: %.4f sec\n", min_time);
    printf("Average GFLOPS: %.2f\n", gflops_avg);
    printf("Peak GFLOPS: %.2f\n", gflops_peak);
    printf("Efficiency vs AMX theoretical (%.1f GFLOPS): %.2f%%\n", theoretical_peak, efficiency);

    free(A);
    free(B);
    free(C);
}

int main(int argc, char *argv[]) {
    int M = 4096, N = 4096, K = 4096;
    int num_iterations = 20;

    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    if (argc >= 5) {
        num_iterations = atoi(argv[4]);
    }

    printf("=== SBGEMM AVX-512 BF16 Optimization (v1.1.0) ===\n");
    printf("Block sizes: BLOCK_M=%d, BLOCK_N=%d, BLOCK_K=%d\n", BLOCK_M, BLOCK_N, BLOCK_K);
    printf("Iterations: %d\n\n", num_iterations);

    benchmark_sbgemm(M, N, K, num_iterations);

    return 0;
}
