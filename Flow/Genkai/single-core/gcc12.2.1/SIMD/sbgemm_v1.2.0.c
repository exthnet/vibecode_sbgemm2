// sbgemm_v1.2.0.c - SIMD最適化版 (AVX-512)
// PG1.2: GCC12.2.1 + AVX-512によるBFloat16 GEMM最適化
// 最適化: キャッシュブロッキング + B行列事前転置 + AVX-512ベクトル化

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

// ブロックサイズ（L2キャッシュ最適化）
#define BLOCK_M 96
#define BLOCK_N 96
#define BLOCK_K 512

/* --- BF16 <-> FP32 変換 --- */

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

#ifdef __AVX512F__
// AVX-512: 16個のBF16をFP32に変換
static inline __m512 bf16x16_to_fp32(__m256i bf16_vec) {
    __m512i extended = _mm512_cvtepu16_epi32(bf16_vec);
    __m512i shifted = _mm512_slli_epi32(extended, 16);
    return _mm512_castsi512_ps(shifted);
}

// AVX-512を使った内積計算（16要素ずつ処理）
static inline float dot_product_avx512(const bf16 *a, const bf16 *b, int k) {
    __m512 sum_vec = _mm512_setzero_ps();
    int i = 0;

    // 16要素ずつ処理
    for (; i + 15 < k; i += 16) {
        __m256i a_bf16 = _mm256_loadu_si256((const __m256i*)(a + i));
        __m256i b_bf16 = _mm256_loadu_si256((const __m256i*)(b + i));

        __m512 a_fp32 = bf16x16_to_fp32(a_bf16);
        __m512 b_fp32 = bf16x16_to_fp32(b_bf16);

        sum_vec = _mm512_fmadd_ps(a_fp32, b_fp32, sum_vec);
    }

    // 水平加算
    float sum = _mm512_reduce_add_ps(sum_vec);

    // 残りの要素をスカラー処理
    for (; i < k; i++) {
        sum += bf16_to_float(a[i]) * bf16_to_float(b[i]);
    }

    return sum;
}
#endif

// AVX2: 8個のBF16をFP32に変換（フォールバック用）
static inline __m256 bf16x8_to_fp32(__m128i bf16_vec) {
    __m256i extended = _mm256_cvtepu16_epi32(bf16_vec);
    __m256i shifted = _mm256_slli_epi32(extended, 16);
    return _mm256_castsi256_ps(shifted);
}

// AVX2を使った内積計算（フォールバック用）
static inline float dot_product_avx2(const bf16 *a, const bf16 *b, int k) {
    __m256 sum_vec = _mm256_setzero_ps();
    int i = 0;

    for (; i + 7 < k; i += 8) {
        __m128i a_bf16 = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i b_bf16 = _mm_loadu_si128((const __m128i*)(b + i));

        __m256 a_fp32 = bf16x8_to_fp32(a_bf16);
        __m256 b_fp32 = bf16x8_to_fp32(b_bf16);

        sum_vec = _mm256_fmadd_ps(a_fp32, b_fp32, sum_vec);
    }

    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum_vec);
    __m128 sum128 = _mm_add_ps(sum_high, sum_low);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float sum = _mm_cvtss_f32(sum128);

    for (; i < k; i++) {
        sum += bf16_to_float(a[i]) * bf16_to_float(b[i]);
    }

    return sum;
}

// B行列を転置
static void transpose_bf16(const bf16 *B, bf16 *B_T, int K, int N) {
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            B_T[n * K + k] = B[k * N + n];
        }
    }
}

/* GEMM実装（AVX-512優先、AVX2フォールバック）*/
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

    // beta * C の適用
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

    if (transA == CblasNoTrans && transB == CblasNoTrans) {
        // B行列を転置
        bf16 *B_T = (bf16*)aligned_alloc(64, sizeof(bf16) * K * N);
        if (!B_T) {
            fprintf(stderr, "Failed to allocate B_T\n");
            exit(1);
        }
        transpose_bf16(B, B_T, K, N);

        // ブロッキング + SIMD計算
        for (int i0 = 0; i0 < M; i0 += BLOCK_M) {
            int i_end = (i0 + BLOCK_M < M) ? i0 + BLOCK_M : M;

            for (int j0 = 0; j0 < N; j0 += BLOCK_N) {
                int j_end = (j0 + BLOCK_N < N) ? j0 + BLOCK_N : N;

                for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
                    int k_end = (k0 + BLOCK_K < K) ? k0 + BLOCK_K : K;
                    int k_len = k_end - k0;

                    for (int i = i0; i < i_end; ++i) {
                        const bf16 *a_row = &A[i * lda + k0];

                        for (int j = j0; j < j_end; ++j) {
                            const bf16 *b_col = &B_T[j * K + k0];

#ifdef __AVX512F__
                            float sum = dot_product_avx512(a_row, b_col, k_len);
#else
                            float sum = dot_product_avx2(a_row, b_col, k_len);
#endif
                            C[i * ldc + j] += alpha * sum;
                        }
                    }
                }
            }
        }

        free(B_T);
    } else {
        // 転置ケースはスカラー実装
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int p = 0; p < K; ++p) {
                    float a_ip, b_pj;
                    if (transA == CblasNoTrans) {
                        a_ip = bf16_to_float(A[i * lda + p]);
                    } else {
                        a_ip = bf16_to_float(A[p * lda + i]);
                    }
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
}

int main(int argc, char *argv[]) {
    int M = 1000, N = 1000, K = 1000;

    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    printf("Matrix size: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Block sizes: BLOCK_M=%d, BLOCK_N=%d, BLOCK_K=%d\n", BLOCK_M, BLOCK_N, BLOCK_K);
#ifdef __AVX512F__
    printf("SIMD: AVX-512 enabled\n");
#else
    printf("SIMD: AVX2 fallback\n");
#endif

    bf16 *A = (bf16*)aligned_alloc(64, sizeof(bf16) * M * K);
    bf16 *B = (bf16*)aligned_alloc(64, sizeof(bf16) * K * N);
    float *C = (float*)aligned_alloc(64, sizeof(float) * M * N);

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    srand(42);
    for (int i = 0; i < M * K; i++) {
        float val = (float)(rand() % 100) / 100.0f;
        A[i] = float_to_bf16_round(val);
    }
    for (int i = 0; i < K * N; i++) {
        float val = (float)(rand() % 100) / 100.0f;
        B[i] = float_to_bf16_round(val);
    }
    memset(C, 0, sizeof(float) * M * N);

    // ウォームアップ
    sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

    int num_runs = 5;
    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int run = 0; run < num_runs; run++) {
        sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                     M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    double avg_time = elapsed / num_runs;
    double flops = 2.0 * M * N * K;
    double gflops = (flops / avg_time) / 1e9;

    printf("Average time: %.4f seconds\n", avg_time);
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("C[0][0] = %.6f\n", C[0]);
    printf("C[M-1][N-1] = %.6f\n", C[(M-1)*N + (N-1)]);

    free(A);
    free(B);
    free(C);

    return 0;
}
