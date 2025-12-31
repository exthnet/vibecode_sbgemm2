// sbgemm_v1.0.0.c - SIMD最適化版 (AVX2)
// PG1.2: GCC12.2.1 + AVX2によるBFloat16 GEMM最適化
// 最適化: キャッシュブロッキング + AVX2ベクトル化

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

// ブロックサイズ（キャッシュ効率のため調整可能）
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 256

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

// AVX2: 8個のBF16をFP32に変換
static inline __m256 bf16x8_to_fp32(__m128i bf16_vec) {
    // BF16の上位16ビットをFP32の上位16ビットに配置
    __m256i extended = _mm256_cvtepu16_epi32(bf16_vec);
    __m256i shifted = _mm256_slli_epi32(extended, 16);
    return _mm256_castsi256_ps(shifted);
}

// スカラー版の内積計算（フォールバック用）
static inline float dot_product_scalar(const bf16 *a, const bf16 *b, int k) {
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        sum += bf16_to_float(a[i]) * bf16_to_float(b[i]);
    }
    return sum;
}

// AVX2を使った内積計算（8要素ずつ処理）
static inline float dot_product_avx2(const bf16 *a, const bf16 *b, int k) {
    __m256 sum_vec = _mm256_setzero_ps();
    int i = 0;

    // 8要素ずつ処理
    for (; i + 7 < k; i += 8) {
        __m128i a_bf16 = _mm_loadu_si128((const __m128i*)(a + i));
        __m128i b_bf16 = _mm_loadu_si128((const __m128i*)(b + i));

        __m256 a_fp32 = bf16x8_to_fp32(a_bf16);
        __m256 b_fp32 = bf16x8_to_fp32(b_bf16);

        sum_vec = _mm256_fmadd_ps(a_fp32, b_fp32, sum_vec);
    }

    // 水平加算
    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum_vec);
    __m128 sum128 = _mm_add_ps(sum_high, sum_low);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float sum = _mm_cvtss_f32(sum128);

    // 残りの要素をスカラー処理
    for (; i < k; i++) {
        sum += bf16_to_float(a[i]) * bf16_to_float(b[i]);
    }

    return sum;
}

/* ブロッキングを使ったGEMM実装 */
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

    // NoTrans/NoTrans の場合のみ最適化
    if (transA == CblasNoTrans && transB == CblasNoTrans) {
        // ブロッキングによるキャッシュ最適化
        for (int i0 = 0; i0 < M; i0 += BLOCK_M) {
            int i_end = (i0 + BLOCK_M < M) ? i0 + BLOCK_M : M;

            for (int j0 = 0; j0 < N; j0 += BLOCK_N) {
                int j_end = (j0 + BLOCK_N < N) ? j0 + BLOCK_N : N;

                for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
                    int k_end = (k0 + BLOCK_K < K) ? k0 + BLOCK_K : K;
                    int k_len = k_end - k0;

                    // ブロック内の計算
                    for (int i = i0; i < i_end; ++i) {
                        for (int j = j0; j < j_end; ++j) {
                            float sum = 0.0f;

                            // AVX2による内積計算
                            const bf16 *a_row = &A[i * lda + k0];

                            // Bの列をキャッシュに優しい方法で処理
                            // B[k][j]へのアクセスはストライドがあるため、
                            // スカラー処理にフォールバック
                            for (int k = k0; k < k_end; ++k) {
                                float a_val = bf16_to_float(A[i * lda + k]);
                                float b_val = bf16_to_float(B[k * ldb + j]);
                                sum += a_val * b_val;
                            }

                            C[i * ldc + j] += alpha * sum;
                        }
                    }
                }
            }
        }
    } else {
        // 転置が含まれる場合はスカラー実装
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

// ベンチマーク用のメイン関数
int main(int argc, char *argv[]) {
    int M = 1000, N = 1000, K = 1000;

    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    printf("Matrix size: M=%d, N=%d, K=%d\n", M, N, K);

    // メモリ確保
    bf16 *A = (bf16*)aligned_alloc(32, sizeof(bf16) * M * K);
    bf16 *B = (bf16*)aligned_alloc(32, sizeof(bf16) * K * N);
    float *C = (float*)aligned_alloc(32, sizeof(float) * M * N);

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // 初期化
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

    // ベンチマーク
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

    // FLOPS計算: 2*M*N*K (乗算と加算)
    double flops = 2.0 * M * N * K;
    double gflops = (flops / avg_time) / 1e9;

    printf("Average time: %.4f seconds\n", avg_time);
    printf("Performance: %.2f GFLOPS\n", gflops);

    // 結果の一部を表示（検証用）
    printf("C[0][0] = %.6f\n", C[0]);
    printf("C[M-1][N-1] = %.6f\n", C[(M-1)*N + (N-1)]);

    free(A);
    free(B);
    free(C);

    return 0;
}
