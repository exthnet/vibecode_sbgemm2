// sbgemm_v1.1.0.c - AVX-512 SIMD最適化版（改良版）
// PG1.6: Intel oneAPI 2024 + AVX-512
//
// v1.0.0からの改善点:
// - K方向4要素アンローリング
// - プリフェッチ命令追加
// - レジスタ再利用最適化
// - ブロックサイズ調整（L2キャッシュ最適化）
//
// コンパイル: icx -O3 -march=native -xCORE-AVX512 sbgemm_v1.1.0.c -o sbgemm

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

// ブロックサイズ (L2キャッシュ2MB想定で最適化)
#define BLOCK_M 96
#define BLOCK_N 96
#define BLOCK_K 512

// プリフェッチ距離
#define PREFETCH_DIST 8

/* AVX-512 SIMD版 sbgemm v1.1.0
 * C = alpha * A * B + beta * C
 * A: M x K (BF16), B: K x N (BF16), C: M x N (FP32)
 * Row-Major, NoTrans のみ対応
 */
void sbgemm_simd(CBLAS_LAYOUT layout,
                 CBLAS_TRANSPOSE transA,
                 CBLAS_TRANSPOSE transB,
                 int M, int N, int K,
                 float alpha,
                 const bf16 *A, int lda,
                 const bf16 *B, int ldb,
                 float beta,
                 float *C, int ldc)
{
    if (layout != CblasRowMajor || transA != CblasNoTrans || transB != CblasNoTrans) {
        fprintf(stderr, "Only CblasRowMajor + NoTrans supported in SIMD version.\n");
        exit(1);
    }

    // beta * C の事前処理（SIMD化）
    if (beta == 0.0f) {
        __m512 zero = _mm512_setzero_ps();
        for (int i = 0; i < M; ++i) {
            int j = 0;
            for (; j + 16 <= N; j += 16) {
                _mm512_storeu_ps(&C[i * ldc + j], zero);
            }
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

    // ブロッキングループ
    for (int i0 = 0; i0 < M; i0 += BLOCK_M) {
        int i_end = (i0 + BLOCK_M < M) ? i0 + BLOCK_M : M;

        for (int j0 = 0; j0 < N; j0 += BLOCK_N) {
            int j_end = (j0 + BLOCK_N < N) ? j0 + BLOCK_N : N;

            for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
                int k_end = (k0 + BLOCK_K < K) ? k0 + BLOCK_K : K;

                // 内側ループ: SIMD最適化 + アンローリング
                for (int i = i0; i < i_end; ++i) {
                    // プリフェッチ: 次の行のA
                    if (i + PREFETCH_DIST < i_end) {
                        _mm_prefetch((const char*)&A[(i + PREFETCH_DIST) * lda + k0], _MM_HINT_T0);
                    }

                    for (int j = j0; j < j_end; j += 16) {
                        int j_len = (j + 16 <= j_end) ? 16 : (j_end - j);
                        __mmask16 mask = (j_len == 16) ? 0xFFFF : ((1 << j_len) - 1);

                        // C[i,j:j+16]をロード
                        __m512 c_vec = _mm512_maskz_loadu_ps(mask, &C[i * ldc + j]);

                        // K方向4要素アンローリング
                        int k = k0;
                        for (; k + 4 <= k_end; k += 4) {
                            // プリフェッチ: 次のBブロック
                            if (k + PREFETCH_DIST < k_end) {
                                _mm_prefetch((const char*)&B[(k + PREFETCH_DIST) * ldb + j], _MM_HINT_T0);
                            }

                            // A[i,k:k+4]をロード
                            float a0 = bf16_to_float(A[i * lda + k]);
                            float a1 = bf16_to_float(A[i * lda + k + 1]);
                            float a2 = bf16_to_float(A[i * lda + k + 2]);
                            float a3 = bf16_to_float(A[i * lda + k + 3]);

                            // B[k:k+4, j:j+16]をロードしてFMA
                            float b_tmp0[16], b_tmp1[16], b_tmp2[16], b_tmp3[16];
                            for (int jj = 0; jj < j_len; ++jj) {
                                b_tmp0[jj] = bf16_to_float(B[k * ldb + j + jj]);
                                b_tmp1[jj] = bf16_to_float(B[(k + 1) * ldb + j + jj]);
                                b_tmp2[jj] = bf16_to_float(B[(k + 2) * ldb + j + jj]);
                                b_tmp3[jj] = bf16_to_float(B[(k + 3) * ldb + j + jj]);
                            }

                            __m512 b0 = _mm512_loadu_ps(b_tmp0);
                            __m512 b1 = _mm512_loadu_ps(b_tmp1);
                            __m512 b2 = _mm512_loadu_ps(b_tmp2);
                            __m512 b3 = _mm512_loadu_ps(b_tmp3);

                            __m512 a0_vec = _mm512_set1_ps(a0);
                            __m512 a1_vec = _mm512_set1_ps(a1);
                            __m512 a2_vec = _mm512_set1_ps(a2);
                            __m512 a3_vec = _mm512_set1_ps(a3);

                            c_vec = _mm512_fmadd_ps(a0_vec, b0, c_vec);
                            c_vec = _mm512_fmadd_ps(a1_vec, b1, c_vec);
                            c_vec = _mm512_fmadd_ps(a2_vec, b2, c_vec);
                            c_vec = _mm512_fmadd_ps(a3_vec, b3, c_vec);
                        }

                        // 端数処理
                        for (; k < k_end; ++k) {
                            float a_val = bf16_to_float(A[i * lda + k]);
                            float b_tmp[16] = {0};
                            for (int jj = 0; jj < j_len; ++jj) {
                                b_tmp[jj] = bf16_to_float(B[k * ldb + j + jj]);
                            }
                            __m512 b_vec = _mm512_loadu_ps(b_tmp);
                            __m512 a_vec = _mm512_set1_ps(a_val);
                            c_vec = _mm512_fmadd_ps(a_vec, b_vec, c_vec);
                        }

                        // 結果をストア
                        _mm512_mask_storeu_ps(&C[i * ldc + j], mask, c_vec);
                    }
                }
            }
        }
    }

    // alpha のスケーリング（SIMD化）
    if (alpha != 1.0f) {
        __m512 alpha_vec = _mm512_set1_ps(alpha);
        for (int i = 0; i < M; ++i) {
            int j = 0;
            for (; j + 16 <= N; j += 16) {
                __m512 c_vec = _mm512_loadu_ps(&C[i * ldc + j]);
                c_vec = _mm512_mul_ps(c_vec, alpha_vec);
                _mm512_storeu_ps(&C[i * ldc + j], c_vec);
            }
            for (; j < N; ++j) {
                C[i * ldc + j] *= alpha;
            }
        }
    }
}

/* ベンチマーク用: 時間測定 */
double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* メイン: ベンチマーク実行 */
int main(int argc, char *argv[]) {
    int M = 1024, N = 1024, K = 1024;
    int repeat = 10;

    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    if (argc >= 5) {
        repeat = atoi(argv[4]);
    }

    printf("SBGEMM SIMD Benchmark v1.1.0 (AVX-512 + Unrolling)\n");
    printf("Matrix size: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Block size: BLOCK_M=%d, BLOCK_N=%d, BLOCK_K=%d\n", BLOCK_M, BLOCK_N, BLOCK_K);
    printf("Repeat: %d times\n", repeat);

    // メモリ確保 (64バイトアラインメント)
    bf16 *A = (bf16*)aligned_alloc(64, sizeof(bf16) * M * K);
    bf16 *B = (bf16*)aligned_alloc(64, sizeof(bf16) * K * N);
    float *C = (float*)aligned_alloc(64, sizeof(float) * M * N);

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // 初期化 (乱数)
    srand(42);
    for (int i = 0; i < M * K; ++i) {
        A[i] = float_to_bf16_round((float)(rand() % 100) / 100.0f);
    }
    for (int i = 0; i < K * N; ++i) {
        B[i] = float_to_bf16_round((float)(rand() % 100) / 100.0f);
    }
    for (int i = 0; i < M * N; ++i) {
        C[i] = 0.0f;
    }

    // ウォームアップ
    sbgemm_simd(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

    // ベンチマーク
    double start = get_time_sec();
    for (int r = 0; r < repeat; ++r) {
        sbgemm_simd(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    }
    double end = get_time_sec();

    double elapsed = (end - start) / repeat;
    double flops = 2.0 * M * N * K;  // 乗算 + 加算
    double gflops = flops / elapsed / 1e9;

    printf("\n=== Results ===\n");
    printf("Average time: %.6f sec\n", elapsed);
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("C[0,0] = %.6f (sanity check)\n", C[0]);

    free(A);
    free(B);
    free(C);

    return 0;
}
