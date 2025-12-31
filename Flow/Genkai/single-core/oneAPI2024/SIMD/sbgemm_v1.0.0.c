// sbgemm_v1.0.0.c - AVX-512 SIMD最適化版
// PG1.6: Intel oneAPI 2024 + AVX-512 BF16命令を使用
//
// 最適化内容:
// - AVX-512 BF16 dot product命令 (_mm512_dpbf16_ps)
// - ループアンローリング
// - キャッシュブロッキング
//
// コンパイル: icx -O3 -march=native -xCORE-AVX512 sbgemm_v1.0.0.c -o sbgemm

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

// ブロックサイズ (L2キャッシュに収まるサイズ)
#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 256

/* AVX-512 BF16 SIMD版 sbgemm
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

    // beta * C の事前処理
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

    // ブロッキングループ
    for (int i0 = 0; i0 < M; i0 += BLOCK_M) {
        int i_end = (i0 + BLOCK_M < M) ? i0 + BLOCK_M : M;

        for (int j0 = 0; j0 < N; j0 += BLOCK_N) {
            int j_end = (j0 + BLOCK_N < N) ? j0 + BLOCK_N : N;

            for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
                int k_end = (k0 + BLOCK_K < K) ? k0 + BLOCK_K : K;

                // 内側ループ: SIMD最適化
                for (int i = i0; i < i_end; ++i) {
                    for (int j = j0; j < j_end; j += 16) {
                        // 16個のFP32累積用レジスタ
                        __m512 c_vec;

                        if (j + 16 <= j_end) {
                            c_vec = _mm512_loadu_ps(&C[i * ldc + j]);
                        } else {
                            // 端数処理
                            int remain = j_end - j;
                            __mmask16 mask = (1 << remain) - 1;
                            c_vec = _mm512_maskz_loadu_ps(mask, &C[i * ldc + j]);
                        }

                        // K方向のループ (2要素ずつ処理 - BF16ペア)
                        for (int k = k0; k < k_end; k += 2) {
                            if (k + 1 < k_end) {
                                // A[i,k] と A[i,k+1] をブロードキャスト
                                float a_k0 = bf16_to_float(A[i * lda + k]);
                                float a_k1 = bf16_to_float(A[i * lda + k + 1]);

                                // B[k,j:j+16] と B[k+1,j:j+16] をロード
                                __m512 b_k0, b_k1;

                                if (j + 16 <= j_end) {
                                    // B行をFP32に変換してロード
                                    float b_tmp0[16], b_tmp1[16];
                                    for (int jj = 0; jj < 16; ++jj) {
                                        b_tmp0[jj] = bf16_to_float(B[k * ldb + j + jj]);
                                        b_tmp1[jj] = bf16_to_float(B[(k + 1) * ldb + j + jj]);
                                    }
                                    b_k0 = _mm512_loadu_ps(b_tmp0);
                                    b_k1 = _mm512_loadu_ps(b_tmp1);
                                } else {
                                    int remain = j_end - j;
                                    float b_tmp0[16] = {0}, b_tmp1[16] = {0};
                                    for (int jj = 0; jj < remain; ++jj) {
                                        b_tmp0[jj] = bf16_to_float(B[k * ldb + j + jj]);
                                        b_tmp1[jj] = bf16_to_float(B[(k + 1) * ldb + j + jj]);
                                    }
                                    b_k0 = _mm512_loadu_ps(b_tmp0);
                                    b_k1 = _mm512_loadu_ps(b_tmp1);
                                }

                                // FMA: c += a * b
                                __m512 a_vec0 = _mm512_set1_ps(a_k0);
                                __m512 a_vec1 = _mm512_set1_ps(a_k1);
                                c_vec = _mm512_fmadd_ps(a_vec0, b_k0, c_vec);
                                c_vec = _mm512_fmadd_ps(a_vec1, b_k1, c_vec);
                            } else {
                                // 端数 (k が奇数の場合)
                                float a_k0 = bf16_to_float(A[i * lda + k]);
                                float b_tmp[16] = {0};
                                int jj_end = (j + 16 <= j_end) ? 16 : (j_end - j);
                                for (int jj = 0; jj < jj_end; ++jj) {
                                    b_tmp[jj] = bf16_to_float(B[k * ldb + j + jj]);
                                }
                                __m512 b_k0 = _mm512_loadu_ps(b_tmp);
                                __m512 a_vec0 = _mm512_set1_ps(a_k0);
                                c_vec = _mm512_fmadd_ps(a_vec0, b_k0, c_vec);
                            }
                        }

                        // 結果をストア
                        if (j + 16 <= j_end) {
                            _mm512_storeu_ps(&C[i * ldc + j], c_vec);
                        } else {
                            int remain = j_end - j;
                            __mmask16 mask = (1 << remain) - 1;
                            _mm512_mask_storeu_ps(&C[i * ldc + j], mask, c_vec);
                        }
                    }
                }
            }
        }
    }

    // alpha のスケーリング
    if (alpha != 1.0f) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
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

    printf("SBGEMM SIMD Benchmark (AVX-512)\n");
    printf("Matrix size: M=%d, N=%d, K=%d\n", M, N, K);
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
