// sbgemm_v1.2.0.c - AVX-512 SIMD最適化版（レジスタブロッキング）
// PG1.6: Intel oneAPI 2024 + AVX-512
//
// v1.1.0からの改善点:
// - レジスタブロッキング（MR=6, NR=16）
// - Cのマイクロタイルをレジスタに保持
// - Bの再利用率向上
//
// コンパイル: icx -O3 -march=native -xCORE-AVX512 sbgemm_v1.2.0.c -o sbgemm

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

// マクロブロックサイズ
#define MC 96    // Aのブロック行数
#define NC 256   // Bのブロック列数
#define KC 512   // K方向ブロック

// マイクロカーネルサイズ
#define MR 6     // Cのレジスタ行数
#define NR 16    // Cのレジスタ列数（512bit = 16 floats）

/* マイクロカーネル: 6x16 のCブロックを計算
 * A: MR x KC パネル (パック済み想定)
 * B: KC x NR パネル (パック済み想定)
 * C: MR x NR ブロック
 */
static inline void micro_kernel_6x16(
    int kc,
    const bf16 *A, int lda,
    const bf16 *B, int ldb,
    float *C, int ldc,
    int m_remain, int n_remain)
{
    // 6行のCレジスタ
    __m512 c0 = _mm512_setzero_ps();
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    __m512 c3 = _mm512_setzero_ps();
    __m512 c4 = _mm512_setzero_ps();
    __m512 c5 = _mm512_setzero_ps();

    // 既存のCをロード
    __mmask16 n_mask = (n_remain >= 16) ? 0xFFFF : ((1 << n_remain) - 1);
    if (m_remain > 0) c0 = _mm512_maskz_loadu_ps(n_mask, &C[0 * ldc]);
    if (m_remain > 1) c1 = _mm512_maskz_loadu_ps(n_mask, &C[1 * ldc]);
    if (m_remain > 2) c2 = _mm512_maskz_loadu_ps(n_mask, &C[2 * ldc]);
    if (m_remain > 3) c3 = _mm512_maskz_loadu_ps(n_mask, &C[3 * ldc]);
    if (m_remain > 4) c4 = _mm512_maskz_loadu_ps(n_mask, &C[4 * ldc]);
    if (m_remain > 5) c5 = _mm512_maskz_loadu_ps(n_mask, &C[5 * ldc]);

    // K方向ループ
    for (int k = 0; k < kc; ++k) {
        // B[k, 0:16] をロード (1回だけ)
        float b_tmp[16] = {0};
        int n_load = (n_remain >= 16) ? 16 : n_remain;
        for (int j = 0; j < n_load; ++j) {
            b_tmp[j] = bf16_to_float(B[k * ldb + j]);
        }
        __m512 b_vec = _mm512_loadu_ps(b_tmp);

        // A[0:6, k] をロードしてFMA
        if (m_remain > 0) {
            float a0 = bf16_to_float(A[0 * lda + k]);
            c0 = _mm512_fmadd_ps(_mm512_set1_ps(a0), b_vec, c0);
        }
        if (m_remain > 1) {
            float a1 = bf16_to_float(A[1 * lda + k]);
            c1 = _mm512_fmadd_ps(_mm512_set1_ps(a1), b_vec, c1);
        }
        if (m_remain > 2) {
            float a2 = bf16_to_float(A[2 * lda + k]);
            c2 = _mm512_fmadd_ps(_mm512_set1_ps(a2), b_vec, c2);
        }
        if (m_remain > 3) {
            float a3 = bf16_to_float(A[3 * lda + k]);
            c3 = _mm512_fmadd_ps(_mm512_set1_ps(a3), b_vec, c3);
        }
        if (m_remain > 4) {
            float a4 = bf16_to_float(A[4 * lda + k]);
            c4 = _mm512_fmadd_ps(_mm512_set1_ps(a4), b_vec, c4);
        }
        if (m_remain > 5) {
            float a5 = bf16_to_float(A[5 * lda + k]);
            c5 = _mm512_fmadd_ps(_mm512_set1_ps(a5), b_vec, c5);
        }
    }

    // Cをストア
    if (m_remain > 0) _mm512_mask_storeu_ps(&C[0 * ldc], n_mask, c0);
    if (m_remain > 1) _mm512_mask_storeu_ps(&C[1 * ldc], n_mask, c1);
    if (m_remain > 2) _mm512_mask_storeu_ps(&C[2 * ldc], n_mask, c2);
    if (m_remain > 3) _mm512_mask_storeu_ps(&C[3 * ldc], n_mask, c3);
    if (m_remain > 4) _mm512_mask_storeu_ps(&C[4 * ldc], n_mask, c4);
    if (m_remain > 5) _mm512_mask_storeu_ps(&C[5 * ldc], n_mask, c5);
}

/* AVX-512 SIMD版 sbgemm v1.2.0
 * C = alpha * A * B + beta * C
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
        fprintf(stderr, "Only CblasRowMajor + NoTrans supported.\n");
        exit(1);
    }

    // beta * C の事前処理
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
                _mm512_storeu_ps(&C[i * ldc + j], _mm512_mul_ps(c_vec, beta_vec));
            }
            for (; j < N; ++j) {
                C[i * ldc + j] *= beta;
            }
        }
    }

    // 3重ブロッキング
    for (int j0 = 0; j0 < N; j0 += NC) {
        int nc = (j0 + NC <= N) ? NC : (N - j0);

        for (int k0 = 0; k0 < K; k0 += KC) {
            int kc = (k0 + KC <= K) ? KC : (K - k0);

            for (int i0 = 0; i0 < M; i0 += MC) {
                int mc = (i0 + MC <= M) ? MC : (M - i0);

                // マイクロカーネルループ
                for (int i = 0; i < mc; i += MR) {
                    int mr = (i + MR <= mc) ? MR : (mc - i);

                    for (int j = 0; j < nc; j += NR) {
                        int nr = (j + NR <= nc) ? NR : (nc - j);

                        // マイクロカーネル実行
                        micro_kernel_6x16(
                            kc,
                            &A[(i0 + i) * lda + k0], lda,
                            &B[k0 * ldb + (j0 + j)], ldb,
                            &C[(i0 + i) * ldc + (j0 + j)], ldc,
                            mr, nr
                        );
                    }
                }
            }
        }
    }

    // alpha のスケーリング
    if (alpha != 1.0f) {
        __m512 alpha_vec = _mm512_set1_ps(alpha);
        for (int i = 0; i < M; ++i) {
            int j = 0;
            for (; j + 16 <= N; j += 16) {
                __m512 c_vec = _mm512_loadu_ps(&C[i * ldc + j]);
                _mm512_storeu_ps(&C[i * ldc + j], _mm512_mul_ps(c_vec, alpha_vec));
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

    printf("SBGEMM SIMD Benchmark v1.2.0 (Register Blocking MR=%d, NR=%d)\n", MR, NR);
    printf("Matrix size: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Block size: MC=%d, NC=%d, KC=%d\n", MC, NC, KC);
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
    double flops = 2.0 * M * N * K;
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
