// sbgemm_v1.0.0.c - メモリアクセス最適化版
// PG1.3: BUFFER_A/BUFFER_Bパッキング + キャッシュブロッキング
// reference.pdfのOpenBLAS手法を参考に実装

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
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

// キャッシュブロッキングパラメータ (reference.pdf推奨値)
// L2キャッシュ: 2MB、最適組み合わせ: n=480, k=1536
#define BLOCK_K 1536
#define BLOCK_N 480
#define BLOCK_M 96

/* --- BF16 <-> FP32 変換 --- */
static inline bf16 float_to_bf16_round(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    u += 0x00008000u;
    return (bf16)(u >> 16);
}

static inline bf16 float_to_bf16_trunc(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return (bf16)(u >> 16);
}

static inline float bf16_to_float(bf16 x) {
    uint32_t u = ((uint32_t)x) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/* --- BUFFER_Aへのパッキング ---
 * 行列Aの部分ブロックを連続メモリ配置にパック
 * 行優先で連続アクセス可能な形式に変換
 */
static void pack_buffer_a(const bf16 *A, int lda, int m_start, int m_size,
                          int k_start, int k_size, float *buffer_a,
                          CBLAS_TRANSPOSE transA) {
    for (int i = 0; i < m_size; i++) {
        for (int k = 0; k < k_size; k++) {
            float val;
            if (transA == CblasNoTrans) {
                val = bf16_to_float(A[(m_start + i) * lda + (k_start + k)]);
            } else {
                val = bf16_to_float(A[(k_start + k) * lda + (m_start + i)]);
            }
            buffer_a[i * k_size + k] = val;
        }
    }
}

/* --- BUFFER_Bへのパッキング ---
 * 行列Bの部分ブロックを転置してパック
 * 列方向アクセスを行方向アクセスに変換
 */
static void pack_buffer_b(const bf16 *B, int ldb, int k_start, int k_size,
                          int n_start, int n_size, float *buffer_b,
                          CBLAS_TRANSPOSE transB) {
    // Bを転置してパック: buffer_b[n][k]の形式で格納
    // これによりカーネル内でのBアクセスが連続になる
    for (int n = 0; n < n_size; n++) {
        for (int k = 0; k < k_size; k++) {
            float val;
            if (transB == CblasNoTrans) {
                val = bf16_to_float(B[(k_start + k) * ldb + (n_start + n)]);
            } else {
                val = bf16_to_float(B[(n_start + n) * ldb + (k_start + k)]);
            }
            buffer_b[n * k_size + k] = val;
        }
    }
}

/* --- マイクロカーネル ---
 * パックされたBUFFER_AとBUFFER_Bを使用して行列積を計算
 * メモリアクセスが連続になるよう最適化
 */
static void micro_kernel(const float *buffer_a, const float *buffer_b,
                         float *C, int ldc, int m_size, int n_size, int k_size,
                         float alpha, int c_row_start, int c_col_start) {
    // ループ順序: i -> j -> k (行列Cへの書き込みを最適化)
    for (int i = 0; i < m_size; i++) {
        for (int j = 0; j < n_size; j++) {
            float sum = 0.0f;
            // k方向は両方のバッファで連続アクセス
            const float *a_row = &buffer_a[i * k_size];
            const float *b_col = &buffer_b[j * k_size];

            // ループアンローリング（4要素ずつ）
            int k = 0;
            for (; k + 3 < k_size; k += 4) {
                sum += a_row[k]     * b_col[k];
                sum += a_row[k + 1] * b_col[k + 1];
                sum += a_row[k + 2] * b_col[k + 2];
                sum += a_row[k + 3] * b_col[k + 3];
            }
            // 残りの要素
            for (; k < k_size; k++) {
                sum += a_row[k] * b_col[k];
            }

            C[(c_row_start + i) * ldc + (c_col_start + j)] += alpha * sum;
        }
    }
}

/* --- メイン関数: sbgemm_nolib ---
 * キャッシュブロッキング + パッキングによるメモリアクセス最適化版
 */
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

    // beta*Cの事前処理
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

    // バッファ確保
    float *buffer_a = (float *)malloc(sizeof(float) * BLOCK_M * BLOCK_K);
    float *buffer_b = (float *)malloc(sizeof(float) * BLOCK_N * BLOCK_K);

    if (!buffer_a || !buffer_b) {
        fprintf(stderr, "Buffer allocation failed\n");
        exit(1);
    }

    // 3重ループブロッキング: K -> N -> M の順
    // (reference.pdfの戦略: BUFFER_Bの再利用を最大化)
    for (int kk = 0; kk < K; kk += BLOCK_K) {
        int k_size = (kk + BLOCK_K <= K) ? BLOCK_K : (K - kk);

        for (int nn = 0; nn < N; nn += BLOCK_N) {
            int n_size = (nn + BLOCK_N <= N) ? BLOCK_N : (N - nn);

            // BUFFER_Bをパック（K方向ブロック × N方向ブロック）
            pack_buffer_b(B, ldb, kk, k_size, nn, n_size, buffer_b, transB);

            for (int mm = 0; mm < M; mm += BLOCK_M) {
                int m_size = (mm + BLOCK_M <= M) ? BLOCK_M : (M - mm);

                // BUFFER_Aをパック（M方向ブロック × K方向ブロック）
                pack_buffer_a(A, lda, mm, m_size, kk, k_size, buffer_a, transA);

                // マイクロカーネル実行
                micro_kernel(buffer_a, buffer_b, C, ldc,
                            m_size, n_size, k_size, alpha, mm, nn);
            }
        }
    }

    free(buffer_a);
    free(buffer_b);
}

/* --- ベンチマーク用ユーティリティ --- */
static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void init_random_bf16(bf16 *arr, int size) {
    for (int i = 0; i < size; i++) {
        float val = (float)(rand() % 100) / 100.0f;
        arr[i] = float_to_bf16_round(val);
    }
}

int main(int argc, char *argv[]) {
    // デフォルトサイズ
    int sizes[] = {1024, 2048, 4096};
    int num_sizes = 3;

    // コマンドライン引数でサイズ指定可能
    if (argc > 1) {
        num_sizes = 1;
        sizes[0] = atoi(argv[1]);
    }

    printf("=== sbgemm Memory Access Optimization v1.0.0 ===\n");
    printf("Block sizes: M=%d, N=%d, K=%d\n", BLOCK_M, BLOCK_N, BLOCK_K);
    printf("Optimization: BUFFER_A/B packing + cache blocking\n\n");

    srand(42);

    for (int s = 0; s < num_sizes; s++) {
        int size = sizes[s];
        int M = size, N = size, K = size;

        // メモリ確保
        bf16 *A = (bf16 *)malloc(sizeof(bf16) * M * K);
        bf16 *B = (bf16 *)malloc(sizeof(bf16) * K * N);
        float *C = (float *)calloc(M * N, sizeof(float));

        if (!A || !B || !C) {
            fprintf(stderr, "Memory allocation failed for size %d\n", size);
            continue;
        }

        // 初期化
        init_random_bf16(A, M * K);
        init_random_bf16(B, K * N);

        // ウォームアップ
        sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                     M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

        // 計測（3回の平均）
        int num_trials = 3;
        double total_time = 0.0;

        for (int t = 0; t < num_trials; t++) {
            memset(C, 0, sizeof(float) * M * N);

            double start = get_time_sec();
            sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                         M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
            double end = get_time_sec();

            total_time += (end - start);
        }

        double avg_time = total_time / num_trials;
        double flops = 2.0 * M * N * K;
        double gflops = flops / avg_time / 1e9;

        printf("Size: %5d x %5d x %5d | Time: %.4f sec | %.2f GFLOPS\n",
               M, N, K, avg_time, gflops);

        free(A);
        free(B);
        free(C);
    }

    // 検証テスト（小サイズ）
    printf("\n=== Verification Test ===\n");
    const int M = 2, K = 3, N = 2;
    float A_f32[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    float B_f32[6] = {7.f, 8.f, 9.f, 10.f, 11.f, 12.f};
    bf16 *A_bf16 = (bf16 *)malloc(sizeof(bf16) * M * K);
    bf16 *B_bf16 = (bf16 *)malloc(sizeof(bf16) * K * N);
    float *C_f32 = (float *)calloc(M * N, sizeof(float));

    for (int i = 0; i < M * K; i++) A_bf16[i] = float_to_bf16_round(A_f32[i]);
    for (int i = 0; i < K * N; i++) B_bf16[i] = float_to_bf16_round(B_f32[i]);

    sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 M, N, K, 1.0f, A_bf16, K, B_bf16, N, 0.0f, C_f32, N);

    printf("Result (expected: [[58, 64], [139, 154]]):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%8.2f ", C_f32[i * N + j]);
        }
        printf("\n");
    }

    free(A_bf16);
    free(B_bf16);
    free(C_f32);

    return 0;
}
