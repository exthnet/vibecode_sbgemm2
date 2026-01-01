// sbgemm_amx_v1.0.0.c - Intel AMX最適化版BFloat16 GEMM
// PG1.1: SIMD最適化担当
// 参考: reference.pdf - Tiling_B手法（n=480, k=1536で65%効率達成）
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
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

// AMXタイル設定用構造体
#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILEDATA      18

#define TILE_M 16
#define TILE_N 16
#define TILE_K 32

// キャッシュブロッキングサイズ（reference.pdfより最適値）
#define BLOCK_K 1536
#define BLOCK_N 480

// タイル設定構造体
typedef struct __tile_config {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} __tilecfg;

// AMX初期化
static int init_amx(void) {
    unsigned long bitmask = 0;
    if (syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask) != 0) {
        return -1;
    }
    if (bitmask & (1UL << XFEATURE_XTILEDATA)) {
        return 0;
    }
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0) {
        return -1;
    }
    return 0;
}

// タイル設定
static void configure_tiles(void) {
    __tilecfg config = {0};
    config.palette_id = 1;
    config.start_row = 0;

    // タイル0-5: 結果C用 (16x16 float32)
    for (int i = 0; i < 6; i++) {
        config.rows[i] = TILE_M;
        config.colsb[i] = TILE_N * sizeof(float);
    }
    // タイル6: A用 (16x32 bf16)
    config.rows[6] = TILE_M;
    config.colsb[6] = TILE_K * sizeof(bf16);
    // タイル7: B用 (16x32 bf16)
    config.rows[7] = TILE_K / 2;
    config.colsb[7] = TILE_N * 2 * sizeof(bf16);

    _tile_loadconfig(&config);
}

// BF16 <-> FP32 変換
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

// スカラーフォールバック実装
static void sbgemm_scalar(int M, int N, int K, float alpha,
                          const bf16 *A, int lda,
                          const bf16 *B, int ldb,
                          float beta, float *C, int ldc) {
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

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += bf16_to_float(A[i * lda + k]) * bf16_to_float(B[k * ldb + j]);
            }
            C[i * ldc + j] += alpha * sum;
        }
    }
}

// 行列Aをパッキング（16x32ブロック単位）
static void pack_A(const bf16 *A, bf16 *buffer, int M, int K, int lda, int k_start, int k_block) {
    for (int i = 0; i < M; i += TILE_M) {
        int actual_m = (i + TILE_M <= M) ? TILE_M : (M - i);
        for (int k = 0; k < k_block; k += TILE_K) {
            int actual_k = (k + TILE_K <= k_block) ? TILE_K : (k_block - k);
            bf16 *dest = buffer + (i / TILE_M) * ((k_block + TILE_K - 1) / TILE_K) * TILE_M * TILE_K
                       + (k / TILE_K) * TILE_M * TILE_K;
            for (int ii = 0; ii < TILE_M; ii++) {
                for (int kk = 0; kk < TILE_K; kk++) {
                    if (ii < actual_m && kk < actual_k) {
                        dest[ii * TILE_K + kk] = A[(i + ii) * lda + (k_start + k + kk)];
                    } else {
                        dest[ii * TILE_K + kk] = 0;
                    }
                }
            }
        }
    }
}

// 行列Bをパッキング・転置（Tiling_B用インターリーブ配置）
static void pack_B_transpose(const bf16 *B, bf16 *buffer, int K, int N, int ldb,
                             int k_start, int k_block, int n_start, int n_block) {
    for (int j = 0; j < n_block; j += TILE_N) {
        int actual_n = (j + TILE_N <= n_block) ? TILE_N : (n_block - j);
        for (int k = 0; k < k_block; k += TILE_K) {
            int actual_k = (k + TILE_K <= k_block) ? TILE_K : (k_block - k);
            bf16 *dest = buffer + (j / TILE_N) * ((k_block + TILE_K - 1) / TILE_K) * (TILE_K / 2) * (TILE_N * 2)
                       + (k / TILE_K) * (TILE_K / 2) * (TILE_N * 2);
            for (int kk = 0; kk < TILE_K / 2; kk++) {
                for (int jj = 0; jj < TILE_N; jj++) {
                    int k_idx0 = k_start + k + kk * 2;
                    int k_idx1 = k_start + k + kk * 2 + 1;
                    int n_idx = n_start + j + jj;
                    if (kk * 2 < actual_k && jj < actual_n) {
                        dest[kk * TILE_N * 2 + jj * 2] = B[k_idx0 * ldb + n_idx];
                        dest[kk * TILE_N * 2 + jj * 2 + 1] = (kk * 2 + 1 < actual_k) ? B[k_idx1 * ldb + n_idx] : 0;
                    } else {
                        dest[kk * TILE_N * 2 + jj * 2] = 0;
                        dest[kk * TILE_N * 2 + jj * 2 + 1] = 0;
                    }
                }
            }
        }
    }
}

// AMXカーネル（Tiling_B: 2x3タイル構成）
static void amx_kernel_tiling_b(const bf16 *A_pack, const bf16 *B_pack, float *C,
                                 int M, int n_block, int k_block, int ldc,
                                 int n_start) {
    int k_tiles = (k_block + TILE_K - 1) / TILE_K;
    int n_tiles = (n_block + TILE_N - 1) / TILE_N;

    for (int i = 0; i < M; i += TILE_M * 2) {
        int m_end = (i + TILE_M * 2 <= M) ? 2 : 1;

        for (int j = 0; j < n_block; j += TILE_N * 3) {
            int n_end = 3;
            if (j + TILE_N * 3 > n_block) {
                n_end = (n_block - j + TILE_N - 1) / TILE_N;
            }

            // ローカルCバッファ
            float C_local[2][3][TILE_M * TILE_N] __attribute__((aligned(64)));

            // Cをロード
            for (int ti = 0; ti < m_end; ti++) {
                for (int tj = 0; tj < n_end; tj++) {
                    int row_base = i + ti * TILE_M;
                    int col_base = n_start + j + tj * TILE_N;
                    for (int ii = 0; ii < TILE_M; ii++) {
                        for (int jj = 0; jj < TILE_N; jj++) {
                            int row = row_base + ii;
                            int col = col_base + jj;
                            if (row < M && col < n_start + n_block) {
                                C_local[ti][tj][ii * TILE_N + jj] = C[row * ldc + col];
                            } else {
                                C_local[ti][tj][ii * TILE_N + jj] = 0.0f;
                            }
                        }
                    }
                }
            }

            // K方向ループでAMX演算
            for (int kt = 0; kt < k_tiles; kt++) {
                // Aの最初の行をロード
                const bf16 *A_ptr0 = A_pack + (i / TILE_M) * k_tiles * TILE_M * TILE_K + kt * TILE_M * TILE_K;
                _tile_loadd(6, A_ptr0, TILE_K * sizeof(bf16));

                // Bの各列と演算
                for (int tj = 0; tj < n_end; tj++) {
                    const bf16 *B_ptr = B_pack + ((j / TILE_N) + tj) * k_tiles * (TILE_K / 2) * (TILE_N * 2)
                                      + kt * (TILE_K / 2) * (TILE_N * 2);
                    _tile_loadd(7, B_ptr, TILE_N * 2 * sizeof(bf16));
                    _tile_loadd(tj, C_local[0][tj], TILE_N * sizeof(float));
                    _tile_dpbf16ps(tj, 6, 7);
                    _tile_stored(tj, C_local[0][tj], TILE_N * sizeof(float));
                }

                // 2行目（存在する場合）
                if (m_end > 1) {
                    const bf16 *A_ptr1 = A_pack + ((i / TILE_M) + 1) * k_tiles * TILE_M * TILE_K + kt * TILE_M * TILE_K;
                    _tile_loadd(6, A_ptr1, TILE_K * sizeof(bf16));

                    for (int tj = 0; tj < n_end; tj++) {
                        const bf16 *B_ptr = B_pack + ((j / TILE_N) + tj) * k_tiles * (TILE_K / 2) * (TILE_N * 2)
                                          + kt * (TILE_K / 2) * (TILE_N * 2);
                        _tile_loadd(7, B_ptr, TILE_N * 2 * sizeof(bf16));
                        _tile_loadd(tj + 3, C_local[1][tj], TILE_N * sizeof(float));
                        _tile_dpbf16ps(tj + 3, 6, 7);
                        _tile_stored(tj + 3, C_local[1][tj], TILE_N * sizeof(float));
                    }
                }
            }

            // 結果をCに書き戻し
            for (int ti = 0; ti < m_end; ti++) {
                for (int tj = 0; tj < n_end; tj++) {
                    int row_base = i + ti * TILE_M;
                    int col_base = n_start + j + tj * TILE_N;
                    for (int ii = 0; ii < TILE_M; ii++) {
                        for (int jj = 0; jj < TILE_N; jj++) {
                            int row = row_base + ii;
                            int col = col_base + jj;
                            if (row < M && col < n_start + n_block) {
                                C[row * ldc + col] = C_local[ti][tj][ii * TILE_N + jj];
                            }
                        }
                    }
                }
            }
        }
    }
}

// メイン関数: AMX最適化sbgemm_nolib
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

    if (transA != CblasNoTrans || transB != CblasNoTrans) {
        sbgemm_scalar(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    if (M < TILE_M || N < TILE_N || K < TILE_K) {
        sbgemm_scalar(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    if (init_amx() != 0) {
        fprintf(stderr, "AMX initialization failed, using scalar fallback.\n");
        sbgemm_scalar(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    configure_tiles();

    // beta処理
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

    // ワークバッファ確保
    int m_tiles = (M + TILE_M - 1) / TILE_M;
    int k_blocks = (K + BLOCK_K - 1) / BLOCK_K;
    size_t buf_A_size = m_tiles * TILE_M * BLOCK_K;
    size_t buf_B_size = ((BLOCK_N + TILE_N - 1) / TILE_N) * TILE_N * BLOCK_K;

    bf16 *buffer_A = (bf16*)aligned_alloc(64, buf_A_size * sizeof(bf16));
    bf16 *buffer_B = (bf16*)aligned_alloc(64, buf_B_size * sizeof(bf16));

    if (!buffer_A || !buffer_B) {
        fprintf(stderr, "Memory allocation failed.\n");
        free(buffer_A);
        free(buffer_B);
        sbgemm_scalar(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    // キャッシュブロッキングループ
    for (int k_start = 0; k_start < K; k_start += BLOCK_K) {
        int k_block = (k_start + BLOCK_K <= K) ? BLOCK_K : (K - k_start);

        pack_A(A, buffer_A, M, K, lda, k_start, k_block);

        for (int n_start = 0; n_start < N; n_start += BLOCK_N) {
            int n_block = (n_start + BLOCK_N <= N) ? BLOCK_N : (N - n_start);

            pack_B_transpose(B, buffer_B, K, N, ldb, k_start, k_block, n_start, n_block);
            amx_kernel_tiling_b(buffer_A, buffer_B, C, M, n_block, k_block, ldc, n_start);
        }
    }

    // alpha適用
    if (alpha != 1.0f) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] *= alpha;
            }
        }
    }

    free(buffer_A);
    free(buffer_B);
    _tile_release();
}

// ベンチマーク用メイン関数
int main(int argc, char *argv[]) {
    int sizes[] = {256, 512, 1024, 2048, 4096, 8192, 10000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("Intel AMX BF16 GEMM Benchmark (v1.0.0 - Tiling_B)\n");
    printf("Block sizes: k=%d, n=%d\n", BLOCK_K, BLOCK_N);
    printf("Theoretical peak: 1945.6 GFLOPS (1 core @ 1.9GHz)\n");
    printf("================================================\n");
    printf("%8s %12s %12s %12s\n", "Size", "Time(sec)", "GFLOPS", "Efficiency");
    printf("------------------------------------------------\n");

    for (int s = 0; s < num_sizes; s++) {
        int M = sizes[s], N = sizes[s], K = sizes[s];

        bf16 *A = (bf16*)aligned_alloc(64, M * K * sizeof(bf16));
        bf16 *B = (bf16*)aligned_alloc(64, K * N * sizeof(bf16));
        float *C = (float*)aligned_alloc(64, M * N * sizeof(float));

        if (!A || !B || !C) {
            printf("Memory allocation failed for size %d\n", M);
            free(A); free(B); free(C);
            continue;
        }

        for (int i = 0; i < M * K; i++) A[i] = float_to_bf16((float)(i % 100) / 100.0f);
        for (int i = 0; i < K * N; i++) B[i] = float_to_bf16((float)(i % 100) / 100.0f);
        for (int i = 0; i < M * N; i++) C[i] = 0.0f;

        // ウォームアップ
        sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                     M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

        // 計測
        struct timespec start, end;
        int iterations = (M >= 4096) ? 3 : 10;

        clock_gettime(CLOCK_MONOTONIC, &start);
        for (int iter = 0; iter < iterations; iter++) {
            sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                         M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
        }
        clock_gettime(CLOCK_MONOTONIC, &end);

        double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        double avg_time = elapsed / iterations;
        double flops = 2.0 * M * N * K;
        double gflops = flops / avg_time / 1e9;
        double efficiency = gflops / 1945.6 * 100.0;

        printf("%8d %12.4f %12.2f %11.1f%%\n", M, avg_time, gflops, efficiency);

        free(A);
        free(B);
        free(C);
    }

    printf("================================================\n");
    return 0;
}
