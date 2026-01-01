// sbgemm_amx_v1.1.0.c - Intel AMX最適化版BFloat16 GEMM
// PG1.1: SIMD最適化担当
// v1.1.0: Tiling_B (2x3カーネル) + プリフェッチ追加
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <time.h>

typedef uint16_t bf16;

typedef enum { CblasRowMajor = 101, CblasColMajor = 102 } CBLAS_LAYOUT;
typedef enum { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 } CBLAS_TRANSPOSE;

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILEDATA      18

#define TILE_M 16
#define TILE_N 16
#define TILE_K 32
#define BLOCK_K 1536
#define BLOCK_N 480

typedef struct __tile_config {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} __tilecfg;

static int init_amx(void) {
    unsigned long bitmask = 0;
    if (syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask) != 0) return -1;
    if (bitmask & (1UL << XFEATURE_XTILEDATA)) return 0;
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0) return -1;
    return 0;
}

static void configure_tiles_2x3(void) {
    __tilecfg config = {0};
    config.palette_id = 1;
    // タイル0-5: 結果C用 (2行x3列)
    for (int i = 0; i < 6; i++) {
        config.rows[i] = TILE_M;
        config.colsb[i] = TILE_N * sizeof(float);
    }
    // タイル6: A用
    config.rows[6] = TILE_M;
    config.colsb[6] = TILE_K * sizeof(bf16);
    // タイル7: B用
    config.rows[7] = TILE_K / 2;
    config.colsb[7] = TILE_N * 2 * sizeof(bf16);
    _tile_loadconfig(&config);
}

static inline bf16 float_to_bf16(float x) {
    uint32_t u; memcpy(&u, &x, sizeof(u));
    u += 0x00008000u;
    return (bf16)(u >> 16);
}

static inline float bf16_to_float(bf16 x) {
    uint32_t u = ((uint32_t)x) << 16;
    float f; memcpy(&f, &u, sizeof(f));
    return f;
}

static void sbgemm_scalar(int M, int N, int K, float alpha,
                          const bf16 *A, int lda, const bf16 *B, int ldb,
                          float beta, float *C, int ldc) {
    if (beta == 0.0f) {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) C[i * ldc + j] = 0.0f;
    } else if (beta != 1.0f) {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) C[i * ldc + j] *= beta;
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += bf16_to_float(A[i * lda + k]) * bf16_to_float(B[k * ldb + j]);
            C[i * ldc + j] += alpha * sum;
        }
    }
}

// AVX-512でAをパッキング
static void pack_A_avx512(const bf16 *A, bf16 *buffer, int M, int K, int lda, int k_start, int k_block) {
    int k_tiles = (k_block + TILE_K - 1) / TILE_K;
    int m_tiles = (M + TILE_M - 1) / TILE_M;

    for (int mt = 0; mt < m_tiles; mt++) {
        for (int kt = 0; kt < k_tiles; kt++) {
            bf16 *dest = buffer + (mt * k_tiles + kt) * TILE_M * TILE_K;
            int m_base = mt * TILE_M;
            int k_base = k_start + kt * TILE_K;

            // プリフェッチ次のブロック
            if (kt + 1 < k_tiles) {
                _mm_prefetch((const char*)(A + m_base * lda + k_base + TILE_K), _MM_HINT_T0);
            }

            for (int ii = 0; ii < TILE_M; ii++) {
                int m_idx = m_base + ii;
                if (m_idx < M) {
                    // AVX-512でコピー（32要素 = 64バイト）
                    if (k_base + TILE_K <= k_start + k_block) {
                        __m512i v = _mm512_loadu_si512((const __m512i*)(A + m_idx * lda + k_base));
                        _mm512_storeu_si512((__m512i*)(dest + ii * TILE_K), v);
                    } else {
                        for (int kk = 0; kk < TILE_K; kk++) {
                            int k_idx = k_base + kk;
                            dest[ii * TILE_K + kk] = (k_idx < k_start + k_block) ? A[m_idx * lda + k_idx] : 0;
                        }
                    }
                } else {
                    memset(dest + ii * TILE_K, 0, TILE_K * sizeof(bf16));
                }
            }
        }
    }
}

// Bをパッキング（AMXインターリーブ形式）
static void pack_B_avx512(const bf16 *B, bf16 *buffer, int K, int N, int ldb,
                          int k_start, int k_block, int n_start, int n_block) {
    int k_tiles = (k_block + TILE_K - 1) / TILE_K;
    int n_tiles = (n_block + TILE_N - 1) / TILE_N;

    for (int nt = 0; nt < n_tiles; nt++) {
        for (int kt = 0; kt < k_tiles; kt++) {
            bf16 *dest = buffer + (nt * k_tiles + kt) * (TILE_K / 2) * (TILE_N * 2);
            int n_base = n_start + nt * TILE_N;
            int k_base = k_start + kt * TILE_K;

            for (int kk = 0; kk < TILE_K / 2; kk++) {
                for (int nn = 0; nn < TILE_N; nn++) {
                    int n_idx = n_base + nn;
                    int k_idx0 = k_base + kk * 2;
                    int k_idx1 = k_base + kk * 2 + 1;

                    if (n_idx < n_start + n_block && k_idx0 < k_start + k_block) {
                        dest[kk * TILE_N * 2 + nn * 2] = B[k_idx0 * ldb + n_idx];
                        dest[kk * TILE_N * 2 + nn * 2 + 1] = (k_idx1 < k_start + k_block) ? B[k_idx1 * ldb + n_idx] : 0;
                    } else {
                        dest[kk * TILE_N * 2 + nn * 2] = 0;
                        dest[kk * TILE_N * 2 + nn * 2 + 1] = 0;
                    }
                }
            }
        }
    }
}

// Tiling_B: 2x3カーネル（固定タイル番号、プリフェッチ付き）
static void amx_kernel_tiling_b(const bf16 *A_pack, const bf16 *B_pack, float *C,
                                 int M, int n_block, int k_block, int ldc, int n_start,
                                 int k_tiles, int n_tiles) {
    int m_tiles = (M + TILE_M - 1) / TILE_M;

    for (int mt = 0; mt < m_tiles; mt += 2) {
        int m0 = mt * TILE_M;
        int m1 = (mt + 1) * TILE_M;
        int m_valid1 = (mt + 1 < m_tiles);

        for (int nt = 0; nt < n_tiles; nt += 3) {
            int n0 = n_start + nt * TILE_N;
            int n1 = n_start + (nt + 1) * TILE_N;
            int n2 = n_start + (nt + 2) * TILE_N;
            int n_valid1 = (nt + 1 < n_tiles);
            int n_valid2 = (nt + 2 < n_tiles);

            // ローカルCバッファ（2x3 = 6タイル）
            float C00[TILE_M * TILE_N] __attribute__((aligned(64)));
            float C01[TILE_M * TILE_N] __attribute__((aligned(64)));
            float C02[TILE_M * TILE_N] __attribute__((aligned(64)));
            float C10[TILE_M * TILE_N] __attribute__((aligned(64)));
            float C11[TILE_M * TILE_N] __attribute__((aligned(64)));
            float C12[TILE_M * TILE_N] __attribute__((aligned(64)));

            // Cをロード
            for (int ii = 0; ii < TILE_M; ii++) {
                for (int jj = 0; jj < TILE_N; jj++) {
                    C00[ii * TILE_N + jj] = (m0 + ii < M && n0 + jj < n_start + n_block) ? C[(m0 + ii) * ldc + n0 + jj] : 0.0f;
                    C01[ii * TILE_N + jj] = (m0 + ii < M && n1 + jj < n_start + n_block && n_valid1) ? C[(m0 + ii) * ldc + n1 + jj] : 0.0f;
                    C02[ii * TILE_N + jj] = (m0 + ii < M && n2 + jj < n_start + n_block && n_valid2) ? C[(m0 + ii) * ldc + n2 + jj] : 0.0f;
                    C10[ii * TILE_N + jj] = (m1 + ii < M && n0 + jj < n_start + n_block && m_valid1) ? C[(m1 + ii) * ldc + n0 + jj] : 0.0f;
                    C11[ii * TILE_N + jj] = (m1 + ii < M && n1 + jj < n_start + n_block && m_valid1 && n_valid1) ? C[(m1 + ii) * ldc + n1 + jj] : 0.0f;
                    C12[ii * TILE_N + jj] = (m1 + ii < M && n2 + jj < n_start + n_block && m_valid1 && n_valid2) ? C[(m1 + ii) * ldc + n2 + jj] : 0.0f;
                }
            }

            // Cをタイルにロード
            _tile_loadd(0, C00, TILE_N * sizeof(float));
            _tile_loadd(1, C01, TILE_N * sizeof(float));
            _tile_loadd(2, C02, TILE_N * sizeof(float));
            _tile_loadd(3, C10, TILE_N * sizeof(float));
            _tile_loadd(4, C11, TILE_N * sizeof(float));
            _tile_loadd(5, C12, TILE_N * sizeof(float));

            // K方向ループ
            for (int kt = 0; kt < k_tiles; kt++) {
                const bf16 *A0 = A_pack + (mt * k_tiles + kt) * TILE_M * TILE_K;
                const bf16 *A1 = A_pack + ((mt + 1) * k_tiles + kt) * TILE_M * TILE_K;
                const bf16 *B0 = B_pack + (nt * k_tiles + kt) * (TILE_K / 2) * (TILE_N * 2);
                const bf16 *B1 = B_pack + ((nt + 1) * k_tiles + kt) * (TILE_K / 2) * (TILE_N * 2);
                const bf16 *B2 = B_pack + ((nt + 2) * k_tiles + kt) * (TILE_K / 2) * (TILE_N * 2);

                // プリフェッチ次のKブロック
                if (kt + 1 < k_tiles) {
                    _mm_prefetch((const char*)(A0 + TILE_M * TILE_K), _MM_HINT_T0);
                    _mm_prefetch((const char*)(B0 + (TILE_K / 2) * (TILE_N * 2)), _MM_HINT_T0);
                }

                // A0をロード
                _tile_loadd(6, A0, TILE_K * sizeof(bf16));

                // A0 * B0 -> C00
                _tile_loadd(7, B0, TILE_N * 2 * sizeof(bf16));
                _tile_dpbf16ps(0, 6, 7);

                // A0 * B1 -> C01
                if (n_valid1) {
                    _tile_loadd(7, B1, TILE_N * 2 * sizeof(bf16));
                    _tile_dpbf16ps(1, 6, 7);
                }

                // A0 * B2 -> C02
                if (n_valid2) {
                    _tile_loadd(7, B2, TILE_N * 2 * sizeof(bf16));
                    _tile_dpbf16ps(2, 6, 7);
                }

                // A1をロード（2行目）
                if (m_valid1) {
                    _tile_loadd(6, A1, TILE_K * sizeof(bf16));

                    // A1 * B0 -> C10
                    _tile_loadd(7, B0, TILE_N * 2 * sizeof(bf16));
                    _tile_dpbf16ps(3, 6, 7);

                    // A1 * B1 -> C11
                    if (n_valid1) {
                        _tile_loadd(7, B1, TILE_N * 2 * sizeof(bf16));
                        _tile_dpbf16ps(4, 6, 7);
                    }

                    // A1 * B2 -> C12
                    if (n_valid2) {
                        _tile_loadd(7, B2, TILE_N * 2 * sizeof(bf16));
                        _tile_dpbf16ps(5, 6, 7);
                    }
                }
            }

            // 結果をストア
            _tile_stored(0, C00, TILE_N * sizeof(float));
            _tile_stored(1, C01, TILE_N * sizeof(float));
            _tile_stored(2, C02, TILE_N * sizeof(float));
            _tile_stored(3, C10, TILE_N * sizeof(float));
            _tile_stored(4, C11, TILE_N * sizeof(float));
            _tile_stored(5, C12, TILE_N * sizeof(float));

            // Cに書き戻し
            for (int ii = 0; ii < TILE_M; ii++) {
                for (int jj = 0; jj < TILE_N; jj++) {
                    if (m0 + ii < M && n0 + jj < n_start + n_block) C[(m0 + ii) * ldc + n0 + jj] = C00[ii * TILE_N + jj];
                    if (m0 + ii < M && n1 + jj < n_start + n_block && n_valid1) C[(m0 + ii) * ldc + n1 + jj] = C01[ii * TILE_N + jj];
                    if (m0 + ii < M && n2 + jj < n_start + n_block && n_valid2) C[(m0 + ii) * ldc + n2 + jj] = C02[ii * TILE_N + jj];
                    if (m1 + ii < M && n0 + jj < n_start + n_block && m_valid1) C[(m1 + ii) * ldc + n0 + jj] = C10[ii * TILE_N + jj];
                    if (m1 + ii < M && n1 + jj < n_start + n_block && m_valid1 && n_valid1) C[(m1 + ii) * ldc + n1 + jj] = C11[ii * TILE_N + jj];
                    if (m1 + ii < M && n2 + jj < n_start + n_block && m_valid1 && n_valid2) C[(m1 + ii) * ldc + n2 + jj] = C12[ii * TILE_N + jj];
                }
            }
        }
    }
}

void sbgemm_nolib(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                  int M, int N, int K, float alpha,
                  const bf16 *A, int lda, const bf16 *B, int ldb,
                  float beta, float *C, int ldc) {
    if (layout != CblasRowMajor) { fprintf(stderr, "Only RowMajor supported.\n"); exit(1); }
    if (transA != CblasNoTrans || transB != CblasNoTrans || M < TILE_M || N < TILE_N || K < TILE_K) {
        sbgemm_scalar(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
    if (init_amx() != 0) {
        fprintf(stderr, "AMX init failed.\n");
        sbgemm_scalar(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    configure_tiles_2x3();

    if (beta == 0.0f) {
        for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) C[i * ldc + j] = 0.0f;
    } else if (beta != 1.0f) {
        for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) C[i * ldc + j] *= beta;
    }

    int m_tiles = (M + TILE_M - 1) / TILE_M;
    size_t buf_A_size = m_tiles * TILE_M * ((BLOCK_K + TILE_K - 1) / TILE_K) * TILE_K;
    size_t buf_B_size = ((BLOCK_N + TILE_N - 1) / TILE_N) * ((BLOCK_K + TILE_K - 1) / TILE_K) * (TILE_K / 2) * (TILE_N * 2);

    bf16 *buffer_A = (bf16*)aligned_alloc(64, buf_A_size * sizeof(bf16));
    bf16 *buffer_B = (bf16*)aligned_alloc(64, buf_B_size * sizeof(bf16));

    if (!buffer_A || !buffer_B) {
        free(buffer_A); free(buffer_B);
        sbgemm_scalar(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    for (int k_start = 0; k_start < K; k_start += BLOCK_K) {
        int k_block = (k_start + BLOCK_K <= K) ? BLOCK_K : (K - k_start);
        int k_tiles = (k_block + TILE_K - 1) / TILE_K;

        pack_A_avx512(A, buffer_A, M, K, lda, k_start, k_block);

        for (int n_start = 0; n_start < N; n_start += BLOCK_N) {
            int n_block = (n_start + BLOCK_N <= N) ? BLOCK_N : (N - n_start);
            int n_tiles = (n_block + TILE_N - 1) / TILE_N;

            pack_B_avx512(B, buffer_B, K, N, ldb, k_start, k_block, n_start, n_block);
            amx_kernel_tiling_b(buffer_A, buffer_B, C, M, n_block, k_block, ldc, n_start, k_tiles, n_tiles);
        }
    }

    if (alpha != 1.0f) {
        for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) C[i * ldc + j] *= alpha;
    }

    free(buffer_A);
    free(buffer_B);
    _tile_release();
}

int main(int argc, char *argv[]) {
    int sizes[] = {256, 512, 1024, 2048, 4096, 8192, 10000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("Intel AMX BF16 GEMM Benchmark (v1.1.0 - Tiling_B 2x3)\n");
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

        if (!A || !B || !C) { printf("Alloc failed %d\n", M); free(A); free(B); free(C); continue; }

        for (int i = 0; i < M * K; i++) A[i] = float_to_bf16((float)(i % 100) / 100.0f);
        for (int i = 0; i < K * N; i++) B[i] = float_to_bf16((float)(i % 100) / 100.0f);
        for (int i = 0; i < M * N; i++) C[i] = 0.0f;

        sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

        struct timespec start, end;
        int iterations = (M >= 4096) ? 3 : 10;

        clock_gettime(CLOCK_MONOTONIC, &start);
        for (int iter = 0; iter < iterations; iter++) {
            sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
        }
        clock_gettime(CLOCK_MONOTONIC, &end);

        double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        double avg_time = elapsed / iterations;
        double flops = 2.0 * M * N * K;
        double gflops = flops / avg_time / 1e9;
        double efficiency = gflops / 1945.6 * 100.0;

        printf("%8d %12.4f %12.2f %11.1f%%\n", M, avg_time, gflops, efficiency);

        free(A); free(B); free(C);
    }

    printf("================================================\n");
    return 0;
}
