// sbgemm_v1.0.0.c - Intel AMX Tiling_B Implementation
// Based on reference.pdf: "Optimization of a GEMM Implementation using Intel AMX"
// Target: 65% theoretical efficiency using Tiling_B method (n=480, k=1536)

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <sys/time.h>

typedef uint16_t bf16;

// AMX tile configuration
#define TILE_M 16
#define TILE_N 16
#define TILE_K 32

// Cache blocking parameters (optimized from reference.pdf)
#define BLOCK_K 1536
#define BLOCK_N 480

// Tile configuration structure for AMX
typedef struct __tile_config {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} __tilecfg;

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

// Initialize AMX tile configuration
static void init_tile_config(__tilecfg *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->palette_id = 1;

    // Tiles 0-5: Result tiles (16x16 float32 = 16 rows, 64 bytes/row)
    for (int i = 0; i < 6; i++) {
        cfg->rows[i] = TILE_M;
        cfg->colsb[i] = TILE_N * sizeof(float);
    }

    // Tile 6: A tile (16x32 bf16 = 16 rows, 64 bytes/row)
    cfg->rows[6] = TILE_M;
    cfg->colsb[6] = TILE_K * sizeof(bf16);

    // Tile 7: B tile (16x32 bf16 = 16 rows, 64 bytes/row)
    cfg->rows[7] = TILE_K / 2;  // 16 rows for transposed B
    cfg->colsb[7] = TILE_N * 2 * sizeof(bf16);  // 32 columns paired

    _tile_loadconfig(cfg);
}

// Pack matrix A into BUFFER_A (16x32 block layout)
static void pack_A(const bf16 *A, bf16 *buffer, int M, int K, int lda, int k_start, int k_len) {
    for (int i = 0; i < M; i += TILE_M) {
        int rows = (i + TILE_M <= M) ? TILE_M : (M - i);
        for (int k = 0; k < k_len; k += TILE_K) {
            int cols = (k + TILE_K <= k_len) ? TILE_K : (k_len - k);
            bf16 *dst = buffer + (i / TILE_M) * (k_len / TILE_K) * TILE_M * TILE_K
                      + (k / TILE_K) * TILE_M * TILE_K;

            for (int ii = 0; ii < rows; ii++) {
                for (int kk = 0; kk < cols; kk++) {
                    dst[ii * TILE_K + kk] = A[(i + ii) * lda + (k_start + k + kk)];
                }
                // Zero padding
                for (int kk = cols; kk < TILE_K; kk++) {
                    dst[ii * TILE_K + kk] = 0;
                }
            }
            // Zero padding for rows
            for (int ii = rows; ii < TILE_M; ii++) {
                for (int kk = 0; kk < TILE_K; kk++) {
                    dst[ii * TILE_K + kk] = 0;
                }
            }
        }
    }
}

// Pack and transpose matrix B into BUFFER_B (32x16 transposed block layout)
// For Tiling_B: process 3 columns of B tiles at a time
static void pack_B_transpose(const bf16 *B, bf16 *buffer, int K, int N, int ldb,
                             int k_start, int k_len, int n_start, int n_len) {
    for (int j = 0; j < n_len; j += TILE_N) {
        int cols = (j + TILE_N <= n_len) ? TILE_N : (n_len - j);
        for (int k = 0; k < k_len; k += TILE_K) {
            int rows = (k + TILE_K <= k_len) ? TILE_K : (k_len - k);
            bf16 *dst = buffer + (j / TILE_N) * (k_len / TILE_K) * TILE_K * TILE_N
                      + (k / TILE_K) * TILE_K * TILE_N;

            // Transpose and interleave for VNNI format
            for (int kk = 0; kk < rows; kk += 2) {
                for (int jj = 0; jj < cols; jj++) {
                    int dst_row = kk / 2;
                    int dst_col = jj * 2;
                    dst[dst_row * (TILE_N * 2) + dst_col + 0] =
                        (kk < rows) ? B[(k_start + k + kk) * ldb + (n_start + j + jj)] : 0;
                    dst[dst_row * (TILE_N * 2) + dst_col + 1] =
                        (kk + 1 < rows) ? B[(k_start + k + kk + 1) * ldb + (n_start + j + jj)] : 0;
                }
                // Zero padding for columns
                for (int jj = cols; jj < TILE_N; jj++) {
                    int dst_row = kk / 2;
                    int dst_col = jj * 2;
                    dst[dst_row * (TILE_N * 2) + dst_col + 0] = 0;
                    dst[dst_row * (TILE_N * 2) + dst_col + 1] = 0;
                }
            }
        }
    }
}

// Tiling_B kernel: Process 2 rows of A, 3 columns of B at a time
// 6 result tiles in registers 0-5, A/B tiles in registers 6-7
static void tiling_b_kernel(const bf16 *buf_A, const bf16 *buf_B, float *C,
                           int M, int N, int K, int ldc,
                           int m_tiles, int n_tiles, int k_tiles) {

    for (int m = 0; m < m_tiles; m += 2) {
        for (int n = 0; n < n_tiles; n += 3) {
            // Load C tiles into registers 0-5
            float *C00 = C + (m * TILE_M) * ldc + (n * TILE_N);
            float *C01 = C + (m * TILE_M) * ldc + ((n + 1) * TILE_N);
            float *C02 = C + (m * TILE_M) * ldc + ((n + 2) * TILE_N);
            float *C10 = C + ((m + 1) * TILE_M) * ldc + (n * TILE_N);
            float *C11 = C + ((m + 1) * TILE_M) * ldc + ((n + 1) * TILE_N);
            float *C12 = C + ((m + 1) * TILE_M) * ldc + ((n + 2) * TILE_N);

            // Check bounds
            int valid_m1 = (m + 1 < m_tiles);
            int valid_n1 = (n + 1 < n_tiles);
            int valid_n2 = (n + 2 < n_tiles);

            _tile_loadd(0, C00, ldc * sizeof(float));
            if (valid_n1) _tile_loadd(1, C01, ldc * sizeof(float));
            if (valid_n2) _tile_loadd(2, C02, ldc * sizeof(float));
            if (valid_m1) _tile_loadd(3, C10, ldc * sizeof(float));
            if (valid_m1 && valid_n1) _tile_loadd(4, C11, ldc * sizeof(float));
            if (valid_m1 && valid_n2) _tile_loadd(5, C12, ldc * sizeof(float));

            // Process K dimension
            for (int k = 0; k < k_tiles; k++) {
                const bf16 *A_tile0 = buf_A + m * k_tiles * TILE_M * TILE_K + k * TILE_M * TILE_K;
                const bf16 *A_tile1 = buf_A + (m + 1) * k_tiles * TILE_M * TILE_K + k * TILE_M * TILE_K;
                const bf16 *B_tile0 = buf_B + n * k_tiles * TILE_K * TILE_N + k * TILE_K * TILE_N;
                const bf16 *B_tile1 = buf_B + (n + 1) * k_tiles * TILE_K * TILE_N + k * TILE_K * TILE_N;
                const bf16 *B_tile2 = buf_B + (n + 2) * k_tiles * TILE_K * TILE_N + k * TILE_K * TILE_N;

                // Prefetch next A tile
                if (k + 1 < k_tiles) {
                    _mm_prefetch((const char*)(A_tile0 + TILE_M * TILE_K), _MM_HINT_T0);
                }

                // Tiling_B computation order (from reference.pdf Figure 9)
                // Step 1: Load A[2m,k], B[k,2n], compute C[2m,2n]
                _tile_loadd(6, A_tile0, TILE_K * sizeof(bf16));
                _tile_loadd(7, B_tile0, TILE_N * 2 * sizeof(bf16));
                _tile_dpbf16ps(0, 6, 7);

                // Step 2: Load B[k,2n+1], compute C[2m,2n+1]
                if (valid_n1) {
                    _tile_loadd(7, B_tile1, TILE_N * 2 * sizeof(bf16));
                    _tile_dpbf16ps(1, 6, 7);
                }

                // Step 3: Load B[k,2n+2], compute C[2m,2n+2]
                if (valid_n2) {
                    _tile_loadd(7, B_tile2, TILE_N * 2 * sizeof(bf16));
                    _tile_dpbf16ps(2, 6, 7);
                }

                // Step 4: Load A[2m+1,k], compute C[2m+1,2n+2]
                if (valid_m1) {
                    _tile_loadd(6, A_tile1, TILE_K * sizeof(bf16));
                    if (valid_n2) _tile_dpbf16ps(5, 6, 7);

                    // Step 5: Reload B[k,2n], compute C[2m+1,2n] (L1 cache hit)
                    _tile_loadd(7, B_tile0, TILE_N * 2 * sizeof(bf16));
                    _tile_dpbf16ps(3, 6, 7);

                    // Step 6: Reload B[k,2n+1], compute C[2m+1,2n+1] (L1 cache hit)
                    if (valid_n1) {
                        _tile_loadd(7, B_tile1, TILE_N * 2 * sizeof(bf16));
                        _tile_dpbf16ps(4, 6, 7);
                    }
                }
            }

            // Store C tiles
            _tile_stored(0, C00, ldc * sizeof(float));
            if (valid_n1) _tile_stored(1, C01, ldc * sizeof(float));
            if (valid_n2) _tile_stored(2, C02, ldc * sizeof(float));
            if (valid_m1) _tile_stored(3, C10, ldc * sizeof(float));
            if (valid_m1 && valid_n1) _tile_stored(4, C11, ldc * sizeof(float));
            if (valid_m1 && valid_n2) _tile_stored(5, C12, ldc * sizeof(float));
        }
    }
}

// Main SBGEMM function with Tiling_B optimization
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
        fprintf(stderr, "Only NoTrans is supported in this version.\n");
        exit(1);
    }

    // Initialize AMX
    __tilecfg tile_cfg;
    init_tile_config(&tile_cfg);

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

    // Allocate buffers for blocking
    int k_block = (K < BLOCK_K) ? K : BLOCK_K;
    int n_block = (N < BLOCK_N) ? N : BLOCK_N;

    // Round up to tile boundaries
    int m_tiles = (M + TILE_M - 1) / TILE_M;
    int k_tiles_per_block = (k_block + TILE_K - 1) / TILE_K;
    int n_tiles_per_block = (n_block + TILE_N - 1) / TILE_N;

    bf16 *buf_A = (bf16*)aligned_alloc(64, m_tiles * k_tiles_per_block * TILE_M * TILE_K * sizeof(bf16));
    bf16 *buf_B = (bf16*)aligned_alloc(64, n_tiles_per_block * k_tiles_per_block * TILE_K * TILE_N * sizeof(bf16));

    if (!buf_A || !buf_B) {
        fprintf(stderr, "Buffer allocation failed\n");
        exit(1);
    }

    // Main computation with cache blocking
    for (int k = 0; k < K; k += BLOCK_K) {
        int k_len = (k + BLOCK_K <= K) ? BLOCK_K : (K - k);
        int k_tiles = (k_len + TILE_K - 1) / TILE_K;

        // Pack A for this k-block
        pack_A(A, buf_A, M, K, lda, k, k_len);

        for (int n = 0; n < N; n += BLOCK_N) {
            int n_len = (n + BLOCK_N <= N) ? BLOCK_N : (N - n);
            int n_tiles = (n_len + TILE_N - 1) / TILE_N;

            // Pack B for this (k,n) block
            pack_B_transpose(B, buf_B, K, N, ldb, k, k_len, n, n_len);

            // Execute Tiling_B kernel
            tiling_b_kernel(buf_A, buf_B, C + n, M, n_len, k_len, ldc,
                           m_tiles, n_tiles, k_tiles);
        }
    }

    // Apply alpha if needed
    if (alpha != 1.0f) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                C[i * ldc + j] *= alpha;
            }
        }
    }

    free(buf_A);
    free(buf_B);

    // Release AMX state
    _tile_release();
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

    // Allocate matrices
    bf16 *A = (bf16*)aligned_alloc(64, M * K * sizeof(bf16));
    bf16 *B = (bf16*)aligned_alloc(64, K * N * sizeof(bf16));
    float *C = (float*)aligned_alloc(64, M * N * sizeof(float));

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Initialize matrices with random values
    srand(42);
    for (int i = 0; i < M * K; i++) {
        A[i] = float_to_bf16((float)(rand() % 100) / 100.0f);
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = float_to_bf16((float)(rand() % 100) / 100.0f);
    }
    for (int i = 0; i < M * N; i++) {
        C[i] = 0.0f;
    }

    // Warmup
    sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

    // Benchmark
    double total_time = 0.0;
    double min_time = 1e9;

    for (int iter = 0; iter < num_iterations; iter++) {
        // Reset C
        for (int i = 0; i < M * N; i++) C[i] = 0.0f;

        double start = get_time();
        sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                     M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
        double end = get_time();

        double elapsed = end - start;
        total_time += elapsed;
        if (elapsed < min_time) min_time = elapsed;
    }

    // Calculate performance
    double ops = 2.0 * M * N * K;
    double avg_time = total_time / num_iterations;
    double gflops_avg = (ops / avg_time) / 1e9;
    double gflops_peak = (ops / min_time) / 1e9;

    // Theoretical peak: 1945.6 GFLOPS (1024 ops/cycle * 1.9 GHz)
    double theoretical_peak = 1945.6;
    double efficiency = (gflops_peak / theoretical_peak) * 100.0;

    printf("Average time: %.4f sec\n", avg_time);
    printf("Min time: %.4f sec\n", min_time);
    printf("Average GFLOPS: %.2f\n", gflops_avg);
    printf("Peak GFLOPS: %.2f\n", gflops_peak);
    printf("Efficiency: %.2f%% of theoretical peak (%.1f GFLOPS)\n", efficiency, theoretical_peak);

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

    printf("=== SBGEMM Tiling_B Optimization (v1.0.0) ===\n");
    printf("Block sizes: BLOCK_K=%d, BLOCK_N=%d\n", BLOCK_K, BLOCK_N);
    printf("Tile sizes: TILE_M=%d, TILE_N=%d, TILE_K=%d\n", TILE_M, TILE_N, TILE_K);
    printf("Iterations: %d\n\n", num_iterations);

    benchmark_sbgemm(M, N, K, num_iterations);

    return 0;
}
