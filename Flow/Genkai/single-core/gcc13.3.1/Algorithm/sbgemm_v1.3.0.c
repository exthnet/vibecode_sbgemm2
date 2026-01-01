// sbgemm_v1.3.0.c - Intel AMX with proper initialization
// Fixed: AMX state permission request via arch_prctl syscall

#define _GNU_SOURCE
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <sys/time.h>
#include <sys/syscall.h>
#include <unistd.h>

typedef uint16_t bf16;

// AMX state permission constants
#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

// Tile configuration
#define TILE_M 16
#define TILE_N 16
#define TILE_K 32

// Cache blocking (from reference.pdf)
#define BLOCK_K 1536
#define BLOCK_N 480

typedef struct __tile_config {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} __tilecfg __attribute__((aligned(64)));

typedef enum { CblasRowMajor = 101 } CBLAS_LAYOUT;
typedef enum { CblasNoTrans = 111, CblasTrans = 112 } CBLAS_TRANSPOSE;

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

// Request AMX permission from kernel
static int request_amx_permission(void) {
    unsigned long bitmask = 0;

    // Get current permissions
    if (syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask) != 0) {
        return -1;
    }

    // Check if XTILEDATA is already permitted
    if (bitmask & (1UL << XFEATURE_XTILEDATA)) {
        return 0;  // Already permitted
    }

    // Request permission for XTILEDATA
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0) {
        return -2;
    }

    return 0;
}

// Initialize tile configuration for Tiling_B method
static void init_tiles_for_tiling_b(__tilecfg *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->palette_id = 1;

    // Tiles 0-5: Result C tiles (16x16 float32)
    // Each row: 16 floats = 64 bytes
    for (int i = 0; i < 6; i++) {
        cfg->rows[i] = TILE_M;
        cfg->colsb[i] = TILE_N * sizeof(float);  // 64 bytes
    }

    // Tile 6: A tile (16 rows x 32 BF16 = 16 rows x 64 bytes)
    cfg->rows[6] = TILE_M;
    cfg->colsb[6] = TILE_K * sizeof(bf16);  // 64 bytes

    // Tile 7: B tile (16 rows x 32 BF16 in VNNI format)
    cfg->rows[7] = TILE_K / 2;  // 16 rows
    cfg->colsb[7] = TILE_N * 2 * sizeof(bf16);  // 64 bytes

    _tile_loadconfig(cfg);
}

// Pack A into tile-friendly format (row-major, 16x32 blocks)
static void pack_A_tile(const bf16 *A, bf16 *packed, int M, int K, int lda,
                        int k_start, int k_len) {
    int m_tiles = (M + TILE_M - 1) / TILE_M;
    int k_tiles = (k_len + TILE_K - 1) / TILE_K;

    for (int mt = 0; mt < m_tiles; mt++) {
        for (int kt = 0; kt < k_tiles; kt++) {
            bf16 *dst = packed + (mt * k_tiles + kt) * TILE_M * TILE_K;
            int m_base = mt * TILE_M;
            int k_base = kt * TILE_K;

            for (int i = 0; i < TILE_M; i++) {
                for (int j = 0; j < TILE_K; j++) {
                    int m_idx = m_base + i;
                    int k_idx = k_start + k_base + j;
                    if (m_idx < M && k_idx < k_start + k_len) {
                        dst[i * TILE_K + j] = A[m_idx * lda + k_idx];
                    } else {
                        dst[i * TILE_K + j] = 0;
                    }
                }
            }
        }
    }
}

// Pack B into VNNI format (pairs of BF16 for dpbf16ps)
static void pack_B_vnni(const bf16 *B, bf16 *packed, int K, int N, int ldb,
                        int k_start, int k_len, int n_start, int n_len) {
    int n_tiles = (n_len + TILE_N - 1) / TILE_N;
    int k_tiles = (k_len + TILE_K - 1) / TILE_K;

    for (int nt = 0; nt < n_tiles; nt++) {
        for (int kt = 0; kt < k_tiles; kt++) {
            bf16 *dst = packed + (nt * k_tiles + kt) * (TILE_K / 2) * (TILE_N * 2);
            int n_base = nt * TILE_N;
            int k_base = kt * TILE_K;

            // Pack in VNNI format: pairs (k, k+1) interleaved with n
            for (int k2 = 0; k2 < TILE_K / 2; k2++) {
                for (int n = 0; n < TILE_N; n++) {
                    int k0_idx = k_start + k_base + k2 * 2;
                    int k1_idx = k0_idx + 1;
                    int n_idx = n_start + n_base + n;

                    bf16 v0 = 0, v1 = 0;
                    if (k0_idx < k_start + k_len && n_idx < n_start + n_len) {
                        v0 = B[k0_idx * ldb + n_idx];
                    }
                    if (k1_idx < k_start + k_len && n_idx < n_start + n_len) {
                        v1 = B[k1_idx * ldb + n_idx];
                    }

                    // VNNI format: (v0, v1) pairs
                    dst[k2 * (TILE_N * 2) + n * 2] = v0;
                    dst[k2 * (TILE_N * 2) + n * 2 + 1] = v1;
                }
            }
        }
    }
}

// Tiling_B kernel using AMX instructions
static void tiling_b_kernel_amx(
    const bf16 *A_packed, const bf16 *B_packed,
    float *C, int ldc,
    int m_tiles, int n_tiles, int k_tiles)
{
    int stride_A = TILE_M * TILE_K * sizeof(bf16);  // 1024 bytes
    int stride_B = (TILE_K / 2) * (TILE_N * 2) * sizeof(bf16);  // 1024 bytes
    int stride_C = ldc * sizeof(float);

    for (int mt = 0; mt < m_tiles; mt += 2) {
        int mt1_valid = (mt + 1 < m_tiles);

        for (int nt = 0; nt < n_tiles; nt += 3) {
            int nt1_valid = (nt + 1 < n_tiles);
            int nt2_valid = (nt + 2 < n_tiles);

            float *C00 = C + (mt * TILE_M) * ldc + (nt * TILE_N);
            float *C01 = C00 + TILE_N;
            float *C02 = C00 + 2 * TILE_N;
            float *C10 = C00 + TILE_M * ldc;
            float *C11 = C10 + TILE_N;
            float *C12 = C10 + 2 * TILE_N;

            // Load C tiles
            _tile_loadd(0, C00, stride_C);
            if (nt1_valid) _tile_loadd(1, C01, stride_C);
            if (nt2_valid) _tile_loadd(2, C02, stride_C);
            if (mt1_valid) _tile_loadd(3, C10, stride_C);
            if (mt1_valid && nt1_valid) _tile_loadd(4, C11, stride_C);
            if (mt1_valid && nt2_valid) _tile_loadd(5, C12, stride_C);

            // Process K tiles
            for (int kt = 0; kt < k_tiles; kt++) {
                const bf16 *A0 = A_packed + (mt * k_tiles + kt) * TILE_M * TILE_K;
                const bf16 *A1 = A_packed + ((mt + 1) * k_tiles + kt) * TILE_M * TILE_K;
                const bf16 *B0 = B_packed + (nt * k_tiles + kt) * (TILE_K / 2) * (TILE_N * 2);
                const bf16 *B1 = B_packed + ((nt + 1) * k_tiles + kt) * (TILE_K / 2) * (TILE_N * 2);
                const bf16 *B2 = B_packed + ((nt + 2) * k_tiles + kt) * (TILE_K / 2) * (TILE_N * 2);

                // Tiling_B order (from reference.pdf Figure 9)
                // Step 1: A[2m,k], B[k,2n] -> C[2m,2n]
                _tile_loadd(6, A0, stride_A);
                _tile_loadd(7, B0, TILE_N * 2 * sizeof(bf16));
                _tile_dpbf16ps(0, 6, 7);

                // Step 2: B[k,2n+1] -> C[2m,2n+1]
                if (nt1_valid) {
                    _tile_loadd(7, B1, TILE_N * 2 * sizeof(bf16));
                    _tile_dpbf16ps(1, 6, 7);
                }

                // Step 3: B[k,2n+2] -> C[2m,2n+2]
                if (nt2_valid) {
                    _tile_loadd(7, B2, TILE_N * 2 * sizeof(bf16));
                    _tile_dpbf16ps(2, 6, 7);
                }

                // Step 4-6: A[2m+1,k]
                if (mt1_valid) {
                    _tile_loadd(6, A1, stride_A);

                    // Step 4: -> C[2m+1,2n+2]
                    if (nt2_valid) _tile_dpbf16ps(5, 6, 7);

                    // Step 5: Reload B[k,2n] -> C[2m+1,2n]
                    _tile_loadd(7, B0, TILE_N * 2 * sizeof(bf16));
                    _tile_dpbf16ps(3, 6, 7);

                    // Step 6: Reload B[k,2n+1] -> C[2m+1,2n+1]
                    if (nt1_valid) {
                        _tile_loadd(7, B1, TILE_N * 2 * sizeof(bf16));
                        _tile_dpbf16ps(4, 6, 7);
                    }
                }
            }

            // Store C tiles
            _tile_stored(0, C00, stride_C);
            if (nt1_valid) _tile_stored(1, C01, stride_C);
            if (nt2_valid) _tile_stored(2, C02, stride_C);
            if (mt1_valid) _tile_stored(3, C10, stride_C);
            if (mt1_valid && nt1_valid) _tile_stored(4, C11, stride_C);
            if (mt1_valid && nt2_valid) _tile_stored(5, C12, stride_C);
        }
    }
}

void sbgemm_nolib(CBLAS_LAYOUT layout,
                  CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                  int M, int N, int K,
                  float alpha,
                  const bf16 *A, int lda,
                  const bf16 *B, int ldb,
                  float beta,
                  float *C, int ldc)
{
    // Apply beta
    if (beta == 0.0f) {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                C[i * ldc + j] = 0.0f;
    } else if (beta != 1.0f) {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                C[i * ldc + j] *= beta;
    }

    // Initialize AMX
    __tilecfg cfg;
    init_tiles_for_tiling_b(&cfg);

    int m_tiles = (M + TILE_M - 1) / TILE_M;
    int k_block = (K < BLOCK_K) ? K : BLOCK_K;
    int n_block = (N < BLOCK_N) ? N : BLOCK_N;
    int k_tiles_per_block = (k_block + TILE_K - 1) / TILE_K;
    int n_tiles_per_block = (n_block + TILE_N - 1) / TILE_N;

    // Allocate packed buffers
    bf16 *A_packed = aligned_alloc(64, m_tiles * k_tiles_per_block * TILE_M * TILE_K * sizeof(bf16));
    bf16 *B_packed = aligned_alloc(64, n_tiles_per_block * k_tiles_per_block * (TILE_K / 2) * (TILE_N * 2) * sizeof(bf16));

    if (!A_packed || !B_packed) {
        fprintf(stderr, "Buffer allocation failed\n");
        exit(1);
    }

    // Cache blocking
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        int k_len = (k0 + BLOCK_K <= K) ? BLOCK_K : (K - k0);
        int k_tiles = (k_len + TILE_K - 1) / TILE_K;

        pack_A_tile(A, A_packed, M, K, lda, k0, k_len);

        for (int n0 = 0; n0 < N; n0 += BLOCK_N) {
            int n_len = (n0 + BLOCK_N <= N) ? BLOCK_N : (N - n0);
            int n_tiles = (n_len + TILE_N - 1) / TILE_N;

            pack_B_vnni(B, B_packed, K, N, ldb, k0, k_len, n0, n_len);

            tiling_b_kernel_amx(A_packed, B_packed, C + n0, ldc,
                               m_tiles, n_tiles, k_tiles);
        }
    }

    free(A_packed);
    free(B_packed);

    // Apply alpha
    if (alpha != 1.0f) {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                C[i * ldc + j] *= alpha;
    }

    _tile_release();
}

static double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void benchmark(int M, int N, int K, int iters) {
    printf("Matrix: %dx%dx%d\n", M, N, K);

    bf16 *A = aligned_alloc(64, M * K * sizeof(bf16));
    bf16 *B = aligned_alloc(64, K * N * sizeof(bf16));
    float *C = aligned_alloc(64, M * N * sizeof(float));

    srand(42);
    for (int i = 0; i < M * K; i++) A[i] = float_to_bf16((rand() % 100) / 100.0f);
    for (int i = 0; i < K * N; i++) B[i] = float_to_bf16((rand() % 100) / 100.0f);
    memset(C, 0, M * N * sizeof(float));

    // Warmup
    sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

    double total = 0, best = 1e9;
    for (int t = 0; t < iters; t++) {
        memset(C, 0, M * N * sizeof(float));
        double t0 = get_time();
        sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                     M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
        double t1 = get_time();
        double dt = t1 - t0;
        total += dt;
        if (dt < best) best = dt;
    }

    double ops = 2.0 * M * N * K;
    double gflops = (ops / best) / 1e9;
    double eff = gflops / 1945.6 * 100;

    printf("Time: %.4f s (best), %.4f s (avg)\n", best, total / iters);
    printf("Performance: %.2f GFLOPS (%.2f%% of 1945.6 GFLOPS theoretical)\n\n",
           gflops, eff);

    free(A);
    free(B);
    free(C);
}

int main(int argc, char **argv) {
    // Request AMX permission
    int ret = request_amx_permission();
    if (ret < 0) {
        fprintf(stderr, "Warning: Failed to request AMX permission (ret=%d)\n", ret);
        fprintf(stderr, "AMX may not work. Continuing anyway...\n");
    } else {
        printf("AMX permission granted.\n");
    }

    int M = 4096, N = 4096, K = 4096, iters = 20;
    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    if (argc >= 5) iters = atoi(argv[4]);

    printf("=== SBGEMM v1.3.0 - AMX Tiling_B with proper init ===\n");
    printf("Blocks: K=%d, N=%d\n", BLOCK_K, BLOCK_N);
    printf("Tile: %dx%dx%d\n\n", TILE_M, TILE_N, TILE_K);

    benchmark(M, N, K, iters);
    return 0;
}
