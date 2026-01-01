// sbgemm_v1.5.0.c - Optimized Tiling_B with minimal tile loads
// Target: 65% efficiency (reference.pdf benchmark)
// Key insight: Minimize tile register loads, maximize L1 reuse

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

#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILEDATA      18

#define TILE_M 16
#define TILE_N 16
#define TILE_K 32

// Block sizes from reference.pdf Table 10
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
typedef enum { CblasNoTrans = 111 } CBLAS_TRANSPOSE;

static inline bf16 f2bf(float x) {
    uint32_t u; memcpy(&u, &x, 4); u += 0x8000u; return (bf16)(u >> 16);
}

static inline float bf2f(bf16 x) {
    uint32_t u = ((uint32_t)x) << 16; float f; memcpy(&f, &u, 4); return f;
}

static int init_amx(void) {
    return syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
}

static void init_tiles(__tilecfg *cfg) {
    memset(cfg, 0, 64);
    cfg->palette_id = 1;
    // Tiles 0-5: C accumulators (16x16 FP32 = 1024 bytes each)
    // Tile 6: A tile (16x32 BF16 = 1024 bytes)
    // Tile 7: B tile (16x32 BF16 in VNNI format = 1024 bytes)
    for (int i = 0; i < 8; i++) {
        cfg->rows[i] = 16;
        cfg->colsb[i] = 64;
    }
    _tile_loadconfig(cfg);
}

// Pack A: row-major to tile format (16 rows x 32 cols per tile)
static void pack_A(const bf16 *A, bf16 *packed, int M, int K, int lda,
                   int k_start, int k_len) {
    int m_tiles = (M + 15) / 16;
    int k_tiles = (k_len + 31) / 32;

    for (int mt = 0; mt < m_tiles; mt++) {
        for (int kt = 0; kt < k_tiles; kt++) {
            bf16 *dst = packed + (mt * k_tiles + kt) * 512;
            int mb = mt * 16, kb = kt * 32;

            for (int i = 0; i < 16; i++) {
                int mi = mb + i;
                if (mi < M) {
                    for (int j = 0; j < 32; j++) {
                        int kj = k_start + kb + j;
                        dst[i * 32 + j] = (kj < k_start + k_len) ? A[mi * lda + kj] : 0;
                    }
                } else {
                    memset(dst + i * 32, 0, 64);
                }
            }
        }
    }
}

// Pack B: row-major to VNNI format (pairs of BF16 for dpbf16ps)
static void pack_B(const bf16 *B, bf16 *packed, int K, int N, int ldb,
                   int k_start, int k_len, int n_start, int n_len) {
    int n_tiles = (n_len + 15) / 16;
    int k_tiles = (k_len + 31) / 32;

    for (int nt = 0; nt < n_tiles; nt++) {
        for (int kt = 0; kt < k_tiles; kt++) {
            bf16 *dst = packed + (nt * k_tiles + kt) * 512;
            int nb = nt * 16, kb = kt * 32;

            // VNNI format: pairs of k values interleaved
            for (int k2 = 0; k2 < 16; k2++) {
                int k0 = k_start + kb + k2 * 2;
                int k1 = k0 + 1;
                for (int n = 0; n < 16; n++) {
                    int ni = n_start + nb + n;
                    bf16 v0 = (k0 < k_start + k_len && ni < n_start + n_len) ? B[k0 * ldb + ni] : 0;
                    bf16 v1 = (k1 < k_start + k_len && ni < n_start + n_len) ? B[k1 * ldb + ni] : 0;
                    dst[k2 * 32 + n * 2] = v0;
                    dst[k2 * 32 + n * 2 + 1] = v1;
                }
            }
        }
    }
}

// Optimized 2x3 Tiling_B kernel
// Uses tiles: 0-5 for C, 6 for A, 7 for B
// Key: Load each B tile once per k-iteration, reuse for both A rows
static void kernel_2x3(const bf16 *Ap, const bf16 *Bp, float *C, int ldc,
                       int m_tiles, int n_tiles, int k_tiles) {
    const int stride_a = k_tiles * 512;  // Stride between A tile rows
    const int stride_b = k_tiles * 512;  // Stride between B tile columns
    const int stride_c = ldc * sizeof(float);  // C stride in bytes

    for (int mt = 0; mt < m_tiles; mt += 2) {
        int has_m1 = (mt + 1 < m_tiles);

        for (int nt = 0; nt < n_tiles; nt += 3) {
            int has_n1 = (nt + 1 < n_tiles);
            int has_n2 = (nt + 2 < n_tiles);

            float *C00 = C + mt * 16 * ldc + nt * 16;

            // Load C tiles (accumulators)
            _tile_loadd(0, C00, stride_c);
            if (has_n1) _tile_loadd(1, C00 + 16, stride_c);
            if (has_n2) _tile_loadd(2, C00 + 32, stride_c);
            if (has_m1) {
                _tile_loadd(3, C00 + 16 * ldc, stride_c);
                if (has_n1) _tile_loadd(4, C00 + 16 * ldc + 16, stride_c);
                if (has_n2) _tile_loadd(5, C00 + 16 * ldc + 32, stride_c);
            }

            // K-loop: accumulate products
            for (int kt = 0; kt < k_tiles; kt++) {
                const bf16 *A0 = Ap + mt * stride_a + kt * 512;
                const bf16 *A1 = Ap + (mt + 1) * stride_a + kt * 512;
                const bf16 *B0 = Bp + nt * stride_b + kt * 512;
                const bf16 *B1 = Bp + (nt + 1) * stride_b + kt * 512;
                const bf16 *B2 = Bp + (nt + 2) * stride_b + kt * 512;

                // Tiling_B order (reference.pdf Figure 9):
                // Load A0, then cycle through B columns
                // Then load A1 and cycle through B columns (B tiles in L1)

                _tile_loadd(6, A0, 64);  // Load A row 0

                _tile_loadd(7, B0, 64);  // Load B col 0
                _tile_dpbf16ps(0, 6, 7); // C[0,0] += A0 * B0

                if (has_n1) {
                    _tile_loadd(7, B1, 64);  // Load B col 1
                    _tile_dpbf16ps(1, 6, 7); // C[0,1] += A0 * B1
                }

                if (has_n2) {
                    _tile_loadd(7, B2, 64);  // Load B col 2
                    _tile_dpbf16ps(2, 6, 7); // C[0,2] += A0 * B2
                }

                if (has_m1) {
                    _tile_loadd(6, A1, 64);  // Load A row 1

                    // B tiles should be in L1 now
                    if (has_n2) {
                        _tile_dpbf16ps(5, 6, 7); // C[1,2] += A1 * B2 (B2 still in tile 7)
                    }

                    _tile_loadd(7, B0, 64);  // Reload B col 0 (L1 hit)
                    _tile_dpbf16ps(3, 6, 7); // C[1,0] += A1 * B0

                    if (has_n1) {
                        _tile_loadd(7, B1, 64);  // Reload B col 1 (L1 hit)
                        _tile_dpbf16ps(4, 6, 7); // C[1,1] += A1 * B1
                    }
                }
            }

            // Store C tiles
            _tile_stored(0, C00, stride_c);
            if (has_n1) _tile_stored(1, C00 + 16, stride_c);
            if (has_n2) _tile_stored(2, C00 + 32, stride_c);
            if (has_m1) {
                _tile_stored(3, C00 + 16 * ldc, stride_c);
                if (has_n1) _tile_stored(4, C00 + 16 * ldc + 16, stride_c);
                if (has_n2) _tile_stored(5, C00 + 16 * ldc + 32, stride_c);
            }
        }
    }
}

void sbgemm_nolib(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                  int M, int N, int K, float alpha,
                  const bf16 *A, int lda, const bf16 *B, int ldb,
                  float beta, float *C, int ldc) {

    // Apply beta scaling
    if (beta == 0.0f) {
        memset(C, 0, M * N * sizeof(float));
    } else if (beta != 1.0f) {
        for (int i = 0; i < M * N; i++) C[i] *= beta;
    }

    __tilecfg cfg;
    init_tiles(&cfg);

    int mt = (M + 15) / 16;
    int kb = (K < BLOCK_K) ? K : BLOCK_K;
    int nb = (N < BLOCK_N) ? N : BLOCK_N;
    int ktb = (kb + 31) / 32;
    int ntb = (nb + 15) / 16;

    // Allocate packed buffers
    bf16 *Ap = aligned_alloc(64, mt * ktb * 1024);
    bf16 *Bp = aligned_alloc(64, ntb * ktb * 1024);

    // Block over K (outer), then N
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        int kl = (k0 + BLOCK_K <= K) ? BLOCK_K : K - k0;
        int kt = (kl + 31) / 32;

        pack_A(A, Ap, M, K, lda, k0, kl);

        for (int n0 = 0; n0 < N; n0 += BLOCK_N) {
            int nl = (n0 + BLOCK_N <= N) ? BLOCK_N : N - n0;
            int nt = (nl + 15) / 16;

            pack_B(B, Bp, K, N, ldb, k0, kl, n0, nl);
            kernel_2x3(Ap, Bp, C + n0, ldc, mt, nt, kt);
        }
    }

    free(Ap);
    free(Bp);

    // Apply alpha scaling
    if (alpha != 1.0f) {
        for (int i = 0; i < M * N; i++) C[i] *= alpha;
    }

    _tile_release();
}

static double timer(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void bench(int M, int N, int K, int iters) {
    printf("Matrix: %dx%dx%d\n", M, N, K);

    bf16 *A = aligned_alloc(64, M * K * 2);
    bf16 *B = aligned_alloc(64, K * N * 2);
    float *C = aligned_alloc(64, M * N * 4);

    srand(42);
    for (int i = 0; i < M * K; i++) A[i] = f2bf((rand() % 100) / 100.0f);
    for (int i = 0; i < K * N; i++) B[i] = f2bf((rand() % 100) / 100.0f);
    memset(C, 0, M * N * 4);

    // Warmup
    sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, K, B, N, 0, C, N);

    double tot = 0, best = 1e9;
    for (int t = 0; t < iters; t++) {
        memset(C, 0, M * N * 4);
        double t0 = timer();
        sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, K, B, N, 0, C, N);
        double dt = timer() - t0;
        tot += dt;
        if (dt < best) best = dt;
    }

    double gf = (2.0 * M * N * K / best) / 1e9;
    printf("Time: %.4f s, GFLOPS: %.2f (%.2f%% of 1945.6)\n\n", best, gf, gf / 19.456);

    free(A); free(B); free(C);
}

int main(int argc, char **argv) {
    if (init_amx() < 0) printf("AMX init warning\n");
    else printf("AMX OK\n");

    int M = 4096, N = 4096, K = 4096, it = 20;
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    if (argc >= 5) it = atoi(argv[4]);

    printf("=== v1.5.0 Optimized Tiling_B ===\n");
    printf("Blocks: K=%d, N=%d\n\n", BLOCK_K, BLOCK_N);

    bench(M, N, K, it);
    return 0;
}
