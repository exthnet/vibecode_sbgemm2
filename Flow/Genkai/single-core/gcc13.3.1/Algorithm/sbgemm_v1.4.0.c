// sbgemm_v1.4.0.c - Intel AMX Tiling_B with prefetch optimization
// Target: 65% efficiency (reference.pdf benchmark)

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

// Optimized block sizes from reference.pdf Table 10
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
    for (int i = 0; i < 6; i++) { cfg->rows[i] = 16; cfg->colsb[i] = 64; }
    cfg->rows[6] = 16; cfg->colsb[6] = 64;
    cfg->rows[7] = 16; cfg->colsb[7] = 64;
    _tile_loadconfig(cfg);
}

// Optimized packing with prefetch
static void pack_A(const bf16 *A, bf16 *packed, int M, int K, int lda,
                   int k_start, int k_len) {
    int m_tiles = (M + 15) / 16;
    int k_tiles = (k_len + 31) / 32;

    for (int mt = 0; mt < m_tiles; mt++) {
        for (int kt = 0; kt < k_tiles; kt++) {
            bf16 *dst = packed + (mt * k_tiles + kt) * 512;
            int mb = mt * 16, kb = kt * 32;

            // Prefetch next tile
            if (kt + 1 < k_tiles) {
                _mm_prefetch((char*)(A + mb * lda + k_start + kb + 32), _MM_HINT_T0);
            }

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

// VNNI packing for B with prefetch
static void pack_B(const bf16 *B, bf16 *packed, int K, int N, int ldb,
                   int k_start, int k_len, int n_start, int n_len) {
    int n_tiles = (n_len + 15) / 16;
    int k_tiles = (k_len + 31) / 32;

    for (int nt = 0; nt < n_tiles; nt++) {
        for (int kt = 0; kt < k_tiles; kt++) {
            bf16 *dst = packed + (nt * k_tiles + kt) * 512;
            int nb = nt * 16, kb = kt * 32;

            // Prefetch next row of B
            _mm_prefetch((char*)(B + (k_start + kb + 2) * ldb + n_start + nb), _MM_HINT_T0);

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

// Tiling_B kernel with prefetch optimization
static void kernel_tiling_b(const bf16 *Ap, const bf16 *Bp, float *C, int ldc,
                            int m_tiles, int n_tiles, int k_tiles) {
    const int sa = 1024, sb = 1024;
    int sc = ldc * 4;

    for (int mt = 0; mt < m_tiles; mt += 2) {
        int mt1 = (mt + 1 < m_tiles);

        for (int nt = 0; nt < n_tiles; nt += 3) {
            int nt1 = (nt + 1 < n_tiles);
            int nt2 = (nt + 2 < n_tiles);

            float *C00 = C + mt * 16 * ldc + nt * 16;

            // Load C tiles
            _tile_loadd(0, C00, sc);
            if (nt1) _tile_loadd(1, C00 + 16, sc);
            if (nt2) _tile_loadd(2, C00 + 32, sc);
            if (mt1) {
                _tile_loadd(3, C00 + 16 * ldc, sc);
                if (nt1) _tile_loadd(4, C00 + 16 * ldc + 16, sc);
                if (nt2) _tile_loadd(5, C00 + 16 * ldc + 32, sc);
            }

            for (int kt = 0; kt < k_tiles; kt++) {
                const bf16 *A0 = Ap + (mt * k_tiles + kt) * 512;
                const bf16 *A1 = Ap + ((mt + 1) * k_tiles + kt) * 512;
                const bf16 *B0 = Bp + (nt * k_tiles + kt) * 512;
                const bf16 *B1 = Bp + ((nt + 1) * k_tiles + kt) * 512;
                const bf16 *B2 = Bp + ((nt + 2) * k_tiles + kt) * 512;

                // Prefetch next A and B tiles
                if (kt + 1 < k_tiles) {
                    _mm_prefetch((char*)(A0 + 512), _MM_HINT_T0);
                    _mm_prefetch((char*)(B0 + 512), _MM_HINT_T0);
                }

                // Tiling_B order (Figure 9 in reference.pdf)
                _tile_loadd(6, A0, 64);
                _tile_loadd(7, B0, 64);
                _tile_dpbf16ps(0, 6, 7);

                if (nt1) { _tile_loadd(7, B1, 64); _tile_dpbf16ps(1, 6, 7); }
                if (nt2) { _tile_loadd(7, B2, 64); _tile_dpbf16ps(2, 6, 7); }

                if (mt1) {
                    _tile_loadd(6, A1, 64);
                    if (nt2) _tile_dpbf16ps(5, 6, 7);
                    _tile_loadd(7, B0, 64);  // L1 hit
                    _tile_dpbf16ps(3, 6, 7);
                    if (nt1) { _tile_loadd(7, B1, 64); _tile_dpbf16ps(4, 6, 7); }  // L1 hit
                }
            }

            // Store C tiles
            _tile_stored(0, C00, sc);
            if (nt1) _tile_stored(1, C00 + 16, sc);
            if (nt2) _tile_stored(2, C00 + 32, sc);
            if (mt1) {
                _tile_stored(3, C00 + 16 * ldc, sc);
                if (nt1) _tile_stored(4, C00 + 16 * ldc + 16, sc);
                if (nt2) _tile_stored(5, C00 + 16 * ldc + 32, sc);
            }
        }
    }
}

void sbgemm_nolib(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                  int M, int N, int K, float alpha,
                  const bf16 *A, int lda, const bf16 *B, int ldb,
                  float beta, float *C, int ldc) {

    if (beta == 0.0f) {
        for (int i = 0; i < M * N; i++) C[i] = 0;
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

    bf16 *Ap = aligned_alloc(64, mt * ktb * 1024);
    bf16 *Bp = aligned_alloc(64, ntb * ktb * 1024);

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        int kl = (k0 + BLOCK_K <= K) ? BLOCK_K : K - k0;
        int kt = (kl + 31) / 32;

        pack_A(A, Ap, M, K, lda, k0, kl);

        for (int n0 = 0; n0 < N; n0 += BLOCK_N) {
            int nl = (n0 + BLOCK_N <= N) ? BLOCK_N : N - n0;
            int nt = (nl + 15) / 16;

            pack_B(B, Bp, K, N, ldb, k0, kl, n0, nl);
            kernel_tiling_b(Ap, Bp, C + n0, ldc, mt, nt, kt);
        }
    }

    free(Ap); free(Bp);

    if (alpha != 1.0f) {
        for (int i = 0; i < M * N; i++) C[i] *= alpha;
    }

    _tile_release();
}

static double timer(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
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

    printf("=== v1.4.0 AMX Tiling_B + Prefetch ===\n");
    printf("Blocks: K=%d, N=%d\n\n", BLOCK_K, BLOCK_N);

    bench(M, N, K, it);
    return 0;
}
