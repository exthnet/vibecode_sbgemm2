// sbgemm_v1.2.0.c - Optimized AVX-512 Implementation
// Focus: Efficient register blocking, cache optimization, loop unrolling

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <sys/time.h>

typedef uint16_t bf16;

// Optimized blocking parameters
#define BLOCK_M 48      // Smaller M block for better L1 utilization
#define BLOCK_N 256     // N block fits in L2 with B data
#define BLOCK_K 512     // K block for inner loop

// Register blocking: 6 rows x 32 columns (2 AVX-512 vectors)
#define MR 6
#define NR 32

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

// Convert 16 BF16 values to FP32 AVX-512 vector
static inline __m512 cvt_bf16_fp32(__m256i bf16_vec) {
    __m512i ext = _mm512_cvtepu16_epi32(bf16_vec);
    return _mm512_castsi512_ps(_mm512_slli_epi32(ext, 16));
}

// High-performance micro-kernel: 6x32 output tile
// C[0:6][0:32] += A[0:6][0:k] * B[0:k][0:32]
static void microkernel_6x32(
    int k,
    const bf16 * restrict A, int lda,
    const bf16 * restrict B, int ldb,
    float * restrict C, int ldc)
{
    // 6 rows x 2 vectors (32 floats) = 12 accumulators
    __m512 c00 = _mm512_loadu_ps(C + 0*ldc);
    __m512 c01 = _mm512_loadu_ps(C + 0*ldc + 16);
    __m512 c10 = _mm512_loadu_ps(C + 1*ldc);
    __m512 c11 = _mm512_loadu_ps(C + 1*ldc + 16);
    __m512 c20 = _mm512_loadu_ps(C + 2*ldc);
    __m512 c21 = _mm512_loadu_ps(C + 2*ldc + 16);
    __m512 c30 = _mm512_loadu_ps(C + 3*ldc);
    __m512 c31 = _mm512_loadu_ps(C + 3*ldc + 16);
    __m512 c40 = _mm512_loadu_ps(C + 4*ldc);
    __m512 c41 = _mm512_loadu_ps(C + 4*ldc + 16);
    __m512 c50 = _mm512_loadu_ps(C + 5*ldc);
    __m512 c51 = _mm512_loadu_ps(C + 5*ldc + 16);

    for (int p = 0; p < k; p++) {
        // Load B row: 32 BF16 values -> 2 FP32 vectors
        __m256i b0_bf16 = _mm256_loadu_si256((const __m256i*)(B + p*ldb));
        __m256i b1_bf16 = _mm256_loadu_si256((const __m256i*)(B + p*ldb + 16));
        __m512 b0 = cvt_bf16_fp32(b0_bf16);
        __m512 b1 = cvt_bf16_fp32(b1_bf16);

        // Prefetch next B row
        _mm_prefetch((const char*)(B + (p+4)*ldb), _MM_HINT_T0);

        // Broadcast A values and FMA
        __m512 a0 = _mm512_set1_ps(bf16_to_float(A[0*lda + p]));
        c00 = _mm512_fmadd_ps(a0, b0, c00);
        c01 = _mm512_fmadd_ps(a0, b1, c01);

        __m512 a1 = _mm512_set1_ps(bf16_to_float(A[1*lda + p]));
        c10 = _mm512_fmadd_ps(a1, b0, c10);
        c11 = _mm512_fmadd_ps(a1, b1, c11);

        __m512 a2 = _mm512_set1_ps(bf16_to_float(A[2*lda + p]));
        c20 = _mm512_fmadd_ps(a2, b0, c20);
        c21 = _mm512_fmadd_ps(a2, b1, c21);

        __m512 a3 = _mm512_set1_ps(bf16_to_float(A[3*lda + p]));
        c30 = _mm512_fmadd_ps(a3, b0, c30);
        c31 = _mm512_fmadd_ps(a3, b1, c31);

        __m512 a4 = _mm512_set1_ps(bf16_to_float(A[4*lda + p]));
        c40 = _mm512_fmadd_ps(a4, b0, c40);
        c41 = _mm512_fmadd_ps(a4, b1, c41);

        __m512 a5 = _mm512_set1_ps(bf16_to_float(A[5*lda + p]));
        c50 = _mm512_fmadd_ps(a5, b0, c50);
        c51 = _mm512_fmadd_ps(a5, b1, c51);
    }

    // Store results
    _mm512_storeu_ps(C + 0*ldc, c00);
    _mm512_storeu_ps(C + 0*ldc + 16, c01);
    _mm512_storeu_ps(C + 1*ldc, c10);
    _mm512_storeu_ps(C + 1*ldc + 16, c11);
    _mm512_storeu_ps(C + 2*ldc, c20);
    _mm512_storeu_ps(C + 2*ldc + 16, c21);
    _mm512_storeu_ps(C + 3*ldc, c30);
    _mm512_storeu_ps(C + 3*ldc + 16, c31);
    _mm512_storeu_ps(C + 4*ldc, c40);
    _mm512_storeu_ps(C + 4*ldc + 16, c41);
    _mm512_storeu_ps(C + 5*ldc, c50);
    _mm512_storeu_ps(C + 5*ldc + 16, c51);
}

// Flexible micro-kernel for edge cases
static void microkernel_flexible(
    int m, int n, int k,
    const bf16 * restrict A, int lda,
    const bf16 * restrict B, int ldb,
    float * restrict C, int ldc)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j += 16) {
            int jlen = (j + 16 <= n) ? 16 : (n - j);

            if (jlen == 16) {
                __m512 c_vec = _mm512_loadu_ps(C + i*ldc + j);
                for (int p = 0; p < k; p++) {
                    float a_val = bf16_to_float(A[i*lda + p]);
                    __m256i b_bf16 = _mm256_loadu_si256((const __m256i*)(B + p*ldb + j));
                    __m512 b_vec = cvt_bf16_fp32(b_bf16);
                    c_vec = _mm512_fmadd_ps(_mm512_set1_ps(a_val), b_vec, c_vec);
                }
                _mm512_storeu_ps(C + i*ldc + j, c_vec);
            } else {
                // Scalar for remaining elements
                for (int jj = j; jj < n; jj++) {
                    float sum = C[i*ldc + jj];
                    for (int p = 0; p < k; p++) {
                        sum += bf16_to_float(A[i*lda + p]) * bf16_to_float(B[p*ldb + jj]);
                    }
                    C[i*ldc + jj] = sum;
                }
            }
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
                C[i*ldc + j] = 0.0f;
    } else if (beta != 1.0f) {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                C[i*ldc + j] *= beta;
    }

    // Three-level blocking: L3 -> L2 -> L1/registers
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        int kb = (k0 + BLOCK_K <= K) ? BLOCK_K : (K - k0);

        for (int j0 = 0; j0 < N; j0 += BLOCK_N) {
            int nb = (j0 + BLOCK_N <= N) ? BLOCK_N : (N - j0);

            for (int i0 = 0; i0 < M; i0 += BLOCK_M) {
                int mb = (i0 + BLOCK_M <= M) ? BLOCK_M : (M - i0);

                // Micro-kernel dispatch
                for (int i = 0; i < mb; i += MR) {
                    int mr = (i + MR <= mb) ? MR : (mb - i);

                    for (int j = 0; j < nb; j += NR) {
                        int nr = (j + NR <= nb) ? NR : (nb - j);

                        if (mr == MR && nr == NR) {
                            microkernel_6x32(kb,
                                A + (i0+i)*lda + k0, lda,
                                B + k0*ldb + (j0+j), ldb,
                                C + (i0+i)*ldc + (j0+j), ldc);
                        } else {
                            microkernel_flexible(mr, nr, kb,
                                A + (i0+i)*lda + k0, lda,
                                B + k0*ldb + (j0+j), ldb,
                                C + (i0+i)*ldc + (j0+j), ldc);
                        }
                    }
                }
            }
        }
    }

    // Apply alpha
    if (alpha != 1.0f) {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                C[i*ldc + j] *= alpha;
    }
}

static double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void benchmark(int M, int N, int K, int iters) {
    printf("Matrix: %dx%dx%d\n", M, N, K);

    bf16 *A = aligned_alloc(64, M*K*sizeof(bf16));
    bf16 *B = aligned_alloc(64, K*N*sizeof(bf16));
    float *C = aligned_alloc(64, M*N*sizeof(float));

    srand(42);
    for (int i = 0; i < M*K; i++) A[i] = float_to_bf16((rand()%100)/100.0f);
    for (int i = 0; i < K*N; i++) B[i] = float_to_bf16((rand()%100)/100.0f);
    memset(C, 0, M*N*sizeof(float));

    // Warmup
    sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

    double total = 0, best = 1e9;
    for (int t = 0; t < iters; t++) {
        memset(C, 0, M*N*sizeof(float));
        double t0 = get_time();
        sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                     M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
        double t1 = get_time();
        double dt = t1 - t0;
        total += dt;
        if (dt < best) best = dt;
    }

    double ops = 2.0*M*N*K;
    double gflops = (ops/best)/1e9;
    double eff = gflops/1945.6*100;  // vs AMX theoretical

    printf("Time: %.4f s (best), %.4f s (avg)\n", best, total/iters);
    printf("Performance: %.2f GFLOPS (%.2f%% of 1945.6 GFLOPS)\n\n", gflops, eff);

    free(A); free(B); free(C);
}

int main(int argc, char **argv) {
    int M = 4096, N = 4096, K = 4096, iters = 20;
    if (argc >= 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    if (argc >= 5) iters = atoi(argv[4]);

    printf("=== SBGEMM v1.2.0 - Optimized AVX-512 ===\n");
    printf("Blocks: M=%d, N=%d, K=%d, Register: %dx%d\n\n",
           BLOCK_M, BLOCK_N, BLOCK_K, MR, NR);

    benchmark(M, N, K, iters);
    return 0;
}
