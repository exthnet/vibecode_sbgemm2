// sbgemm_v1.3.0.c - SIMD最適化版 (AVX-512 + ループアンローリング)
// PG1.2: GCC12.2.1 + AVX-512 + レジスタブロッキングによるBFloat16 GEMM最適化
// 最適化: キャッシュブロッキング + B転置 + AVX-512 + 4x4アンローリング

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
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

// ブロックサイズ
#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 512
// アンローリング係数
#define UNROLL_M 4
#define UNROLL_N 4

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

#ifdef __AVX512F__
static inline __m512 bf16x16_to_fp32(__m256i bf16_vec) {
    __m512i extended = _mm512_cvtepu16_epi32(bf16_vec);
    __m512i shifted = _mm512_slli_epi32(extended, 16);
    return _mm512_castsi512_ps(shifted);
}

// 4x4マイクロカーネル（AVX-512）
// Cの4x4ブロックを16個のAVX-512レジスタで累積
static void microkernel_4x4_avx512(
    const bf16 *A_panel, int lda_panel,
    const bf16 *B_panel, int ldb_panel,
    float *C_block, int ldc,
    int K_len, float alpha)
{
    // 16個の累積レジスタ (4x4)
    __m512 c00 = _mm512_setzero_ps();
    __m512 c01 = _mm512_setzero_ps();
    __m512 c02 = _mm512_setzero_ps();
    __m512 c03 = _mm512_setzero_ps();
    __m512 c10 = _mm512_setzero_ps();
    __m512 c11 = _mm512_setzero_ps();
    __m512 c12 = _mm512_setzero_ps();
    __m512 c13 = _mm512_setzero_ps();
    __m512 c20 = _mm512_setzero_ps();
    __m512 c21 = _mm512_setzero_ps();
    __m512 c22 = _mm512_setzero_ps();
    __m512 c23 = _mm512_setzero_ps();
    __m512 c30 = _mm512_setzero_ps();
    __m512 c31 = _mm512_setzero_ps();
    __m512 c32 = _mm512_setzero_ps();
    __m512 c33 = _mm512_setzero_ps();

    int k = 0;
    for (; k + 15 < K_len; k += 16) {
        // Aの4行をロード
        __m256i a0_bf16 = _mm256_loadu_si256((const __m256i*)(A_panel + 0 * lda_panel + k));
        __m256i a1_bf16 = _mm256_loadu_si256((const __m256i*)(A_panel + 1 * lda_panel + k));
        __m256i a2_bf16 = _mm256_loadu_si256((const __m256i*)(A_panel + 2 * lda_panel + k));
        __m256i a3_bf16 = _mm256_loadu_si256((const __m256i*)(A_panel + 3 * lda_panel + k));

        __m512 a0 = bf16x16_to_fp32(a0_bf16);
        __m512 a1 = bf16x16_to_fp32(a1_bf16);
        __m512 a2 = bf16x16_to_fp32(a2_bf16);
        __m512 a3 = bf16x16_to_fp32(a3_bf16);

        // Bの4列をロード（転置済みなので行としてアクセス）
        __m256i b0_bf16 = _mm256_loadu_si256((const __m256i*)(B_panel + 0 * ldb_panel + k));
        __m256i b1_bf16 = _mm256_loadu_si256((const __m256i*)(B_panel + 1 * ldb_panel + k));
        __m256i b2_bf16 = _mm256_loadu_si256((const __m256i*)(B_panel + 2 * ldb_panel + k));
        __m256i b3_bf16 = _mm256_loadu_si256((const __m256i*)(B_panel + 3 * ldb_panel + k));

        __m512 b0 = bf16x16_to_fp32(b0_bf16);
        __m512 b1 = bf16x16_to_fp32(b1_bf16);
        __m512 b2 = bf16x16_to_fp32(b2_bf16);
        __m512 b3 = bf16x16_to_fp32(b3_bf16);

        // 4x4の外積累積
        c00 = _mm512_fmadd_ps(a0, b0, c00);
        c01 = _mm512_fmadd_ps(a0, b1, c01);
        c02 = _mm512_fmadd_ps(a0, b2, c02);
        c03 = _mm512_fmadd_ps(a0, b3, c03);

        c10 = _mm512_fmadd_ps(a1, b0, c10);
        c11 = _mm512_fmadd_ps(a1, b1, c11);
        c12 = _mm512_fmadd_ps(a1, b2, c12);
        c13 = _mm512_fmadd_ps(a1, b3, c13);

        c20 = _mm512_fmadd_ps(a2, b0, c20);
        c21 = _mm512_fmadd_ps(a2, b1, c21);
        c22 = _mm512_fmadd_ps(a2, b2, c22);
        c23 = _mm512_fmadd_ps(a2, b3, c23);

        c30 = _mm512_fmadd_ps(a3, b0, c30);
        c31 = _mm512_fmadd_ps(a3, b1, c31);
        c32 = _mm512_fmadd_ps(a3, b2, c32);
        c33 = _mm512_fmadd_ps(a3, b3, c33);
    }

    // 水平加算して結果をスカラーに
    float r00 = _mm512_reduce_add_ps(c00);
    float r01 = _mm512_reduce_add_ps(c01);
    float r02 = _mm512_reduce_add_ps(c02);
    float r03 = _mm512_reduce_add_ps(c03);
    float r10 = _mm512_reduce_add_ps(c10);
    float r11 = _mm512_reduce_add_ps(c11);
    float r12 = _mm512_reduce_add_ps(c12);
    float r13 = _mm512_reduce_add_ps(c13);
    float r20 = _mm512_reduce_add_ps(c20);
    float r21 = _mm512_reduce_add_ps(c21);
    float r22 = _mm512_reduce_add_ps(c22);
    float r23 = _mm512_reduce_add_ps(c23);
    float r30 = _mm512_reduce_add_ps(c30);
    float r31 = _mm512_reduce_add_ps(c31);
    float r32 = _mm512_reduce_add_ps(c32);
    float r33 = _mm512_reduce_add_ps(c33);

    // 残りの要素をスカラー処理
    for (; k < K_len; k++) {
        float a0 = bf16_to_float(A_panel[0 * lda_panel + k]);
        float a1 = bf16_to_float(A_panel[1 * lda_panel + k]);
        float a2 = bf16_to_float(A_panel[2 * lda_panel + k]);
        float a3 = bf16_to_float(A_panel[3 * lda_panel + k]);
        float b0 = bf16_to_float(B_panel[0 * ldb_panel + k]);
        float b1 = bf16_to_float(B_panel[1 * ldb_panel + k]);
        float b2 = bf16_to_float(B_panel[2 * ldb_panel + k]);
        float b3 = bf16_to_float(B_panel[3 * ldb_panel + k]);

        r00 += a0 * b0; r01 += a0 * b1; r02 += a0 * b2; r03 += a0 * b3;
        r10 += a1 * b0; r11 += a1 * b1; r12 += a1 * b2; r13 += a1 * b3;
        r20 += a2 * b0; r21 += a2 * b1; r22 += a2 * b2; r23 += a2 * b3;
        r30 += a3 * b0; r31 += a3 * b1; r32 += a3 * b2; r33 += a3 * b3;
    }

    // Cに加算
    C_block[0 * ldc + 0] += alpha * r00;
    C_block[0 * ldc + 1] += alpha * r01;
    C_block[0 * ldc + 2] += alpha * r02;
    C_block[0 * ldc + 3] += alpha * r03;
    C_block[1 * ldc + 0] += alpha * r10;
    C_block[1 * ldc + 1] += alpha * r11;
    C_block[1 * ldc + 2] += alpha * r12;
    C_block[1 * ldc + 3] += alpha * r13;
    C_block[2 * ldc + 0] += alpha * r20;
    C_block[2 * ldc + 1] += alpha * r21;
    C_block[2 * ldc + 2] += alpha * r22;
    C_block[2 * ldc + 3] += alpha * r23;
    C_block[3 * ldc + 0] += alpha * r30;
    C_block[3 * ldc + 1] += alpha * r31;
    C_block[3 * ldc + 2] += alpha * r32;
    C_block[3 * ldc + 3] += alpha * r33;
}
#endif

// B行列を転置
static void transpose_bf16(const bf16 *B, bf16 *B_T, int K, int N) {
    for (int k = 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
            B_T[n * K + k] = B[k * N + n];
        }
    }
}

// スカラーフォールバック
static inline float dot_scalar(const bf16 *a, const bf16 *b, int k) {
    float sum = 0.0f;
    for (int i = 0; i < k; i++) {
        sum += bf16_to_float(a[i]) * bf16_to_float(b[i]);
    }
    return sum;
}

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

    // beta * C の適用
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

    if (transA == CblasNoTrans && transB == CblasNoTrans) {
        bf16 *B_T = (bf16*)aligned_alloc(64, sizeof(bf16) * K * N);
        if (!B_T) {
            fprintf(stderr, "Failed to allocate B_T\n");
            exit(1);
        }
        transpose_bf16(B, B_T, K, N);

#ifdef __AVX512F__
        // ブロッキング + 4x4マイクロカーネル
        for (int i0 = 0; i0 < M; i0 += BLOCK_M) {
            int i_end = (i0 + BLOCK_M < M) ? i0 + BLOCK_M : M;

            for (int j0 = 0; j0 < N; j0 += BLOCK_N) {
                int j_end = (j0 + BLOCK_N < N) ? j0 + BLOCK_N : N;

                for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
                    int k_end = (k0 + BLOCK_K < K) ? k0 + BLOCK_K : K;
                    int k_len = k_end - k0;

                    // 4x4タイルでループ
                    int i;
                    for (i = i0; i + UNROLL_M <= i_end; i += UNROLL_M) {
                        int j;
                        for (j = j0; j + UNROLL_N <= j_end; j += UNROLL_N) {
                            microkernel_4x4_avx512(
                                &A[i * lda + k0], lda,
                                &B_T[j * K + k0], K,
                                &C[i * ldc + j], ldc,
                                k_len, alpha);
                        }
                        // 端処理 (j)
                        for (; j < j_end; j++) {
                            for (int ii = i; ii < i + UNROLL_M; ii++) {
                                float sum = dot_scalar(&A[ii * lda + k0], &B_T[j * K + k0], k_len);
                                C[ii * ldc + j] += alpha * sum;
                            }
                        }
                    }
                    // 端処理 (i)
                    for (; i < i_end; i++) {
                        for (int j = j0; j < j_end; j++) {
                            float sum = dot_scalar(&A[i * lda + k0], &B_T[j * K + k0], k_len);
                            C[i * ldc + j] += alpha * sum;
                        }
                    }
                }
            }
        }
#else
        // AVX2フォールバック（簡易版）
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = dot_scalar(&A[i * lda], &B_T[j * K], K);
                C[i * ldc + j] += alpha * sum;
            }
        }
#endif
        free(B_T);
    } else {
        // 転置ケースはスカラー
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int p = 0; p < K; ++p) {
                    float a_ip = (transA == CblasNoTrans) ?
                        bf16_to_float(A[i * lda + p]) : bf16_to_float(A[p * lda + i]);
                    float b_pj = (transB == CblasNoTrans) ?
                        bf16_to_float(B[p * ldb + j]) : bf16_to_float(B[j * ldb + p]);
                    sum += a_ip * b_pj;
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int M = 1000, N = 1000, K = 1000;
    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }

    printf("Matrix size: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Block sizes: BLOCK_M=%d, BLOCK_N=%d, BLOCK_K=%d\n", BLOCK_M, BLOCK_N, BLOCK_K);
    printf("Unroll: %dx%d microkernel\n", UNROLL_M, UNROLL_N);
#ifdef __AVX512F__
    printf("SIMD: AVX-512 enabled\n");
#else
    printf("SIMD: Scalar fallback\n");
#endif

    bf16 *A = (bf16*)aligned_alloc(64, sizeof(bf16) * M * K);
    bf16 *B = (bf16*)aligned_alloc(64, sizeof(bf16) * K * N);
    float *C = (float*)aligned_alloc(64, sizeof(float) * M * N);

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    srand(42);
    for (int i = 0; i < M * K; i++) A[i] = float_to_bf16_round((float)(rand() % 100) / 100.0f);
    for (int i = 0; i < K * N; i++) B[i] = float_to_bf16_round((float)(rand() % 100) / 100.0f);
    memset(C, 0, sizeof(float) * M * N);

    sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

    int num_runs = 5;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int run = 0; run < num_runs; run++) {
        sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                     M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double avg_time = elapsed / num_runs;
    double gflops = (2.0 * M * N * K / avg_time) / 1e9;

    printf("Average time: %.4f seconds\n", avg_time);
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("C[0][0] = %.6f\n", C[0]);

    free(A); free(B); free(C);
    return 0;
}
