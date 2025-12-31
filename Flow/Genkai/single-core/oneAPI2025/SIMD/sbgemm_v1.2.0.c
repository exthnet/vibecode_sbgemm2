// sbgemm_v1.2.0.c - SIMD最適化版 BF16 GEMM (高度最適化)
// PG1.8 - Intel oneAPI 2025 / AVX-512
//
// v1.1.0からの改善:
// - 2x2タイリング（Cの2行x2列を同時計算）
// - レジスタブロッキング強化
// - 依存関係を減らしたパイプライン効率向上

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

static inline __m512 bf16x16_to_fp32(__m256i bf16_vec) {
    __m512i shifted = _mm512_cvtepu16_epi32(bf16_vec);
    shifted = _mm512_slli_epi32(shifted, 16);
    return _mm512_castsi512_ps(shifted);
}

/* --- キャッシュブロッキング定数 (v1.2.0) --- */
// L2キャッシュ(2MB)とL1キャッシュ(32KB)を考慮
#define BLOCK_M 96   // reference.pdf: m=96
#define BLOCK_N 64
#define BLOCK_K 512

#define PREFETCH_L1(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)

/* --- 2x2タイリング マイクロカーネル --- */
// Cの2行x2列(各16要素)を同時に計算
// 合計4つのAVX-512レジスタを結果用に使用
static inline void microkernel_2x2(
    int K_block,
    float alpha,
    const bf16 *A_ptr0, const bf16 *A_ptr1, int lda,
    const bf16 *B_ptr,  int ldb,
    float *C_ptr0, float *C_ptr1, int ldc)
{
    // C[i,j:j+16], C[i,j+16:j+32], C[i+1,j:j+16], C[i+1,j+16:j+32]
    __m512 c00 = _mm512_loadu_ps(C_ptr0);
    __m512 c01 = _mm512_loadu_ps(C_ptr0 + 16);
    __m512 c10 = _mm512_loadu_ps(C_ptr1);
    __m512 c11 = _mm512_loadu_ps(C_ptr1 + 16);

    for (int k = 0; k < K_block; ++k) {
        // A[i,k], A[i+1,k]をブロードキャスト
        float a0 = bf16_to_float(A_ptr0[k]) * alpha;
        float a1 = bf16_to_float(A_ptr1[k]) * alpha;
        __m512 a0_vec = _mm512_set1_ps(a0);
        __m512 a1_vec = _mm512_set1_ps(a1);

        // B[k,j:j+16], B[k,j+16:j+32]
        __m256i b_lo_bf16 = _mm256_loadu_si256((const __m256i*)(B_ptr + k * ldb));
        __m256i b_hi_bf16 = _mm256_loadu_si256((const __m256i*)(B_ptr + k * ldb + 16));
        __m512 b0 = bf16x16_to_fp32(b_lo_bf16);
        __m512 b1 = bf16x16_to_fp32(b_hi_bf16);

        // プリフェッチ
        if (k + 2 < K_block) {
            PREFETCH_L1(B_ptr + (k + 2) * ldb);
        }

        // FMA: 4つの積和演算
        c00 = _mm512_fmadd_ps(a0_vec, b0, c00);
        c01 = _mm512_fmadd_ps(a0_vec, b1, c01);
        c10 = _mm512_fmadd_ps(a1_vec, b0, c10);
        c11 = _mm512_fmadd_ps(a1_vec, b1, c11);
    }

    _mm512_storeu_ps(C_ptr0, c00);
    _mm512_storeu_ps(C_ptr0 + 16, c01);
    _mm512_storeu_ps(C_ptr1, c10);
    _mm512_storeu_ps(C_ptr1 + 16, c11);
}

/* --- SIMD最適化版 sbgemm v1.2.0 (NoTrans, NoTrans専用) --- */
void sbgemm_simd_nn_v12(int M, int N, int K,
                        float alpha,
                        const bf16 *A, int lda,
                        const bf16 *B, int ldb,
                        float beta,
                        float *C, int ldc)
{
    // beta処理
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

    // キャッシュブロッキングループ
    for (int kb = 0; kb < K; kb += BLOCK_K) {
        int k_block = (kb + BLOCK_K < K) ? BLOCK_K : K - kb;

        for (int ib = 0; ib < M; ib += BLOCK_M) {
            int i_end = (ib + BLOCK_M < M) ? ib + BLOCK_M : M;

            for (int jb = 0; jb < N; jb += BLOCK_N) {
                int j_end = (jb + BLOCK_N < N) ? jb + BLOCK_N : N;

                // 2x2タイリング: 2行ずつ、32列ずつ処理
                int i = ib;
                for (; i + 2 <= i_end; i += 2) {
                    int j = jb;
                    for (; j + 32 <= j_end; j += 32) {
                        microkernel_2x2(
                            k_block, alpha,
                            &A[i * lda + kb], &A[(i + 1) * lda + kb], lda,
                            &B[kb * ldb + j], ldb,
                            &C[i * ldc + j], &C[(i + 1) * ldc + j], ldc
                        );
                    }

                    // 残りの列: 16要素単位
                    for (; j + 16 <= j_end; j += 16) {
                        __m512 c0 = _mm512_loadu_ps(&C[i * ldc + j]);
                        __m512 c1 = _mm512_loadu_ps(&C[(i + 1) * ldc + j]);

                        for (int k = kb; k < kb + k_block; ++k) {
                            float a0 = bf16_to_float(A[i * lda + k]) * alpha;
                            float a1 = bf16_to_float(A[(i + 1) * lda + k]) * alpha;
                            __m256i b_bf16 = _mm256_loadu_si256(
                                (const __m256i*)&B[k * ldb + j]);
                            __m512 b = bf16x16_to_fp32(b_bf16);

                            c0 = _mm512_fmadd_ps(_mm512_set1_ps(a0), b, c0);
                            c1 = _mm512_fmadd_ps(_mm512_set1_ps(a1), b, c1);
                        }

                        _mm512_storeu_ps(&C[i * ldc + j], c0);
                        _mm512_storeu_ps(&C[(i + 1) * ldc + j], c1);
                    }

                    // 残りのスカラ
                    for (; j < j_end; ++j) {
                        float sum0 = C[i * ldc + j];
                        float sum1 = C[(i + 1) * ldc + j];
                        for (int k = kb; k < kb + k_block; ++k) {
                            float a0 = bf16_to_float(A[i * lda + k]) * alpha;
                            float a1 = bf16_to_float(A[(i + 1) * lda + k]) * alpha;
                            float b_val = bf16_to_float(B[k * ldb + j]);
                            sum0 += a0 * b_val;
                            sum1 += a1 * b_val;
                        }
                        C[i * ldc + j] = sum0;
                        C[(i + 1) * ldc + j] = sum1;
                    }
                }

                // 残りの行（1行）
                for (; i < i_end; ++i) {
                    int j = jb;
                    for (; j + 16 <= j_end; j += 16) {
                        __m512 sum = _mm512_loadu_ps(&C[i * ldc + j]);
                        for (int k = kb; k < kb + k_block; ++k) {
                            float a_val = bf16_to_float(A[i * lda + k]) * alpha;
                            __m256i b_bf16 = _mm256_loadu_si256(
                                (const __m256i*)&B[k * ldb + j]);
                            __m512 b = bf16x16_to_fp32(b_bf16);
                            sum = _mm512_fmadd_ps(_mm512_set1_ps(a_val), b, sum);
                        }
                        _mm512_storeu_ps(&C[i * ldc + j], sum);
                    }
                    for (; j < j_end; ++j) {
                        float sum = C[i * ldc + j];
                        for (int k = kb; k < kb + k_block; ++k) {
                            float a_val = bf16_to_float(A[i * lda + k]) * alpha;
                            float b_val = bf16_to_float(B[k * ldb + j]);
                            sum += a_val * b_val;
                        }
                        C[i * ldc + j] = sum;
                    }
                }
            }
        }
    }
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

    if (transA == CblasNoTrans && transB == CblasNoTrans) {
        sbgemm_simd_nn_v12(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }

    // スカラ版フォールバック
    if (beta == 0.0f) {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                C[i * ldc + j] = 0.0f;
    } else if (beta != 1.0f) {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                C[i * ldc + j] *= beta;
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < K; ++p) {
                float a_ip, b_pj;
                uint32_t ua = ((uint32_t)A[(transA == CblasNoTrans) ? i * lda + p : p * lda + i]) << 16;
                uint32_t ub = ((uint32_t)B[(transB == CblasNoTrans) ? p * ldb + j : j * ldb + p]) << 16;
                memcpy(&a_ip, &ua, sizeof(float));
                memcpy(&b_pj, &ub, sizeof(float));
                sum += a_ip * b_pj;
            }
            C[i * ldc + j] += alpha * sum;
        }
    }
}

int main(int argc, char *argv[]) {
    int M = 1024, N = 1024, K = 1024;
    int num_trials = 5;

    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    if (argc >= 5) {
        num_trials = atoi(argv[4]);
    }

    printf("SBGEMM SIMD v1.2.0 Benchmark (oneAPI 2025)\n");
    printf("Improvements: 2x2 Tiling + Register blocking\n");
    printf("Matrix size: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Trials: %d\n", num_trials);

    bf16 *A = (bf16*)aligned_alloc(64, sizeof(bf16) * M * K);
    bf16 *B = (bf16*)aligned_alloc(64, sizeof(bf16) * K * N);
    float *C = (float*)aligned_alloc(64, sizeof(float) * M * N);

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    srand(42);
    for (int i = 0; i < M * K; ++i) A[i] = float_to_bf16_round((float)(rand() % 100) / 100.0f);
    for (int i = 0; i < K * N; ++i) B[i] = float_to_bf16_round((float)(rand() % 100) / 100.0f);

    memset(C, 0, sizeof(float) * M * N);
    sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

    double total_time = 0.0, min_time = 1e9;

    for (int t = 0; t < num_trials; ++t) {
        memset(C, 0, sizeof(float) * M * N);
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
        total_time += elapsed;
        if (elapsed < min_time) min_time = elapsed;
        printf("Trial %d: %.4f sec\n", t + 1, elapsed);
    }

    double avg_time = total_time / num_trials;
    double flops = 2.0 * M * N * K;
    printf("\n=== Results ===\n");
    printf("Average time: %.4f sec\n", avg_time);
    printf("Min time:     %.4f sec\n", min_time);
    printf("Average GFLOPS: %.2f\n", flops / avg_time / 1e9);
    printf("Peak GFLOPS:    %.2f\n", flops / min_time / 1e9);

    printf("\nC[0:3,0:3]:\n");
    for (int i = 0; i < 3 && i < M; ++i) {
        for (int j = 0; j < 3 && j < N; ++j) printf("%8.4f ", C[i * N + j]);
        printf("\n");
    }

    free(A); free(B); free(C);
    return 0;
}
