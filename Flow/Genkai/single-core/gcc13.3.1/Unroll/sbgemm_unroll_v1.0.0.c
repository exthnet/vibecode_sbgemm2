// sbgemm_unroll_v1.0.0.c
// PG1.4: ループアンローリング最適化 - Kループ4倍展開
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

/* --- BF16 <-> FP32 変換 --- */
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

/* --- 行列要素アクセス（Row-Major）--- */
static inline float get_elem_bf16(const bf16 *A, int lda, int row, int col) {
    return bf16_to_float(A[row * lda + col]);
}

static inline void set_elem_f32(float *C, int ldc, int row, int col, float v) {
    C[row * ldc + col] = v;
}

/* C = alpha * op(A) * op(B) + beta * C
 * ループアンローリング最適化版（Kループ4倍展開）
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

    // beta*C を事前処理
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

    // NoTrans/NoTransケースに最適化
    if (transA == CblasNoTrans && transB == CblasNoTrans) {
        int K4 = K - (K % 4);  // 4の倍数部分

        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;

                // Kループ4倍展開
                for (int p = 0; p < K4; p += 4) {
                    float a0 = bf16_to_float(A[i * lda + p]);
                    float a1 = bf16_to_float(A[i * lda + p + 1]);
                    float a2 = bf16_to_float(A[i * lda + p + 2]);
                    float a3 = bf16_to_float(A[i * lda + p + 3]);

                    float b0 = bf16_to_float(B[p * ldb + j]);
                    float b1 = bf16_to_float(B[(p + 1) * ldb + j]);
                    float b2 = bf16_to_float(B[(p + 2) * ldb + j]);
                    float b3 = bf16_to_float(B[(p + 3) * ldb + j]);

                    sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
                }

                // 残り処理
                for (int p = K4; p < K; ++p) {
                    float a_ip = bf16_to_float(A[i * lda + p]);
                    float b_pj = bf16_to_float(B[p * ldb + j]);
                    sum += a_ip * b_pj;
                }

                C[i * ldc + j] += alpha * sum;
            }
        }
    } else {
        // 汎用版（転置対応）
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int p = 0; p < K; ++p) {
                    float a_ip;
                    if (transA == CblasNoTrans) {
                        a_ip = bf16_to_float(A[i * lda + p]);
                    } else {
                        a_ip = bf16_to_float(A[p * lda + i]);
                    }

                    float b_pj;
                    if (transB == CblasNoTrans) {
                        b_pj = bf16_to_float(B[p * ldb + j]);
                    } else {
                        b_pj = bf16_to_float(B[j * ldb + p]);
                    }

                    sum += a_ip * b_pj;
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    }
}

// 時間計測用
static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char *argv[]) {
    int M = 1024, N = 1024, K = 1024;
    int niter = 10;

    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    if (argc > 4) niter = atoi(argv[4]);

    printf("Matrix size: M=%d, N=%d, K=%d, iterations=%d\n", M, N, K, niter);

    // メモリ確保
    bf16 *A = (bf16*)malloc(sizeof(bf16) * M * K);
    bf16 *B = (bf16*)malloc(sizeof(bf16) * K * N);
    float *C = (float*)calloc(M * N, sizeof(float));

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // 初期化（ランダム値）
    srand(42);
    for (int i = 0; i < M * K; ++i) {
        float val = (float)(rand() % 100) / 100.0f;
        A[i] = float_to_bf16_round(val);
    }
    for (int i = 0; i < K * N; ++i) {
        float val = (float)(rand() % 100) / 100.0f;
        B[i] = float_to_bf16_round(val);
    }

    // ウォームアップ
    sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

    // 計測
    double t_start = get_time();
    for (int iter = 0; iter < niter; ++iter) {
        sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                     M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    }
    double t_end = get_time();

    double elapsed = (t_end - t_start) / niter;
    double flops = 2.0 * M * N * K;  // 乗算と加算
    double gflops = flops / elapsed / 1e9;

    printf("Average time: %.6f sec\n", elapsed);
    printf("Performance: %.2f GFLOPS\n", gflops);

    // 検証（C[0][0]の値を表示）
    printf("C[0][0] = %.6f\n", C[0]);

    free(A);
    free(B);
    free(C);

    return 0;
}
