// sbgemm_unroll_v1.2.0.c
// PG1.4: 4x4タイル + ループ交換(i-k-j)
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef uint16_t bf16;
typedef enum { CblasRowMajor = 101, CblasColMajor = 102 } CBLAS_LAYOUT;
typedef enum { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 } CBLAS_TRANSPOSE;

static inline bf16 float_to_bf16_round(float x) {
    uint32_t u; memcpy(&u, &x, sizeof(u)); u += 0x00008000u; return (bf16)(u >> 16);
}
static inline float bf16_to_float(bf16 x) {
    uint32_t u = ((uint32_t)x) << 16; float f; memcpy(&f, &u, sizeof(f)); return f;
}

// 4x4タイル + ループ交換(i-k-j)でAのキャッシュ効率向上
void sbgemm_nolib(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                  int M, int N, int K, float alpha, const bf16 *A, int lda,
                  const bf16 *B, int ldb, float beta, float *C, int ldc) {
    if (layout != CblasRowMajor) { fprintf(stderr, "Only CblasRowMajor.\n"); exit(1); }

    // beta*C処理
    if (beta == 0.0f) {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                C[i * ldc + j] = 0.0f;
    } else if (beta != 1.0f) {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                C[i * ldc + j] *= beta;
    }

    if (transA == CblasNoTrans && transB == CblasNoTrans) {
        int M4 = M - (M % 4);
        int N4 = N - (N % 4);

        // メイン: 4x4タイル、i-k-j順序
        for (int i = 0; i < M4; i += 4) {
            for (int j = 0; j < N4; j += 4) {
                // 4x4 アキュムレータ
                float c00=0,c01=0,c02=0,c03=0;
                float c10=0,c11=0,c12=0,c13=0;
                float c20=0,c21=0,c22=0,c23=0;
                float c30=0,c31=0,c32=0,c33=0;

                for (int p = 0; p < K; ++p) {
                    // Aの4要素を読み込み（連続アクセス）
                    float a0 = bf16_to_float(A[i * lda + p]);
                    float a1 = bf16_to_float(A[(i+1) * lda + p]);
                    float a2 = bf16_to_float(A[(i+2) * lda + p]);
                    float a3 = bf16_to_float(A[(i+3) * lda + p]);

                    // Bの4要素を読み込み
                    float b0 = bf16_to_float(B[p * ldb + j]);
                    float b1 = bf16_to_float(B[p * ldb + j + 1]);
                    float b2 = bf16_to_float(B[p * ldb + j + 2]);
                    float b3 = bf16_to_float(B[p * ldb + j + 3]);

                    // 外積: 4x4
                    c00 += a0*b0; c01 += a0*b1; c02 += a0*b2; c03 += a0*b3;
                    c10 += a1*b0; c11 += a1*b1; c12 += a1*b2; c13 += a1*b3;
                    c20 += a2*b0; c21 += a2*b1; c22 += a2*b2; c23 += a2*b3;
                    c30 += a3*b0; c31 += a3*b1; c32 += a3*b2; c33 += a3*b3;
                }

                // Cに書き込み
                C[i*ldc+j] += alpha*c00; C[i*ldc+j+1] += alpha*c01;
                C[i*ldc+j+2] += alpha*c02; C[i*ldc+j+3] += alpha*c03;
                C[(i+1)*ldc+j] += alpha*c10; C[(i+1)*ldc+j+1] += alpha*c11;
                C[(i+1)*ldc+j+2] += alpha*c12; C[(i+1)*ldc+j+3] += alpha*c13;
                C[(i+2)*ldc+j] += alpha*c20; C[(i+2)*ldc+j+1] += alpha*c21;
                C[(i+2)*ldc+j+2] += alpha*c22; C[(i+2)*ldc+j+3] += alpha*c23;
                C[(i+3)*ldc+j] += alpha*c30; C[(i+3)*ldc+j+1] += alpha*c31;
                C[(i+3)*ldc+j+2] += alpha*c32; C[(i+3)*ldc+j+3] += alpha*c33;
            }

            // N残り
            for (int j = N4; j < N; ++j) {
                float c0=0,c1=0,c2=0,c3=0;
                for (int p = 0; p < K; ++p) {
                    float a0 = bf16_to_float(A[i*lda+p]);
                    float a1 = bf16_to_float(A[(i+1)*lda+p]);
                    float a2 = bf16_to_float(A[(i+2)*lda+p]);
                    float a3 = bf16_to_float(A[(i+3)*lda+p]);
                    float b = bf16_to_float(B[p*ldb+j]);
                    c0 += a0*b; c1 += a1*b; c2 += a2*b; c3 += a3*b;
                }
                C[i*ldc+j] += alpha*c0; C[(i+1)*ldc+j] += alpha*c1;
                C[(i+2)*ldc+j] += alpha*c2; C[(i+3)*ldc+j] += alpha*c3;
            }
        }

        // M残り
        for (int i = M4; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int p = 0; p < K; ++p)
                    sum += bf16_to_float(A[i*lda+p]) * bf16_to_float(B[p*ldb+j]);
                C[i*ldc+j] += alpha * sum;
            }
        }
    } else {
        // 汎用版
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int p = 0; p < K; ++p) {
                    float a = (transA==CblasNoTrans) ? bf16_to_float(A[i*lda+p]) : bf16_to_float(A[p*lda+i]);
                    float b = (transB==CblasNoTrans) ? bf16_to_float(B[p*ldb+j]) : bf16_to_float(B[j*ldb+p]);
                    sum += a * b;
                }
                C[i*ldc+j] += alpha * sum;
            }
        }
    }
}

static double get_time(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char *argv[]) {
    int M = 1024, N = 1024, K = 1024, niter = 10;
    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    if (argc > 4) niter = atoi(argv[4]);

    printf("Matrix size: M=%d, N=%d, K=%d, iterations=%d\n", M, N, K, niter);

    bf16 *A = (bf16*)malloc(sizeof(bf16) * M * K);
    bf16 *B = (bf16*)malloc(sizeof(bf16) * K * N);
    float *C = (float*)calloc(M * N, sizeof(float));
    if (!A || !B || !C) { fprintf(stderr, "Memory allocation failed\n"); return 1; }

    srand(42);
    for (int i = 0; i < M * K; ++i) A[i] = float_to_bf16_round((float)(rand() % 100) / 100.0f);
    for (int i = 0; i < K * N; ++i) B[i] = float_to_bf16_round((float)(rand() % 100) / 100.0f);

    sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

    double t_start = get_time();
    for (int iter = 0; iter < niter; ++iter)
        sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    double t_end = get_time();

    double elapsed = (t_end - t_start) / niter;
    double flops = 2.0 * M * N * K;
    double gflops = flops / elapsed / 1e9;

    printf("Average time: %.6f sec\n", elapsed);
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("C[0][0] = %.6f\n", C[0]);

    free(A); free(B); free(C);
    return 0;
}
