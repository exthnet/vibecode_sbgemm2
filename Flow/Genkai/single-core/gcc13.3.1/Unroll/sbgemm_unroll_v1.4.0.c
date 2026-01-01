// sbgemm_unroll_v1.4.0.c
// PG1.4: ループ交換(k-i-j) + 4x4タイル
// Bの行を外側ループで再利用
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

// k-i-j順序: Bの行を外側で固定し、Aの列とCを更新
void sbgemm_nolib(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                  int M, int N, int K, float alpha, const bf16 * restrict A, int lda,
                  const bf16 * restrict B, int ldb, float beta, float * restrict C, int ldc) {
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

        // k-i-j順序: Kを外側ループに
        for (int p = 0; p < K; ++p) {
            // Bのp行目を4要素ずつ処理
            for (int j = 0; j < N4; j += 4) {
                float b0 = alpha * bf16_to_float(B[p * ldb + j]);
                float b1 = alpha * bf16_to_float(B[p * ldb + j + 1]);
                float b2 = alpha * bf16_to_float(B[p * ldb + j + 2]);
                float b3 = alpha * bf16_to_float(B[p * ldb + j + 3]);

                // Aのp列目を使ってCを更新
                for (int i = 0; i < M4; i += 4) {
                    float a0 = bf16_to_float(A[i * lda + p]);
                    float a1 = bf16_to_float(A[(i+1) * lda + p]);
                    float a2 = bf16_to_float(A[(i+2) * lda + p]);
                    float a3 = bf16_to_float(A[(i+3) * lda + p]);

                    C[i*ldc+j] += a0*b0; C[i*ldc+j+1] += a0*b1;
                    C[i*ldc+j+2] += a0*b2; C[i*ldc+j+3] += a0*b3;
                    C[(i+1)*ldc+j] += a1*b0; C[(i+1)*ldc+j+1] += a1*b1;
                    C[(i+1)*ldc+j+2] += a1*b2; C[(i+1)*ldc+j+3] += a1*b3;
                    C[(i+2)*ldc+j] += a2*b0; C[(i+2)*ldc+j+1] += a2*b1;
                    C[(i+2)*ldc+j+2] += a2*b2; C[(i+2)*ldc+j+3] += a2*b3;
                    C[(i+3)*ldc+j] += a3*b0; C[(i+3)*ldc+j+1] += a3*b1;
                    C[(i+3)*ldc+j+2] += a3*b2; C[(i+3)*ldc+j+3] += a3*b3;
                }

                // M残り
                for (int i = M4; i < M; ++i) {
                    float a = bf16_to_float(A[i * lda + p]);
                    C[i*ldc+j] += a*b0; C[i*ldc+j+1] += a*b1;
                    C[i*ldc+j+2] += a*b2; C[i*ldc+j+3] += a*b3;
                }
            }

            // N残り
            for (int j = N4; j < N; ++j) {
                float b = alpha * bf16_to_float(B[p * ldb + j]);
                for (int i = 0; i < M; ++i) {
                    C[i * ldc + j] += bf16_to_float(A[i * lda + p]) * b;
                }
            }
        }
    } else {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int p = 0; p < K; ++p) {
                    float a = (transA == CblasNoTrans) ? bf16_to_float(A[i*lda+p]) : bf16_to_float(A[p*lda+i]);
                    float b = (transB == CblasNoTrans) ? bf16_to_float(B[p*ldb+j]) : bf16_to_float(B[j*ldb+p]);
                    sum += a * b;
                }
                C[i * ldc + j] += alpha * sum;
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
