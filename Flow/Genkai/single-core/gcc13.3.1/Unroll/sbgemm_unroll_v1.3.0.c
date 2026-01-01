// sbgemm_unroll_v1.3.0.c
// PG1.4: 8x8タイル + __builtin_prefetch
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

void sbgemm_nolib(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
                  int M, int N, int K, float alpha, const bf16 * restrict A, int lda,
                  const bf16 * restrict B, int ldb, float beta, float * restrict C, int ldc) {
    if (layout != CblasRowMajor) { fprintf(stderr, "Only CblasRowMajor.\n"); exit(1); }

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
        int M8 = M - (M % 8);
        int N8 = N - (N % 8);

        for (int i = 0; i < M8; i += 8) {
            for (int j = 0; j < N8; j += 8) {
                // 8x8 アキュムレータ
                float c[8][8] = {{0}};

                for (int p = 0; p < K; ++p) {
                    // プリフェッチ
                    if (p + 8 < K) {
                        __builtin_prefetch(&A[i * lda + p + 8], 0, 3);
                        __builtin_prefetch(&B[(p + 8) * ldb + j], 0, 3);
                    }

                    float a[8], b[8];
                    for (int ii = 0; ii < 8; ++ii)
                        a[ii] = bf16_to_float(A[(i + ii) * lda + p]);
                    for (int jj = 0; jj < 8; ++jj)
                        b[jj] = bf16_to_float(B[p * ldb + j + jj]);

                    for (int ii = 0; ii < 8; ++ii)
                        for (int jj = 0; jj < 8; ++jj)
                            c[ii][jj] += a[ii] * b[jj];
                }

                for (int ii = 0; ii < 8; ++ii)
                    for (int jj = 0; jj < 8; ++jj)
                        C[(i + ii) * ldc + j + jj] += alpha * c[ii][jj];
            }

            // N残り
            for (int j = N8; j < N; ++j) {
                float cv[8] = {0};
                for (int p = 0; p < K; ++p) {
                    float b = bf16_to_float(B[p * ldb + j]);
                    for (int ii = 0; ii < 8; ++ii)
                        cv[ii] += bf16_to_float(A[(i + ii) * lda + p]) * b;
                }
                for (int ii = 0; ii < 8; ++ii)
                    C[(i + ii) * ldc + j] += alpha * cv[ii];
            }
        }

        // M残り
        for (int i = M8; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int p = 0; p < K; ++p)
                    sum += bf16_to_float(A[i * lda + p]) * bf16_to_float(B[p * ldb + j]);
                C[i * ldc + j] += alpha * sum;
            }
        }
    } else {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int p = 0; p < K; ++p) {
                    float a = (transA == CblasNoTrans) ? bf16_to_float(A[i * lda + p]) : bf16_to_float(A[p * lda + i]);
                    float b = (transB == CblasNoTrans) ? bf16_to_float(B[p * ldb + j]) : bf16_to_float(B[j * ldb + p]);
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
