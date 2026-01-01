// sbgemm_v1.0.1.c - Cache Blocking (小行列テスト版)
// PG1.2: 動作確認用に小行列サイズでテスト

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

#define BLOCK_K 1024
#define BLOCK_N 256

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

    for (int kk = 0; kk < K; kk += BLOCK_K) {
        int k_end = (kk + BLOCK_K < K) ? kk + BLOCK_K : K;

        for (int jj = 0; jj < N; jj += BLOCK_N) {
            int j_end = (jj + BLOCK_N < N) ? jj + BLOCK_N : N;

            for (int i = 0; i < M; ++i) {
                for (int j = jj; j < j_end; ++j) {
                    float sum = 0.0f;

                    for (int k = kk; k < k_end; ++k) {
                        float a_ik, b_kj;

                        if (transA == CblasNoTrans) {
                            uint32_t ua = ((uint32_t)A[i * lda + k]) << 16;
                            memcpy(&a_ik, &ua, sizeof(float));
                        } else {
                            uint32_t ua = ((uint32_t)A[k * lda + i]) << 16;
                            memcpy(&a_ik, &ua, sizeof(float));
                        }

                        if (transB == CblasNoTrans) {
                            uint32_t ub = ((uint32_t)B[k * ldb + j]) << 16;
                            memcpy(&b_kj, &ub, sizeof(float));
                        } else {
                            uint32_t ub = ((uint32_t)B[j * ldb + k]) << 16;
                            memcpy(&b_kj, &ub, sizeof(float));
                        }

                        sum += a_ik * b_kj;
                    }
                    C[i * ldc + j] += alpha * sum;
                }
            }
        }
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char *argv[]) {
    // 小行列サイズでテスト（時間短縮）
    int sizes[] = {500, 1000, 1500, 2000, 2500, 3000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("Cache Blocking SBGEMM Benchmark (BLOCK_K=%d, BLOCK_N=%d)\n", BLOCK_K, BLOCK_N);
    printf("Size,Time(s),GFLOPS\n");
    fflush(stdout);

    for (int s = 0; s < num_sizes; ++s) {
        int M = sizes[s], N = sizes[s], K = sizes[s];

        bf16 *A = (bf16*)malloc(sizeof(bf16) * M * K);
        bf16 *B = (bf16*)malloc(sizeof(bf16) * K * N);
        float *C = (float*)calloc(M * N, sizeof(float));

        if (!A || !B || !C) {
            fprintf(stderr, "Memory allocation failed for size %d\n", M);
            free(A); free(B); free(C);
            continue;
        }

        srand(42);
        for (int i = 0; i < M * K; ++i) {
            A[i] = float_to_bf16_round((float)(rand() % 100) / 100.0f);
        }
        for (int i = 0; i < K * N; ++i) {
            B[i] = float_to_bf16_round((float)(rand() % 100) / 100.0f);
        }

        // Timed run (no warm-up for faster execution)
        double start = get_time();
        sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                     M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
        double end = get_time();

        double elapsed = end - start;
        double flops = 2.0 * M * N * K;
        double gflops = flops / elapsed / 1e9;

        printf("%d,%.4f,%.2f\n", M, elapsed, gflops);
        fflush(stdout);

        free(A);
        free(B);
        free(C);
    }

    return 0;
}
