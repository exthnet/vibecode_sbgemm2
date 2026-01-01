// sbgemm_v1.1.0.c - Optimized Cache Blocking
// PG1.2: 最適化ブロックサイズ n=480, k=1536 (reference.pdf Table 10より)
// L2キャッシュ2MB/コアに最適化

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

// BF16 <-> FP32 conversion
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

// Optimized cache blocking parameters (from reference.pdf Table 10-12)
// For L2 cache 2MB/core: n=480, k=1536 achieves best performance
#define BLOCK_K 1536
#define BLOCK_N 480

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

    // Apply beta to C
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

    // Cache-blocked matrix multiplication with optimized block sizes
    // Loop order: kk (K-blocking) -> jj (N-blocking) -> i -> j -> k
    for (int kk = 0; kk < K; kk += BLOCK_K) {
        int k_end = (kk + BLOCK_K < K) ? kk + BLOCK_K : K;

        for (int jj = 0; jj < N; jj += BLOCK_N) {
            int j_end = (jj + BLOCK_N < N) ? jj + BLOCK_N : N;

            // Inner loops over the block
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

// Performance measurement
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char *argv[]) {
    int sizes[] = {1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("Optimized Cache Blocking SBGEMM Benchmark (BLOCK_K=%d, BLOCK_N=%d)\n", BLOCK_K, BLOCK_N);
    printf("Size,Time(s),GFLOPS\n");

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

        // Initialize with random values
        srand(42);
        for (int i = 0; i < M * K; ++i) {
            A[i] = float_to_bf16_round((float)(rand() % 100) / 100.0f);
        }
        for (int i = 0; i < K * N; ++i) {
            B[i] = float_to_bf16_round((float)(rand() % 100) / 100.0f);
        }

        // Warm-up run
        sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                     M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

        // Reset C
        memset(C, 0, sizeof(float) * M * N);

        // Timed run
        double start = get_time();
        sbgemm_nolib(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                     M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
        double end = get_time();

        double elapsed = end - start;
        double flops = 2.0 * M * N * K;
        double gflops = flops / elapsed / 1e9;

        printf("%d,%.4f,%.2f\n", M, elapsed, gflops);

        free(A);
        free(B);
        free(C);
    }

    return 0;
}
