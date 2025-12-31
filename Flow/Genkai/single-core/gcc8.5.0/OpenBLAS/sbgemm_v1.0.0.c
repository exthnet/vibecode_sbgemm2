// sbgemm_openblas_v1.0.0.c
// OpenBLAS cblas_sgemm を利用したBF16行列積和計算
// BF16 -> FP32変換後、cblas_sgemmで計算
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cblas.h>

typedef uint16_t bf16;

typedef enum {
    CblasRowMajor_Custom = 101,
    CblasColMajor_Custom = 102
} CBLAS_LAYOUT_CUSTOM;

typedef enum {
    CblasNoTrans_Custom  = 111,
    CblasTrans_Custom    = 112,
    CblasConjTrans_Custom= 113
} CBLAS_TRANSPOSE_CUSTOM;

/* --- BF16 <-> FP32 変換 --- */

// FP32 -> BF16（簡易: round-to-nearest, ties-away）
static inline bf16 float_to_bf16_round(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    u += 0x00008000u;
    return (bf16)(u >> 16);
}

// BF16 -> FP32（下位 16bit を 0）
static inline float bf16_to_float(bf16 x) {
    uint32_t u = ((uint32_t)x) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

/* BF16配列をFP32配列に変換 */
static void bf16_to_float_array(const bf16 *src, float *dst, int n) {
    for (int i = 0; i < n; ++i) {
        dst[i] = bf16_to_float(src[i]);
    }
}

/* sbgemm_openblas: OpenBLAS cblas_sgemmを利用
 * C = alpha * op(A) * op(B) + beta * C
 * A, B: BF16配列
 * C: FP32配列
 * Row-Major前提
 */
void sbgemm_openblas(CBLAS_LAYOUT_CUSTOM layout,
                     CBLAS_TRANSPOSE_CUSTOM transA,
                     CBLAS_TRANSPOSE_CUSTOM transB,
                     int M, int N, int K,
                     float alpha,
                     const bf16 *A, int lda,
                     const bf16 *B, int ldb,
                     float beta,
                     float *C, int ldc)
{
    if (layout != CblasRowMajor_Custom) {
        fprintf(stderr, "Only CblasRowMajor is supported.\n");
        exit(1);
    }

    // Aのサイズを決定
    int A_rows, A_cols;
    if (transA == CblasNoTrans_Custom) {
        A_rows = M;
        A_cols = K;
    } else {
        A_rows = K;
        A_cols = M;
    }
    int A_size = A_rows * lda;

    // Bのサイズを決定
    int B_rows, B_cols;
    if (transB == CblasNoTrans_Custom) {
        B_rows = K;
        B_cols = N;
    } else {
        B_rows = N;
        B_cols = K;
    }
    int B_size = B_rows * ldb;

    // FP32バッファを確保
    float *A_f32 = (float*)malloc(sizeof(float) * A_size);
    float *B_f32 = (float*)malloc(sizeof(float) * B_size);
    if (!A_f32 || !B_f32) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // BF16 -> FP32 変換
    bf16_to_float_array(A, A_f32, A_size);
    bf16_to_float_array(B, B_f32, B_size);

    // OpenBLAS cblas_sgemm呼び出し
    CBLAS_TRANSPOSE transA_blas = (transA == CblasNoTrans_Custom) ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE transB_blas = (transB == CblasNoTrans_Custom) ? CblasNoTrans : CblasTrans;

    cblas_sgemm(CblasRowMajor, transA_blas, transB_blas,
                M, N, K,
                alpha,
                A_f32, lda,
                B_f32, ldb,
                beta,
                C, ldc);

    free(A_f32);
    free(B_f32);
}

/* ベンチマーク用の sbgemm_nolib（オリジナル実装） */
void sbgemm_nolib(CBLAS_LAYOUT_CUSTOM layout,
                  CBLAS_TRANSPOSE_CUSTOM transA,
                  CBLAS_TRANSPOSE_CUSTOM transB,
                  int M, int N, int K,
                  float alpha,
                  const bf16 *A, int lda,
                  const bf16 *B, int ldb,
                  float beta,
                  float *C, int ldc)
{
    if (layout != CblasRowMajor_Custom) {
        fprintf(stderr, "Only CblasRowMajor is supported.\n");
        exit(1);
    }

    // beta*C を事前適用
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

    // 3重ループ計算
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < K; ++p) {
                float a_ip, b_pj;
                if (transA == CblasNoTrans_Custom) {
                    a_ip = bf16_to_float(A[i * lda + p]);
                } else {
                    a_ip = bf16_to_float(A[p * lda + i]);
                }
                if (transB == CblasNoTrans_Custom) {
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

double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char *argv[]) {
    // テストサイズ
    int sizes[] = {100, 500, 1000, 2000, 3000, 5000, 7000, 10000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("# OpenBLAS sbgemm benchmark (BF16 -> FP32 -> cblas_sgemm)\n");
    printf("# Size, GFLOPS_OpenBLAS, Time_sec\n");

    for (int s = 0; s < num_sizes; ++s) {
        int N = sizes[s];
        int M = N, K = N;

        // メモリ確保
        bf16 *A_bf16 = (bf16*)malloc(sizeof(bf16) * M * K);
        bf16 *B_bf16 = (bf16*)malloc(sizeof(bf16) * K * N);
        float *C_f32 = (float*)calloc(M * N, sizeof(float));

        if (!A_bf16 || !B_bf16 || !C_f32) {
            fprintf(stderr, "Memory allocation failed for size %d\n", N);
            continue;
        }

        // 初期化（ランダム値をBF16に変換）
        srand(42);
        for (int i = 0; i < M * K; ++i) {
            float val = (float)(rand() % 100) / 100.0f;
            A_bf16[i] = float_to_bf16_round(val);
        }
        for (int i = 0; i < K * N; ++i) {
            float val = (float)(rand() % 100) / 100.0f;
            B_bf16[i] = float_to_bf16_round(val);
        }

        // ウォームアップ
        sbgemm_openblas(CblasRowMajor_Custom, CblasNoTrans_Custom, CblasNoTrans_Custom,
                        M, N, K, 1.0f, A_bf16, K, B_bf16, N, 0.0f, C_f32, N);

        // ベンチマーク
        int num_runs = (N <= 1000) ? 5 : (N <= 3000) ? 3 : 1;
        double total_time = 0.0;

        for (int r = 0; r < num_runs; ++r) {
            memset(C_f32, 0, sizeof(float) * M * N);
            double start = get_time_sec();
            sbgemm_openblas(CblasRowMajor_Custom, CblasNoTrans_Custom, CblasNoTrans_Custom,
                            M, N, K, 1.0f, A_bf16, K, B_bf16, N, 0.0f, C_f32, N);
            double end = get_time_sec();
            total_time += (end - start);
        }

        double avg_time = total_time / num_runs;
        double flops = 2.0 * M * N * K;
        double gflops = flops / avg_time / 1e9;

        printf("%d, %.2f, %.6f\n", N, gflops, avg_time);

        free(A_bf16);
        free(B_bf16);
        free(C_f32);
    }

    return 0;
}
