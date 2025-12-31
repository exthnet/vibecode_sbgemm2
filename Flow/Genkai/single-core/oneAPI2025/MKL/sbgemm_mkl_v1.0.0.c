// sbgemm_mkl_v1.0.0.c
// Intel MKL cblas_sbgemm を使用した BF16 行列積
// PG1.9 - oneAPI 2025.1.3 / MKL

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <mkl.h>

typedef uint16_t bf16;

// FP32 -> BF16 変換（round-to-nearest）
static inline bf16 float_to_bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    u += 0x00008000u;  // 丸め
    return (bf16)(u >> 16);
}

// BF16 -> FP32 変換
static inline float bf16_to_float(bf16 x) {
    uint32_t u = ((uint32_t)x) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

// 行列初期化（ランダム値）
void init_matrix_bf16(bf16 *mat, int rows, int cols, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < rows * cols; i++) {
        float val = (float)(rand() % 100) / 10.0f - 5.0f;  // -5.0 ~ 5.0
        mat[i] = float_to_bf16(val);
    }
}

// 結果検証用: ナイーブ実装
void sbgemm_naive(int M, int N, int K,
                  float alpha,
                  const bf16 *A, int lda,
                  const bf16 *B, int ldb,
                  float beta,
                  float *C, int ldc) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float a_val = bf16_to_float(A[i * lda + k]);
                float b_val = bf16_to_float(B[k * ldb + j]);
                sum += a_val * b_val;
            }
            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
        }
    }
}

// 相対誤差チェック
int verify_result(const float *C_mkl, const float *C_ref, int M, int N, int ldc, float tol) {
    int errors = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float mkl_val = C_mkl[i * ldc + j];
            float ref_val = C_ref[i * ldc + j];
            float diff = fabsf(mkl_val - ref_val);
            float rel_err = diff / (fabsf(ref_val) + 1e-6f);
            if (rel_err > tol) {
                if (errors < 5) {
                    printf("Mismatch at [%d,%d]: MKL=%.6f, Ref=%.6f, RelErr=%.6f\n",
                           i, j, mkl_val, ref_val, rel_err);
                }
                errors++;
            }
        }
    }
    return errors;
}

int main(int argc, char *argv[]) {
    // デフォルト行列サイズ
    int M = 1024, N = 1024, K = 1024;
    int num_iterations = 10;

    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    if (argc >= 5) {
        num_iterations = atoi(argv[4]);
    }

    printf("=== MKL sbgemm Benchmark ===\n");
    printf("Matrix size: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Iterations: %d\n", num_iterations);

    // MKLスレッド数を1に設定（single-core）
    mkl_set_num_threads(1);
    printf("MKL threads: %d\n", mkl_get_max_threads());

    // メモリ確保
    bf16 *A = (bf16 *)mkl_malloc(M * K * sizeof(bf16), 64);
    bf16 *B = (bf16 *)mkl_malloc(K * N * sizeof(bf16), 64);
    float *C = (float *)mkl_malloc(M * N * sizeof(float), 64);
    float *C_ref = (float *)mkl_malloc(M * N * sizeof(float), 64);

    if (!A || !B || !C || !C_ref) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // 行列初期化
    init_matrix_bf16(A, M, K, 12345);
    init_matrix_bf16(B, K, N, 67890);
    memset(C, 0, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));

    float alpha = 1.0f;
    float beta = 0.0f;
    int lda = K;
    int ldb = N;
    int ldc = N;

    // ウォームアップ
    cblas_sbgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 M, N, K,
                 alpha,
                 (const MKL_BF16 *)A, lda,
                 (const MKL_BF16 *)B, ldb,
                 beta,
                 C, ldc);

    // ベンチマーク実行
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int iter = 0; iter < num_iterations; iter++) {
        memset(C, 0, M * N * sizeof(float));
        cblas_sbgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                     M, N, K,
                     alpha,
                     (const MKL_BF16 *)A, lda,
                     (const MKL_BF16 *)B, ldb,
                     beta,
                     C, ldc);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    double avg_time = elapsed / num_iterations;

    // FLOPS計算: 2*M*N*K (乗算 + 加算)
    double flops = 2.0 * M * N * K;
    double gflops = (flops / avg_time) / 1e9;

    printf("\n=== Results ===\n");
    printf("Total time: %.6f sec\n", elapsed);
    printf("Average time per iteration: %.6f sec\n", avg_time);
    printf("Performance: %.2f GFLOPS\n", gflops);

    // 精度検証（小さい行列で実施）
    if (M <= 256 && N <= 256 && K <= 256) {
        printf("\n=== Verification ===\n");
        sbgemm_naive(M, N, K, alpha, A, lda, B, ldb, 0.0f, C_ref, ldc);
        int errors = verify_result(C, C_ref, M, N, ldc, 0.01f);
        if (errors == 0) {
            printf("Verification: PASSED\n");
        } else {
            printf("Verification: FAILED (%d errors)\n", errors);
        }
    }

    // メモリ解放
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
    mkl_free(C_ref);

    return 0;
}
