// sbgemm_mkl_v1.1.0.c
// Intel MKL cblas_sbgemm - 詳細ベンチマーク版
// PG1.9 - oneAPI 2025.1.3 / MKL
// 改良点: 複数サイズ自動テスト、詳細計測、CSV出力

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <mkl.h>

typedef uint16_t bf16;

// FP32 -> BF16 変換
static inline bf16 float_to_bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    u += 0x00008000u;
    return (bf16)(u >> 16);
}

// BF16 -> FP32 変換
static inline float bf16_to_float(bf16 x) {
    uint32_t u = ((uint32_t)x) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

// 行列初期化
void init_matrix_bf16(bf16 *mat, int rows, int cols, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < rows * cols; i++) {
        float val = (float)(rand() % 100) / 10.0f - 5.0f;
        mat[i] = float_to_bf16(val);
    }
}

// ナイーブ実装（検証用）
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
    double max_rel_err = 0.0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float mkl_val = C_mkl[i * ldc + j];
            float ref_val = C_ref[i * ldc + j];
            float diff = fabsf(mkl_val - ref_val);
            float rel_err = diff / (fabsf(ref_val) + 1e-6f);
            if (rel_err > max_rel_err) max_rel_err = rel_err;
            if (rel_err > tol) errors++;
        }
    }
    printf("  Max relative error: %.2e\n", max_rel_err);
    return errors;
}

// 単一サイズのベンチマーク
double benchmark_size(int M, int N, int K, int num_iterations, int verify) {
    bf16 *A = (bf16 *)mkl_malloc(M * K * sizeof(bf16), 64);
    bf16 *B = (bf16 *)mkl_malloc(K * N * sizeof(bf16), 64);
    float *C = (float *)mkl_malloc(M * N * sizeof(float), 64);
    float *C_ref = NULL;

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed for size %dx%dx%d\n", M, N, K);
        return -1.0;
    }

    init_matrix_bf16(A, M, K, 12345);
    init_matrix_bf16(B, K, N, 67890);
    memset(C, 0, M * N * sizeof(float));

    float alpha = 1.0f;
    float beta = 0.0f;
    int lda = K, ldb = N, ldc = N;

    // ウォームアップ
    cblas_sbgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 M, N, K, alpha,
                 (const MKL_BF16 *)A, lda,
                 (const MKL_BF16 *)B, ldb,
                 beta, C, ldc);

    // ベンチマーク
    double *times = (double *)malloc(num_iterations * sizeof(double));
    struct timespec start, end;

    for (int iter = 0; iter < num_iterations; iter++) {
        memset(C, 0, M * N * sizeof(float));
        clock_gettime(CLOCK_MONOTONIC, &start);

        cblas_sbgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                     M, N, K, alpha,
                     (const MKL_BF16 *)A, lda,
                     (const MKL_BF16 *)B, ldb,
                     beta, C, ldc);

        clock_gettime(CLOCK_MONOTONIC, &end);
        times[iter] = (end.tv_sec - start.tv_sec) +
                      (end.tv_nsec - start.tv_nsec) / 1e9;
    }

    // 統計計算
    double sum = 0.0, sum_sq = 0.0;
    double min_time = times[0], max_time = times[0];
    for (int i = 0; i < num_iterations; i++) {
        sum += times[i];
        sum_sq += times[i] * times[i];
        if (times[i] < min_time) min_time = times[i];
        if (times[i] > max_time) max_time = times[i];
    }
    double avg_time = sum / num_iterations;
    double variance = (sum_sq / num_iterations) - (avg_time * avg_time);
    double stddev = sqrt(variance > 0 ? variance : 0);

    double flops = 2.0 * M * N * K;
    double gflops = (flops / avg_time) / 1e9;
    double gflops_max = (flops / min_time) / 1e9;

    // メモリ帯域幅計算
    double bytes = (M * K + K * N) * sizeof(bf16) + M * N * sizeof(float);
    double bandwidth = (bytes / avg_time) / 1e9;

    printf("  Size: %4d x %4d x %4d\n", M, N, K);
    printf("  Avg time: %.6f sec (stddev: %.6f)\n", avg_time, stddev);
    printf("  Min time: %.6f sec, Max time: %.6f sec\n", min_time, max_time);
    printf("  Avg GFLOPS: %.2f, Peak GFLOPS: %.2f\n", gflops, gflops_max);
    printf("  Memory bandwidth: %.2f GB/s\n", bandwidth);

    // 検証
    if (verify && M <= 512 && N <= 512 && K <= 512) {
        C_ref = (float *)mkl_malloc(M * N * sizeof(float), 64);
        memset(C_ref, 0, M * N * sizeof(float));
        sbgemm_naive(M, N, K, alpha, A, lda, B, ldb, 0.0f, C_ref, ldc);
        int errors = verify_result(C, C_ref, M, N, ldc, 0.01f);
        printf("  Verification: %s (%d errors)\n", errors == 0 ? "PASSED" : "FAILED", errors);
        mkl_free(C_ref);
    }

    free(times);
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    return gflops;
}

int main(int argc, char *argv[]) {
    printf("=== MKL sbgemm Benchmark v1.1.0 ===\n");
    printf("Detailed benchmark with multiple sizes\n\n");

    mkl_set_num_threads(1);
    printf("MKL threads: %d\n", mkl_get_max_threads());
    printf("MKL version: %s\n\n", mkl_get_version_string());

    // テストサイズ配列
    int sizes[] = {128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int num_iterations = 10;

    if (argc >= 2) {
        num_iterations = atoi(argv[1]);
    }

    printf("Iterations per size: %d\n\n", num_iterations);
    printf("=== Benchmark Results ===\n");

    // CSV出力準備
    FILE *csv = fopen("benchmark_results.csv", "w");
    if (csv) {
        fprintf(csv, "M,N,K,GFLOPS,Time_sec\n");
    }

    double max_gflops = 0.0;
    int best_size = 0;

    for (int i = 0; i < num_sizes; i++) {
        int size = sizes[i];
        printf("\n--- Test %d/%d ---\n", i + 1, num_sizes);

        // 大きいサイズはイテレーション数を減らす
        int iters = (size > 2048) ? 5 : num_iterations;

        double gflops = benchmark_size(size, size, size, iters, 1);

        if (gflops > max_gflops) {
            max_gflops = gflops;
            best_size = size;
        }

        if (csv && gflops > 0) {
            double flops = 2.0 * size * size * size;
            double time = flops / (gflops * 1e9);
            fprintf(csv, "%d,%d,%d,%.2f,%.6f\n", size, size, size, gflops, time);
        }
    }

    if (csv) fclose(csv);

    printf("\n=== Summary ===\n");
    printf("Best performance: %.2f GFLOPS at size %dx%dx%d\n", max_gflops, best_size, best_size, best_size);
    printf("Results saved to: benchmark_results.csv\n");

    return 0;
}
