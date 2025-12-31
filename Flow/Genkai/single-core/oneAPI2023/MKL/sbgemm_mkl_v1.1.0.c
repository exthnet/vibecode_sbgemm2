// sbgemm_mkl_v1.1.0.c
// Intel MKL cblas_gemm_bf16bf16f32 を利用したBF16行列積
// PG1.5 - oneAPI 2023.2

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <mkl.h>

// FP32 -> BF16 変換（round-to-nearest）
static inline MKL_BF16 float_to_bf16(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    u += 0x00008000u;  // round
    return (MKL_BF16)(u >> 16);
}

// BF16 -> FP32 変換
static inline float bf16_to_float(MKL_BF16 x) {
    uint32_t u = ((uint32_t)x) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

// 時間計測用
static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char *argv[]) {
    // 行列サイズ（デフォルト: 1024x1024）
    int M = 1024, N = 1024, K = 1024;
    int num_iterations = 10;

    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    if (argc > 4) num_iterations = atoi(argv[4]);

    printf("=== MKL cblas_gemm_bf16bf16f32 Benchmark ===\n");
    printf("Matrix size: M=%d, N=%d, K=%d\n", M, N, K);
    printf("Iterations: %d\n", num_iterations);
    printf("MKL Version: %d.%d.%d\n",
           __INTEL_MKL__, __INTEL_MKL_MINOR__, __INTEL_MKL_UPDATE__);

    // メモリ確保（64バイトアラインメント）
    MKL_BF16 *A = (MKL_BF16*)mkl_malloc(sizeof(MKL_BF16) * M * K, 64);
    MKL_BF16 *B = (MKL_BF16*)mkl_malloc(sizeof(MKL_BF16) * K * N, 64);
    float *C = (float*)mkl_malloc(sizeof(float) * M * N, 64);

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // 行列初期化（ランダム値）
    srand(42);
    for (int i = 0; i < M * K; i++) {
        A[i] = float_to_bf16((float)(rand() % 100) / 100.0f);
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = float_to_bf16((float)(rand() % 100) / 100.0f);
    }
    for (int i = 0; i < M * N; i++) {
        C[i] = 0.0f;
    }

    // ウォームアップ
    printf("Warming up...\n");
    cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                           M, N, K,
                           1.0f,
                           A, K,
                           B, N,
                           0.0f,
                           C, N);

    // ベンチマーク
    printf("Running benchmark...\n");
    double total_time = 0.0;
    double min_time = 1e30;
    double max_time = 0.0;

    for (int iter = 0; iter < num_iterations; iter++) {
        // Cをリセット
        memset(C, 0, sizeof(float) * M * N);

        double start = get_time_sec();
        cblas_gemm_bf16bf16f32(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                               M, N, K,
                               1.0f,
                               A, K,
                               B, N,
                               0.0f,
                               C, N);
        double end = get_time_sec();

        double elapsed = end - start;
        total_time += elapsed;
        if (elapsed < min_time) min_time = elapsed;
        if (elapsed > max_time) max_time = elapsed;
    }

    double avg_time = total_time / num_iterations;

    // FLOPS計算: 2*M*N*K (乗算+加算)
    double flops = 2.0 * M * N * K;
    double gflops = flops / avg_time / 1e9;
    double gflops_max = flops / min_time / 1e9;

    printf("\n=== Results ===\n");
    printf("Average time: %.6f sec\n", avg_time);
    printf("Min time:     %.6f sec\n", min_time);
    printf("Max time:     %.6f sec\n", max_time);
    printf("Average:      %.2f GFLOPS\n", gflops);
    printf("Peak:         %.2f GFLOPS\n", gflops_max);

    // 結果検証（サンプル出力）
    printf("\nSample output C[0..3]: ");
    for (int i = 0; i < 4 && i < M * N; i++) {
        printf("%.4f ", C[i]);
    }
    printf("\n");

    // メモリ解放
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    return 0;
}
