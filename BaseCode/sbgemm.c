// sbgemm_nolib.c
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef uint16_t bf16;

typedef enum {
    CblasRowMajor = 101,
    CblasColMajor = 102  // 今回は RowMajor のみ実装
} CBLAS_LAYOUT;

typedef enum {
    CblasNoTrans  = 111,
    CblasTrans    = 112,
    CblasConjTrans= 113  // 実装は CblasTrans と同一扱い
} CBLAS_TRANSPOSE;

/* --- BF16 <-> FP32 変換 --- */

// FP32 -> BF16（簡易: round-to-nearest, ties-away）
static inline bf16 float_to_bf16_round(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    // 16bit 目に 1 を加えて丸め（ties away）。NaN 等の詳細は省略。
    u += 0x00008000u;
    return (bf16)(u >> 16);
}

// FP32 -> BF16（単純切り捨て）
static inline bf16 float_to_bf16_trunc(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    return (bf16)(u >> 16);
}

// BF16 -> FP32（下位 16bit を 0）
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
 *  - A, B は BF16 配列
 *  - C は FP32 配列
 *  - Row-Major 前提（CblasRowMajor のみ対応）
 *  - 転置指定: CblasNoTrans / CblasTrans
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
        fprintf(stderr, "Only CblasRowMajor is supported in this sample.\n");
        exit(1);
    }

    // 事前に beta*C を反映（beta==0ならゼロ初期化）
    if (beta == 0.0f) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                set_elem_f32(C, ldc, i, j, 0.0f);
            }
        }
    } else if (beta != 1.0f) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float cij = C[i * ldc + j];
                set_elem_f32(C, ldc, i, j, beta * cij);
            }
        }
    }
    // beta==1.0f の場合はそのまま上書き加算。

    // コア計算
    // op(A) のサイズ: M x K   （A: NoTrans） / K x M （A: Trans）
    // op(B) のサイズ: K x N   （B: NoTrans） / N x K （B: Trans）
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < K; ++p) {
                // a_ip = op(A)[i, p]
                float a_ip;
                if (transA == CblasNoTrans) {
                    // A: (M x K), lda=K  (Row-Major)
                    a_ip = get_elem_bf16(A, lda, i, p);
                } else {
                    // A^T: (K x M), 位置は (p, i)
                    a_ip = get_elem_bf16(A, lda, p, i);
                }

                // b_pj = op(B)[p, j]
                float b_pj;
                if (transB == CblasNoTrans) {
                    // B: (K x N), ldb=N
                    b_pj = get_elem_bf16(B, ldb, p, j);
                } else {
                    // B^T: (N x K), 位置は (j, p)
                    b_pj = get_elem_bf16(B, ldb, j, p);
                }

                sum += a_ip * b_pj;
            }
            // C_ij = alpha * sum + C_ij（事前に beta*C 済み）
            C[i * ldc + j] += alpha * sum;
        }
    }
}


int main(void) {
    // 例: C = A(2x3) * B(3x2)
    const int M = 2, K = 3, N = 2;

    // Row-Major の A(2x3), B(3x2) を FP32 で用意
    float A_f32[2*3] = { 1.f, 2.f, 3.f,
                         4.f, 5.f, 6.f };
    float B_f32[3*2] = { 7.f,  8.f,
                         9.f, 10.f,
                         11.f,12.f };

    // BF16 バッファへ変換
    bf16 *A_bf16 = (bf16*)malloc(sizeof(bf16) * M * K);
    bf16 *B_bf16 = (bf16*)malloc(sizeof(bf16) * K * N);
    float *C_f32  = (float*)calloc(M * N, sizeof(float));
    if (!A_bf16 || !B_bf16 || !C_f32) {
        fprintf(stderr, "alloc failed\n");
        return 1;
    }

    // 丸めありで BF16 化
    for (int i = 0; i < M*K; ++i) A_bf16[i] = float_to_bf16_round(A_f32[i]);
    for (int i = 0; i < K*N; ++i) B_bf16[i] = float_to_bf16_round(B_f32[i]);

    const int lda = K; // Row-Major & NoTrans なら列数
    const int ldb = N;
    const int ldc = N;

    // C = A * B
    sbgemm_nolib(CblasRowMajor,
                 CblasNoTrans, CblasNoTrans,
                 M, N, K, 1.0f,
                 A_bf16, lda,
                 B_bf16, ldb,
                 0.0f,
                 C_f32,  ldc);

    // 期待値: [[58, 64], [139, 154]]
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%8.2f ", C_f32[i * N + j]);
        }
        printf("\n");
    }

    free(A_bf16);
    free(B_bf16);
    free(C_f32);
    return 0;
}
