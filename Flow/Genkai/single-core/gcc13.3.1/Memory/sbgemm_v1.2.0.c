// sbgemm_v1.2.0.c - メモリアクセス最適化版 (8x8レジスタブロッキング + プリフェッチ)
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef uint16_t bf16;
typedef enum { CblasRowMajor = 101 } CBLAS_LAYOUT;
typedef enum { CblasNoTrans = 111, CblasTrans = 112 } CBLAS_TRANSPOSE;

#define BLOCK_K 512
#define BLOCK_N 512
#define BLOCK_M 64
#define MR 8
#define NR 8

static inline bf16 float_to_bf16_round(float x) {
    uint32_t u; memcpy(&u, &x, sizeof(u)); u += 0x00008000u; return (bf16)(u >> 16);
}
static inline float bf16_to_float(bf16 x) {
    uint32_t u = ((uint32_t)x) << 16; float f; memcpy(&f, &u, sizeof(f)); return f;
}

// 8x8マイクロカーネル（64要素のアキュムレータ）
static inline void micro_kernel_8x8(const float *a, const float *b, float *c, int ldc, int klen) {
    // 8x8 = 64個のアキュムレータ
    float c00=0,c01=0,c02=0,c03=0,c04=0,c05=0,c06=0,c07=0;
    float c10=0,c11=0,c12=0,c13=0,c14=0,c15=0,c16=0,c17=0;
    float c20=0,c21=0,c22=0,c23=0,c24=0,c25=0,c26=0,c27=0;
    float c30=0,c31=0,c32=0,c33=0,c34=0,c35=0,c36=0,c37=0;
    float c40=0,c41=0,c42=0,c43=0,c44=0,c45=0,c46=0,c47=0;
    float c50=0,c51=0,c52=0,c53=0,c54=0,c55=0,c56=0,c57=0;
    float c60=0,c61=0,c62=0,c63=0,c64=0,c65=0,c66=0,c67=0;
    float c70=0,c71=0,c72=0,c73=0,c74=0,c75=0,c76=0,c77=0;

    for (int k = 0; k < klen; k++) {
        // プリフェッチ（次のk反復のデータ）
        if (k + 2 < klen) {
            __builtin_prefetch(&a[(k+2)*MR], 0, 3);
            __builtin_prefetch(&b[(k+2)*NR], 0, 3);
        }

        float a0=a[k*MR], a1=a[k*MR+1], a2=a[k*MR+2], a3=a[k*MR+3];
        float a4=a[k*MR+4], a5=a[k*MR+5], a6=a[k*MR+6], a7=a[k*MR+7];
        float b0=b[k*NR], b1=b[k*NR+1], b2=b[k*NR+2], b3=b[k*NR+3];
        float b4=b[k*NR+4], b5=b[k*NR+5], b6=b[k*NR+6], b7=b[k*NR+7];

        // Row 0
        c00+=a0*b0; c01+=a0*b1; c02+=a0*b2; c03+=a0*b3;
        c04+=a0*b4; c05+=a0*b5; c06+=a0*b6; c07+=a0*b7;
        // Row 1
        c10+=a1*b0; c11+=a1*b1; c12+=a1*b2; c13+=a1*b3;
        c14+=a1*b4; c15+=a1*b5; c16+=a1*b6; c17+=a1*b7;
        // Row 2
        c20+=a2*b0; c21+=a2*b1; c22+=a2*b2; c23+=a2*b3;
        c24+=a2*b4; c25+=a2*b5; c26+=a2*b6; c27+=a2*b7;
        // Row 3
        c30+=a3*b0; c31+=a3*b1; c32+=a3*b2; c33+=a3*b3;
        c34+=a3*b4; c35+=a3*b5; c36+=a3*b6; c37+=a3*b7;
        // Row 4
        c40+=a4*b0; c41+=a4*b1; c42+=a4*b2; c43+=a4*b3;
        c44+=a4*b4; c45+=a4*b5; c46+=a4*b6; c47+=a4*b7;
        // Row 5
        c50+=a5*b0; c51+=a5*b1; c52+=a5*b2; c53+=a5*b3;
        c54+=a5*b4; c55+=a5*b5; c56+=a5*b6; c57+=a5*b7;
        // Row 6
        c60+=a6*b0; c61+=a6*b1; c62+=a6*b2; c63+=a6*b3;
        c64+=a6*b4; c65+=a6*b5; c66+=a6*b6; c67+=a6*b7;
        // Row 7
        c70+=a7*b0; c71+=a7*b1; c72+=a7*b2; c73+=a7*b3;
        c74+=a7*b4; c75+=a7*b5; c76+=a7*b6; c77+=a7*b7;
    }
    // Store results
    c[0*ldc+0]+=c00; c[0*ldc+1]+=c01; c[0*ldc+2]+=c02; c[0*ldc+3]+=c03;
    c[0*ldc+4]+=c04; c[0*ldc+5]+=c05; c[0*ldc+6]+=c06; c[0*ldc+7]+=c07;
    c[1*ldc+0]+=c10; c[1*ldc+1]+=c11; c[1*ldc+2]+=c12; c[1*ldc+3]+=c13;
    c[1*ldc+4]+=c14; c[1*ldc+5]+=c15; c[1*ldc+6]+=c16; c[1*ldc+7]+=c17;
    c[2*ldc+0]+=c20; c[2*ldc+1]+=c21; c[2*ldc+2]+=c22; c[2*ldc+3]+=c23;
    c[2*ldc+4]+=c24; c[2*ldc+5]+=c25; c[2*ldc+6]+=c26; c[2*ldc+7]+=c27;
    c[3*ldc+0]+=c30; c[3*ldc+1]+=c31; c[3*ldc+2]+=c32; c[3*ldc+3]+=c33;
    c[3*ldc+4]+=c34; c[3*ldc+5]+=c35; c[3*ldc+6]+=c36; c[3*ldc+7]+=c37;
    c[4*ldc+0]+=c40; c[4*ldc+1]+=c41; c[4*ldc+2]+=c42; c[4*ldc+3]+=c43;
    c[4*ldc+4]+=c44; c[4*ldc+5]+=c45; c[4*ldc+6]+=c46; c[4*ldc+7]+=c47;
    c[5*ldc+0]+=c50; c[5*ldc+1]+=c51; c[5*ldc+2]+=c52; c[5*ldc+3]+=c53;
    c[5*ldc+4]+=c54; c[5*ldc+5]+=c55; c[5*ldc+6]+=c56; c[5*ldc+7]+=c57;
    c[6*ldc+0]+=c60; c[6*ldc+1]+=c61; c[6*ldc+2]+=c62; c[6*ldc+3]+=c63;
    c[6*ldc+4]+=c64; c[6*ldc+5]+=c65; c[6*ldc+6]+=c66; c[6*ldc+7]+=c67;
    c[7*ldc+0]+=c70; c[7*ldc+1]+=c71; c[7*ldc+2]+=c72; c[7*ldc+3]+=c73;
    c[7*ldc+4]+=c74; c[7*ldc+5]+=c75; c[7*ldc+6]+=c76; c[7*ldc+7]+=c77;
}

static void do_pack_a(const bf16 *A, int lda, int ms, int mlen, int ks, int klen, float *abuf, CBLAS_TRANSPOSE tA) {
    for (int k = 0; k < klen; k++) {
        for (int i = 0; i < mlen; i++) {
            float v = (tA==CblasNoTrans) ? bf16_to_float(A[(ms+i)*lda+(ks+k)]) : bf16_to_float(A[(ks+k)*lda+(ms+i)]);
            abuf[k*MR + (i%MR)] = v;
        }
    }
}

static void do_pack_b(const bf16 *B, int ldb, int ks, int klen, int ns, int nlen, float *bbuf, CBLAS_TRANSPOSE tB) {
    for (int k = 0; k < klen; k++) {
        for (int j = 0; j < nlen; j++) {
            float v = (tB==CblasNoTrans) ? bf16_to_float(B[(ks+k)*ldb+(ns+j)]) : bf16_to_float(B[(ns+j)*ldb+(ks+k)]);
            bbuf[k*NR + (j%NR)] = v;
        }
    }
}

void sbgemm_nolib(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE tA, CBLAS_TRANSPOSE tB,
                  int M, int N, int K, float alpha, const bf16 *A, int lda, const bf16 *B, int ldb,
                  float beta, float *C, int ldc) {
    if (layout != CblasRowMajor) exit(1);
    if (beta == 0.0f) { for (int i=0;i<M;i++) for (int j=0;j<N;j++) C[i*ldc+j]=0.0f; }
    else if (beta != 1.0f) { for (int i=0;i<M;i++) for (int j=0;j<N;j++) C[i*ldc+j]*=beta; }

    float *abuf = (float*)malloc(sizeof(float)*BLOCK_K*MR);
    float *bbuf = (float*)malloc(sizeof(float)*BLOCK_K*NR);
    if (!abuf || !bbuf) exit(1);

    for (int kk = 0; kk < K; kk += BLOCK_K) {
        int kb = (kk+BLOCK_K<=K) ? BLOCK_K : (K-kk);
        for (int nn = 0; nn < N; nn += NR) {
            int nb = (nn+NR<=N) ? NR : (N-nn);
            if (nb < NR) continue;
            do_pack_b(B, ldb, kk, kb, nn, nb, bbuf, tB);
            for (int mm = 0; mm < M; mm += MR) {
                int mb = (mm+MR<=M) ? MR : (M-mm);
                if (mb < MR) continue;
                do_pack_a(A, lda, mm, mb, kk, kb, abuf, tA);
                micro_kernel_8x8(abuf, bbuf, &C[mm*ldc+nn], ldc, kb);
            }
        }
    }
    // 端数処理
    for (int i=(M/MR)*MR; i<M; i++) for (int j=0; j<N; j++) {
        float s=0; for (int k=0;k<K;k++) s+=bf16_to_float(A[i*lda+k])*bf16_to_float(B[k*ldb+j]);
        C[i*ldc+j]+=alpha*s;
    }
    for (int i=0; i<(M/MR)*MR; i++) for (int j=(N/NR)*NR; j<N; j++) {
        float s=0; for (int k=0;k<K;k++) s+=bf16_to_float(A[i*lda+k])*bf16_to_float(B[k*ldb+j]);
        C[i*ldc+j]+=alpha*s;
    }
    free(abuf); free(bbuf);
}

static double get_time(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }
static void init_bf16(bf16 *a, int n) { for(int i=0;i<n;i++) a[i]=float_to_bf16_round((float)(rand()%100)/100.0f); }

int main(int argc, char **argv) {
    int sizes[]={1024,2048,4096}; int ns=3;
    if(argc>1){ns=1;sizes[0]=atoi(argv[1]);}
    printf("=== sbgemm v1.2.0 (Register Blocking MR=%d NR=%d + Prefetch) ===\n",MR,NR);
    srand(42);
    for(int s=0;s<ns;s++){
        int sz=sizes[s];
        bf16 *A=(bf16*)malloc(sz*sz*sizeof(bf16)), *B=(bf16*)malloc(sz*sz*sizeof(bf16));
        float *C=(float*)calloc(sz*sz,sizeof(float));
        init_bf16(A,sz*sz); init_bf16(B,sz*sz);
        sbgemm_nolib(CblasRowMajor,CblasNoTrans,CblasNoTrans,sz,sz,sz,1.0f,A,sz,B,sz,0.0f,C,sz);
        double tot=0; for(int t=0;t<3;t++){
            memset(C,0,sz*sz*sizeof(float)); double st=get_time();
            sbgemm_nolib(CblasRowMajor,CblasNoTrans,CblasNoTrans,sz,sz,sz,1.0f,A,sz,B,sz,0.0f,C,sz);
            tot+=get_time()-st;
        }
        printf("Size:%5d | Time:%.4f sec | %.2f GFLOPS\n",sz,tot/3,2.0*sz*sz*sz/(tot/3)/1e9);
        free(A);free(B);free(C);
    }
    printf("\n=== Verification ===\n");
    bf16 Ab[6],Bb[6]; float Cf[4]={0}, Af[]={1,2,3,4,5,6}, Bf[]={7,8,9,10,11,12};
    for(int i=0;i<6;i++){Ab[i]=float_to_bf16_round(Af[i]);Bb[i]=float_to_bf16_round(Bf[i]);}
    sbgemm_nolib(CblasRowMajor,CblasNoTrans,CblasNoTrans,2,2,3,1.0f,Ab,3,Bb,2,0.0f,Cf,2);
    printf("Expected:[[58,64],[139,154]]\nResult:[[%.0f,%.0f],[%.0f,%.0f]]\n",Cf[0],Cf[1],Cf[2],Cf[3]);
    return 0;
}
