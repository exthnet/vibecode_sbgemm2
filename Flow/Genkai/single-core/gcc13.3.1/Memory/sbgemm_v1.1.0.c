// sbgemm_v1.1.0.c - メモリアクセス最適化版 (レジスタブロッキング)
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef uint16_t bf16;
typedef enum { CblasRowMajor = 101 } CBLAS_LAYOUT;
typedef enum { CblasNoTrans = 111, CblasTrans = 112 } CBLAS_TRANSPOSE;

#define BLOCK_K 256
#define BLOCK_N 256
#define BLOCK_M 64
#define MR 4
#define NR 4

static inline bf16 float_to_bf16_round(float x) {
    uint32_t u; memcpy(&u, &x, sizeof(u)); u += 0x00008000u; return (bf16)(u >> 16);
}
static inline float bf16_to_float(bf16 x) {
    uint32_t u = ((uint32_t)x) << 16; float f; memcpy(&f, &u, sizeof(f)); return f;
}

static inline void micro_kernel_4x4(const float *a, const float *b, float *c, int ldc, int klen) {
    float c00=0,c01=0,c02=0,c03=0, c10=0,c11=0,c12=0,c13=0;
    float c20=0,c21=0,c22=0,c23=0, c30=0,c31=0,c32=0,c33=0;
    for (int k = 0; k < klen; k++) {
        float a0=a[k*MR], a1=a[k*MR+1], a2=a[k*MR+2], a3=a[k*MR+3];
        float b0=b[k*NR], b1=b[k*NR+1], b2=b[k*NR+2], b3=b[k*NR+3];
        c00+=a0*b0; c01+=a0*b1; c02+=a0*b2; c03+=a0*b3;
        c10+=a1*b0; c11+=a1*b1; c12+=a1*b2; c13+=a1*b3;
        c20+=a2*b0; c21+=a2*b1; c22+=a2*b2; c23+=a2*b3;
        c30+=a3*b0; c31+=a3*b1; c32+=a3*b2; c33+=a3*b3;
    }
    c[0*ldc+0]+=c00; c[0*ldc+1]+=c01; c[0*ldc+2]+=c02; c[0*ldc+3]+=c03;
    c[1*ldc+0]+=c10; c[1*ldc+1]+=c11; c[1*ldc+2]+=c12; c[1*ldc+3]+=c13;
    c[2*ldc+0]+=c20; c[2*ldc+1]+=c21; c[2*ldc+2]+=c22; c[2*ldc+3]+=c23;
    c[3*ldc+0]+=c30; c[3*ldc+1]+=c31; c[3*ldc+2]+=c32; c[3*ldc+3]+=c33;
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
                micro_kernel_4x4(abuf, bbuf, &C[mm*ldc+nn], ldc, kb);
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
    printf("=== sbgemm v1.1.0 (Register Blocking MR=%d NR=%d) ===\n",MR,NR);
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
