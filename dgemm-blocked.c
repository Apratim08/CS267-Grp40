#include <stdio.h>
#include<stdlib.h>
#include "immintrin.h"

const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 68
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

static void micro_kernal(int lda, double* A, double* C, __m256d b1, __m256d b2, __m256d b3, __m256d b4) {
    __m256d a1 = _mm256_loadu_pd(A);
    __m256d a2 = _mm256_loadu_pd(A + lda);
    __m256d a3 = _mm256_loadu_pd(A + 2 * lda);
    __m256d a4 = _mm256_loadu_pd(A + 3 * lda);

    __m256d c1 = _mm256_loadu_pd(C);
    __m256d c2 = _mm256_loadu_pd(C + lda);
    __m256d c3 = _mm256_loadu_pd(C + 2 * lda);
    __m256d c4 = _mm256_loadu_pd(C + 3 * lda);

    // First column of b
    __m256d broadcast_b = _mm256_set1_pd(b1[0]);
    c1 = _mm256_fmadd_pd(a1, broadcast_b, c1);

    broadcast_b = _mm256_set1_pd(b2[0]);
    c2 = _mm256_fmadd_pd(a1, broadcast_b, c2);
    
    broadcast_b = _mm256_set1_pd(b3[0]);
    c3 = _mm256_fmadd_pd(a1, broadcast_b, c3);

    broadcast_b = _mm256_set1_pd(b4[0]);
    c4 = _mm256_fmadd_pd(a1, broadcast_b, c4);

    // Second column of b
    broadcast_b = _mm256_set1_pd(b1[1]);
    c1 = _mm256_fmadd_pd(a2, broadcast_b, c1);

    broadcast_b = _mm256_set1_pd(b2[1]);
    c2 = _mm256_fmadd_pd(a2, broadcast_b, c2);
    
    broadcast_b = _mm256_set1_pd(b3[1]);
    c3  =_mm256_fmadd_pd(a2, broadcast_b, c3);

    broadcast_b = _mm256_set1_pd(b4[1]);
    c4 = _mm256_fmadd_pd(a2, broadcast_b, c4);

    // Third column of b
    broadcast_b = _mm256_set1_pd(b1[2]);
    c1 = _mm256_fmadd_pd(a3, broadcast_b, c1);

    broadcast_b = _mm256_set1_pd(b2[2]);
    c2 = _mm256_fmadd_pd(a3, broadcast_b, c2);
    
    broadcast_b = _mm256_set1_pd(b3[2]);
    c3 = _mm256_fmadd_pd(a3, broadcast_b, c3);

    broadcast_b = _mm256_set1_pd(b4[2]);
    c4 = _mm256_fmadd_pd(a3, broadcast_b, c4);


    // Fourth column of b
    broadcast_b = _mm256_set1_pd(b1[3]);
    c1 = _mm256_fmadd_pd(a4, broadcast_b, c1);

    broadcast_b = _mm256_set1_pd(b2[3]);
    c2 = _mm256_fmadd_pd(a4, broadcast_b, c2);
    
    broadcast_b = _mm256_set1_pd(b3[3]);
    c3 = _mm256_fmadd_pd(a4, broadcast_b, c3);

    broadcast_b = _mm256_set1_pd(b4[3]);
    c4 = _mm256_fmadd_pd(a4, broadcast_b, c4);

    // Store C
    _mm256_storeu_pd(C, c1);
    _mm256_storeu_pd(C + 1 * lda, c2);
    _mm256_storeu_pd(C + 2 * lda, c3);
    _mm256_storeu_pd(C + 3 * lda, c4);

}


/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    
    for (int k = 0; k < (K/4)*4; k = k + 4) {
        // For each column j of B
        for (int j = 0; j < (N/4)*4; j = j + 4) {
            __m256d b1 = _mm256_loadu_pd(B + k + j * lda);
            __m256d b2  = _mm256_loadu_pd(B + k + (j + 1) * lda);
            __m256d b3  = _mm256_loadu_pd(B + k + (j + 2) * lda);
            __m256d b4  = _mm256_loadu_pd(B + k + (j + 3) * lda);
            

            // For each row i of A
            for (int i = 0; i < (M/4)*4; i = i + 4) {
                micro_kernal(lda, A + i + k * lda, C + i + j * lda, b1, b2, b3, b4);
            }

            // printf("Kernel Result:\n");
            // for (int q = 0; q < 4; q++) {
            //     for(int qq = 0; qq < 4; ++qq){
            //         printf("%lf\t", C[(0 + q) + lda*(j + qq)]);
            //     }
            //     printf("\n");
            // }
            // printf("\n");

            for (int i = (M/4)*4; i < M; ++i) {
                for (int c = 0; c < 4; c++) {
                    for (int d = 0; d < 4; d++) {
                        C[i + (j + c) * lda] += A[i + (k + d) * lda] * B[(k + d) + (j + c) * lda];
                    }   
                }
                
            }
        }
        for (int j = (N/4)*4; j < N; ++j) {
            for (int i = 0; i < M; ++i) {
                for (int c = 0; c < 4; c++) {
                    C[i + j * lda] += A[i + (k + c) * lda] * B[(k + c) + j * lda];
                }
            }
        }
    }


    for (int k = (K/4)*4; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < M; ++i) {
                C[i + j * lda] += A[i + k * lda] * B[k + j * lda];
            }
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double* A, double* B, double* C) {
    // For each block-row of A
    for (int k = 0; k < lda; k += BLOCK_SIZE) {
        for (int j = 0; j < lda; j += BLOCK_SIZE) {
        
            // For each block-column of B
            for (int i = 0; i < lda; i += BLOCK_SIZE) {
                // Accumulate block dgemms into block of C
            
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min(BLOCK_SIZE, lda - i);
                int N = min(BLOCK_SIZE, lda - j);
                int K = min(BLOCK_SIZE, lda - k);
                // Perform individual block dgemm
                do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
}

/*
 * This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 */
void square_dgemm_naive(int n, double* A, double* B, double* C) {
    // For each row i of A
    for (int i = 0; i < n; ++i) {
        // For each column j of B
        for (int j = 0; j < n; ++j) {
            // Compute C(i,j)
            double cij = C[i + j * n];
            for (int k = 0; k < n; k++) {
                cij += A[i + k * n] * B[k + j * n];
            }
            C[i + j * n] = cij;
        }
    }
}

// int main() {
//     // double a[16] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
//     // double b[16] = {1,2,2,1,2,1,1,2,1,1,2,1,1,1,2,1};
//     int lda = 7;
//     int dim = lda*lda;
//     double *b = malloc(dim*sizeof(double));
//     for(int i = 0; i < dim; ++i){
//         b[i] = rand() % 2;
//     }
//     double *a = malloc(dim*sizeof(double));
//     for(int i = 0; i < dim; ++i){
//         a[i] = rand() % 2;
//     }
//     double *c = calloc(dim, sizeof(double));
//     for(int i = 0; i < dim; ++i){
//         c[i] = 1;
//     }

//     // __m256d b1 = _mm256_loadu_pd(b + 0 + 0 * lda);
//     // __m256d b2  = _mm256_loadu_pd(b + 0 + (0 + 1) * lda);
//     // __m256d b3  = _mm256_loadu_pd(b + 0 + (0 + 2) * lda);
//     // __m256d b4  = _mm256_loadu_pd(b + 0 + (0 + 3) * lda);

//     // micro_kernal(lda, a, c, b1, b2, b3, b4);
//     square_dgemm(lda, a, b, c);

//     printf("A:\n");
//     for(int i = 0; i < lda; ++i){
//         for(int j = 0; j < lda; ++j){
//             printf("%lf\t", a[i + lda*j]);
//         }
//         printf("\n");
//     }
//     printf("\n\n\n");

//     printf("B:\n");
//     for(int i = 0; i < lda; ++i){
//         for(int j = 0; j < lda; ++j){
//             printf("%lf\t", b[i + lda*j]);
//         }
//         printf("\n");
//     }
//     printf("\n\n\n");

//     for(int i = 0; i < lda; ++i){
//         for(int j = 0; j < lda; ++j){
//             printf("%lf\t", c[i + lda*j]);
//         }
//         printf("\n");
//     }

//     for(int i = 0; i < dim; ++i){
//         c[i] = 1;
//     }
//     square_dgemm_naive(lda, a, b, c);
//     printf("\n\nCorrect:\n");

//     for(int i = 0; i < lda; ++i){
//         for(int j = 0; j < lda; ++j){
//             printf("%lf\t", c[i + lda*j]);
//         }
//         printf("\n");
//     }


//     return 0;
// }