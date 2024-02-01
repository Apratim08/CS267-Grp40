#include <stdio.h>
#include "immintrin.h"
const char* dgemm_desc = "Simple blocked dgemm.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 40
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */
static void do_block(int lda, int M, int N, int K, double* A, double* B, double* C) {
    // For each row i of A
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            // For each column j of B
            double bkj = B[k + j * lda];
            __m256d bv = _mm256_set1_pd(bkj);

            for (int i = 0; i < (M/4)*4; i = i + 4){
                __m256d cv = _mm256_loadu_pd(C + i + j * lda);
                __m256d av = _mm256_loadu_pd(A + i + k * lda);
                _mm256_storeu_pd(C + i + j * lda, _mm256_fmadd_pd(av, bv, cv));
            }
            for (int i = (M/4)*4; i < M; ++i) {
                // Compute C(i,j)
                C[i + j * lda] += A[i + k * lda] * bkj;
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

