#ifndef MATRIX_KERNELS_H
#define MATRIX_KERNELS_H

void matmul_classic(const double *A, const double *B, double *C, int n);
void matmul_omp(const double *A, const double *B, double *C, int n, int num_threads);
void matmul_vectorized(const double *A, const double *B, double *C, int n);
void matmul_omp_vectorized(const double *A, const double *B, double *C, int n, int num_threads);

#endif // MATRIX_KERNELS_H
