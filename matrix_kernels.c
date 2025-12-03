// matrix_kernels.c
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <immintrin.h>   // AVX/AVX2 Intrinsics

#ifdef _WIN32
    #include <malloc.h>  // _aligned_malloc, _aligned_free
#endif

#ifdef _OPENMP
    #include <omp.h>
#endif

#include "matrix_kernels.h"

// -----------------------------------------------------------
//  Cross-Platform aligned_malloc / aligned_free
// -----------------------------------------------------------
static void* aligned_malloc(size_t alignment, size_t size) {
#ifdef _WIN32
    // Windows: _aligned_malloc(size, alignment)
    return _aligned_malloc(size, alignment);
#else
    // POSIX: posix_memalign
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
#endif
}

static void aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

// -----------------------------------------------------------
//  Klassische Matrixmultiplikation: C = A * B
// -----------------------------------------------------------
void matmul_classic(const double *A, const double *B, double *C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// -----------------------------------------------------------
//  OpenMP-parallele Matrixmultiplikation (klassisch)
// -----------------------------------------------------------
void matmul_omp(const double *A, const double *B, double *C, int n, int num_threads) {
#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
#else
    (void) num_threads;
    matmul_classic(A, B, C, n);
#endif
}

// -----------------------------------------------------------
//  AVX2-Helper: Dot-Product zweier double-Vektoren
//  a, b: Pointer auf n contiguous double-Werte
// -----------------------------------------------------------
static inline double dot_avx2(const double *a, const double *b, int n) {
    __m256d acc = _mm256_setzero_pd();

    int k = 0;
    int limit = n - (n % 4); // größtes Vielfaches von 4 <= n

    for (; k < limit; k += 4) {
        __m256d va = _mm256_loadu_pd(a + k);
        __m256d vb = _mm256_loadu_pd(b + k);
        __m256d prod = _mm256_mul_pd(va, vb);
        acc = _mm256_add_pd(acc, prod);
    }

    // horizontale Summe acc[0] + acc[1] + acc[2] + acc[3]
    __m128d low  = _mm256_castpd256_pd128(acc);   // untere 2 doubles
    __m128d high = _mm256_extractf128_pd(acc, 1); // obere 2 doubles
    __m128d sum2 = _mm_add_pd(low, high);         // [a0+a2, a1+a3]
    __m128d shuf = _mm_permute_pd(sum2, 0x1);     // swap
    __m128d sum1 = _mm_add_sd(sum2, shuf);        // [gesamt, x]

    double result = _mm_cvtsd_f64(sum1);

    // Rest (falls n kein Vielfaches von 4 ist)
    for (; k < n; ++k) {
        result += a[k] * b[k];
    }

    return result;
}

// -----------------------------------------------------------
//  AVX2-Matrixmultiplikation mit transponiertem B
//
//  B_T[j, i] = B[i, j]
//  C[i, j] = dot( A[i, :], B_T[j, :] )
//
//  => beide Operanden des Dot-Products sind in Row-Major
//     und damit contiguous -> perfekt für SIMD
// -----------------------------------------------------------
static void matmul_avx2_transposed(const double *A, const double *B, double *C, int n) {
    size_t bytes = (size_t)n * (size_t)n * sizeof(double);

    double *B_T = (double *) aligned_malloc(32, bytes);
    if (!B_T) {
        // Fallback, wenn Alloc fehlschlägt
        matmul_classic(A, B, C, n);
        return;
    }

    // B transponieren: B_T[j*n + i] = B[i*n + j]
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            B_T[j * n + i] = B[i * n + j];
        }
    }

    // C = A * B mit AVX2-Dot-Product
    for (int i = 0; i < n; ++i) {
        const double *rowA = A + i * n;
        for (int j = 0; j < n; ++j) {
            const double *rowBT = B_T + j * n;
            double sum = dot_avx2(rowA, rowBT, n);
            C[i * n + j] = sum;
        }
    }

    aligned_free(B_T);
}

// -----------------------------------------------------------
//  Öffentliche vektorisierte Variante (seriell)
// -----------------------------------------------------------
void matmul_vectorized(const double *A, const double *B, double *C, int n) {
#if defined(__AVX2__)
    matmul_avx2_transposed(A, B, C, n);
#else
    // Falls ohne AVX2 kompiliert wird -> Fallback
    matmul_classic(A, B, C, n);
#endif
}

// -----------------------------------------------------------
//  OpenMP + AVX2 kombinierte Version
//
//  - B wird EINMAL transponiert
//  - äußere Schleife über i wird parallelisiert
//  - inneres Produkt läuft mit AVX2
// -----------------------------------------------------------
void matmul_omp_vectorized(const double *A, const double *B, double *C, int n, int num_threads) {
#if defined(__AVX2__) && defined(_OPENMP)
    size_t bytes = (size_t)n * (size_t)n * sizeof(double);

    double *B_T = (double *) aligned_malloc(32, bytes);
    if (!B_T) {
        // Fallback bei fehlendem Speicher
        matmul_omp(A, B, C, n, num_threads);
        return;
    }

    // B transponieren
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            B_T[j * n + i] = B[i * n + j];
        }
    }

    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const double *rowA = A + i * n;
        for (int j = 0; j < n; ++j) {
            const double *rowBT = B_T + j * n;
            double sum = dot_avx2(rowA, rowBT, n);
            C[i * n + j] = sum;
        }
    }

    aligned_free(B_T);
#elif defined(_OPENMP)
    // Kein AVX2, aber OpenMP vorhanden -> nur parallel
    matmul_omp(A, B, C, n, num_threads);
#else
    // Weder AVX2 noch OpenMP
    (void) num_threads;
    matmul_vectorized(A, B, C, n);
#endif
}
