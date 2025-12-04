// benchmark.c
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
    #include <windows.h>
    #include <psapi.h>
#else
    #include <sys/resource.h>
    #include <sys/time.h>
#endif


void matmul_classic(const double *A, const double *B, double *C, int n);
void matmul_omp(const double *A, const double *B, double *C, int n, int num_threads);
void matmul_vectorized(const double *A, const double *B, double *C, int n);
void matmul_omp_vectorized(const double *A, const double *B, double *C, int n, int num_threads);

#ifdef _WIN32

static long long now_us(void) {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (long long)(counter.QuadPart * 1000000LL / freq.QuadPart);
}
#else
// Hilfsfunktion: aktuelle Zeit (us) – POSIX-Version
static long long now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long) ts.tv_sec * 1000000LL + ts.tv_nsec / 1000LL;
}
#endif

static long get_peak_rss_kb(void) {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        // PeakWorkingSetSize ist das Maximum des Working Sets
        return (long)(pmc.PeakWorkingSetSize / 1024);
    } else {
        return 0; // Fallback, falls API fehlschlägt
    }
#else
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    // Auf Linux: ru_maxrss in kB, auf macOS in Bytes (aber wir nehmen es so)
    return usage.ru_maxrss;
#endif
}


static void init_matrix(double *M, int n, double seed) {
    for (int i = 0; i < n * n; ++i) {
        M[i] = (double)((i + 1) % 100) * seed;
    }
}


static double max_abs_diff(const double *A, const double *B, int n) {
    double max_diff = 0.0;
    for (int i = 0; i < n * n; ++i) {
        double diff = A[i] - B[i];
        if (diff < 0.0) diff = -diff;
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

void run_benchmarks(void) {
    int sizes[] = {512, 1024, 2048, 4069};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);


    int runs = 3;
    int num_threads = 0; // 0 => OpenMP default, automatically choose optimized value

    printf("Algorithm,Size,Run,Time_us,PeakRSS_kB\n");

    for (int s = 0; s < num_sizes; ++s) {
        int n = sizes[s];
        size_t bytes = (size_t)n * (size_t)n * sizeof(double);

        double *A = (double *) malloc(bytes);
        double *B = (double *) malloc(bytes);
        double *C = (double *) malloc(bytes);
        double *C_ref = (double *) malloc(bytes);

        if (!A || !B || !C || !C_ref) {
            fprintf(stderr, "Allocation failed for n=%d\n", n);
            free(A); free(B); free(C); free(C_ref);
            continue;
        }

        init_matrix(A, n, 0.01);
        init_matrix(B, n, 0.02);

        matmul_classic(A, B, C_ref, n);

        // 1) Classic
        for (int r = 0; r < runs; ++r) {
            memset(C, 0, bytes);
            long long t0 = now_us();
            matmul_classic(A, B, C, n);
            long long t1 = now_us();
            long rss = get_peak_rss_kb();

            double diff = max_abs_diff(C, C_ref, n);
            if (diff > 1e-6) {
                fprintf(stderr, "WARNING: classic result mismatch diff=%g for n=%d\n", diff, n);
            }

            printf("classic,%d,%d,%lld,%ld\n", n, r + 1, (t1 - t0), rss);
            fflush(stdout);
        }

        // 2) OpenMP
        for (int r = 0; r < runs; ++r) {
            memset(C, 0, bytes);
            long long t0 = now_us();
            matmul_omp(A, B, C, n, num_threads);
            long long t1 = now_us();
            long rss = get_peak_rss_kb();

            double diff = max_abs_diff(C, C_ref, n);
            if (diff > 1e-6) {
                fprintf(stderr, "WARNING: omp result mismatch diff=%g for n=%d\n", diff, n);
            }

            printf("omp,%d,%d,%lld,%ld\n", n, r + 1, (t1 - t0), rss);
            fflush(stdout);
        }

        // 3) Vectorized
        for (int r = 0; r < runs; ++r) {
            memset(C, 0, bytes);
            long long t0 = now_us();
            matmul_vectorized(A, B, C, n);
            long long t1 = now_us();
            long rss = get_peak_rss_kb();

            double diff = max_abs_diff(C, C_ref, n);
            if (diff > 1e-6) {
                fprintf(stderr, "WARNING: vectorized result mismatch diff=%g for n=%d\n", diff, n);
            }

            printf("vectorized,%d,%d,%lld,%ld\n", n, r + 1, (t1 - t0), rss);
            fflush(stdout);
        }

        // 4) OMP + Vectorized
        for (int r = 0; r < runs; ++r) {
            memset(C, 0, bytes);
            long long t0 = now_us();
            matmul_omp_vectorized(A, B, C, n, num_threads);
            long long t1 = now_us();
            long rss = get_peak_rss_kb();

            double diff = max_abs_diff(C, C_ref, n);
            if (diff > 1e-6) {
                fprintf(stderr, "WARNING: omp_vectorized result mismatch diff=%g for n=%d\n", diff, n);
            }

            printf("omp_vectorized,%d,%d,%lld,%ld\n", n, r + 1, (t1 - t0), rss);
            fflush(stdout);
        }

        free(A);
        free(B);
        free(C);
        free(C_ref);
    }
}
