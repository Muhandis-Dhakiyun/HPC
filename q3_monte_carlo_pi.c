/*
 * Tutorial-05: OpenMP
 * Question 3: Monte Carlo estimation of π using OpenMP
 *
 * HOW IT WORKS:
 * -------------
 * A unit circle is inscribed in a 2×2 square.
 *   Area of circle  = π·R² = π  (R=1)
 *   Area of square  = (2R)² = 4
 *   → Ratio = π/4
 *
 * We sample N random points (x,y) ∈ [-1,1]×[-1,1].
 * A point is "inside" the circle if x²+y² ≤ 1.
 * Let hits = number of points inside.
 * Then  hits/N ≈ π/4  →  π ≈ 4·hits/N
 *
 * PARALLELISM:
 * ------------
 * Each thread has its own seed so random number streams don't conflict.
 * A reduction on hits safely aggregates the count across threads.
 *
 * OUTPUTS:
 * --------
 *   i.  Table of N vs estimated π for N = 10, 10², 10⁴, 10⁶, 10⁷, 10⁸, 10⁹
 *  ii.  Error = |π_estimate - π_true| printed alongside
 *
 * Compile: gcc -O2 -fopenmp q3_monte_carlo_pi.c -o q3 -lm
 * Run:     ./q3
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define PI_TRUE 3.14159265358979323846

/*
 * Thread‑safe LCG random number generator in [0, 1).
 * Using a local seed per thread avoids a shared-state race.
 */
static inline double rand_local(unsigned long long *seed)
{
    *seed = (*seed) * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)((*seed) >> 33) / (double)(1ULL << 31);
}

double estimate_pi(long long N)
{
    long long hits = 0;
    unsigned long long base_seed = (unsigned long long)time(NULL);

    #pragma omp parallel reduction(+:hits)
    {
        int tid = omp_get_thread_num();
        unsigned long long seed = base_seed + (unsigned long long)tid * 1234567ULL;

        #pragma omp for schedule(static)
        for (long long i = 0; i < N; i++) {
            double x = rand_local(&seed) * 2.0 - 1.0;
            double y = rand_local(&seed) * 2.0 - 1.0;
            if (x*x + y*y <= 1.0)
                hits++;
        }
    }

    return 4.0 * (double)hits / (double)N;
}

int main(void)
{
    long long sample_sizes[] = {
        10LL,
        100LL,
        10000LL,
        1000000LL,
        10000000LL,
        100000000LL,
        1000000000LL
    };
    int num_cases = (int)(sizeof(sample_sizes) / sizeof(sample_sizes[0]));

    printf("=== Q3: Monte Carlo π Estimation ===\n");
    printf("Threads available: %d\n\n", omp_get_max_threads());
    printf("%-15s  %-18s  %-18s  %-10s\n",
           "N", "Estimated π", "True π", "Error");
    printf("%-15s  %-18s  %-18s  %-10s\n",
           "---------------", "------------------",
           "------------------", "----------");

    for (int i = 0; i < num_cases; i++) {
        long long N = sample_sizes[i];
        double t0 = omp_get_wtime();
        double pi_est = estimate_pi(N);
        double t1 = omp_get_wtime();
        double err = fabs(pi_est - PI_TRUE);

        printf("%-15lld  %-18.10f  %-18.10f  %-10.6e  (%.3fs)\n",
               N, pi_est, PI_TRUE, err, t1 - t0);
    }

    printf("\n--- i.  Graph data (N vs π estimate) printed above ---\n");
    printf("--- ii. Error column shows |estimate - true π|    ---\n");
    printf("        Error decreases ~as 1/sqrt(N) (Monte Carlo rate).\n");

    return 0;
}
