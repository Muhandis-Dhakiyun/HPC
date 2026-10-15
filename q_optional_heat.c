/*
 * Tutorial-05: OpenMP
 * Optional: Steady-state heat conduction on a thin square plate (Jacobi iteration)
 *
 * PROBLEM:
 * --------
 *   Simulate heat diffusion on an N×N plate until temperatures stabilise.
 *   - Initial temperature of interior: 50°C
 *   - Boundary conditions:
 *       Left, Top, Right edges : 100°C  (constant)
 *       Bottom edge            :   0°C  (constant)
 *
 * NUMERICAL METHOD — Jacobi iteration:
 * --------------------------------------
 *   Each interior cell is updated as the average of its four neighbours:
 *       T_new[i][j] = 0.25 * (T[i-1][j] + T[i+1][j] + T[i][j-1] + T[i][j+1])
 *   We iterate until max change < TOLERANCE or we hit MAX_ITER.
 *
 *   Jacobi is embarrassingly parallel: T_new[i][j] depends only on OLD values,
 *   so all interior cells can be updated simultaneously without data races.
 *
 * PARALLELISM:
 * ------------
 *   The outer "i" loop is parallelised with #pragma omp parallel for.
 *   The reduction for max_diff uses a custom reduction.
 *
 * Compile: gcc -O2 -fopenmp q_optional_heat.c -o q_opt -lm
 * Run:     ./q_opt
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N          100        /* grid size (NxN interior + boundary) */
#define TOLERANCE  1e-4       /* convergence threshold                */
#define MAX_ITER   100000     /* maximum iterations                   */

/* Boundary temperatures */
#define T_LEFT   100.0
#define T_TOP    100.0
#define T_RIGHT  100.0
#define T_BOTTOM   0.0
#define T_INIT    50.0

static double T[N+2][N+2];      /* current grid (with ghost boundary) */
static double T_new[N+2][N+2];  /* next grid                          */

static void init_grid(void)
{
    /* Interior: 50°C */
    for (int i = 1; i <= N; i++)
        for (int j = 1; j <= N; j++)
            T[i][j] = T_INIT;

    /* Boundaries */
    for (int j = 0; j <= N+1; j++) {
        T[0][j]   = T_TOP;       /* top row    */
        T[N+1][j] = T_BOTTOM;    /* bottom row */
    }
    for (int i = 0; i <= N+1; i++) {
        T[i][0]   = T_LEFT;      /* left col   */
        T[i][N+1] = T_RIGHT;     /* right col  */
    }
}

int main(void)
{
    init_grid();

    printf("=== Optional: Steady-State Heat Conduction ===\n");
    printf("Grid: %dx%d interior points\n", N, N);
    printf("Boundaries: Top=%g°C, Left=%g°C, Right=%g°C, Bottom=%g°C\n",
           T_TOP, T_LEFT, T_RIGHT, T_BOTTOM);
    printf("Initial interior temp: %g°C\n", T_INIT);
    printf("Threads: %d\n\n", omp_get_max_threads());

    double t_start = omp_get_wtime();

    int iter;
    double max_diff = 1e9;

    for (iter = 0; iter < MAX_ITER && max_diff > TOLERANCE; iter++) {
        max_diff = 0.0;

        #pragma omp parallel for schedule(static) reduction(max:max_diff)
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= N; j++) {
                T_new[i][j] = 0.25 * (T[i-1][j] + T[i+1][j]
                                     + T[i][j-1] + T[i][j+1]);
                double diff = fabs(T_new[i][j] - T[i][j]);
                if (diff > max_diff) max_diff = diff;
            }
        }

        /* Copy T_new → T (boundaries are fixed, skip them) */
        #pragma omp parallel for schedule(static)
        for (int i = 1; i <= N; i++)
            for (int j = 1; j <= N; j++)
                T[i][j] = T_new[i][j];

        if (iter % 1000 == 0)
            printf("  iter %6d: max_diff = %.6f\n", iter, max_diff);
    }

    double t_end = omp_get_wtime();

    printf("\nConverged after %d iterations (max_diff=%.2e) in %.4f s\n",
           iter, max_diff, t_end - t_start);

    /* Print a small sample of the final temperature field */
    printf("\nFinal temperature grid (every 10th row/col):\n");
    printf("     ");
    for (int j = 0; j <= N+1; j += 10) printf("%6d", j);
    printf("\n");
    for (int i = 0; i <= N+1; i += 10) {
        printf("%4d:", i);
        for (int j = 0; j <= N+1; j += 10)
            printf("%6.1f", T[i][j]);
        printf("\n");
    }

    return 0;
}
