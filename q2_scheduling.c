/*
 * Tutorial-05: OpenMP
 * Question 2: Scheduling analysis
 *
 * Given code (executed by 2 threads, 2 big chunks, f(i) takes i ms):
 *
 *   #pragma omp parallel
 *   {
 *       #pragma omp for schedule(static)
 *       for (i = 1; i < n; i++)  f(i);
 *
 *       #pragma omp for schedule(static)
 *       for (i = 1; i < n; i++)  f(n-i);
 *   }
 *
 * ANALYSIS (n iterations, 2 threads):
 *
 * With schedule(static) — 2 big chunks:
 *   Thread 0 gets iterations [1 .. n/2]
 *   Thread 1 gets iterations [n/2+1 .. n-1]
 *
 * --- First loop (f(i), costs i ms) ---
 *   Thread 0 work = sum(1..n/2)        ≈ n²/8
 *   Thread 1 work = sum(n/2+1..n-1)    ≈ 3n²/8
 *   Thread 0 finishes MUCH earlier and waits for Thread 1 at implicit barrier.
 *   Thread 0 wait time ≈ 3n²/8 - n²/8 = n²/4 ms.
 *
 * --- Second loop (f(n-i), costs (n-i) ms) ---
 *   Thread 0 gets i in [1..n/2]  → costs (n-1), (n-2), … (n/2) ≈ 3n²/8
 *   Thread 1 gets i in [n/2+1..n-1] → costs (n/2-1)..1          ≈ n²/8
 *   Thread 1 finishes first and waits for Thread 0 ≈ n²/4 ms.
 *
 * a) Each thread total time ≈ n²/2 ms; each waits ≈ n²/4 ms.
 *
 * b) schedule(static,1) interleaves 1 iteration at a time → better load
 *    balance → threads finish closer together → less waiting.
 *    schedule(dynamic,1) assigns one chunk at a time on demand → near-perfect
 *    balance but has higher overhead per iteration.
 *
 * c) The "nowait" clause on the first for removes the implicit barrier between
 *    the two loops.  A thread that finishes loop‑1 early immediately starts
 *    loop‑2 rather than waiting.  Because loop‑1 imbalance and loop‑2 imbalance
 *    are opposite in direction, they cancel out → total time ≈ n²/2 per thread
 *    with zero waiting.
 *
 * This program demonstrates and measures all three cases.
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

/* Simulate f(i) taking proportional time (busy‑wait) */
static void busy_wait_us(int us)
{
    double t = omp_get_wtime();
    while ((omp_get_wtime() - t) * 1e6 < us) { /* spin */ }
}

/* Scale factor so the demo runs quickly */
#define SCALE 100

static void run_case(int n, const char *label, int use_nowait, int chunk)
{
    double t_start = omp_get_wtime();
    #pragma omp parallel num_threads(2)
    {
        if (chunk > 0) {
            if (use_nowait) {
                #pragma omp for schedule(static,1) nowait
                for (int i = 1; i < n; i++) busy_wait_us(i / SCALE);
                #pragma omp for schedule(static,1)
                for (int i = 1; i < n; i++) busy_wait_us((n-i) / SCALE);
            } else {
                #pragma omp for schedule(static,1)
                for (int i = 1; i < n; i++) busy_wait_us(i / SCALE);
                #pragma omp for schedule(static,1)
                for (int i = 1; i < n; i++) busy_wait_us((n-i) / SCALE);
            }
        } else {
            if (use_nowait) {
                #pragma omp for schedule(static) nowait
                for (int i = 1; i < n; i++) busy_wait_us(i / SCALE);
                #pragma omp for schedule(static)
                for (int i = 1; i < n; i++) busy_wait_us((n-i) / SCALE);
            } else {
                #pragma omp for schedule(static)
                for (int i = 1; i < n; i++) busy_wait_us(i / SCALE);
                #pragma omp for schedule(static)
                for (int i = 1; i < n; i++) busy_wait_us((n-i) / SCALE);
            }
        }
    }
    double elapsed = omp_get_wtime() - t_start;
    printf("%-40s  wall time = %.4f s\n", label, elapsed);
}

int main(void)
{
    int n = 200;
    printf("=== Q2: Schedule Analysis (n=%d, 2 threads) ===\n\n", n);

    run_case(n, "schedule(static) — 2 chunks",        0, 0);
    run_case(n, "schedule(static,1) — interleaved",   0, 1);
    run_case(n, "schedule(static) + nowait",           1, 0);

    printf("\nSee source comments for full analytical answers.\n");
    return 0;
}
