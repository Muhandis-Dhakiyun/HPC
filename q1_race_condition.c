/*
 * Tutorial-05: OpenMP
 * Question 1: Differences between the three pieces of code
 *
 * The PDF shows two variants; we analyze both + add a third correct one.
 *
 * Array v has length m.  indices[] maps loop iteration i -> some index in v.
 * Values in indices[] may REPEAT, so multiple iterations can write to the
 * same element of v — this is the core race‑condition scenario.
 *
 * VARIANT 1 — INCORRECT (data race)
 * -----------------------------------
 *   #pragma omp parallel for default(none) shared(v, indices, n)
 *   for (i = 0; i < n; i++)
 *       v[ indices[i] ] += f(i);
 *
 *   Problem: if two threads have the same indices[i] value they both do
 *   read‑modify‑write on the same v[] cell simultaneously → race condition.
 *   Result is non‑deterministic / wrong.
 *
 * VARIANT 2 — CORRECT but slow (critical section)
 * -------------------------------------------------
 *   #pragma omp parallel for default(none) shared(v, indices, n)
 *   for (i = 0; i < n; i++) {
 *       #pragma omp critical
 *       v[ indices[i] ] += f(i);
 *   }
 *
 *   The critical directive serialises every update.  This is correct but
 *   effectively sequential — no speedup at all.
 *
 * VARIANT 3 — CORRECT and parallel (atomic)
 * ------------------------------------------
 *   #pragma omp parallel for default(none) shared(v, indices, n)
 *   for (i = 0; i < n; i++) {
 *       #pragma omp atomic
 *       v[ indices[i] ] += f(i);
 *   }
 *
 *   atomic is cheaper than critical when only a single memory operation
 *   needs protection.  Multiple different indices are updated in parallel;
 *   only colliding indices are serialised by hardware.
 *
 * Summary:
 *   Variant 1  — WRONG   (race condition)
 *   Variant 2  — CORRECT but no speedup (full critical section)
 *   Variant 3  — CORRECT and faster (atomic update)
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* A simple function that takes i ms to compute — here we just return i */
static double f(int i) { return (double)i; }

int main(void)
{
    const int n = 1000;  /* number of iterations */
    const int m = 50;    /* size of v            */

    double *v       = calloc(m, sizeof(double));
    int    *indices = malloc(n * sizeof(int));

    /* Fill indices with values in [0, m) so collisions happen */
    for (int i = 0; i < n; i++)
        indices[i] = i % m;

    /* ---- VARIANT 1: INCORRECT ---- */
    for (int j = 0; j < m; j++) v[j] = 0.0;
    #pragma omp parallel for default(none) shared(v, indices, n)
    for (int i = 0; i < n; i++)
        v[ indices[i] ] += f(i);
    printf("[Variant 1 - RACE] v[0] = %.2f  (result may be wrong)\n", v[0]);

    /* ---- VARIANT 2: CORRECT, slow ---- */
    for (int j = 0; j < m; j++) v[j] = 0.0;
    #pragma omp parallel for default(none) shared(v, indices, n)
    for (int i = 0; i < n; i++) {
        #pragma omp critical
        v[ indices[i] ] += f(i);
    }
    printf("[Variant 2 - CRITICAL] v[0] = %.2f  (correct)\n", v[0]);

    /* ---- VARIANT 3: CORRECT, faster ---- */
    for (int j = 0; j < m; j++) v[j] = 0.0;
    #pragma omp parallel for default(none) shared(v, indices, n)
    for (int i = 0; i < n; i++) {
        #pragma omp atomic
        v[ indices[i] ] += f(i);
    }
    printf("[Variant 3 - ATOMIC] v[0] = %.2f  (correct and faster)\n", v[0]);

    free(v);
    free(indices);
    return 0;
}
