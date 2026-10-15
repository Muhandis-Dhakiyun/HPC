/*
 * Tutorial-05: OpenMP
 * Question 6: Determine the default scheduling policy of your OpenMP implementation
 *
 * HOW IT WORKS:
 * -------------
 * The OpenMP standard does NOT mandate a specific default schedule.
 * Implementations typically default to static, but this can vary.
 *
 * Strategy:
 *   1. Run a parallel for loop with the DEFAULT schedule (no schedule clause).
 *   2. Record which thread handles which iteration.
 *   3. Infer the schedule by inspecting the iteration-to-thread mapping:
 *        - static (big chunks):  contiguous blocks per thread
 *        - static,1 (round-robin): alternating 1 iteration per thread
 *        - dynamic or guided:    non-uniform, non-contiguous distribution
 *
 * Compile: gcc -O2 -fopenmp q6_default_schedule.c -o q6
 * Run:     ./q6
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define N 32   /* number of iterations — small so output is readable */

int main(void)
{
    int owner[N];   /* owner[i] = thread that executed iteration i */
    memset(owner, -1, sizeof(owner));

    int num_threads;

    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();

        /* Default schedule — NO schedule clause */
        #pragma omp for
        for (int i = 0; i < N; i++)
            owner[i] = omp_get_thread_num();
    }

    printf("=== Q6: Default Schedule Detection ===\n");
    printf("Threads: %d,  Iterations: %d\n\n", num_threads, N);

    printf("Iteration-to-thread mapping:\n");
    for (int i = 0; i < N; i++)
        printf("  iter %2d -> thread %d\n", i, owner[i]);

    /* ---- Classify the observed schedule ---- */
    int chunk_size    = 0;
    int is_contiguous = 1;
    int is_roundrobin = 1;

    /* Check contiguous chunks (static big-block) */
    int current_thread = owner[0];
    int current_chunk  = 1;
    int min_chunk      = N;
    int max_chunk      = 0;
    for (int i = 1; i < N; i++) {
        if (owner[i] == current_thread) {
            current_chunk++;
        } else {
            if (current_chunk < min_chunk) min_chunk = current_chunk;
            if (current_chunk > max_chunk) max_chunk = current_chunk;
            current_thread = owner[i];
            current_chunk  = 1;
        }
    }
    if (current_chunk < min_chunk) min_chunk = current_chunk;
    if (current_chunk > max_chunk) max_chunk = current_chunk;

    /* Check round-robin (chunk=1, alternating) */
    for (int i = 0; i < N - num_threads; i++) {
        if (owner[i] != owner[i + num_threads]) {
            is_roundrobin = 0;
            break;
        }
    }

    printf("\n--- Analysis ---\n");
    printf("Min contiguous run: %d,  Max contiguous run: %d\n", min_chunk, max_chunk);

    if (max_chunk > 1 && (max_chunk - min_chunk) <= 1) {
        printf("Detected: schedule(STATIC) — large contiguous chunks\n");
        printf("Chunk size ≈ N / num_threads = %d / %d ≈ %d\n",
               N, num_threads, N / num_threads);
    } else if (is_roundrobin) {
        printf("Detected: schedule(STATIC, 1) — round-robin\n");
    } else {
        printf("Detected: schedule(DYNAMIC or GUIDED) — irregular distribution\n");
    }

    printf("\nConclusion: Most OpenMP implementations default to schedule(static),\n");
    printf("which assigns floor(N/T) or ceil(N/T) consecutive iterations per thread.\n");

    return 0;
}
