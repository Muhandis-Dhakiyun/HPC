/*
 * Tutorial-05: OpenMP
 * Question 4: Parallel binary tree traversal to compute sum of all node values
 *
 * HOW IT WORKS:
 * -------------
 * We build a binary tree and compute the sum of all node values in parallel.
 *
 * Strategy:
 *   - Use OpenMP tasks (#pragma omp task) — each recursive call can be a task.
 *   - A taskwait waits for both child tasks before combining their results.
 *   - The root task is started inside a "single" block so only one thread
 *     kicks off the recursion; all threads in the parallel team pick up tasks.
 *
 * Thread safety:
 *   - Each task writes to its own local variable (sum).
 *   - The parent reads child sums only after taskwait → no race condition.
 *
 * Compile: gcc -O2 -fopenmp q4_parallel_tree_sum.c -o q4
 * Run:     ./q4
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* ---- Binary tree node ---- */
typedef struct Node {
    int          value;
    struct Node *left;
    struct Node *right;
} Node;

/* Allocate a new node */
static Node *new_node(int val)
{
    Node *n = malloc(sizeof(Node));
    n->value = val;
    n->left  = NULL;
    n->right = NULL;
    return n;
}

/* Build a complete binary tree of given depth with sequential values */
static Node *build_tree(int depth, int *counter)
{
    if (depth == 0) return NULL;
    Node *n  = new_node((*counter)++);
    n->left  = build_tree(depth - 1, counter);
    n->right = build_tree(depth - 1, counter);
    return n;
}

/* Free the tree */
static void free_tree(Node *n)
{
    if (!n) return;
    free_tree(n->left);
    free_tree(n->right);
    free(n);
}

/* ---- Sequential sum (for verification) ---- */
static long long seq_sum(Node *n)
{
    if (!n) return 0;
    return (long long)n->value + seq_sum(n->left) + seq_sum(n->right);
}

/* ---- Parallel sum using OpenMP tasks ---- */
static long long par_sum(Node *n)
{
    if (!n) return 0;

    long long left_sum  = 0;
    long long right_sum = 0;

    /* Spawn tasks for each subtree only if they are non-trivial */
    #pragma omp task shared(left_sum)
    left_sum  = par_sum(n->left);

    #pragma omp task shared(right_sum)
    right_sum = par_sum(n->right);

    /* Wait for both child tasks to finish before using their results */
    #pragma omp taskwait

    return (long long)n->value + left_sum + right_sum;
}

int main(void)
{
    const int DEPTH = 18;  /* 2^18 - 1 = 262143 nodes */

    int counter = 0;
    Node *root  = build_tree(DEPTH, &counter);
    int   total = counter;

    printf("=== Q4: Parallel Binary Tree Sum ===\n");
    printf("Tree depth: %d,  Total nodes: %d\n\n", DEPTH, total);
    printf("Threads: %d\n", omp_get_max_threads());

    /* Sequential */
    double t0  = omp_get_wtime();
    long long s = seq_sum(root);
    double t1  = omp_get_wtime();
    printf("Sequential sum = %lld  (%.4f s)\n", s, t1 - t0);

    /* Parallel */
    long long p = 0;
    t0 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        p = par_sum(root);
    }
    t1 = omp_get_wtime();
    printf("Parallel   sum = %lld  (%.4f s)\n", p, t1 - t0);

    if (s == p)
        printf("\nResults MATCH ✓  (thread‑safe and correct)\n");
    else
        printf("\nResults MISMATCH ✗  (bug!)\n");

    free_tree(root);
    return 0;
}
