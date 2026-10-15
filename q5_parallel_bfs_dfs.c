/*
 * Tutorial-05: OpenMP
 * Question 5: Parallel BFS and DFS traversal using OpenMP
 *
 * HOW IT WORKS:
 * -------------
 * We use a generic adjacency-list graph.
 *
 * PARALLEL BFS:
 *   - Process one level (frontier) at a time.
 *   - All nodes in the current frontier can be expanded in parallel.
 *   - Use an atomic counter + a shared "next frontier" array to collect
 *     newly discovered nodes without races.
 *
 * PARALLEL DFS (task-based):
 *   - Each recursive DFS call becomes an OpenMP task.
 *   - Visited[] is protected: once a node is marked visited (with an atomic
 *     CAS-style approach using omp atomic capture), only one thread recurses.
 *
 * Compile: gcc -O2 -fopenmp q5_parallel_bfs_dfs.c -o q5
 * Run:     ./q5
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define MAX_NODES  1024
#define MAX_EDGES  8192

/* ---- Adjacency list graph ---- */
typedef struct {
    int  adj[MAX_NODES][64];   /* neighbour list per node  */
    int  deg[MAX_NODES];       /* degree of each node       */
    int  n;                    /* number of nodes           */
} Graph;

static Graph G;

static void graph_init(int n)
{
    G.n = n;
    memset(G.deg, 0, sizeof(G.deg));
}

static void graph_add_edge(int u, int v)
{
    G.adj[u][G.deg[u]++] = v;
    G.adj[v][G.deg[v]++] = u;
}

/* ---- Build a random graph ---- */
static void build_random_graph(int n, int extra_edges)
{
    graph_init(n);
    /* Spanning tree first so graph is connected */
    for (int i = 1; i < n; i++)
        graph_add_edge(i, rand() % i);
    /* Extra random edges */
    for (int e = 0; e < extra_edges; e++) {
        int u = rand() % n;
        int v = rand() % n;
        if (u != v) graph_add_edge(u, v);
    }
}

/* ======================================================
 * PARALLEL BFS
 * ====================================================== */
static int bfs_visited[MAX_NODES];
static int bfs_order[MAX_NODES];
static int bfs_order_cnt = 0;

void parallel_bfs(int src)
{
    memset(bfs_visited, 0, sizeof(int) * G.n);
    bfs_order_cnt = 0;

    /* frontier and next_frontier are plain arrays with a shared counter */
    int *frontier      = malloc(G.n * sizeof(int));
    int *next_frontier = malloc(G.n * sizeof(int));

    int frontier_size      = 1;
    int next_frontier_size = 0;

    frontier[0]       = src;
    bfs_visited[src]  = 1;

    while (frontier_size > 0) {
        next_frontier_size = 0;

        #pragma omp parallel for schedule(dynamic) \
                shared(next_frontier, next_frontier_size)
        for (int fi = 0; fi < frontier_size; fi++) {
            int u = frontier[fi];

            /* Record visit (atomic so no two threads stomp on the counter) */
            int pos;
            #pragma omp atomic capture
            pos = bfs_order_cnt++;
            bfs_order[pos] = u;

            for (int j = 0; j < G.deg[u]; j++) {
                int v = G.adj[u][j];
                int old_visited;
                /* Atomically set visited[v] = 1 and capture old value */
                #pragma omp atomic capture
                { old_visited = bfs_visited[v]; bfs_visited[v] = 1; }

                if (!old_visited) {
                    /* Only one thread will see old_visited == 0 */
                    int npos;
                    #pragma omp atomic capture
                    npos = next_frontier_size++;
                    next_frontier[npos] = v;
                }
            }
        }

        /* Swap frontiers */
        int *tmp    = frontier;
        frontier    = next_frontier;
        next_frontier = tmp;
        frontier_size      = next_frontier_size;
    }

    free(frontier);
    free(next_frontier);
}

/* ======================================================
 * PARALLEL DFS  (task‑based)
 * ====================================================== */
static int dfs_visited[MAX_NODES];
static int dfs_order[MAX_NODES];
static int dfs_order_cnt = 0;

void dfs_task(int u)
{
    int old;
    #pragma omp atomic capture
    { old = dfs_visited[u]; dfs_visited[u] = 1; }

    if (old) return;   /* already visited by another thread */

    int pos;
    #pragma omp atomic capture
    pos = dfs_order_cnt++;
    dfs_order[pos] = u;

    for (int j = 0; j < G.deg[u]; j++) {
        int v = G.adj[u][j];
        if (!dfs_visited[v]) {
            #pragma omp task
            dfs_task(v);
        }
    }
    #pragma omp taskwait
}

void parallel_dfs(int src)
{
    memset(dfs_visited, 0, sizeof(int) * G.n);
    dfs_order_cnt = 0;

    #pragma omp parallel
    {
        #pragma omp single
        dfs_task(src);
    }
}

/* ======================================================
 * MAIN
 * ====================================================== */
int main(void)
{
    srand(42);
    const int N    = 20;
    const int EXTRA = 15;

    build_random_graph(N, EXTRA);

    printf("=== Q5: Parallel BFS & DFS ===\n");
    printf("Graph: %d nodes, threads: %d\n\n", N, omp_get_max_threads());

    /* --- BFS --- */
    double t0 = omp_get_wtime();
    parallel_bfs(0);
    double t1 = omp_get_wtime();

    printf("BFS visit order (%.6f s):\n  ", t1 - t0);
    for (int i = 0; i < bfs_order_cnt; i++)
        printf("%d ", bfs_order[i]);
    printf("\n\n");

    /* --- DFS --- */
    t0 = omp_get_wtime();
    parallel_dfs(0);
    t1 = omp_get_wtime();

    printf("DFS visit order (%.6f s):\n  ", t1 - t0);
    for (int i = 0; i < dfs_order_cnt; i++)
        printf("%d ", dfs_order[i]);
    printf("\n\n");

    /* Verify all nodes visited */
    int bfs_ok = (bfs_order_cnt == N);
    int dfs_ok = (dfs_order_cnt == N);
    printf("BFS visited all %d nodes: %s\n", N, bfs_ok ? "YES ✓" : "NO ✗");
    printf("DFS visited all %d nodes: %s\n", N, dfs_ok ? "YES ✓" : "NO ✗");

    return 0;
}
