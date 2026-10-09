# HPC

# Evaluation Platform & Targeted HPC Optimizations

---

## 1. Evaluation Platform & Architectural Justification

### Platform Specifications
* **Host CPU:** AMD Ryzen 7 7840HS (8 Cores / 16 Threads, Zen 4, 3.8 GHz base / 5.1 GHz boost, 16 MB L3 cache, 16 GB DDR5 RAM).
* **Device GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (Ada Lovelace, 3,072 CUDA Cores, 24 SMs, 8 GB GDDR6 VRAM, 32 MB on-chip L2 cache, SM 8.9, 88W TGP).
* **Software Stack:** Ubuntu 24.04 LTS (WSL2 direct GPU kernel pass-through), CUDA 12.8, OpenMP 4.5, GCC 13.3, Nsight Systems 2024.5.

### Platform Justification
1. **Modern Architectural Verification (Ada Lovelace vs. Pascal):**  
   The 2018 paper evaluated Hornet on an enterprise Pascal Tesla P100 (SM 6.0, 4 MB L2). Testing on Ada Lovelace (SM 8.9, 32 MB L2) verifies whether the paper's claims and bottlenecks hold across a 6-year architectural leap with an $8\times$ larger L2 cache and modern SM instruction issue.
2. **Accessible "Edge HPC" Feasibility:**  
   Demonstrates that state-of-the-art dynamic graph streaming ($200\text{M}$ updates/s, $10.5\text{ GTEPS}$) is fully achievable on commodity, portable consumer laptop hardware ($<\$1,000$), without requiring multi-thousand-dollar server accelerators (A100/H100).
3. **Exposing Heterogeneous Bottlenecks:**  
   Pairing a 16-thread Zen 4 CPU with a 3,072-core Ada GPU provides the exact compute balance needed to isolate host-side runtime bounds versus device execution bounds under Amdahl’s Law.

---

## 2. Master Optimizations Table (Slide Format)

| # | Optimization | Programming Model / Tool | Platform & Hardware Target | Concrete Technique | Expected Impact |
| :-: | :--- | :--- | :--- | :--- | :-: |
| **1** | **Device-Side Lock-Free Slab Allocator** | **CUDA** (Device Atomics) | **GPU Device:**<br>RTX 4060 VRAM + 24 SMs | Move block free-lists directly into GPU VRAM; claim/free blocks via GPU `atomicSub`. | **Up to $1.85\times$ overall speedup** (saturates Amdahl bound; eliminates 63.8% idle gap). |
| **2** | **Pre-Allocated Memory Arenas** | **CUDA** (Runtime API) | **GPU VRAM:**<br>8 GB GDDR6 Reserve Pool | Pre-allocate a 20% growth reserve during graph setup; make dynamic batch updates zero-allocation. | **Eliminates multi-ms tail latency spikes** (avoids 2.06 ms `cudaMalloc` stalls). |
| **3** | **Warp-Aggregated Atomics** | **CUDA** (Warp Intrinsics) | **GPU SM Hardware:**<br>Ada Lovelace Warp Units | Use register shuffles (`__ballot_sync`, `__popc`, `__shfl_sync`) to issue 1 atomic per 32-thread warp. | **$32\times$ reduction in atomic bus transactions**; 60% faster queue building. |
| **4** | **Multi-Threaded Initialization** | **OpenMP** (C++ Multithreading) | **Host CPU:**<br>Ryzen 7 7840HS (16 Threads) | Parallelize vertex degree computation and staging across all 16 CPU threads with `#pragma omp parallel for`. | **$10\times\text{--}14\times$ faster graph loading** (drops initialization from 350 ms to $<25$ ms). |
| **5** | **Asynchronous Dual-Stream Pipelining** | **CUDA Streams** (Asynchronous API) | **Heterogeneous Concurrency:**<br>Dual CUDA Streams + Hardware Engines | Double-buffer updates; overlap sorting/dedup of Batch $K+1$ with edge appending of Batch $K$. | **Hides up to 3.2 ms preprocessing latency** per batch ($1.3\times$ streaming gain). |
| **6** | **Unordered Map + Cache Scan (over B+Tree)** | **C++ STL** (Cache-Conscious) | **Host CPU Cache:**<br>Zen 4 L1 Data Cache (32 KB/core) | Retain flat hash table + linear scan over theoretical B+Tree; leverage 64-byte L1 cache line locality. | **$2.7\times$ faster allocation** and $4.3\times$ faster deallocation than B+Tree. |

---

## 3. High-Impact Slide Summary: Division of Labor

```
                                      PROGRAMMING MODEL & PLATFORM
                   ┌────────────────────────────────┴────────────────────────────────┐
                   ▼                                                                 ▼
         OPENMP & C++ STL (HOST)                                     CUDA & CUDA STREAMS (DEVICE)
       AMD Ryzen 7 7840HS (16 Threads)                             NVIDIA RTX 4060 (3,072 CUDA Cores)
  ──────────────────────────────────────────                     ──────────────────────────────────────────
  • OpenMP: Parallel Graph Init (16 threads)                     • CUDA Atomics: Device-Side Slab Allocator
  • C++ STL: L1 Cache Linear Scan (over B+Tree)                  • CUDA Runtime: Pre-Allocated Growth Arenas
                                                                 • CUDA Warp Intrinsics: Warp-Aggregated Shuffles
                                                                 • CUDA Streams: Dual-Stream Asynchronous Pipeline
```

### Presentation Talking Points
1. **Tooling & Platform Justification:**  
   *"We use **OpenMP** on the AMD Ryzen 7 7840HS (16 threads) and **CUDA / CUDA Streams** on the NVIDIA RTX 4060 Laptop GPU. This modern testbed proves that high-throughput dynamic graph processing ($200\text{M}$ upd/s) is achievable on an accessible consumer system without enterprise servers."*
2. **The Core Hotspot:**  
   *"Nsight Systems reveals that our GPU kernels finish in only $4.69\text{ ms}$, but the GPU sits idle for $63.8\%$ of the time because a single-threaded CPU memory manager takes $5.54\text{ ms}$ to manage blocks over PCIe."*
3. **The Solution Division:**  
   *"We divide the work strictly by the best tool for the job:  
   - **CUDA Atomics & Warp Intrinsics** keep block allocation and mutation 100% on the GPU, eliminating the 8.2 ms PCIe idle gap.  
   - **CUDA Streams** pipeline preprocessing and insertion concurrently.  
   - **OpenMP** parallelizes host graph initialization across all 16 CPU threads."*
