# Sparse Matrix-Vector Multiplication (SpMV) Optimization

## Overview
This project implements and optimizes Sparse Matrix-Vector Multiplication (SpMV) operations using the CSR (Compressed Sparse Row) format. It demonstrates various optimization techniques for both CPU and GPU implementations, focusing on handling irregular memory access patterns and improving performance through different optimization strategies.

## Project Structure

- `common/` - Shared utilities for CUDA operations and timing
- `v0_cpu/` - Baseline CPU implementation using CSR format
- `v1_scalar_gpu/` - Basic GPU implementation with one thread per row
- `v2_vector_gpu/` - Improved GPU version with warp/block per row
- `v3_warp_sync/` - Optimized using warp-level primitives
- `v4_sharedmem_cache_x/` - Enhanced with shared memory caching
- `v5_format_comparison/` - (Optional) Implementation using ELLPACK format

## Optimization Stages

### v0_cpu: Baseline CPU Implementation
- Sequential CSR SpMV implementation
- Reference for correctness verification
- Performance baseline for comparison

### v1_scalar_gpu: Basic GPU Implementation
- One thread per matrix row
- Direct port to CUDA
- Known bottlenecks:
  - Load imbalance for varying row lengths
  - Uncoalesced reads from input vector

### v2_vector_gpu: Warp/Block-based Implementation
- One warp/thread block per matrix row
- Better load balancing for long rows
- Improved vector element reuse
- Remaining challenges:
  - Uncoalesced matrix value/index reads
  - Underutilization for short rows

### v3_warp_sync: Warp-Level Optimization
- Utilizes warp-level primitives for reduction
- Improved coalescing for matrix data access
- Enhanced thread cooperation within warps

### v4_sharedmem_cache_x: Shared Memory Optimization
- Caches input vector in shared memory
- Optimized for matrices with locality in column indices
- Reduced global memory access

### v5_format_comparison: Alternative Format (Optional)
- ELLPACK format implementation
- Performance comparison with CSR
- Analysis of format-specific trade-offs

## Key Concepts
- Irregular memory access patterns
- Sparse matrix storage formats
- Load balancing strategies
- Warp-level programming
- Shared memory optimization
- Data structure impact on performance

## Building and Running
Detailed instructions for building and running each version will be provided in their respective directories.

## Performance Analysis
Each implementation includes performance measurements and analysis, comparing different optimization strategies and their effectiveness for various matrix patterns.