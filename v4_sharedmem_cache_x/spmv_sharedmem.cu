#include <cuda_runtime.h>
#include "../common/cuda_utils.h"
#include <vector>
#include <iostream>
#include <random>

// CSR matrix structure for GPU
struct CSRMatrix {
    double* values;        // Non-zero values
    int* col_indices;      // Column indices for values
    int* row_ptr;         // Row pointers into values/col_indices
    int num_rows;
    int num_cols;
    int nnz;              // Number of non-zero elements
};

// Constants for kernel configuration
const int WARP_SIZE = 32;
const int BLOCK_SIZE = 256;
const int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;
const int CACHE_SIZE = 1024;  // Size of shared memory cache for input vector

// Warp-level reduction using shuffle instructions
__device__ __forceinline__ double warp_reduce_sum(double val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel for shared memory cached SpMV implementation
__global__ void spmv_sharedmem_kernel(const double* __restrict__ values,
                                     const int* __restrict__ col_indices,
                                     const int* __restrict__ row_ptr,
                                     const double* __restrict__ x,
                                     double* __restrict__ y,
                                     const int num_rows) {
    __shared__ double x_cache[CACHE_SIZE];
    
    const int warp_id = (blockIdx.x * WARPS_PER_BLOCK) + (threadIdx.x / WARP_SIZE);
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int row = warp_id;
    
    // Load frequently accessed elements of x into shared memory
    for (int i = threadIdx.x; i < CACHE_SIZE; i += BLOCK_SIZE) {
        x_cache[i] = x[i];
    }
    __syncthreads();
    
    if (row < num_rows) {
        const int row_start = row_ptr[row];
        const int row_end = row_ptr[row + 1];
        double sum = 0.0;
        
        // Process elements using cached x values when possible
        for (int j = row_start + lane_id; j < row_end; j += WARP_SIZE) {
            const int col = col_indices[j];
            const double x_val = (col < CACHE_SIZE) ? x_cache[col] : x[col];
            sum += values[j] * x_val;
        }
        
        // Warp-level reduction
        sum = warp_reduce_sum(sum);
        
        // First thread in warp writes the result
        if (lane_id == 0) {
            y[row] = sum;
        }
    }
}

// Host function to allocate GPU memory and copy data
CSRMatrix create_gpu_matrix(const std::vector<double>& h_values,
                           const std::vector<int>& h_col_indices,
                           const std::vector<int>& h_row_ptr,
                           int num_rows, int num_cols) {
    CSRMatrix d_matrix;
    d_matrix.num_rows = num_rows;
    d_matrix.num_cols = num_cols;
    d_matrix.nnz = h_values.size();
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix.values, h_values.size() * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix.col_indices, h_col_indices.size() * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix.row_ptr, h_row_ptr.size() * sizeof(int)));
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix.values, h_values.data(),
                               h_values.size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix.col_indices, h_col_indices.data(),
                               h_col_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix.row_ptr, h_row_ptr.data(),
                               h_row_ptr.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    return d_matrix;
}

// Free GPU memory
void free_gpu_matrix(CSRMatrix& matrix) {
    CHECK_CUDA_ERROR(cudaFree(matrix.values));
    CHECK_CUDA_ERROR(cudaFree(matrix.col_indices));
    CHECK_CUDA_ERROR(cudaFree(matrix.row_ptr));
}

// Generate random sparse matrix (host)
void generate_random_sparse_matrix(int rows, int cols, float density,
                                 std::vector<double>& values,
                                 std::vector<int>& col_indices,
                                 std::vector<int>& row_ptr) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    row_ptr.resize(rows + 1, 0);
    values.clear();
    col_indices.clear();
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (dis(gen) < density) {
                values.push_back(dis(gen));
                col_indices.push_back(j);
            }
        }
        row_ptr[i + 1] = values.size();
    }
}

int main() {
    // Test parameters
    const int num_rows = 10000;
    const int num_cols = 10000;
    const float density = 0.01f; // 1% non-zero elements
    
    // Generate random sparse matrix on host
    std::vector<double> h_values;
    std::vector<int> h_col_indices;
    std::vector<int> h_row_ptr;
    generate_random_sparse_matrix(num_rows, num_cols, density,
                                h_values, h_col_indices, h_row_ptr);
    
    // Create GPU matrix
    CSRMatrix d_matrix = create_gpu_matrix(h_values, h_col_indices, h_row_ptr,
                                          num_rows, num_cols);
    
    // Generate random input vector
    std::vector<double> h_x(num_cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < num_cols; ++i) {
        h_x[i] = dis(gen);
    }
    
    // Allocate device vectors
    double *d_x, *d_y;
    CHECK_CUDA_ERROR(cudaMalloc(&d_x, num_cols * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_y, num_rows * sizeof(double)));
    
    // Copy input vector to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_x, h_x.data(), num_cols * sizeof(double),
                               cudaMemcpyHostToDevice));
    
    // Launch configuration
    int num_blocks = (num_rows + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    
    // Warm-up run
    spmv_sharedmem_kernel<<<num_blocks, BLOCK_SIZE>>>(d_matrix.values,
                                                     d_matrix.col_indices,
                                                     d_matrix.row_ptr,
                                                     d_x, d_y, num_rows);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Benchmark
    const int num_iterations = 100;
    Timer timer;
    
    for (int i = 0; i < num_iterations; ++i) {
        spmv_sharedmem_kernel<<<num_blocks, BLOCK_SIZE>>>(d_matrix.values,
                                                         d_matrix.col_indices,
                                                         d_matrix.row_ptr,
                                                         d_x, d_y, num_rows);
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    double avg_time = timer.elapsed() / num_iterations;
    
    // Copy result back to host for verification
    std::vector<double> h_y(num_rows);
    CHECK_CUDA_ERROR(cudaMemcpy(h_y.data(), d_y, num_rows * sizeof(double),
                               cudaMemcpyDeviceToHost));
    
    // Print statistics
    std::cout << "Matrix size: " << num_rows << "x" << num_cols << std::endl;
    std::cout << "Non-zero elements: " << d_matrix.nnz << std::endl;
    std::cout << "Density: " << density * 100 << "%" << std::endl;
    std::cout << "Average execution time: " << avg_time << " ms" << std::endl;
    std::cout << "Configuration: Shared memory cache size: " << CACHE_SIZE << std::endl;
    
    // Cleanup
    free_gpu_matrix(d_matrix);
    CHECK_CUDA_ERROR(cudaFree(d_x));
    CHECK_CUDA_ERROR(cudaFree(d_y));
    
    return 0;
}