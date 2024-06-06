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

// Kernel for scalar SpMV implementation (one thread per row)
__global__ void spmv_scalar_kernel(const double* values,
                                  const int* col_indices,
                                  const int* row_ptr,
                                  const double* x,
                                  double* y,
                                  const int num_rows) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (row < num_rows) {
        double sum = 0.0;
        const int row_start = row_ptr[row];
        const int row_end = row_ptr[row + 1];
        
        // Each thread processes all elements in its row
        for (int j = row_start; j < row_end; j++) {
            sum += values[j] * x[col_indices[j]];
        }
        
        y[row] = sum;
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
    
    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix.values, h_values.size() * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix.col_indices, h_col_indices.size() * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix.row_ptr, h_row_ptr.size() * sizeof(int)));
    
    // Copy data to device
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
    
    // First pass: count non-zeros per row
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
    int block_size = 256;
    int grid_size = (num_rows + block_size - 1) / block_size;
    
    // Warm-up run
    spmv_scalar_kernel<<<grid_size, block_size>>>(d_matrix.values,
                                                 d_matrix.col_indices,
                                                 d_matrix.row_ptr,
                                                 d_x, d_y, num_rows);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Benchmark
    const int num_iterations = 100;
    Timer timer;
    
    for (int i = 0; i < num_iterations; ++i) {
        spmv_scalar_kernel<<<grid_size, block_size>>>(d_matrix.values,
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
    
    // Cleanup
    free_gpu_matrix(d_matrix);
    CHECK_CUDA_ERROR(cudaFree(d_x));
    CHECK_CUDA_ERROR(cudaFree(d_y));
    
    return 0;
}