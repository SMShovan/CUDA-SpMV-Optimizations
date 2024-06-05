#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <cassert>

// CSR format sparse matrix structure
struct CSRMatrix {
    std::vector<double> values;     // Non-zero values
    std::vector<int> col_indices;   // Column indices for values
    std::vector<int> row_ptr;      // Row pointers into values/col_indices
    int num_rows;
    int num_cols;
    
    CSRMatrix(int rows, int cols) : num_rows(rows), num_cols(cols) {
        row_ptr.resize(rows + 1, 0);
    }
};

// Generate a random sparse matrix in CSR format
CSRMatrix generate_random_sparse_matrix(int rows, int cols, float density) {
    CSRMatrix matrix(rows, cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    // First pass: count non-zeros per row
    int total_nnz = 0;
    for (int i = 0; i < rows; ++i) {
        int row_nnz = 0;
        for (int j = 0; j < cols; ++j) {
            if (dis(gen) < density) {
                row_nnz++;
            }
        }
        matrix.row_ptr[i + 1] = matrix.row_ptr[i] + row_nnz;
        total_nnz += row_nnz;
    }
    
    // Allocate space for values and column indices
    matrix.values.reserve(total_nnz);
    matrix.col_indices.reserve(total_nnz);
    
    // Second pass: fill in values and column indices
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (dis(gen) < density) {
                matrix.values.push_back(dis(gen));
                matrix.col_indices.push_back(j);
            }
        }
    }
    
    return matrix;
}

// CPU implementation of SpMV
void spmv_cpu(const CSRMatrix& matrix, const std::vector<double>& x, std::vector<double>& y) {
    for (int i = 0; i < matrix.num_rows; ++i) {
        double sum = 0.0;
        for (int j = matrix.row_ptr[i]; j < matrix.row_ptr[i + 1]; ++j) {
            sum += matrix.values[j] * x[matrix.col_indices[j]];
        }
        y[i] = sum;
    }
}

// Verify result against a simple dense matrix multiplication
bool verify_result(const CSRMatrix& matrix, const std::vector<double>& x,
                   const std::vector<double>& y) {
    std::vector<double> y_dense(matrix.num_rows, 0.0);
    
    // Compute dense matrix-vector product
    for (int i = 0; i < matrix.num_rows; ++i) {
        for (int j = matrix.row_ptr[i]; j < matrix.row_ptr[i + 1]; ++j) {
            y_dense[i] += matrix.values[j] * x[matrix.col_indices[j]];
        }
    }
    
    // Compare results
    const double epsilon = 1e-10;
    for (int i = 0; i < matrix.num_rows; ++i) {
        if (std::abs(y[i] - y_dense[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

int main() {
    // Test parameters
    const int num_rows = 1000;
    const int num_cols = 1000;
    const float density = 0.01; // 1% non-zero elements
    
    // Generate random sparse matrix
    CSRMatrix matrix = generate_random_sparse_matrix(num_rows, num_cols, density);
    
    // Generate random input vector
    std::vector<double> x(num_cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < num_cols; ++i) {
        x[i] = dis(gen);
    }
    
    // Output vector
    std::vector<double> y(num_rows, 0.0);
    
    // Warm-up run
    spmv_cpu(matrix, x, y);
    
    // Benchmark
    const int num_iterations = 100;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_iterations; ++i) {
        spmv_cpu(matrix, x, y);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    double avg_time = diff.count() * 1000 / num_iterations; // Convert to milliseconds
    
    // Verify result
    bool is_correct = verify_result(matrix, x, y);
    
    // Print statistics
    std::cout << "Matrix size: " << num_rows << "x" << num_cols << std::endl;
    std::cout << "Non-zero elements: " << matrix.values.size() << std::endl;
    std::cout << "Density: " << density * 100 << "%" << std::endl;
    std::cout << "Average execution time: " << avg_time << " ms" << std::endl;
    std::cout << "Result verification: " << (is_correct ? "PASSED" : "FAILED") << std::endl;
    
    return 0;
}