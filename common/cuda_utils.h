#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

// Error checking macro
#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)

inline void check_cuda_error(cudaError_t result, const char *func, const char *file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n%s\n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}

// Timer class for performance measurement
class Timer {
public:
    Timer() : start_time(std::chrono::high_resolution_clock::now()) {}
    
    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed() {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> diff = end_time - start_time;
        return diff.count();
    }
    
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
};

// Utility function to calculate grid and block dimensions
inline void calculate_grid_block_dim(int n, int& grid_dim, int& block_dim) {
    block_dim = 256; // Default block size
    grid_dim = (n + block_dim - 1) / block_dim;
}

#endif // CUDA_UTILS_H