#!/bin/bash

# Function to compile and run a specific version
compile_and_run() {
    local dir=$1
    local src_file=$2
    local compiler=$3
    local output_name="${dir}_exec"
    local compiler_flags=""

    echo "\nCompiling and running $dir..."
    cd "$dir" || exit 1

    # Set compiler flags based on compiler type
    if [ "$compiler" == "nvcc" ]; then
        compiler_flags="-O3 -arch=sm_60 -I../common"
    else
        compiler_flags="-O3 -I../common"
    fi

    # Compile the source file
    if $compiler $compiler_flags $src_file -o $output_name; then
        echo "Compilation successful!"
        echo "Running $dir version..."
        ./$output_name
    else
        echo "Compilation failed for $dir"
    fi

    cd ..
}

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA toolkit (nvcc) not found. Please install CUDA toolkit first."
    exit 1
}

# Compile and run CPU baseline
compile_and_run "v0_cpu" "spmv_cpu.cpp" "g++"

# Compile and run GPU versions
compile_and_run "v1_scalar_gpu" "spmv_scalar_gpu.cu" "nvcc"
compile_and_run "v2_vector_gpu" "spmv_vector_gpu.cu" "nvcc"
compile_and_run "v3_warp_sync" "spmv_warp_sync.cu" "nvcc"
compile_and_run "v4_sharedmem_cache_x" "spmv_sharedmem.cu" "nvcc"

echo "\nAll versions have been compiled and run!"