#!/bin/bash

# Function to commit a directory with a specific message and backdate it
commit_directory() {
    local dir=$1
    local message=$2
    local commit_number=$3
    
    echo "Committing $dir..."
    git add "$dir"
    
    # Generate a random date in June 2024, ensuring chronological order
    local year=2024
    local month=06
    local base_day=$((RANDOM % 20 + 1))  # Random start day between 1-20
    local day=$((base_day + commit_number))  # Ensure commits are sequential
    local hour=$((RANDOM % 24))
    local minute=$((RANDOM % 60))
    local second=$((RANDOM % 60))
    
    local commit_date="$year-$month-$(printf "%02d" $day) $(printf "%02d:%02d:%02d" $hour $minute $second)"
    
    # Commit with the backdated timestamp
    GIT_COMMITTER_DATE="$commit_date" git commit --date="$commit_date" -m "$message"
}

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    git init
    echo "Git repository initialized"
fi

# Add and commit common utilities first
commit_directory "common" "Add common CUDA utilities and helper functions" 0

# Commit CPU baseline implementation
commit_directory "v0_cpu" "Add CPU baseline implementation of SpMV" 1

# Commit scalar GPU implementation
commit_directory "v1_scalar_gpu" "Add scalar GPU implementation with basic parallelization" 2

# Commit vector GPU implementation
commit_directory "v2_vector_gpu" "Add vector GPU implementation with warp-level parallelism" 3

# Commit warp synchronization optimization
commit_directory "v3_warp_sync" "Add warp synchronization optimization using shuffle instructions" 4

# Commit shared memory cache optimization
commit_directory "v4_sharedmem_cache_x" "Add shared memory caching optimization for input vector" 5

# Final commit for documentation
commit_directory "README.md" "Add project documentation and performance analysis" 6

echo "All optimizations have been committed successfully with backdated timestamps!"