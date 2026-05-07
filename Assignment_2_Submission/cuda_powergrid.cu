/*
 * SL PowerGrid HPC Simulator — CUDA Implementation
 * SE3082 Parallel Computing, Assignment 2
 *
 * Task: Parallel Power Demand Satisfaction for a Grid Network
 *
 * Each city node has a power demand. This CUDA program simulates parallel
 * load assignment where each GPU thread handles one city node and calculates
 * how much power it receives based on a simplified generator model.
 *
 * Two kernels are implemented:
 *   1. Unoptimized: each thread reads from global memory independently
 *   2. Optimized: uses shared memory tiling to cache generator data,
 *      reducing redundant global memory reads
 *
 * Compile on Google Colab:
 *   !nvcc cuda_powergrid.cu -o cuda_powergrid
 *   !./cuda_powergrid
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------
#define NUM_CITIES      10000    // Number of city (consumer) nodes
#define NUM_GENERATORS  100      // Number of generator nodes
#define BLOCK_SIZE      256      // Threads per block
#define TILE_SIZE       256      // Shared memory tile for generators

// ---------------------------------------------------------------
// Data Structures
// ---------------------------------------------------------------
typedef struct {
    float demand;    // Power demanded by this city (kW)
    float load;      // Power actually received (kW)
} City;

typedef struct {
    float supply;    // Total power supply available (kW)
    int   zone;      // Zone ID (cities in the same zone get priority)
} Generator;

// ---------------------------------------------------------------
// UNOPTIMIZED KERNEL
// Each thread reads all generator data directly from global memory.
// This causes high global memory latency due to repeated uncoalesced reads.
// ---------------------------------------------------------------
__global__ void assignPower_unoptimized(City *cities,
                                        Generator *generators,
                                        int numCities,
                                        int numGenerators,
                                        float *totalSupplyPerGen) {
    int city_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (city_id >= numCities) return;

    float needed = cities[city_id].demand;
    float received = 0.0f;

    // Naively iterate over all generators from global memory
    for (int g = 0; g < numGenerators; g++) {
        float available = totalSupplyPerGen[g];
        float share = available / (float)numCities;  // equal share model
        received += share;
        if (received >= needed) {
            received = needed;
            break;
        }
    }

    cities[city_id].load = received;
}

// ---------------------------------------------------------------
// OPTIMIZED KERNEL — Shared Memory Tiling
// Generators are loaded into shared memory in tiles. Each block
// reuses the same tile for all threads, drastically reducing
// global memory bandwidth usage.
// ---------------------------------------------------------------
__global__ void assignPower_optimized(City *cities,
                                      Generator *generators,
                                      int numCities,
                                      int numGenerators,
                                      float *totalSupplyPerGen) {
    int city_id = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float tile_supply[TILE_SIZE];

    float needed   = (city_id < numCities) ? cities[city_id].demand : 0.0f;
    float received = 0.0f;

    // Process generators in tiles loaded into shared memory
    for (int tile_start = 0; tile_start < numGenerators; tile_start += TILE_SIZE) {
        int tile_idx = tile_start + threadIdx.x;

        // Cooperatively load one tile of generator supply into shared memory
        if (tile_idx < numGenerators) {
            tile_supply[threadIdx.x] = totalSupplyPerGen[tile_idx];
        } else {
            tile_supply[threadIdx.x] = 0.0f;
        }
        __syncthreads();  // Ensure all threads have loaded their tile element

        // Use the cached tile to compute the city's share
        if (city_id < numCities) {
            int tile_len = min(TILE_SIZE, numGenerators - tile_start);
            for (int g = 0; g < tile_len; g++) {
                float share = tile_supply[g] / (float)numCities;
                received += share;
                if (received >= needed) {
                    received = needed;
                    break;
                }
            }
        }
        __syncthreads();  // Ready for next tile
    }

    if (city_id < numCities) {
        cities[city_id].load = received;
    }
}

// ---------------------------------------------------------------
// Main
// ---------------------------------------------------------------
int main() {
    printf("SL PowerGrid HPC Simulator — CUDA\n");
    printf("Cities: %d | Generators: %d | Block size: %d\n\n",
           NUM_CITIES, NUM_GENERATORS, BLOCK_SIZE);

    // --- Host memory allocation ---
    City      *h_cities     = (City*)     malloc(NUM_CITIES     * sizeof(City));
    Generator *h_generators = (Generator*)malloc(NUM_GENERATORS * sizeof(Generator));
    float     *h_supply     = (float*)    malloc(NUM_GENERATORS * sizeof(float));

    // Initialize cities with random demands (50–200 kW)
    srand(42);
    float total_demand = 0.0f;
    for (int i = 0; i < NUM_CITIES; i++) {
        h_cities[i].demand = (float)(rand() % 151 + 50);
        h_cities[i].load   = 0.0f;
        total_demand += h_cities[i].demand;
    }

    // Initialize generators with generous supply
    for (int g = 0; g < NUM_GENERATORS; g++) {
        h_generators[g].supply = total_demand / NUM_GENERATORS * 1.2f; // 20% surplus
        h_generators[g].zone   = g % 10;
        h_supply[g]            = h_generators[g].supply;
    }

    printf("Total city demand: %.2f kW\n", total_demand);
    printf("Total generator supply: %.2f kW\n\n",
           h_generators[0].supply * NUM_GENERATORS);

    // --- Device memory allocation ---
    City      *d_cities;
    Generator *d_generators;
    float     *d_supply;

    cudaMalloc(&d_cities,     NUM_CITIES     * sizeof(City));
    cudaMalloc(&d_generators, NUM_GENERATORS * sizeof(Generator));
    cudaMalloc(&d_supply,     NUM_GENERATORS * sizeof(float));

    cudaMemcpy(d_cities,     h_cities,     NUM_CITIES     * sizeof(City),      cudaMemcpyHostToDevice);
    cudaMemcpy(d_generators, h_generators, NUM_GENERATORS * sizeof(Generator), cudaMemcpyHostToDevice);
    cudaMemcpy(d_supply,     h_supply,     NUM_GENERATORS * sizeof(float),     cudaMemcpyHostToDevice);

    // Grid/block dimensions
    dim3 block(BLOCK_SIZE);
    dim3 grid((NUM_CITIES + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("Grid: %d blocks × %d threads = %d total threads\n\n",
           grid.x, block.x, grid.x * block.x);

    // --- CUDA timing events ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ==========================================
    // RUN UNOPTIMIZED KERNEL
    // ==========================================
    printf("--- Unoptimized Kernel (Global Memory) ---\n");
    // Reset city loads
    cudaMemset(d_cities, 0, NUM_CITIES * sizeof(City));
    cudaMemcpy(d_cities, h_cities, NUM_CITIES * sizeof(City), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    assignPower_unoptimized<<<grid, block>>>(d_cities, d_generators,
                                             NUM_CITIES, NUM_GENERATORS, d_supply);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_unopt = 0;
    cudaEventElapsedTime(&ms_unopt, start, stop);
    printf("Execution Time: %.4f ms\n\n", ms_unopt);

    // Read back results
    City *h_result_unopt = (City*)malloc(NUM_CITIES * sizeof(City));
    cudaMemcpy(h_result_unopt, d_cities, NUM_CITIES * sizeof(City), cudaMemcpyDeviceToHost);
    float served_unopt = 0.0f;
    for (int i = 0; i < NUM_CITIES; i++) served_unopt += h_result_unopt[i].load;
    printf("Total Demand Served (Unoptimized): %.2f / %.2f kW\n\n", served_unopt, total_demand);

    // ==========================================
    // RUN OPTIMIZED KERNEL
    // ==========================================
    printf("--- Optimized Kernel (Shared Memory Tiling) ---\n");
    // Reset city loads
    cudaMemcpy(d_cities, h_cities, NUM_CITIES * sizeof(City), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    assignPower_optimized<<<grid, block>>>(d_cities, d_generators,
                                           NUM_CITIES, NUM_GENERATORS, d_supply);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_opt = 0;
    cudaEventElapsedTime(&ms_opt, start, stop);
    printf("Execution Time: %.4f ms\n\n", ms_opt);

    // Read back results
    City *h_result_opt = (City*)malloc(NUM_CITIES * sizeof(City));
    cudaMemcpy(h_result_opt, d_cities, NUM_CITIES * sizeof(City), cudaMemcpyDeviceToHost);
    float served_opt = 0.0f;
    for (int i = 0; i < NUM_CITIES; i++) served_opt += h_result_opt[i].load;
    printf("Total Demand Served (Optimized): %.2f / %.2f kW\n\n", served_opt, total_demand);

    // ==========================================
    // SUMMARY
    // ==========================================
    printf("============================================\n");
    printf("PERFORMANCE SUMMARY\n");
    printf("============================================\n");
    printf("Unoptimized (Global Memory): %.4f ms\n", ms_unopt);
    printf("Optimized (Shared Memory) : %.4f ms\n",  ms_opt);
    if (ms_opt > 0)
        printf("Speedup                  : %.2fx\n", ms_unopt / ms_opt);
    printf("============================================\n");

    // --- Cleanup ---
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_cities);
    cudaFree(d_generators);
    cudaFree(d_supply);
    free(h_cities);
    free(h_generators);
    free(h_supply);
    free(h_result_unopt);
    free(h_result_opt);

    return 0;
}
