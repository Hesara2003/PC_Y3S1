# SL PowerGrid HPC Simulator
> **High-Performance Computing Project (Serial → OpenMP → MPI → OpenCL)**

A scalable power grid load balancing simulator that demonstrates the evolution of a graph algorithm from a single-threaded implementation to massively parallel architectures.

## 📂 Project Structure

This project is organized into modular directories and follows a strict Git branching strategy for isolation.

```
├── bin/                 # Compiled executables (serial.exe, openmp.exe, etc.)
├── data/                # Generated input grid datasets
├── serial/              # Source code for Serial implementation
├── openmp/              # Source code for OpenMP implementation
├── mpi/                 # Source code for MPI implementation
├── opencl/              # Source code for OpenCL implementation
├── tests/               # Validation, benchmarking, and stress-test scripts
└── results/             # Output logs (ignored by git)
```

## 🌿 Git Branches

- **`main`**: The complete project containing all versions.
- **`serial`**: Isolated Serial implementation (baseline).
- **`openmp`**: Isolated OpenMP implementation.
- **`mpi`**: Isolated MPI implementation.
- **`opencl`**: Isolated OpenCL implementation.

## 🚀 Building & Running

**Prerequisites:**
- GCC Compiler (MinGW-w64 recommended for Windows)
- Microsoft MPI SDK (for MPI build)
- OpenCL SDK (e.g., AMD APP SDK or NVIDIA CUDA Toolkit)

### 1. Build All Versions
Run the automated build script in the root directory:
```cmd
.\build.bat
```

### 2. Verify Correctness
Run the functional verification suite to compare outputs across all versions:
```cmd
cd tests
.\run_tests.bat
```

### 3. Run Performance Benchmarks
Execute the full performance suite (Small, Medium, Large datasets):
```cmd
cd tests
.\run_benchmarks.bat
```

## ⚡ Performance Results

Benchmarks were executed on a Windows machine with GCC 15.2.0.

### Execution Time (Lower is Better)

| Dataset | Serial | OpenMP (8 Threads) | OpenCL (GPU) | MPI |
| :--- | :--- | :--- | :--- | :--- |
| **Small** (100 Nodes) | < 1ms | < 1ms | 2ms | N/A |
| **Medium** (1k Nodes) | 8ms | 1ms | 1ms | N/A |
| **Large** (10k Nodes) | 908ms | 354ms | 4ms | N/A |
| **Extra Large** (50k) | *N/A* | 6.55s | *Untested* | N/A |

### Stress Test (Extra Large Dataset)
We successfully simulated a **50,000 Node / 150,000 Edge** grid with ~1.25 Million demand load.
- **Implementation**: OpenMP (16 Threads)
- **Time**: ~6.55 seconds
- **Stability**: Passed 5 consecutive runs with consistent demand satisfaction.

## 🔍 Key Findings & Optimization

### 1. Non-Determinism in Parallel Greedy Algorithms
The "Greedy BFS" algorithm assigns power to the first city that claims it. In parallel executions (OpenMP/MPI), the order in which threads process cities is non-deterministic.
- **Observation**: Different runs produce different per-node allocations.
- **Conclusion**: This is **valid behavior**. The total demand served remains high, but the specific distribution varies based on thread scheduling.
- **Fix**: Implemented **Optimistic Concurrency Control** (Search → Critical Section Verify → Atomic Update) to ensure thread safety despite randomness.

### 2. Stack vs. Heap Allocation
Processing large graphs (50k+ nodes) caused stack overflows in the OpenMP version when using thread-local arrays.
- **Fix**: Migrated `visited`, `queue`, and `parent_edge` arrays to **Heap Allocation (`malloc`)** for robust scalability.

### 3. OpenCL Speedup
OpenCL provided the most significant speedup (**~227x** on Large inputs) because the graph traversal workload (updates per edge) is highly parallelizable on GPU architecture.

## 🛠 Troubleshooting

- **"mpiexec not found"**: Ensure Microsoft MPI Runtime is installed and added to PATH. The SDK alone allows compilation but not execution.
- **"cl.h not found"**: Verify your OpenCL SDK path in `build.bat`. The script attempts to auto-locate common paths.
- **"Run differs"**: Segments of output may vary between runs due to the greedy nature of the algorithm. Use the provided verification scripts which account for this.
