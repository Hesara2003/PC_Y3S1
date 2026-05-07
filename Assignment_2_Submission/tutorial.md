# Environment Setup Tutorial
## SE3082 – Parallel Computing | SL PowerGrid HPC Simulator

**Target Audience:** 3rd-year Computer Science undergraduates with basic command-line knowledge.  
**Environments:** macOS (Apple M5, ARM64) for OpenMP & MPI | Google Colab (NVIDIA T4 GPU) for CUDA

---

## 1. Hardware & OS Specifications

### Local Machine (OpenMP & MPI)

| Component | Detail |
|-----------|--------|
| **OS** | macOS (Apple Silicon) |
| **CPU** | Apple M5 (8-core: 4 Performance + 4 Efficiency) |
| **RAM** | Unified Memory Architecture (shared CPU/GPU pool) |
| **Compiler** | Apple Clang 21.0.0 (Xcode Command Line Tools) |

**Why this setup?** The Apple M5's heterogeneous CPU design (fast Performance cores + efficient Efficiency cores) makes OpenMP's `schedule(dynamic)` essential — it lets faster cores pick up extra work rather than idling. MPI runs 4 processes mapped to the physical cores for distributed-style parallelism on a single machine.

### Cloud Environment (CUDA)

| Component | Detail |
|-----------|--------|
| **Platform** | Google Colab (Google Cloud Platform) |
| **GPU** | NVIDIA T4 (2560 CUDA cores, 16 GB GDDR6, 320 GB/s bandwidth) |
| **CUDA Toolkit** | CUDA 12.x (pre-installed on Colab T4 runtime) |
| **OS** | Ubuntu Linux (Colab) |

**Why Google Colab?** Apple Silicon (M-series chips) uses an integrated GPU that does not support NVIDIA CUDA. Google Colab provides free access to NVIDIA T4 GPU instances, making it the ideal cloud environment (non-AWS) for CUDA development. The T4's 2560 CUDA cores and high-bandwidth GDDR6 memory are well-suited to demonstrating shared memory optimization.

---

## 2. Prerequisites

### For Local Setup (OpenMP & MPI)

1. **Xcode Command Line Tools** (provides Clang compiler):
   ```bash
   xcode-select --install
   ```
   Verify:
   ```bash
   clang --version
   # Expected: Apple clang version 21.x.x or later
   ```

2. **Homebrew** package manager:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

3. **Python 3** (for the grid data generator):
   ```bash
   brew install python3
   ```

### For Cloud Setup (CUDA)

1. A **Google Account** — to access Google Colab at [colab.research.google.com](https://colab.research.google.com)
2. No local software installation required — `nvcc` (NVIDIA CUDA Compiler) is pre-installed in the Colab T4 GPU runtime.

---

## 3. OpenMP Setup (macOS)

OpenMP is an API for shared-memory multi-threading. Apple's default Clang compiler requires an external OpenMP runtime library because Apple's own toolchain excludes it.

### Installation

```bash
brew install libomp
```

This installs `libomp` (LLVM's OpenMP runtime). It is **keg-only** (not symlinked into `/opt/homebrew`), so you must reference it explicitly during compilation.

Verify the installation path:
```bash
brew --prefix libomp
# Output: /opt/homebrew/opt/libomp
```

### Compilation Command

```bash
clang -Xpreprocessor -fopenmp \
  -I$(brew --prefix libomp)/include \
  -L$(brew --prefix libomp)/lib \
  -lomp \
  openmp/grid_openmp.c -o bin/openmp
```

**Flag Breakdown:**
- `-Xpreprocessor -fopenmp`: Passes the OpenMP flag through Apple Clang's preprocessor.
- `-I$(brew --prefix libomp)/include`: Adds the `omp.h` header path.
- `-L$(brew --prefix libomp)/lib -lomp`: Links the OpenMP runtime library.

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `omp.h not found` | Run `brew --prefix libomp` and use that exact path in the `-I` flag. |
| `dyld: Library not loaded` at runtime | Add `export DYLD_LIBRARY_PATH=$(brew --prefix libomp)/lib` to your `~/.zshrc`. |
| OpenMP not using all cores | Set `OMP_NUM_THREADS=8` before running: `OMP_NUM_THREADS=8 ./bin/openmp input.txt` |

---

## 4. MPI Setup (macOS)

MPI (Message Passing Interface) enables distributed-memory parallel programming via message passing between processes. We use **OpenMPI**, the most widely used open-source MPI implementation.

### Installation

```bash
brew install open-mpi
```

This installs `mpicc` (MPI-aware C compiler wrapper) and `mpirun` (process launcher) at `/opt/homebrew/bin/`.

Verify:
```bash
mpirun --version
# Expected: mpirun (Open MPI) 5.x.x
```

### Compilation and Execution

```bash
# Compile using the mpicc wrapper
mpicc mpi/grid_mpi.c -o bin/mpi

# Run with 4 parallel processes
mpirun -np 4 ./bin/mpi data/grid.txt
```

**Note:** The `mpicc` wrapper automatically links all required MPI headers and libraries; no manual `-I` or `-L` flags are needed.

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `mpirun: command not found` | Ensure `/opt/homebrew/bin` is in your `$PATH`. Add `export PATH="/opt/homebrew/bin:$PATH"` to `~/.zshrc`, then run `source ~/.zshrc`. |
| `There are not enough slots available` | Reduce process count: `mpirun -np 2 ...` or add `--oversubscribe` flag. |
| Slow performance with many processes | On Apple Silicon, limit to physical core count: `mpirun -np 8 ...` |

---

## 5. CUDA Setup (Google Colab — Cloud Environment)

CUDA (Compute Unified Device Architecture) is NVIDIA's GPU parallel computing platform. Since Apple Silicon does not have an NVIDIA GPU, we use **Google Colab** as the cloud environment, which provides free access to NVIDIA T4 GPU instances with CUDA pre-installed.

> **Why Google Colab and not AWS?** The assignment explicitly states the cloud platform must not be AWS. Google Colab (Google Cloud Platform infrastructure) is an ideal choice — it is free, requires no billing setup, has CUDA 12 pre-installed, and is beginner-friendly.

### Step 1 — Create a Notebook and Enable GPU

1. Navigate to [colab.research.google.com](https://colab.research.google.com)
2. Click **New Notebook**
3. Go to **Runtime → Change runtime type**
4. Under **Hardware accelerator**, select **T4 GPU**
5. Click **Save**

### Step 2 — Verify GPU Availability

In the first cell, run:
```bash
!nvidia-smi
```
Expected output will show the NVIDIA T4 GPU details, driver version, and CUDA version.

### Step 3 — Upload the CUDA Source File

In a new cell, run:
```python
from google.colab import files
files.upload()   # select cuda_powergrid.cu from your computer
```
Alternatively, paste the source code directly using the `%%writefile` magic:
```
%%writefile cuda_powergrid.cu
// paste source code here
```

### Step 4 — Compile with nvcc

```bash
!nvcc cuda_powergrid.cu -o cuda_powergrid
```
`nvcc` is NVIDIA's CUDA compiler. It compiles both the host (CPU) C code and the device (GPU) kernel code. Successful compilation produces no output.

**Flag breakdown:**
- `nvcc` — NVIDIA CUDA Compiler (pre-installed in Colab GPU runtime)
- `cuda_powergrid.cu` — source file (`.cu` extension indicates CUDA C++)
- `-o cuda_powergrid` — output binary name

### Step 5 — Run the Program

```bash
!./cuda_powergrid
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `nvcc: command not found` | You are on a CPU-only runtime. Go to **Runtime → Change runtime type → T4 GPU**. |
| `CUDA error: no kernel image` | Ensure the runtime is T4 GPU, not an older GPU type. |
| `out of memory` | Reduce `NUM_CITIES` or `NUM_GENERATORS` in the source file before recompiling. |
| Colab disconnects | Free Colab sessions expire after ~90 min of inactivity. Re-run all cells if reconnected. |

---

## 6. Generating Test Data

The project includes a Python grid generator to create test inputs of various sizes:

```bash
mkdir -p data

# Small dataset: 100 nodes, 300 edges
python3 generate_grid.py data/small.txt 100 300

# Medium dataset: 1000 nodes, 3000 edges  
python3 generate_grid.py data/medium.txt 1000 3000

# Large dataset: 10000 nodes, 30000 edges
python3 generate_grid.py data/large.txt 10000 30000
```

Each generated file contains a section of `# nodes` (with type: GENERATOR, SUBSTATION, or CITY) and `# edges` with capacities.
