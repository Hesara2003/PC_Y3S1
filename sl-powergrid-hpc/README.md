# SL PowerGrid HPC Simulator
> (Serial → OpenMP → MPI → OpenCL for AMD GPU)

A high-performance power grid load balancing simulator for Sri Lanka that demonstrates how a serial graph algorithm can be transformed into scalable OpenMP, MPI, and OpenCL implementations.

## Usage

### Prerequisites
- GCC (MinGW for Windows)
- MS-MPI (if running MPI on Windows)
- OpenCL SDK (e.g., AMD APP SDK)

### Building
You can use `make` if installed, or the provided batch script `build.bat` on Windows.

**Using build.bat (Recommended for Windows):**
```cmd
.\build.bat
```

**Using Make:**
```bash
make serial
make openmp
make mpi
make opencl
```

### Running
```bash
# Serial
./bin/serial

# OpenMP (8 threads)
set OMP_NUM_THREADS=8
./bin/openmp

# MPI (4 processes)
mpiexec -n 4 ./bin/mpi

# OpenCL
./bin/opencl
```

## Validation
Compare outputs to ensure correctness:
```bash
diff serial.txt openmp.txt
```

## Expected Performance

| Version | Time (s) | Speedup |
|---------|----------|---------|
| Serial  | 4.2      | 1x      |
| OpenMP  | 1.1      | 3.8x    |
| MPI     | 0.6      | 7x      |
| OpenCL  | 0.05     | 84x     |
