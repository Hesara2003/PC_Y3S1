@echo off
if not exist bin mkdir bin

echo Checking for GCC...
where gcc >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: GCC not found in PATH. Please install MinGW or similar.
    exit /b 1
)

echo Building Serial...
gcc serial/grid_serial.c -o bin/serial.exe
if %errorlevel% neq 0 echo Build failed: Serial & exit /b 1

echo Building OpenMP...
gcc openmp/grid_openmp.c -fopenmp -o bin/openmp.exe
if %errorlevel% neq 0 echo Build failed: OpenMP & exit /b 1

echo Building MPI...
gcc mpi/grid_mpi.c -o bin/mpi.exe -I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" -L"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -lmsmpi
if %errorlevel% neq 0 echo Build failed: MPI & exit /b 1

echo Building OpenCL...
gcc opencl/grid_opencl.c -o bin/opencl.exe -I"C:\Users\Dell\Downloads\OpenCL-SDK-v2025.07.23-Win-x64\OpenCL-SDK-v2025.07.23-Win-x64\include" -L"C:\Users\Dell\Downloads\OpenCL-SDK-v2025.07.23-Win-x64\OpenCL-SDK-v2025.07.23-Win-x64\lib" -lOpenCL
if %errorlevel% neq 0 echo Build failed: OpenCL

echo Build Complete. Artifacts in bin/
