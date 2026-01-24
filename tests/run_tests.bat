@echo off
setlocal

if not exist ..\bin\serial.exe (
    echo Error: Binaries not found. Please run build.bat in root first.
    exit /b 1
)

echo ==================================================
echo PHASE 3: FUNCTIONAL CORRECTNESS TESTS
echo ==================================================

echo [1/4] Running Serial...
..\bin\serial.exe > serial.txt

echo [2/4] Running OpenMP (8 Threads)...
set OMP_NUM_THREADS=8
..\bin\openmp.exe > openmp.txt

echo [3/4] Running MPI (4 Processes)...
if exist "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" (
    "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" -n 4 ..\bin\mpi.exe > mpi.txt
) else (
    echo Warning: mpiexec not found. Skipping MPI execution.
    echo "MPI skipped" > mpi.txt
)

echo [4/4] Running OpenCL...
..\bin\opencl.exe > opencl.txt

echo.
echo ==================================================
echo VERIFICATION RESULTS
echo ==================================================

echo Comparing OpenMP vs Serial...
findstr /V "Time:" serial.txt > serial_clean.txt
findstr /V "Time:" openmp.txt > openmp_clean.txt
fc /w serial_clean.txt openmp_clean.txt > nul
if %errorlevel% equ 0 (
    echo [PASS] OpenMP Output Matches
) else (
    echo [FAIL] OpenMP Output Differs
    echo See diff_openmp.log or use 'fc serial_clean.txt openmp_clean.txt' for details.
)

echo Comparing MPI vs Serial...
if exist mpi.txt (
    findstr /V "Time:" mpi.txt > mpi_clean.txt
    fc /w serial_clean.txt mpi_clean.txt > nul
    if %errorlevel% equ 0 (
        echo [PASS] MPI Output Matches
    ) else (
        echo [FAIL] MPI Output Differs
    )
)

echo Comparing OpenCL vs Serial...
findstr /V "Time:" opencl.txt > opencl_clean.txt
fc /w serial_clean.txt opencl_clean.txt > nul
if %errorlevel% equ 0 (
    echo [PASS] OpenCL Output Matches
) else (
    echo [FAIL] OpenCL Output Differs
    echo See diff_opencl.log or use 'fc serial_clean.txt opencl_clean.txt' for details.
)
echo ==================================================
