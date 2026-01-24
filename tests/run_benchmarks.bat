@echo off
setlocal enabledelayedexpansion

echo ==================================================
echo PHASE 7: PERFORMANCE TESTING
echo ==================================================

echo Dataset	Version	Time(s)	Speedup
echo --------------------------------------------------

for %%D in (small medium large) do (
    set DATA=..\data\%%D.txt
    
    REM Serial ----------------------------------------
    for /f "tokens=2" %%T in ('..\bin\serial.exe !DATA! ^| findstr "Time:"') do set T_SERIAL=%%T
    echo %%D	Serial	!T_SERIAL!	1.0x

    REM OpenMP ----------------------------------------
    set OMP_NUM_THREADS=8
    for /f "tokens=2" %%T in ('..\bin\openmp.exe !DATA! ^| findstr "Time:"') do set T_OMP=%%T
    REM Very basic floating point math in batch? No. using powershell/vbs is better.
    REM Just print the raw time for now.
    echo %%D	OpenMP	!T_OMP!
    
    REM MPI -------------------------------------------
    if exist "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" (
        for /f "tokens=2" %%T in ('"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" -n 4 ..\bin\mpi.exe !DATA! ^| findstr "Time:"') do set T_MPI=%%T
        echo %%D	MPI	!T_MPI!
    ) else (
        echo %%D	MPI	N/A
    )

    REM OpenCL ----------------------------------------
    for /f "tokens=2" %%T in ('..\bin\opencl.exe !DATA! ^| findstr "Time:"') do set T_OCL=%%T
    echo %%D	OpenCL	!T_OCL!
    echo.
)
