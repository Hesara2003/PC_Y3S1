@echo off
setlocal enabledelayedexpansion

echo ==================================================
echo PHASE 9 & 10: STRESS & STABILITY TESTING
echo ==================================================

echo [1/3] Generating Extra Large Dataset (50k nodes, 150k edges)...
..\bin\serial.exe --generate ..\data\xlarge.txt 50000 150000

echo [2/3] Running Stress Test (OpenMP)...
set OMP_NUM_THREADS=16
..\bin\openmp.exe ..\data\xlarge.txt > stress_run1.txt
type stress_run1.txt | findstr "Time:"
type stress_run1.txt | findstr "Served:"

echo [3/3] Running Stability Loop (5 iterations)...
for /L %%i in (1,1,5) do (
    ..\bin\openmp.exe ..\data\xlarge.txt > stress_loop_%%i.txt
    
    REM Compare with first run
    findstr /V "Time:" stress_loop_%%i.txt > stress_loop_clean_%%i.txt
    findstr /V "Time:" stress_run1.txt > stress_run1_clean.txt
    
    fc /w stress_run1_clean.txt stress_loop_clean_%%i.txt > nul
    if errorlevel 1 (
        echo [FAIL] Run %%i differs!
    ) else (
        echo [PASS] Run %%i matches
        del stress_loop_%%i.txt stress_loop_clean_%%i.txt
    )
)
REM del stress_run1.txt stress_run1_clean.txt
echo ==================================================
