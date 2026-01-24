@echo off
setlocal

if not exist serial_clean.txt (
    echo Error: serial_clean.txt not found. Please run run_tests.bat first.
    exit /b 1
)

echo ==================================================
echo PHASE 4: OPENMP RACE CONDITION TESTING
echo ==================================================

for %%t in (1 2 4 8 16) do (
    set OMP_NUM_THREADS=%%t
    echo Testing with %%t threads...
    ..\bin\openmp.exe > openmp_race_%%t.txt
    
    findstr /V "Time:" openmp_race_%%t.txt > openmp_race_clean_%%t.txt
    fc /w serial_clean.txt openmp_race_clean_%%t.txt > nul
    if errorlevel 1 (
        echo [FAIL] Comparison failed for %%t threads!
    ) else (
        echo [PASS] Output matches Serial
    )
    del openmp_race_%%t.txt openmp_race_clean_%%t.txt
)

echo ==================================================
echo TEST COMPLETE
echo ==================================================
