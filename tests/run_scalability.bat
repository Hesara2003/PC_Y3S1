@echo off
setlocal enabledelayedexpansion

echo ==================================================
echo PHASE 8: SCALABILITY TESTING (OpenMP)
echo ==================================================
echo Dataset: data/large.txt
echo Threads	Time(s)
echo --------------------------------------------------

set DATA=..\data\large.txt

for %%T in (1 2 4 8 16) do (
    set OMP_NUM_THREADS=%%T
    for /f "tokens=2" %%t in ('..\bin\openmp.exe !DATA! ^| findstr "Time:"') do set TIME=%%t
    echo %%T	!TIME!
)

echo ==================================================
