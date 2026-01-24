all: serial openmp mpi opencl

bin:
	if not exist bin mkdir bin

serial: | bin
	gcc serial/grid_serial.c -o bin/serial

openmp: | bin
	gcc openmp/grid_openmp.c -fopenmp -o bin/openmp

mpi: | bin
	mpicc mpi/grid_mpi.c -o bin/mpi

opencl: | bin
	gcc opencl/grid_opencl.c -lOpenCL -o bin/opencl

clean:
	if exist bin del /Q bin\* 2>NUL
