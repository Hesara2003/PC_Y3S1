all: serial openmp mpi

bin:
	mkdir -p bin

serial: | bin
	gcc serial/grid_serial.c -o bin/serial -lm

openmp: | bin
	gcc openmp/grid_openmp.c -fopenmp -o bin/openmp -lm

mpi: | bin
	mpicc mpi/grid_mpi.c -o bin/mpi -lm

clean:
	rm -f bin/serial bin/openmp bin/mpi
