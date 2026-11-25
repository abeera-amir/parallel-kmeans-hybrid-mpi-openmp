# make        -> builds kmeans_serial, kmeans_hybrid, choose_k
# make clean  -> removes them
FLAGS = -O3 -Wall -Wno-unused-function

all: kmeans_serial kmeans_hybrid choose_k

kmeans_serial: src/kmeans_serial.cpp src/common.h
	g++ $(FLAGS) -o kmeans_serial src/kmeans_serial.cpp

kmeans_hybrid: src/kmeans_hybrid.cpp src/common.h
	mpicxx $(FLAGS) -fopenmp -o kmeans_hybrid src/kmeans_hybrid.cpp

choose_k: src/choose_k.cpp src/common.h
	g++ $(FLAGS) -o choose_k src/choose_k.cpp

clean:
	rm -f kmeans_serial kmeans_hybrid choose_k
