# Build:   make            (serial + hybrid)
#          make NATIVE=1   (adds -march=native: uses AVX/FMA of YOUR cpu - build serial and hybrid the same way!)
#          make clean
CXXFLAGS = -O3 -std=c++17 -Wall -Wno-unknown-pragmas
ifdef NATIVE
CXXFLAGS += -march=native
endif

all: kmeans_serial kmeans_hybrid

# SERIAL baseline: NO MPI, NO OpenMP threads.  -fopenmp-simd only enables the "omp simd"
# vectorisation hint inside dist2(), so that the serial and the parallel program run the
# SAME inner loop machine code (fair comparison: speedup then measures parallelism only).
kmeans_serial: src/kmeans_serial.cpp src/common.h
	g++ $(CXXFLAGS) -fopenmp-simd -o $@ $<

# HYBRID: MPI + OpenMP
kmeans_hybrid: src/kmeans_hybrid.cpp src/common.h
	mpicxx $(CXXFLAGS) -fopenmp -o $@ $<

clean:
	rm -f kmeans_serial kmeans_hybrid
	rm -rf results
