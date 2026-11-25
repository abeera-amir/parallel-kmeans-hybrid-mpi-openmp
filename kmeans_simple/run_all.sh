#!/usr/bin/env bash
# =============================================================================
#  run_all.sh  -  does everything:
#     1. build
#     2. choose K (elbow + silhouette)
#     3. time serial and parallel runs for the 3 algorithms
#     4. draw the graphs
#
#  usage:   bash run_all.sh                 (4M points, 50 iterations, 3 repeats)
#  quick:   bash run_all.sh 400000 10 1     (small test)
# =============================================================================
set -e
cd "$(dirname "$0")"
N=${1:-4000000}        # number of points
ITERS=${2:-50}         # iterations
REPEATS=${3:-3}        # each run is repeated, the average time is used
WORKERS="1 2 4"        # total workers p = processes x threads (4 = your 4 cores)

export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
# --bind-to none : let the OpenMP threads of one process use different cores
# --oversubscribe: allow more processes than MPI thinks there are slots
MPIRUN="mpirun --oversubscribe --bind-to none --mca btl self,vader"

make
mkdir -p results

echo "=== Step 1: choose K ==="
./choose_k --n $N
K=$(cut -d' ' -f1 results/chosen_k.txt)
echo "Using K = $K"

CSV=results/timings.csv
echo "algo,setup,procs,threads,workers,time,sse" > $CSV

# average of the numbers given
average() { python3 -c "import sys; v=[float(x) for x in sys.argv[1:]]; print(round(sum(v)/len(v),4))" "$@"; }

# run_one <algo> <setup name> <procs> <threads>    (procs=0 means the serial program)
run_one() {
  local algo=$1 setup=$2 procs=$3 threads=$4 times=() line=""
  for ((r = 0; r < REPEATS; r++)); do
    if [ "$procs" = "0" ]; then
      line=$(./kmeans_serial --algo $algo --n $N --k $K --iters $ITERS | grep RESULT)
    else
      line=$($MPIRUN -np $procs ./kmeans_hybrid --algo $algo --n $N --k $K --iters $ITERS --threads $threads | grep RESULT)
    fi
    times+=( $(echo "$line" | sed -E 's/.* time=([0-9.]+) .*/\1/') )
  done
  local sse=$(echo "$line" | sed -E 's/.* sse=([0-9.e+-]+).*/\1/')
  local workers=$(( procs == 0 ? 1 : procs * threads ))
  local avg=$(average "${times[@]}")
  echo "$algo,$setup,$procs,$threads,$workers,$avg,$sse" >> $CSV
  printf "  %-9s %-12s procs=%s threads=%s  time=%8.3f s  sse=%s\n" $algo $setup $procs $threads $avg $sse
}

echo "=== Step 2: timings (N=$N, K=$K, iterations=$ITERS) ==="
for algo in classic kmeanspp hamerly; do
  run_one $algo serial 0 1
  for p in $WORKERS; do
    run_one $algo openmp 1 $p                       # pure OpenMP: 1 process,  p threads
    if [ $p -gt 1 ]; then
      run_one $algo mpi $p 1                        # pure MPI:    p processes, 1 thread
    fi
    if [ $p -ge 4 ]; then
      run_one $algo hybrid $((p / 2)) 2             # hybrid:      p/2 processes x 2 threads
    fi
  done
done

echo "=== Step 3: graphs ==="
python3 make_graphs.py
echo "Done. See the results folder."
