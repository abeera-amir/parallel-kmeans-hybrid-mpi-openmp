#!/usr/bin/env bash
# =============================================================================
# benchmark.sh  -  runs ALL timing experiments and writes results/results.csv
#   usage:  scripts/benchmark.sh [N] [ITERS] [REPS]
#   default N=4000000 ITERS=50 REPS=3   (REPS runs are averaged, as in the proposal)
#   quick test:  scripts/benchmark.sh 200000 10 1
# Experiments
#   E1 scaling      : serial | pure OpenMP | pure MPI | hybrid     (Table 3.5/3.6/3.7 style)
#   E2 accumulation : reduction / private / atomic / critical / falseshare
#   E3 schedule     : static / dynamic / guided / task (+ chunk size = false-sharing on labels)
#   E4 region       : ONE parallel region vs fork-join per phase    (Table 5.3 style)
#   E5 choose K     : serial vs hybrid (tasks)
# =============================================================================
set -u
cd "$(dirname "$0")/.."
N=${1:-4000000}; IT=${2:-50}; REPS=${3:-3}
N2=$(( N / 4 )); IT2=$(( IT / 5 > 5 ? IT / 5 : 5 ))      # smaller size for the slow experiments (critical, atomic...)
CORES=$(nproc)
mkdir -p results
CSV=results/results.csv
echo "exp,algo,procs,threads,accum,sched,region,n,iters,time" > $CSV
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
# oversubscribe lets you test with more workers than cores;  bind-to none lets OpenMP threads of a rank spread over all cores
MPIRUN="mpirun --oversubscribe --bind-to none --mca btl self,vader"

avg() { python3 -c "import sys; v=[float(x) for x in sys.argv[1:]]; print(sum(v)/len(v))" "$@"; }

# run_serial exp algo n iters
run_serial() {
  local exp=$1 algo=$2 n=$3 it=$4 ts=()
  for ((r=0;r<REPS;r++)); do ts+=( $(./kmeans_serial --n $n --iters $it --algo $algo | grep RESULT | sed -E 's/.* time=([0-9.]+) .*/\1/') ); done
  echo "$exp,$algo,0,1,serial,serial,serial,$n,$it,$(avg "${ts[@]}")" >> $CSV
  echo "  [$exp] serial $algo n=$n -> $(tail -1 $CSV | awk -F, '{print $10}') s"
}
# run_hybrid exp algo procs threads accum sched region n iters
run_hybrid() {
  local exp=$1 algo=$2 p=$3 t=$4 acc=$5 sch=$6 reg=$7 n=$8 it=$9 ts=()
  for ((r=0;r<REPS;r++)); do
    ts+=( $($MPIRUN -np $p ./kmeans_hybrid --n $n --iters $it --threads $t --algo $algo --accum $acc --sched $sch --region $reg | grep RESULT | sed -E 's/.* time=([0-9.]+) .*/\1/') )
  done
  echo "$exp,$algo,$p,$t,$acc,$sch,$reg,$n,$it,$(avg "${ts[@]}")" >> $CSV
  echo "  [$exp] $algo ${p}x${t} $acc/$sch/$reg -> $(tail -1 $CSV | awk -F, '{print $10}') s"
}

# worker counts 1,2,4,8,... up to the number of cores
W=(); w=1; while (( w <= CORES )); do W+=($w); w=$((w*2)); done
echo "cores=$CORES  worker counts: ${W[*]}"

echo "== E1 scaling";
for algo in lloyd hamerly; do
  run_serial E1 $algo $N $IT
  for w in "${W[@]}"; do
    run_hybrid E1 $algo 1 $w private static persistent $N $IT        # pure OpenMP
    (( w > 1 )) && run_hybrid E1 $algo $w 1 private static persistent $N $IT    # pure MPI
    (( w >= 4 )) && run_hybrid E1 $algo $((w/2)) 2 private static persistent $N $IT   # hybrid p/2 x 2
  done
done

echo "== E2 accumulation strategy (threads=$CORES, N=$N2)"
for acc in private reduction atomic critical falseshare; do run_hybrid E2 lloyd 1 $CORES $acc static persistent $N2 $IT2; done

echo "== E3 loop schedule / tasks (threads=$CORES)"
for sch in static dynamic guided task; do run_hybrid E3 lloyd 1 $CORES private $sch persistent $N2 $IT2; done
for ch in 1 16 256 4096; do   # tiny dynamic chunks -> neighbouring threads write the same cache line of lab[]  (false sharing)
  ts=(); for ((r=0;r<REPS;r++)); do ts+=( $($MPIRUN -np 1 ./kmeans_hybrid --n $N2 --iters $IT2 --threads $CORES --sched dynamic --chunk $ch --accum private | grep RESULT | sed -E 's/.* time=([0-9.]+) .*/\1/') ); done
  echo "E3chunk,lloyd,1,$CORES,private,dynamic$ch,persistent,$N2,$IT2,$(avg "${ts[@]}")" >> $CSV; echo "  [E3chunk] dynamic chunk=$ch -> $(tail -1 $CSV | awk -F, '{print $10}') s"
done

echo "== E4 one parallel region vs fork-join (small problem: fork cost visible)"
for reg in persistent forkjoin; do for t in 1 2 3 4; do (( t <= CORES )) && run_hybrid E4 lloyd 1 $t private static $reg 100000 200; done; done

echo "== E5 choose K"
ts=(); for ((r=0;r<REPS;r++)); do ts+=( $(./kmeans_serial --mode choosek --ktrue 6 --kmax 14 --n $N --sample 3000 | grep RESULT | sed -E 's/.* time=([0-9.]+) .*/\1/') ); done
echo "E5,choosek,0,1,serial,serial,serial,$N,0,$(avg "${ts[@]}")" >> $CSV
for w in "${W[@]}"; do
  ts=(); for ((r=0;r<REPS;r++)); do ts+=( $($MPIRUN -np 1 ./kmeans_hybrid --mode choosek --ktrue 6 --kmax 14 --n $N --sample 3000 --threads $w | grep RESULT | sed -E 's/.* time=([0-9.]+) .*/\1/') ); done
  echo "E5,choosek,1,$w,tasks,tasks,persistent,$N,0,$(avg "${ts[@]}")" >> $CSV
done
echo "results written to $CSV  ->  now run:  python3 scripts/make_report.py"
