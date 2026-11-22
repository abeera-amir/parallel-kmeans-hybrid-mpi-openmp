#!/usr/bin/env bash
# verify.sh : correctness check.  Runs the serial baseline once, then MANY parallel
# configurations on the SAME data / SAME start centroids and checks that the final
# centroids are identical (up to floating point rounding).
#   usage: scripts/verify.sh [N] [ITERS]
set -u
cd "$(dirname "$0")/.."
N=${1:-200000}; IT=${2:-20}
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
MPIRUN="mpirun --oversubscribe --bind-to none --mca btl self,vader"
fail=0
check() {   # $1 = reference file, rest = "procs threads extra args"
  local ref=$1 p=$2 t=$3; shift 3
  $MPIRUN -np "$p" ./kmeans_hybrid --n "$N" --iters "$IT" --threads "$t" "$@" --out /tmp/h.txt > /tmp/h.log
  local line; line=$(grep RESULT /tmp/h.log | sed -E 's/.*procs=/procs=/; s/ comm=.*sse=/ sse=/; s/ outlier.*//')
  local msg; msg=$(python3 scripts/compare.py "$ref" /tmp/h.txt 1e-6) || fail=1
  printf '%-90s %s\n' "$line" "$msg"
}
for variant in "lloyd" "hamerly" "lloyd --weighted" "hamerly --weighted"; do
  algo=${variant%% *}; extra=""; [[ $variant == *weighted* ]] && extra="--weighted"
  echo "=== reference: serial $variant"
  ./kmeans_serial --n "$N" --iters "$IT" --algo "$algo" $extra --out /tmp/ref.txt | grep RESULT | sed -E 's/.*(time=[^ ]+ sse=[^ ]+).*/serial \1/'
  for acc in reduction private atomic critical falseshare; do check /tmp/ref.txt 1 4 --algo "$algo" $extra --accum $acc; done
  check /tmp/ref.txt 3 2 --algo "$algo" $extra
  check /tmp/ref.txt 2 3 --algo "$algo" $extra --sched dynamic
  check /tmp/ref.txt 2 3 --algo "$algo" $extra --sched guided
  check /tmp/ref.txt 2 3 --algo "$algo" $extra --sched task
  check /tmp/ref.txt 2 3 --algo "$algo" $extra --sched task --accum atomic
  check /tmp/ref.txt 2 3 --algo "$algo" $extra --region forkjoin
  check /tmp/ref.txt 1 4 --algo "$algo" $extra --region forkjoin --sched task --accum falseshare
done
[[ $fail == 0 ]] && echo "ALL CONFIGURATIONS MATCH THE SERIAL RESULT" || echo "SOME CONFIGURATION MISMATCHED"
exit $fail
