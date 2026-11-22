#!/usr/bin/env bash
# demo_reallife.sh : "customer segmentation" story, end to end, in 4 steps.
#   Each point = one customer with D=16 numeric features (recency, frequency, spend, ...).
#   There are really 6 natural customer groups, but the analyst does NOT know that.
cd "$(dirname "$0")/.."
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
MPIRUN="mpirun --oversubscribe --bind-to none --mca btl self,vader"
N=${1:-1000000}; T=${2:-2}; P=${3:-2}
echo "################ STEP 1: how many segments?  (choice of K: elbow + silhouette, tasks) ################"
$MPIRUN -np $P ./kmeans_hybrid --mode choosek --ktrue 6 --kmin 2 --kmax 12 --n $N --sample 2000 --threads $T
echo
echo "################ STEP 2: random start vs k-means++ start (same K=6, classical algorithm) ################"
for init in random kmeanspp; do
  echo "--- init=$init"; $MPIRUN -np $P ./kmeans_hybrid --n $N --k 6 --ktrue 6 --iters 30 --init $init --threads $T | grep -E "RESULT" | sed -E 's/.*(init=[a-z]+).*(time=[0-9.]+).*(sse=[0-9.]+).*/\1 \2 \3/'
done
echo
echo "################ STEP 3: classical vs OPTIMISED (Hamerly bounds) - identical answer, less work ################"
for algo in lloyd hamerly; do
  echo "--- algo=$algo"; $MPIRUN -np $P ./kmeans_hybrid --n $N --k 6 --ktrue 6 --iters 30 --algo $algo --threads $T | grep -E "RESULT" | sed -E 's/.*(algo=[a-z]+).*(time=[0-9.]+).*(sse=[0-9.]+).*skipped=([0-9]+).*/\1 \2 \3 distance-searches-skipped=\4/'
done
echo
echo "################ STEP 4: WEIGHTED k-means - weight = customer revenue (1..6), big customers pull centroids ################"
# --init random for both runs => IDENTICAL starting centroids, so the difference below is caused ONLY by the weights
$MPIRUN -np $P ./kmeans_hybrid --n $N --k 6 --ktrue 6 --iters 30 --threads $T --init random --out /tmp/c_plain.txt > /dev/null
$MPIRUN -np $P ./kmeans_hybrid --n $N --k 6 --ktrue 6 --iters 30 --threads $T --init random --weighted --out /tmp/c_weighted.txt | grep -E "cluster sizes|outlier"
python3 - <<'PY'
import numpy as np
a=np.loadtxt('/tmp/c_plain.txt'); b=np.loadtxt('/tmp/c_weighted.txt')
print("movement of each centroid caused by the weights:", np.round(np.linalg.norm(a-b,axis=1),3))
print("(weighted centroid = revenue-weighted mean: it moves toward the heavy customers of its segment)")
PY
