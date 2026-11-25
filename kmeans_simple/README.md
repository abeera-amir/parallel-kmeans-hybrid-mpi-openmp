# Parallel K-Means (simple version) - Serial vs Hybrid MPI + OpenMP

Same data as the old project: 4,000,000 points, 16 dimensions, 16 natural groups, seed 12345.

## Files
| File | What it does |
|---|---|
| `src/common.h` | data generator, distance, random start, k-means++ start (shared by all programs) |
| `src/kmeans_serial.cpp` | serial code: classic, k-means++, Hamerly |
| `src/kmeans_hybrid.cpp` | MPI + OpenMP code: same 3 algorithms. Lists all MPI/OpenMP statements at the top, false sharing + padding, loop-carried dependencies |
| `src/choose_k.cpp` | elbow + silhouette, K = 2..20, writes the chosen K |
| `run_all.sh` | build -> choose K -> all timings -> graphs |
| `make_graphs.py` | graphs + tables (time, speedup, efficiency) |

## Run (Ubuntu / WSL)
```bash
sudo apt install -y build-essential openmpi-bin libopenmpi-dev python3-matplotlib
bash run_all.sh                # full run: 4M points, 50 iterations, 3 repeats
bash run_all.sh 400000 10 1    # quick test
```
Single runs:
```bash
./choose_k
./kmeans_serial --algo hamerly --k 16
mpirun -np 2 --bind-to none ./kmeans_hybrid --algo hamerly --k 16 --threads 2
```
`--algo` = `classic` | `kmeanspp` | `hamerly`

## Output (folder `results/`)
- `choose_k.png` - elbow and silhouette graphs
- `time.png` - run time of every setup
- `speedup.png` - speedup vs number of workers
- `efficiency.png` - efficiency vs number of workers
- `algorithms.png` - classic vs k-means++ vs Hamerly
- `tables.txt`, `timings.csv`, `choose_k.csv`

Correctness: the `sse` printed by the serial and every parallel setup must be the same number.
