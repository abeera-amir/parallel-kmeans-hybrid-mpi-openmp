# Parallel K-Means on Large Datasets — Hybrid MPI + OpenMP
**PDC Semester Project #9 · Dua Shafiq (k230825), Abeera Amir (k230896)**

Everything in this folder was compiled and tested (see §9 for exactly what was verified and what was *not* measurable).

---

## 0. Quick start (copy–paste)

```bash
# 1) tools (Ubuntu / WSL)
sudo apt update
sudo apt install -y build-essential openmpi-bin libopenmpi-dev python3 python3-numpy python3-matplotlib

# 2) build
cd kmeans_project
make                                   # makes ./kmeans_serial and ./kmeans_hybrid

# 3) prove it is correct (serial result == every parallel configuration)
bash scripts/verify.sh                 # must end with: ALL CONFIGURATIONS MATCH THE SERIAL RESULT

# 4) real-life story: choose K -> k-means++ -> optimised -> weighted
bash scripts/demo_reallife.sh

# 5) the project run from your proposal (4M points, 16-D, K=16, 50 iterations)
./kmeans_serial --n 4000000 --iters 50                                       # baseline
mpirun -np 1 --bind-to none ./kmeans_hybrid --n 4000000 --iters 50 --threads 4   # pure OpenMP
mpirun -np 4 --bind-to none ./kmeans_hybrid --n 4000000 --iters 50 --threads 1   # pure MPI
mpirun -np 2 --bind-to none ./kmeans_hybrid --n 4000000 --iters 50 --threads 2   # hybrid 2x2

# 6) all timing experiments -> tables + graphs for the report  (takes a while)
bash scripts/benchmark.sh              # writes results/results.csv
python3 scripts/make_report.py         # writes results/tables.txt and results/scaling_*.png
python3 scripts/final_figures.py        # final cleaned tables + graphs (median values)
```
If MPI complains about "not enough slots", add `--oversubscribe`. If you run as root (Docker), `export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1`.
**Always use `--bind-to none`** with several threads per rank: Open MPI's default binds a rank to *one core* (for ≤2 ranks) and all its OpenMP threads would then share that core.

---

## 1. What is in the folder

| File | Purpose |
|---|---|
| `src/common.h` | Code shared by serial and parallel programs: counter-based random numbers, data generator, distance, k-means++, silhouette/elbow. Shared on purpose, so both programs use *identical* data, distance and start. |
| `src/kmeans_serial.cpp` | Sequential baseline (no MPI, no OpenMP threads). Classical **and** optimised, weighted, choose-K. |
| `src/kmeans_hybrid.cpp` | The parallel program (MPI + OpenMP). All the pragmas discussed below live here. |
| `scripts/verify.sh` | Correctness: 12 parallel configurations × 4 variants must equal the serial answer. |
| `scripts/demo_reallife.sh` | Customer-segmentation story in 4 steps. |
| `scripts/benchmark.sh`, `make_report.py` | Run all experiments; print tables in the layout of your slides; draw graphs. |
| `Makefile` | `make`, `make NATIVE=1` (adds `-march=native`), `make clean`. |

### Command-line options (both programs)
| Option | Meaning | Default |
|---|---|---|
| `--n --d --k --iters` | points, dimensions, clusters, fixed iteration count | 4000000, 16, 16, 50 |
| `--ktrue` | how many natural groups the generated data really has | = k |
| `--algo lloyd\|hamerly` | classical / optimised (exact) | lloyd |
| `--init random\|kmeanspp` | starting centroids | kmeanspp |
| `--weighted` | weighted k-means (weight 1..6 per point) | off |
| `--mode run\|choosek` | normal run / find the best K | run |
| `--kmin --kmax --sample` | range of K and sample size for `choosek` | 2, 12, 2000 |
| `--out file` | write final centroids (used by the correctness check) | – |
| *hybrid only* `--threads` | OpenMP threads per MPI process | `OMP_NUM_THREADS` |
| `--accum private\|reduction\|atomic\|critical\|falseshare` | how threads add into the shared sums (§3.1, §5) | private |
| `--sched static\|dynamic\|guided\|task` and `--chunk` | how points are shared among threads | static, 2048 |
| `--region persistent\|forkjoin` | one parallel region vs. a new one per phase | persistent |

---

## 2. Theory (what to write in the report's background section)

### 2.1 Classical K-Means (Lloyd's algorithm)
Given N points in D dimensions and a number K:
1. choose K starting centroids;
2. **assignment**: every point goes to the nearest centroid, `d(p,c)² = Σ_k (p_k − c_k)²`;
3. **update**: every centroid moves to the mean of its points;
4. repeat 2–3 (until nothing changes, or — as in our timing runs — a *fixed* number of iterations).

It minimises the **SSE** (sum of squared errors) `Σ_i d(p_i, c_label(i))²`. Cost per iteration: `N·K·D` operations — for 4 M × 16 × 16 ≈ 10⁹, so 5·10¹⁰ for 50 iterations. It finds a *local* minimum, so the start matters.

### 2.2 Choice of K
K is an input, but in real life nobody knows it. Two standard tools (both implemented in `--mode choosek`):
* **Elbow method** – run k-means for K = 2…Kmax and plot SSE. SSE always falls as K grows; the "elbow" where the curve stops falling steeply is a good K. We pick it automatically as the point farthest from the straight line joining the first and last point of the (normalised) curve.
* **Silhouette** – for each point, `s = (b − a) / max(a, b)` where `a` = mean distance to its own cluster, `b` = mean distance to the nearest other cluster. Mean `s` near 1 = compact, well-separated clusters. Pick the K with the highest mean silhouette.

Measured on the generated data with 6 true groups: SSE collapses from 4.47 M (K=5) to 0.51 M (K=6) and then barely moves; silhouette peaks at **0.847 for K=6**. Both methods answer K=6 (see §7).
Silhouette is O(M²), so it runs on a **sample** (default 2000 points). Each candidate K is an independent job → we run them as **OpenMP tasks** spread over MPI ranks (§3.8).

### 2.3 Weighted K-Means
Point *i* has a weight *wᵢ*. Centroid = `Σ wᵢ pᵢ / Σ wᵢ`, SSE = `Σ wᵢ d²`. Use it when points are not equally important, or when one row stands for many identical items (a *histogram bin* with a count).
In the code the only change is `acc += w*x` and `cnt += w` instead of `+= x` and `+= 1` (see `accumulate()` and `update_cluster()`), plus weighted k-means++ seeding.

### 2.4 Optimised K-Means
Two optimisations, both **exact** (same answer as the classical algorithm — we check this):
* **k-means++ seeding** – first centre random, each next centre chosen with probability ∝ `w·(distance to nearest chosen centre)²`. Spreads the start out. Measured (4 M points, K=16, 50 iterations): final SSE **4.21·10⁹ with k-means++ vs 8.31·10⁹ with random start**.
* **Hamerly's bounds** – for each point keep `u` (upper bound on the distance to its own centroid) and `l` (lower bound on the distance to the 2nd-nearest). When centroids move by `δ`, update `u += δ[own]`, `l −= max δ`. If `u ≤ max(l, s[own])` (s = half the distance from own centroid to its closest other centroid), the point **cannot** change cluster → skip all K distance calculations. Measured: **99.2 % of the assignment steps skipped** (194.5 M of 196 M) and 5.4× faster serially.
  Cost: 2 extra doubles per point (`up[]`, `lo[]`), and still one pass over the data per iteration to accumulate the sums.

### 2.5 Serial vs. Parallel — what changes
| | Serial | Hybrid MPI + OpenMP |
|---|---|---|
| Data | all N points in one array | N/P points per process, generated locally from the *global index* (nothing big is ever sent) |
| Assignment | one loop over N | loop over N/P points shared by T threads |
| Update | sums are added in one place | thread-private sums → merged by threads → merged by `MPI_Allreduce` |
| Communication | none | one `MPI_Allreduce` of `K·D+K+2` doubles (2.2 KB) per iteration |
| Result | reference | identical centroids/SSE (tiny rounding differences allowed) |
| What limits speed | CPU | memory bandwidth, barriers, Allreduce, Amdahl's serial part |

### 2.6 Real-life examples (use these in the report)
| Application | Point = | D | K | Why this variant |
|---|---|---|---|---|
| **Customer segmentation** (retail, telecom, banking) | one customer: recency, frequency, spend, basket size… | 10–50 | 4–10, found with elbow/silhouette | **weighted** by revenue so big customers matter more |
| **Image colour quantisation / compression** | one pixel (R,G,B) | 3 | 16–256 (palette size) | a 12-megapixel photo = 12 M points → exactly our "large N" case; weighted version clusters the colour *histogram* |
| **Depot / hub placement** (logistics, ambulances) | a neighbourhood (x, y) | 2 | number of depots | **weighted** by population / parcel volume |
| **Electricity smart-meter profiles** | a household's 24 hourly readings | 24 | 5–8 load types | points far from every centroid = faulty/fraudulent meters (our program prints the most unusual point) |
| **Document / news clustering** | TF-IDF vector | 100–1000 | 20–100 | large D makes the `N·K·D` cost huge → Hamerly pruning pays off |
| **Sensor / network anomaly detection** | one measurement window | 10–100 | few | large distance to nearest centroid = anomaly |

---

## 3. How each slide topic is used in the code

### 3.1 `#pragma omp critical` vs `atomic` (your slide 1)
Rule from the slide: *atomic = one simple read-modify-write of one memory location; critical = a block with several operations.*

| Place in code | Directive | Why |
|---|---|---|
| `accumulate<M_ATOMIC>`: `tgt[k*D+d] += w*x[d]` | `#pragma omp atomic` | one simple update of one location (hardware instruction) |
| `accumulate<M_CRIT>`: the loop over d **and** the count update | `#pragma omp critical(kmeans_acc)` | a *block* of many updates that must look like one unit |
| end of `assign_phase`: `g_sse += local`, `g_changed += …`, `g_skipped += …` | `#pragma omp atomic` | one scalar `+=` per thread per iteration — a lock would be overkill |
| final pass: `if (worst > gworst) { gworst = worst; gworstI = wi; }` | `#pragma omp critical(outlier)` | **two** variables (value + index) must change together; atomic cannot do that. Runs once per thread, so cost is irrelevant |

Neither is the *fast* way to accumulate. The best way (default, `--accum private`) needs **no** atomic and **no** critical at all: each thread adds into its own buffer and the buffers are merged afterwards (§3.9, §5). Measured on the build machine with 1 thread: private ≈ serial speed, `reduction` clause ≈ +15 %. With several threads `atomic` and `critical` serialise all threads on the same 34 cache lines — `benchmark.sh` experiment E2 / Table D shows how much slower on your machine.

### 3.2 `master` vs `single` (your slide 2)
| Place | Directive | Reason (using the slide's rules) |
|---|---|---|
| `comm_step()` → `MPI_Allreduce` | `#pragma omp master` + explicit `#pragma omp barrier` | We initialise MPI with `MPI_THREAD_FUNNELED`: only the **main thread** may call MPI → must be *master* (a `single` might pick another thread). `master` has no implicit barrier, but the next step needs the result, so we add the barrier ourselves. |
| `begin_iter()` (zero the shared sums) | `#pragma omp single` | any thread may do it, and everybody must **wait** until it is done → `single`'s implicit barrier is exactly what we need. |
| update of centroids (`update_cluster` loop + `finish_update`) | `#pragma omp single` | tiny serial work (K·D = 256 operations), result needed by all threads → barrier wanted. |
| `log_iter()` (print) | `#pragma omp master` (no barrier) | nobody depends on the print → "no barrier overhead" case. (It reads `g_chg_hist[it]`, which is never overwritten, so there is no race.) |
| creating the tasks (§3.8) | `#pragma omp single` | one producer thread, the rest consume tasks while waiting at the barrier |

### 3.3 Data-sharing clauses (your clause table)
| Clause | Where | Note |
|---|---|---|
| `shared` | all state in globals (`G`, `g_acc`, …) | globals are shared automatically; we did this on purpose so tasks never silently get *private* copies |
| `private` | per-thread locals (`tid`, `worst`, loop counters) | every thread declares its own inside the region |
| `firstprivate` | `#pragma omp task firstprivate(lo, hi)` and `firstprivate(ki)` | a task runs *later*, so it must capture the loop values *now* |
| `reduction` | `--accum reduction` (array section `g_acc[0:g_AL]`) and the final SSE (`reduction(+:sse)`) | array reduction = private zero-initialised copy per thread, added at the end |
| `lastprivate`, `collapse`, `default(none)` | **not used** | no loop value is needed after a loop; our nested loops over K and D are tiny (K·D=256), collapsing them would only add overhead; `default(none)` is unnecessary because shared state is in globals |
| `nowait` | **not used** | every `for` here is followed by a step that reads its result, so the barrier is required. (It is the right tool if you add independent work after a loop.) |

### 3.4 Loop scheduling (your section 5.7 slide)
`assign_phase` uses `#pragma omp for schedule(runtime)` and the program calls `omp_set_schedule()` from `--sched`. This is the **`runtime`** row of the slide: change policy without recompiling.
* **static** (default) – every point costs the same in the classical algorithm → zero scheduling overhead, contiguous blocks (good for cache and for false sharing).
* **dynamic / guided** (`--chunk`) – the right choice for the **Hamerly** algorithm, where cost per point varies (a pruned point is almost free, a full search costs K distances).
* **task** – see §3.8.

### 3.5 One parallel region vs. many (your Table 5.3)
`--region persistent` = one `#pragma omp parallel` around the *whole* iteration loop, with `single / for / master / barrier` inside (threads are created once).
`--region forkjoin` = three `#pragma omp parallel [for]` per iteration (zeroing, assignment, centroid update) — "two `parallel for`" in the book's wording. Experiment E4 / Table G reproduces the layout of Table 5.3. Expect the persistent version to win by a few percent on small problems and the difference to vanish at N = 4 M (fork/join costs microseconds; one iteration costs ~0.5 s).

### 3.6 Caches, coherence and false sharing (your section 5.9 slide) → see the full audit in §5.

### 3.7 MPI collectives (your MPI slide)
| Call | Where | Why this one |
|---|---|---|
| `MPI_Bcast` | rank 0 sends the K·D start centroids | all ranks must start identically |
| `MPI_Allreduce(SUM)` | every iteration: partial sums, counts, SSE, #changed in **one** call | *All*reduce, not *Reduce*: every rank needs the new centroids. One message of 2.2 KB instead of four small ones (latency dominates for small messages) |
| `MPI_Reduce(MAX)` | max of per-rank times = parallel time | the slowest process defines the run time |
| `MPI_Allreduce(MAXLOC)` + `MPI_Bcast` | most unusual point (value + owner rank, then its index) | classic "max with location" |
| `MPI_Barrier` | before starting the timer | all ranks start timing together |
| *Scatter / Gather / Allgather* | **not used, deliberately** | the data never has to travel: each rank *generates* its own block from global indices. (With a real file you would `MPI_Scatterv` it once, before the timed loop.) |

### 3.8 Tasking
Two real uses (a task is a unit of work put in a pool, executed by whichever thread is free):
1. **Assignment loop, `--sched task`** – one thread (`single`) creates one task per chunk of `--chunk` points; the other threads execute them. Gives dynamic load balancing like `schedule(dynamic)` but with a producer/consumer pool. Tasks may run on any thread, therefore a task calls `omp_get_thread_num()` *inside* itself to find its own private buffer. The implicit barrier at the end of `single` guarantees all tasks finished before we merge.
2. **Choice of K** – every candidate K (each a complete small k-means + silhouette) is one task. Jobs have different lengths, tasks balance that automatically; the jobs are also dealt cyclically to MPI ranks, so both levels of the machine are used.

Tasks are **not** used where they would lose: updating K = 16 centroids is 256 operations — creating a task costs more than the work. (For K in the thousands you would use `taskloop`.)
**A real bug that the test-suite caught:** with tasks + private buffers, a thread still leaving the previous barrier could *steal a task* and accumulate into its buffer **before** it had run its own `memset` — the `memset` then erased results. Fix: an explicit `#pragma omp barrier` between "clear my buffer" and "tasks may start" (marked in the code). 75 stress runs afterwards: 0 mismatches.

### 3.9 The "best approach" (decided by measurement, not opinion)
Default = **one persistent parallel region + `schedule(static)` + padded per-thread private buffers + merge loop + one `MPI_Allreduce`, with Hamerly + k-means++ for the optimised algorithm.** It is the "private storage, then update shared storage when done" idea from the bottom of your 5.9 slide. Evidence on 1 thread (4 M points, 50 it., same machine): serial 28.5 s, hybrid-private 28.5 s, hybrid-`reduction` ≈ +15 %.

---

## 4. Loop-carried dependencies — where they are and how each is handled
A loop-carried dependency means iteration *i* needs a value produced by iteration *i−1* (or the same memory is written by several iterations), so the iterations cannot simply run in parallel.

| # | Where | Kind | How we handled it |
|---|---|---|---|
| 1 | Outer loop over k-means iterations (`for it`) | **True** dependence: iteration t+1 reads centroids written by t | Cannot be parallelised. Kept sequential; parallelism is *inside* each iteration. Order is enforced by barriers and by the `MPI_Allreduce`. |
| 2 | `sum[label][d] += w·x[d]` over all points | Memory dependence (many iterations write the same sums) — a *reduction* | Thread-private padded buffers, merged afterwards (default); `reduction` array clause; or `atomic` / `critical` (slow alternatives). |
| 3 | `sse`, `changed`, `skipped` counters | Scalar reduction | per-thread struct `TS` (cache-line aligned), then **one atomic add per thread** |
| 4 | Scan over centroids `j` keeping `best`/`second` | Value carried from j−1 to j | Stays sequential inside one point (K is tiny); parallelism is **across points**, which are independent. |
| 5 | `Σ_d (p_d − c_d)²` | Scalar reduction | `#pragma omp simd reduction(+:s)` (vectorised) |
| 6 | Random-number generation | A normal RNG has hidden *state* → number i needs number i−1 | Replaced by a **counter-based** generator: `random = hash(seed, global index, coordinate)`. Every point is independent, generation is parallel, and the data is identical for any number of processes. |
| 7 | k-means++ choosing centre j | True dependence on centres 0…j−1 and a prefix-sum scan | Sequential, but on a 20 000-point sample on rank 0 only (≈ 0.02 s, 0.07 % of the run); result sent with `MPI_Bcast`. |
| 8 | Hamerly bounds `u[i]`, `l[i]` carried from one iteration to the next | Inter-iteration dependence, but *per point* | Each point's bounds are touched only by the thread handling point *i*; barriers separate iterations → no race. |
| 9 | Per-process start offset `off = Σ sizes of earlier ranks` | Prefix sum (dependence on earlier ranks) | Replaced by the closed form `rank·(N/P) + min(rank, N mod P)`. |
| 10 | "Most unusual point": max value + its index | Two linked variables | `#pragma omp critical` once per thread, then `MPI_MAXLOC`. |
| 11 | Merge loop `acc[e] += Σ_t buf[t][e]` | Looks like accumulation but element *e* is owned by exactly one iteration | No dependence between iterations → plain `omp for schedule(static)`. |
| 12 | Writes `lab[i]`, `up[i]`, `lo[i]` | Output dependence? No — index i belongs to one iteration | No action needed (this is why the assignment step is "embarrassingly parallel"). |

---

## 5. Cache coherence & false sharing audit
Cache lines are 64 bytes (8 doubles, 16 ints). If two cores write different variables that live in the **same** line, the line bounces between the cores (MESI invalidations) although there is no race — that is *false sharing* (your 5.9 slide: `y` of the matrix-vector product). We audited every shared structure:

| Structure | Access pattern | Risk | Resolution |
|---|---|---|---|
| `X` points (N×D) | read-only, each thread reads its own contiguous block | none | 64-byte aligned (`posix_memalign`); one point = 128 B = 2 whole lines; parallel first-touch initialisation (NUMA-friendly) |
| `C` centroids (K×D = 2 KB = 32 lines) | read-only during assignment; written once per iteration by one thread | read-sharing is free (lines stay in *Shared* state). One write burst per iteration invalidates 32 lines — negligible | none needed |
| `lab[]`, `up[]`, `lo[]` | each thread writes only its own points | **possible**: at block edges, and everywhere if chunks are tiny (16 ints per line) | `schedule(static)` = a few huge contiguous blocks; dynamic/task chunk default **2048** points (8 KB of labels). Experiment E3chunk (chunk = 1, 16, 256, 4096) shows the penalty of tiny chunks |
| per-thread statistics `TS` | written constantly by each thread | **real**: a 24-byte struct → 2–3 threads per line | `struct alignas(64) TS` → one line per thread |
| per-thread sum buffers | written constantly | **real** if buffers are adjacent | each buffer padded to a multiple of 8 doubles (`g_stride`) and 64-byte aligned |
| `--accum falseshare` | **deliberately bad** layout `buf[e*T + tid]` | thread 0..T−1 write neighbouring doubles of one line (no race, but line ping-pong) | exists to *measure* false sharing (Table D) — it is the "unpadded y vector" of your slide, and `private` is the fix of the slide ("each thread uses its own private storage, then updates the shared storage") |
| `--accum atomic / critical` | all threads write the same 34 lines | **true sharing** (same addresses): cores fight for ownership of the line, or for a lock | not default; measured in Table D |
| `g_sse`, `g_changed`, `g_skipped` | `atomic` add, once per thread per iteration | true sharing but only T operations per iteration | acceptable |

How to *see* it on your machine: run Table D/F of `benchmark.sh`; optional deep check on Linux: `perf c2c record -- mpirun -np 1 ./kmeans_hybrid --accum falseshare …` then `perf c2c report` (shows "HITM" cache-line contention).
Measured on our 4-core machine (§7.3, §7.4): the `falseshare` layout is 1.66x slower than the padded `private` layout, `atomic` 2.5x and `critical` 4.5x slower, and dynamic chunk size 1 is 34 % slower than chunk size 4096.

---

## 6. Step-by-step run guide, with what you should see

**Step 1 – machine facts (needed for the report).** `nproc; lscpu | grep -E "Model name|Core|Thread|L2|L3"`. Write down the model name and core count: the proposal requires it.

**Step 2 – build.** `make`. You get `kmeans_serial` and `kmeans_hybrid`. (The serial program is compiled with `-fopenmp-simd`: this only allows vectorising the distance loop — no threads — so serial and parallel run the *same inner loop* and the speedup measures parallelism only.)

**Step 3 – correctness.** `bash scripts/verify.sh`. For classical, Hamerly, weighted, weighted-Hamerly it runs the serial program and 12 parallel configurations (every accumulation strategy, static/dynamic/guided/task, 1–3 processes, 1–4 threads, both region styles). Each line ends with `max centroid difference = 0.000e+00 -> OK`. This is your "correctness check" (item 8 of the proposal; SSE equal to ~13 digits).

**Step 4 – one run, and how to read the output.**
```
$ mpirun -np 2 --bind-to none ./kmeans_hybrid --n 4000000 --iters 50 --threads 2
iter   0  changed=4000000 ...            <- first iteration: every label changes
iter   1  changed=...                    <- falls quickly: convergence
cluster sizes (weight) after last iteration: ...
most unusual point (outlier): global index ..., distance ...
setup: generate=..s init=..s | MPI_Allreduce total=..s (..% of loop)   <- communication share
RESULT ... time=<seconds of the 50 iterations only> sse=<final SSE>
```
`time` = slowest process, measured with `MPI_Wtime` around the iteration loop only (data generation excluded) — as the proposal says.

**Step 5 – scaling experiments.** `bash scripts/benchmark.sh` (default N = 4 M, 50 iterations, 3 repetitions averaged). Quick trial: `bash scripts/benchmark.sh 200000 10 1`. Then `python3 scripts/make_report.py` prints **Table A** (times), **B** (speedup), **C** (efficiency) in the layout of your Table 3.5/3.6/3.7 and draws `results/scaling_*.png` (time, speedup with ideal line, efficiency; MPI vs OpenMP vs hybrid). Tables D–H cover accumulation strategy, schedule/task, chunk size (false sharing), region style (Table 5.3 layout) and choice-of-K.

**Step 6 – write the analysis.** For every table explain *why* it looks as it does (§7 gives the expected reasons).

---

## 7. Timings (all measured on our own machine)

**Test machine:** Intel Core i7-1165G7 @ 2.80 GHz (4 virtual CPUs given to the VM), 5.3 GiB RAM, Ubuntu 26.04.1 LTS guest in VirtualBox on a Windows laptop, g++ 15.2.0, Open MPI 5.0.10 (details in `results/machine_info.txt`).
**Problem:** 4,000,000 points, D = 16, K = 16, 50 iterations, k-means++ start. Time = main iteration loop only (`MPI_Wtime`, maximum over processes).
**Method:** every time is the **median of 3 runs** with a 30 s cooling pause between runs (classical algorithm) or the mean of 3 runs from `scripts/benchmark.sh` (optimised algorithm). The median is used because a laptop VM occasionally produces one slow run (we saw 41.6 s, 33.7 s and 30.6 s instead of ~11-16 s); the median removes a single spike without hand-picking runs. Raw output: `results/tables.txt`, `results/results.csv`.

### 7.1 Classical algorithm (serial = 29.06 s)
| Configuration | Workers | Time (s) | Speedup | Efficiency |
|---|---|---|---|---|
| OpenMP (1 x p) | 2 | 16.14 | 1.80 | 0.90 |
| OpenMP (1 x p) | 4 | 11.15 | 2.61 | 0.65 |
| MPI (p x 1) | 2 | 15.91 | 1.83 | 0.91 |
| MPI (p x 1) | 4 | 11.47 | 2.53 | 0.63 |
| Hybrid (2 x 2) | 4 | 11.23 | 2.59 | 0.65 |

### 7.2 Optimised algorithm - Hamerly + k-means++ (serial = 3.49 s)
| Configuration | Workers | Time (s) | Speedup | Efficiency |
|---|---|---|---|---|
| OpenMP (1 x p) | 2 | 2.30 | 1.52 | 0.76 |
| OpenMP (1 x p) | 4 | 2.15 | 1.63 | 0.41 |
| MPI (p x 1) | 2 | 2.31 | 1.51 | 0.75 |
| MPI (p x 1) | 4 | 2.15 | 1.62 | 0.41 |
| Hybrid (2 x 2) | 4 | 2.14 | 1.63 | 0.41 |

Graphs: `results/final_lloyd.png`, `results/final_hamerly.png` (made by `scripts/final_figures.py`; the 1-worker point is the true serial program).

**Algorithmic gain.** Serial optimised vs serial classical: 29.06 s / 3.49 s = **8.3x** with the identical final SSE. Optimised on 4 workers vs classical serial: 29.06 / 2.14 = about **13.6x**. The optimisation contributes more than parallelism does.

### 7.3 Accumulation strategy (4 threads, N = 1,000,000, 10 iterations)
| Strategy | Time (s) | Relative to `private` |
|---|---|---|
| private (padded per-thread buffers) | 0.539 | 1.00 |
| reduction clause | 0.503 | 0.93 |
| atomic | 1.362 | 2.53x slower |
| critical | 2.415 | 4.48x slower |
| falseshare (deliberately bad layout) | 0.894 | 1.66x slower |

### 7.4 Schedules, tasks, chunk size (4 threads, same size)
static 0.499 s, task 0.507 s, dynamic 0.518 s, guided 0.572 s. Dynamic chunk size 1 / 16 / 256 / 4096: 0.759 / 0.597 / 0.564 / 0.566 s.

### 7.5 Interpretation and honest limits
* **Hybrid, pure MPI and pure OpenMP perform the same** on this machine at 4 workers (differences of 1-3 % are inside the noise); we do not claim a winner. `MPI_Allreduce` took only 2-3 % of the loop, so communication is not the limit.
* **Classical scaling** is close to ideal on 2 workers (efficiency 0.90-0.91) and drops to about 0.65 on 4. Likely causes (not individually measured): the laptop CPU lowers its clock under all-core load, the VM's virtual CPUs share the physical chip, memory bandwidth and barriers.
* **Optimised scaling is weaker** (1.6x on 4 workers): after pruning, each iteration does little arithmetic per byte read, which is consistent with a memory-bandwidth limit; it is still the fastest version in absolute time.
* **Critical and atomic** are 4.5x and 2.5x slower than private buffers because all threads contend for the same cache lines; the padded private buffers avoid this. The `falseshare` layout, which differs from `private` only in memory layout, is 1.66x slower, and dynamic chunk size 1 is 34 % slower than chunk size 4096 (consistent with false sharing on `lab[]` plus scheduling overhead; the two effects were not separated).
* **Run-to-run variation:** a separate session on the same machine gave classical 4-worker times of about 7.8-8.3 s (speedup ~3.4) instead of 11.2 s. The speed of this laptop VM depends on temperature and power state, so absolute numbers are only comparable inside one session.
* In the benchmark the hybrid 1x1 run was about 6 % faster than the serial program; we could not explain this and use the serial program as T1.
* The choice-of-K benchmark (`results/tables.txt`, Table H) used a serial baseline that was inflated; use its hybrid times (0.905, 0.507, 0.297 s for 1, 2, 4 threads) rather than its speedup column.

## 8. Code walkthrough (read together with the comments in the files)

**`common.h`**
* `splitmix64`, `mix`, `u01` – counter-based random numbers (dependency #6).
* `gen_point(S, i, out)` – point *i* = true centre `hash(i) mod Ktrue` + Gaussian noise (Box–Muller from two hashed uniforms). `gen_weight` – weight 1…6.
* `dist2` – squared distance with `omp simd reduction`. `nearest2` – nearest **and second nearest** centroid.
* `init_random`, `kmeanspp`, `make_init` – seeding; `make_sample` – evenly spaced sample by global index.
* `eval_k` – small k-means (≤ 40 iterations) + SSE + silhouette; `pick_elbow` – knee rule.

**`kmeans_serial.cpp`** – read the main loop top to bottom: assignment (classical branch or Hamerly branch), accumulation, centroid update, bound update. It is the specification the parallel program must reproduce.

**`kmeans_hybrid.cpp`**, in file order:
1. `Shared G` – everything the threads share (global on purpose). `TS` – padded per-thread statistics.
2. `accumulate<M>` – the five ways to add a point to its cluster (§3.1, §5). Template parameter `M` → the strategy is chosen at compile time, no `if` in the hot loop.
3. `process_point<M>` – one point: Hamerly test or full search, label write, statistics, accumulation.
4. `assign_phase<M>` – called by **all** threads: clear private buffer → (barrier if tasks) → `omp for schedule(runtime)` *or* `single`+tasks → merge scalar stats with `atomic` → barrier → merge buffers with a parallel loop.
5. `comm_step` – master thread only: pack SSE and #changed behind the sums and call **one** `MPI_Allreduce`.
6. `update_cluster`, `finish_update` – mean = sum / weight (empty cluster keeps the old centroid), centroid movement `delta`, Hamerly's `s[]`.
7. `iterate_persistent` / `iterate_forkjoin` – the two region styles (§3.5).
8. `main` – MPI init (`FUNNELED`); block distribution `nloc, off`; parallel data generation; init on rank 0 + `Bcast`; **timed region**; final exact SSE with `reduction` + `critical` outlier; result line; centroid file. The `choosek` mode (tasks) is at the top of `main`.

Why `MPI_THREAD_FUNNELED` and not `MPI_THREAD_MULTIPLE`: only one thread talks to MPI, so the cheapest level is enough.

---

## 9. What was verified, and honest limits
* **Verified:** compiles warning-free under `-Wall` (only warnings are inside Open MPI's own headers); 48 configurations (4 variants × 12) reproduce the serial centroids with difference 0.0 and SSE equal to ≥ 13 digits; 75 additional stress runs of the task paths (found and fixed one race, §3.8); uneven N (100 001 points over 3 ranks) works; choose-K gives K = 6 in serial and hybrid.
* **Measured on our machine (i7-1165G7, 4 vCPUs, VirtualBox):** serial and parallel times (§7.1, §7.2), pruning rate, k-means++ effect, accumulation strategies, schedules, chunk sizes. All numbers in §7 are measurements; nothing is modelled or projected.
* **Limits:** timings come from a laptop VM whose speed varies with temperature and power state (see §7.5), so speedups are only comparable within one session; the cause of the loss of efficiency at 4 workers was not isolated.
* Rounding: floating-point addition order differs between versions, so SSE can differ in the last 1–2 digits; the check uses a tolerance of 1e-6 on centroids.
* Dataset is generated (as in the proposal). To use real data, replace `gen_point` by a loader (`MPI_Scatterv` of the file blocks) — nothing else changes.
* Fixed iteration count (50), as in the proposal; the program prints `changed=` per iteration so you can also show convergence.

## 10. Checklist against your proposal
1 sequential baseline ✔ · 2 fixed-seed start by rank 0 + `MPI_Bcast` ✔ · 3 block distribution, local generation ✔ · 4 OpenMP assignment, `schedule(static)`, no races ✔ · 5 private sums + merge ✔ (and `reduction` variant) · 6 `MPI_Allreduce`, empty cluster keeps centroid ✔ · 7 fixed 50 iterations ✔ · 8 correctness check vs serial ✔ (`verify.sh`) · 9 timing with `MPI_Wtime`, max over ranks, only the loop ✔; pure OpenMP / pure MPI / hybrid ✔ · speedup, efficiency, ideal line, graphs ✔ (`make_report.py`) · explain non-linear speedup ✔ (§7.2).
