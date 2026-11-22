// ============================================================================
//  kmeans_hybrid.cpp  -  PARALLEL K-MEANS  (MPI between processes + OpenMP inside)
//
//  build : mpicxx -O3 -std=c++17 -fopenmp -o kmeans_hybrid kmeans_hybrid.cpp
//  run   : OMP_NUM_THREADS=4 mpirun -np 2 --bind-to none ./kmeans_hybrid --n 4000000 --k 16
//
//  MAP OF THE PARALLELISM
//  ----------------------
//   MPI    : the N points are split in P equal blocks (one per process).  Only the
//            tiny K*D centroid sums (K*D+K+2 doubles) travel between processes, once
//            per iteration, with ONE MPI_Allreduce.
//   OpenMP : inside a process, T threads share the block of points.
//
//   One k-means iteration  =  [A] zero accumulators           (omp single)
//                             [B] assign + accumulate points   (omp for | omp task)
//                             [C] MPI_Allreduce                (omp master + barrier)
//                             [D] new centroids                (omp single)
//   The iteration loop itself can NOT be parallel: iteration t+1 needs the centroids
//   of iteration t  (true loop-carried dependency).
// ============================================================================
#include <mpi.h>
#include <omp.h>
#include "common.h"

// ----------------------------------------------------------------------------
//  Accumulation strategies for  sum[k][:] += w*x   (the shared-write problem)
// ----------------------------------------------------------------------------
enum { M_RED = 0, M_PRIV, M_ATOMIC, M_CRIT, M_FALSE };

// ----------------------------------------------------------------------------
//  Shared state.  Kept in globals on purpose: globals are SHARED by all threads,
//  so we never have to reason about firstprivate/private surprises inside tasks.
// ----------------------------------------------------------------------------
struct Shared {
    int K, D;
    long nloc;
    const double* X;     // local points   nloc x D   (row-major, 64-byte aligned)
    const double* W;     // local weights  (nullptr if unweighted)
    double* C;           // centroids K x D  (read-only during the assignment step!)
    int* lab;            // label of every local point
    double *up, *lo;     // Hamerly bounds  (upper to own centroid, lower to 2nd best)
    double *sh, *delta;  // Hamerly: half distance to closest other centroid / centroid movement
    double maxdelta;
    bool ham;
    int iter;
};
static Shared G;

// per-thread statistics.  alignas(64) = every thread's struct sits in its OWN cache
// line.  Without it, neighbouring threads would hammer the same line -> FALSE SHARING.
struct alignas(64) TS { double sse = 0; long changed = 0; long skipped = 0; };
static TS* g_ts;
static double *g_acc, *g_priv, *g_false, *g_Cn;
static long g_AL, g_stride;
static int g_T, g_chunk, g_rank = 0, g_P = 1, g_iters;
static double g_sse; static long g_changed, g_skipped, g_skipped_total;
static double g_comm_time;
static std::vector<double> g_sse_hist, g_chg_hist;
static std::vector<double> g_lastcnt;

static void* amalloc(size_t bytes) {            // 64-byte aligned allocation (= cache-line aligned)
    void* p = nullptr;
    if (posix_memalign(&p, 64, bytes ? bytes : 64)) { std::fprintf(stderr, "alloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    return p;
}

template <int M> static inline double* target(int t) {
    if constexpr (M == M_PRIV) return g_priv + (long)t * g_stride;
    else if constexpr (M == M_FALSE) return g_false;
    else return g_acc;
}

// ----------------------------------------------------------------------------
//  accumulate: add weighted point x to cluster k.   One variant per strategy.
// ----------------------------------------------------------------------------
template <int M>
static inline void accumulate(double* tgt, int k, const double* x, double w, int tid) {
    const int D = G.D, K = G.K, T = g_T;
    const long KD = (long)K * D;
    if constexpr (M == M_RED || M == M_PRIV) {
        // M_RED : tgt is this thread's private copy made by  reduction(+:...)
        // M_PRIV: tgt is this thread's own padded buffer.  No other thread touches it
        //         -> no race, no atomic, no lock, no false sharing.
        double* r = tgt + (long)k * D;
        for (int d = 0; d < D; ++d) r[d] += w * x[d];
        tgt[KD + k] += w;
    } else if constexpr (M == M_ATOMIC) {
        // every statement is ONE simple read-modify-write of ONE location -> atomic is
        // enough (hardware instruction, cheaper than a lock).  Still slow: all threads
        // fight for the same K*D+K cache lines (true sharing).
        for (int d = 0; d < D; ++d) {
            double v = w * x[d];
#pragma omp atomic
            tgt[(long)k * D + d] += v;
        }
#pragma omp atomic
        tgt[KD + k] += w;
    } else if constexpr (M == M_CRIT) {
        // a BLOCK of several updates that must appear as one unit -> critical.
        // (Correct but slowest: threads queue up for a lock for every single point.)
#pragma omp critical(kmeans_acc)
        {
            for (int d = 0; d < D; ++d) tgt[(long)k * D + d] += w * x[d];
            tgt[KD + k] += w;
        }
    } else {
        // M_FALSE: DELIBERATELY BAD layout for the false-sharing experiment.  Element e
        // of thread t lives at e*T+t, so the T threads write into the SAME 64-byte line
        // although they never touch the same variable (no race, but the cache line
        // ping-pongs between cores:  MESI invalidations).
        for (int d = 0; d < D; ++d) tgt[((long)k * D + d) * T + tid] += w * x[d];
        tgt[(KD + k) * T + tid] += w;
    }
}

// ----------------------------------------------------------------------------
//  process_point: assignment of ONE point (+ accumulation).  Writes only to
//  lab[i], up[i], lo[i] (index i is owned by exactly one iteration -> no race)
//  and to the executing thread's private statistics.
// ----------------------------------------------------------------------------
template <int M>
static inline void process_point(long i, double* tgt, int tid, TS& ts) {
    const int D = G.D, K = G.K;
    const double* x = G.X + (size_t)i * D;
    const double w = G.W ? G.W[i] : 1.0;
    const int a = G.lab[i];
    int best; double bd = 0, sd = 0;
    if (!G.ham || G.iter == 0) {
        nearest2(x, G.C, K, D, best, bd, sd);                      // classical: N*K distances
        if (G.ham) { G.up[i] = std::sqrt(bd); G.lo[i] = std::sqrt(sd); }
    } else {                                                        // Hamerly (exact, optimised)
        double u = G.up[i] + G.delta[a];                            // bounds drift with centroid movement
        double l = G.lo[i] - G.maxdelta;
        double m = std::max(G.sh[a], l);
        best = a;
        if (u <= m) ++ts.skipped;                                   // pruned: ZERO distances computed
        else {
            u = std::sqrt(dist2(x, G.C + (size_t)a * D, D));
            if (u <= m) ++ts.skipped;
            else { nearest2(x, G.C, K, D, best, bd, sd); u = std::sqrt(bd); l = std::sqrt(sd); }
        }
        G.up[i] = u; G.lo[i] = l;
    }
    if (best != a) { G.lab[i] = best; ++ts.changed; }
    if (!G.ham) ts.sse += w * bd;
    accumulate<M>(tgt, best, x, w, tid);
}

// ----------------------------------------------------------------------------
//  [B] ASSIGNMENT PHASE - must be called by ALL threads of a parallel region
//  ("orphaned" worksharing constructs bind to the region of the caller).
//  Loop schedule is chosen at run time (omp_set_schedule + schedule(runtime)),
//  exactly the "runtime" row of the scheduling slide.
// ----------------------------------------------------------------------------
template <int M>
static void assign_phase(bool use_task) {
    const int tid = omp_get_thread_num();
    const int T = g_T;

    // each thread clears ITS OWN private buffer (nobody else touches it until the merge)
    if constexpr (M == M_PRIV) std::memset(g_priv + (long)tid * g_stride, 0, sizeof(double) * g_AL);
    if constexpr (M == M_FALSE) for (long e = 0; e < g_AL; ++e) g_false[e * T + tid] = 0.0;

    // RACE FIXED HERE (found by our own correctness test):  a thread that is still
    // leaving the previous barrier may already STEAL a task and add into its buffer
    // BEFORE it executed the memset above - the memset would then erase real results.
    // An explicit barrier guarantees "all buffers cleared" before any task is created.
    if (use_task && (M == M_PRIV || M == M_FALSE)) {
#pragma omp barrier
    }

    if (use_task) {
        // ---- TASKING:  one thread (single) PRODUCES chunks of points as tasks, every
        // thread (incl. the producer, once it is done) CONSUMES them -> dynamic load
        // balancing.  Tasks run on arbitrary threads, so they must use omp_get_thread_num()
        // *inside the task* to find their private buffer.  The implicit barrier at the
        // end of 'single' guarantees all tasks are finished before we continue.
#pragma omp single
        {
            for (long lo = 0; lo < G.nloc; lo += g_chunk) {
                const long hi = std::min(lo + (long)g_chunk, G.nloc);
#pragma omp task firstprivate(lo, hi)
                {
                    const int t = omp_get_thread_num();
                    double* tgt = target<M>(t);
                    for (long i = lo; i < hi; ++i) process_point<M>(i, tgt, t, g_ts[t]);
                }
            }
        }
    } else if constexpr (M == M_RED) {
        // OpenMP ARRAY REDUCTION: runtime gives every thread a private zero-initialised
        // copy of g_acc[0..AL) and adds all copies into g_acc at the end of the loop.
#pragma omp for schedule(runtime) reduction(+ : g_acc[0 : g_AL])
        for (long i = 0; i < G.nloc; ++i) process_point<M>(i, g_acc, tid, g_ts[tid]);
    } else {
        double* tgt = target<M>(tid);
#pragma omp for schedule(runtime)
        for (long i = 0; i < G.nloc; ++i) process_point<M>(i, tgt, tid, g_ts[tid]);
    }

    // merge this thread's scalar statistics into the shared totals.
    // ONE scalar "+=" per thread  ->  '#pragma omp atomic' (critical would be overkill)
#pragma omp atomic
    g_sse += g_ts[tid].sse;
#pragma omp atomic
    g_changed += g_ts[tid].changed;
#pragma omp atomic
    g_skipped += g_ts[tid].skipped;
    g_ts[tid] = TS();
#pragma omp barrier

    // merge the per-thread buffers into g_acc.  Element e is owned by ONE loop
    // iteration -> no dependency between iterations, perfectly parallel.
    if constexpr (M == M_PRIV || M == M_FALSE) {
#pragma omp for schedule(static)
        for (long e = 0; e < g_AL; ++e) {
            double s = 0;
            for (int t = 0; t < T; ++t) s += (M == M_PRIV) ? g_priv[(long)t * g_stride + e] : g_false[e * T + t];
            g_acc[e] += s;
        }
    }
}

// ----------------------------------------------------------------------------
//  [C] MPI combine.  Called by the MASTER thread only (MPI_THREAD_FUNNELED rule:
//  only the thread that called MPI_Init_thread may call MPI).  'master' has no
//  implied barrier, so the caller adds an explicit  omp barrier  afterwards.
// ----------------------------------------------------------------------------
static void comm_step() {
    g_acc[g_AL] = g_sse;
    g_acc[g_AL + 1] = (double)g_changed;
    double t0 = MPI_Wtime();
    // Allreduce (not Reduce): EVERY process needs the new centroids for next iteration
    MPI_Allreduce(MPI_IN_PLACE, g_acc, (int)g_AL + 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    g_comm_time += MPI_Wtime() - t0;
    g_skipped_total += g_skipped;
}

// ----------------------------------------------------------------------------
//  [D] UPDATE: new centroid = (sum of w*x) / (sum of w).  Empty cluster keeps old centroid.
// ----------------------------------------------------------------------------
static void update_cluster(int k) {
    const int D = G.D; const long KD = (long)G.K * D;
    const double cnt = g_acc[KD + k];
    for (int d = 0; d < D; ++d) g_Cn[(long)k * D + d] = cnt > 0 ? g_acc[(long)k * D + d] / cnt : G.C[(long)k * D + d];
    G.delta[k] = std::sqrt(dist2(G.C + (long)k * D, g_Cn + (long)k * D, D));
}
static void finish_update(int it) {
    const int K = G.K, D = G.D;
    double md = 0;
    for (int k = 0; k < K; ++k) md = std::max(md, G.delta[k]);
    G.maxdelta = md;
    std::memcpy(G.C, g_Cn, sizeof(double) * K * D);
    for (int k = 0; k < K; ++k) g_lastcnt[k] = g_acc[(long)K * D + k];
    if (G.ham)
        for (int j = 0; j < K; ++j) {
            double m = std::numeric_limits<double>::infinity();
            for (int j2 = 0; j2 < K; ++j2) if (j2 != j) m = std::min(m, std::sqrt(dist2(G.C + (long)j * D, G.C + (long)j2 * D, D)));
            G.sh[j] = 0.5 * m;
        }
    g_sse_hist[it] = g_acc[g_AL];
    g_chg_hist[it] = g_acc[g_AL + 1];
}
static void log_iter(int it) {
    if (g_rank == 0 && (it < 3 || it == g_iters - 1))
        std::printf("iter %3d  changed=%.0f  %s\n", it, g_chg_hist[it], G.ham ? "(hamerly: SSE only at the end)" : "");
}
static void begin_iter(int it) {
    G.iter = it; g_sse = 0; g_changed = 0; g_skipped = 0;
    std::memset(g_acc, 0, sizeof(double) * (g_AL + 2));
}

// ----------------------------------------------------------------------------
//  THE ITERATION LOOP - variant 1: ONE persistent parallel region
//  (threads are created once; inside we use  single / for / master / barrier)
//  This is the "two for directives inside ONE parallel" row of Table 5.3, which
//  was faster than "two parallel for directives" because fork/join is paid once.
// ----------------------------------------------------------------------------
template <int M> static void iterate_persistent(int iters, bool task) {
#pragma omp parallel num_threads(g_T)
    {
        for (int it = 0; it < iters; ++it) {        // every thread runs the loop; they stay in lock-step
#pragma omp single                                  // [A] any thread, others wait (barrier wanted!)
            begin_iter(it);

            assign_phase<M>(task);                  // [B] (ends with a barrier)

#pragma omp master                                  // [C] only thread 0 may call MPI
            comm_step();
#pragma omp barrier                                 //     master has no barrier -> we add one

#pragma omp single                                  // [D] tiny serial work, result needed by all -> single + barrier
            {
                for (int k = 0; k < G.K; ++k) update_cluster(k);
                finish_update(it);
            }
#pragma omp master                                  // logging: nobody waits for it -> master (no barrier)
            log_iter(it);
        }
    }
}

// variant 2: classical fork-join - a NEW parallel region for every phase, every iteration
template <int M> static void iterate_forkjoin(int iters, bool task) {
    for (int it = 0; it < iters; ++it) {
        G.iter = it; g_sse = 0; g_changed = 0; g_skipped = 0;
#pragma omp parallel for schedule(static) num_threads(g_T)          // fork/join #1
        for (long e = 0; e < g_AL + 2; ++e) g_acc[e] = 0.0;
#pragma omp parallel num_threads(g_T)                               // fork/join #2
        { assign_phase<M>(task); }
        comm_step();                                                // plain serial code, main thread
#pragma omp parallel for schedule(static) num_threads(g_T)          // fork/join #3
        for (int k = 0; k < G.K; ++k) update_cluster(k);
        finish_update(it);
        log_iter(it);
    }
}

template <int M> static void run_iterations(int iters, bool persistent, bool task) {
    if (persistent) iterate_persistent<M>(iters, task); else iterate_forkjoin<M>(iters, task);
}

// ============================================================================
int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);   // only master thread calls MPI
    if (provided < MPI_THREAD_FUNNELED) { std::fprintf(stderr, "MPI has no FUNNELED support\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    g_rank = rank; g_P = P;

    Args A(argc, argv);
    const long N = A.num("--n", 4000000);
    const int D = (int)A.num("--d", 16), K = (int)A.num("--k", 16), iters = (int)A.num("--iters", 50);
    const bool weighted = A.has("--weighted");
    const std::string algo = A.str("--algo", "lloyd"), init = A.str("--init", "kmeanspp");
    const std::string accum = A.str("--accum", "private"), sched = A.str("--sched", "static");
    const std::string region = A.str("--region", "persistent"), mode = A.str("--mode", "run");
    const bool ham = (algo == "hamerly"), task = (sched == "task"), persistent = (region == "persistent");
    g_T = (int)A.num("--threads", omp_get_max_threads());
    g_chunk = (int)A.num("--chunk", 2048);
    g_iters = iters;

    DataSpec S;
    S.D = D; S.Ktrue = (int)A.num("--ktrue", K); S.seed = (uint64_t)A.num("--seed", 12345); S.sigma = A.dbl("--sigma", 4.0);
    S.init();

    // ====================================================== MODE: choose K (tasks!)
    if (mode == "choosek") {
        const long M = A.num("--sample", 2000);
        const int kmin = (int)A.num("--kmin", 2), kmax = (int)A.num("--kmax", 12);
        const int nk = kmax - kmin + 1;
        std::vector<double> Xs, Ws;
        make_sample(S, (uint64_t)N, M, weighted, Xs, Ws);            // every rank builds the SAME sample (no communication)
        std::vector<double> res((size_t)nk * 2, 0.0);
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        // Each candidate K is an independent job of different length (cost grows with K) ->
        //   MPI level   : job ki goes to rank (ki % P)        (cyclic = simple load balance)
        //   OpenMP level: the jobs of a rank are TASKS, created by one thread ('single'),
        //                 executed by whichever thread is free  (dynamic balancing)
        //   Each task writes ONLY res[2*ki], res[2*ki+1]  -> no race.
#pragma omp parallel num_threads(g_T)
#pragma omp single
        {
            for (int ki = 0; ki < nk; ++ki) {
                if (ki % P != rank) continue;
#pragma omp task firstprivate(ki)
                {
                    KEval e = eval_k(Xs.data(), weighted ? Ws.data() : nullptr, M, D, kmin + ki, S.seed);
                    res[2 * ki] = e.sse; res[2 * ki + 1] = e.sil;
                }
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, res.data(), nk * 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);   // others contributed 0
        double t1 = MPI_Wtime(), loc = t1 - t0, tmax;
        MPI_Reduce(&loc, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::vector<int> Ks; std::vector<double> sse; int bs = 0;
            for (int i = 0; i < nk; ++i) {
                Ks.push_back(kmin + i); sse.push_back(res[2 * i]);
                if (res[2 * i + 1] > res[2 * bs + 1]) bs = i;
                std::printf("K=%2d  SSE=%14.2f  silhouette=%.4f\n", kmin + i, res[2 * i], res[2 * i + 1]);
            }
            std::printf("RESULT mode=choosek prog=hybrid sample=%ld procs=%d threads=%d time=%.6f elbowK=%d silhouetteK=%d\n",
                        M, P, g_T, tmax, pick_elbow(Ks, sse), kmin + bs);
        }
        MPI_Finalize();
        return 0;
    }

    // ====================================================== MODE: run
    // ---- MPI data distribution: block of N/P points (+1 for the first N%P ranks)
    const long base = N / P, rem = N % P;
    const long nloc = base + (rank < rem ? 1 : 0);
    const long off = rank * base + std::min<long>(rank, rem);

    G.K = K; G.D = D; G.nloc = nloc; G.ham = ham; G.iter = 0;
    double* X = (double*)amalloc(sizeof(double) * (size_t)nloc * D);
    double* Wt = weighted ? (double*)amalloc(sizeof(double) * (size_t)nloc) : nullptr;
    int* lab = (int*)amalloc(sizeof(int) * (size_t)nloc);
    G.X = X; G.W = Wt; G.lab = lab;
    G.C = (double*)amalloc(sizeof(double) * K * D);
    g_Cn = (double*)amalloc(sizeof(double) * K * D);
    G.delta = (double*)amalloc(sizeof(double) * K);  G.sh = (double*)amalloc(sizeof(double) * K);
    std::fill(G.delta, G.delta + K, 0.0); std::fill(G.sh, G.sh + K, 0.0); G.maxdelta = 0;
    if (ham) { G.up = (double*)amalloc(sizeof(double) * (size_t)nloc); G.lo = (double*)amalloc(sizeof(double) * (size_t)nloc); }
    g_AL = (long)K * D + K;
    g_stride = (g_AL + 7) / 8 * 8;                                    // pad each thread's buffer to a multiple of 64 bytes
    g_acc = (double*)amalloc(sizeof(double) * (g_AL + 2));
    g_priv = (double*)amalloc(sizeof(double) * g_stride * g_T);
    g_false = (double*)amalloc(sizeof(double) * g_AL * g_T);
    g_ts = new TS[g_T];                                               // C++17 honours alignas(64)
    g_sse_hist.assign(iters, 0.0); g_chg_hist.assign(iters, 0.0); g_lastcnt.assign(K, 0.0);

    // ---- parallel data generation.  schedule(static) = the SAME thread->points mapping as the
    //      compute loop, so with "first touch" each thread's memory pages are allocated on
    //      its own NUMA node.  Every point depends only on its global index (no RNG state).
    double tg0 = MPI_Wtime();
#pragma omp parallel for schedule(static) num_threads(g_T)
    for (long i = 0; i < nloc; ++i) {
        gen_point(S, (uint64_t)(off + i), X + (size_t)i * D);
        if (weighted) Wt[i] = gen_weight(S, (uint64_t)(off + i));
        lab[i] = -1;
        if (ham) { G.up[i] = 0; G.lo[i] = 0; }
    }
    double tgen = MPI_Wtime() - tg0;

    // ---- identical start for everybody: rank 0 chooses, MPI_Bcast distributes
    double ti0 = MPI_Wtime();
    if (rank == 0) make_init(S, (uint64_t)N, K, weighted, init, G.C);
    MPI_Bcast(G.C, K * D, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double tinit = MPI_Wtime() - ti0;

    // loop schedule for  schedule(runtime)
    if (sched == "dynamic") omp_set_schedule(omp_sched_dynamic, g_chunk);
    else if (sched == "guided") omp_set_schedule(omp_sched_guided, g_chunk);
    else omp_set_schedule(omp_sched_static, 0);

    // strategy dispatch.  DEFAULT = private (padded per-thread buffers): measured ~15% faster
    // than the OpenMP reduction clause on 1 thread and no slower on many threads.
    // Tasks + reduction clause do not combine -> fall back to private buffers.
    int Mode = M_PRIV;
    if (accum == "reduction") Mode = M_RED; else if (accum == "atomic") Mode = M_ATOMIC;
    else if (accum == "critical") Mode = M_CRIT; else if (accum == "falseshare") Mode = M_FALSE;
    if (task && Mode == M_RED) Mode = M_PRIV;

    // ================================================================ TIMED REGION
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    switch (Mode) {
        case M_RED:    run_iterations<M_RED>(iters, persistent, task); break;
        case M_PRIV:   run_iterations<M_PRIV>(iters, persistent, task); break;
        case M_ATOMIC: run_iterations<M_ATOMIC>(iters, persistent, task); break;
        case M_CRIT:   run_iterations<M_CRIT>(iters, persistent, task); break;
        default:       run_iterations<M_FALSE>(iters, persistent, task); break;
    }
    double t1 = MPI_Wtime();
    double tloc = t1 - t0, tmax, tcomm = g_comm_time, tcomm_max;
    MPI_Reduce(&tloc, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);     // slowest process = parallel time
    MPI_Reduce(&tcomm, &tcomm_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    // ================================================================

    // ---- exact final SSE and the most "unusual" point (outlier / anomaly), not timed
    double sse = 0, gworst = -1; long gworstI = 0;
#pragma omp parallel num_threads(g_T)
    {
        double worst = -1; long wi = 0;
#pragma omp for schedule(static) reduction(+ : sse)
        for (long i = 0; i < nloc; ++i) {
            int b; double bd, sd;
            nearest2(X + (size_t)i * D, G.C, K, D, b, bd, sd);
            sse += (Wt ? Wt[i] : 1.0) * bd;
            if (bd > worst) { worst = bd; wi = off + i; }
        }
        // two variables (value + index) must change TOGETHER -> atomic cannot do that -> critical.
        // (executed once per thread, so its cost is irrelevant)
#pragma omp critical(outlier)
        { if (worst > gworst) { gworst = worst; gworstI = wi; } }
    }
    MPI_Allreduce(MPI_IN_PLACE, &sse, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    struct { double v; int r; } in = {gworst, rank}, out;
    MPI_Allreduce(&in, &out, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    MPI_Bcast(&gworstI, 1, MPI_LONG, out.r, MPI_COMM_WORLD);
    long skipped_all = 0;
    MPI_Reduce(&g_skipped_total, &skipped_all, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::printf("cluster sizes (weight) after last iteration:");
        for (int k = 0; k < K; ++k) std::printf(" %.0f", g_lastcnt[k]);
        std::printf("\nmost unusual point (outlier): global index %ld, distance %.3f\n", gworstI, std::sqrt(out.v));
        std::printf("setup: generate=%.3fs init=%.3fs | MPI_Allreduce total=%.4fs (%.1f%% of loop)\n", tgen, tinit, tcomm_max, 100.0 * tcomm_max / tmax);
        std::printf("RESULT mode=run prog=hybrid algo=%s init=%s weighted=%d N=%ld D=%d K=%d iters=%d procs=%d threads=%d accum=%s sched=%s region=%s time=%.6f comm=%.6f sse=%.6f outlier=%ld skipped=%ld\n",
                    algo.c_str(), init.c_str(), (int)weighted, N, D, K, iters, P, g_T, accum.c_str(), sched.c_str(), region.c_str(), tmax, tcomm_max, sse, gworstI, skipped_all);
        std::string outf = A.str("--out", "");
        if (!outf.empty()) { FILE* f = std::fopen(outf.c_str(), "w"); for (int k = 0; k < K; ++k) { for (int d = 0; d < D; ++d) std::fprintf(f, "%.10g ", G.C[(size_t)k * D + d]); std::fprintf(f, "\n"); } std::fclose(f); }
    }
    MPI_Finalize();
    return 0;
}
