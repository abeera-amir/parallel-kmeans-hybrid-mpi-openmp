// ============================================================================
//  common.h  -  code shared by the SERIAL and the HYBRID (MPI+OpenMP) programs
//
//  Everything here is plain C++ (no MPI calls, no OpenMP *constructs* except one
//  harmless "omp simd" hint) so the serial baseline and the parallel version are
//  guaranteed to use the SAME data, the SAME distance function and the SAME
//  initial centroids.  That is what makes the correctness check meaningful.
// ============================================================================
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

// ----------------------------------------------------------------------------
//  1. COUNTER-BASED RANDOM NUMBERS  (this removes a loop-carried dependency!)
//
//  A normal rand() keeps an internal state: number i can only be produced after
//  number i-1.  That is a LOOP-CARRIED DEPENDENCY and it would force the data
//  generation loop to be sequential, and it would give different data for a
//  different number of processes.
//  Instead we hash (seed, global point index, coordinate) -> random number.
//  Point i always gets exactly the same value, no matter which process/thread
//  generates it, and all points can be generated independently in parallel.
// ----------------------------------------------------------------------------
static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
static inline uint64_t mix(uint64_t a, uint64_t b) { return splitmix64(a ^ splitmix64(b + 0x1234567ULL)); }
// uniform number in (0,1)
static inline double u01(uint64_t key) {
    return ((splitmix64(key) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

// ----------------------------------------------------------------------------
//  2. SYNTHETIC "REAL-LIFE LIKE" DATA SET
//
//  Think of every point as one CUSTOMER described by D numeric features
//  (recency, frequency, spend, basket size, ...).  Customers form Ktrue natural
//  groups (budget shoppers, loyal big spenders, ...).  Each point = centre of its
//  group + Gaussian noise.  Weight of a point = e.g. the customer's yearly revenue
//  (used by WEIGHTED k-means, so big customers pull the centroid more).
// ----------------------------------------------------------------------------
struct DataSpec {
    int D = 16;
    int Ktrue = 16;
    uint64_t seed = 12345;
    double sigma = 4.0;            // noise (std-dev) around each true centre
    std::vector<double> centers;   // Ktrue x D
    void init() {
        centers.resize((size_t)Ktrue * D);
        for (int c = 0; c < Ktrue; ++c)
            for (int d = 0; d < D; ++d)
                centers[(size_t)c * D + d] = 100.0 * u01(mix(mix(seed, 99991), (uint64_t)c * 4096 + d));
    }
};

// global index i  ->  coordinates of point i (writes D doubles to out)
static inline void gen_point(const DataSpec& S, uint64_t i, double* out) {
    const int c = (int)(splitmix64(mix(S.seed, i)) % (uint64_t)S.Ktrue);
    const double* ctr = &S.centers[(size_t)c * S.D];
    const uint64_t base = mix(S.seed + 7, i);
    for (int d = 0; d < S.D; ++d) {
        double u1 = u01(mix(base, 2 * (uint64_t)d));
        double u2 = u01(mix(base, 2 * (uint64_t)d + 1));
        double g = std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);  // Box-Muller
        out[d] = ctr[d] + S.sigma * g;
    }
}
// weight of point i  (1 .. 6)  e.g. "yearly revenue of this customer"
static inline double gen_weight(const DataSpec& S, uint64_t i) {
    return 1.0 + 5.0 * u01(mix(S.seed + 31337, i));
}

// ----------------------------------------------------------------------------
//  3. DISTANCE  d(p,c)^2 = sum_k (p_k - c_k)^2
//  "omp simd reduction" tells the compiler: the accumulation into s is a
//  reduction, you may vectorise it (SIMD).  Ignored by the serial build.
// ----------------------------------------------------------------------------
static inline double dist2(const double* __restrict a, const double* __restrict b, int D) {
    double s = 0.0;
#pragma omp simd reduction(+ : s)
    for (int d = 0; d < D; ++d) {
        double t = a[d] - b[d];
        s += t * t;
    }
    return s;
}

// nearest + second nearest centroid (second is needed by Hamerly's algorithm)
// NOTE the loop over j has a loop-carried dependency (bd/sd/b depend on the
// previous j).  K is tiny, so we keep it sequential and parallelise over points.
static inline void nearest2(const double* x, const double* C, int K, int D, int& b, double& bd, double& sd) {
    bd = std::numeric_limits<double>::infinity();
    sd = bd;
    b = 0;
    for (int j = 0; j < K; ++j) {
        double d = dist2(x, C + (size_t)j * D, D);
        if (d < bd) { sd = bd; bd = d; b = j; }
        else if (d < sd) sd = d;
    }
}

// ----------------------------------------------------------------------------
//  4. INITIALISATION
//     random   : K distinct points picked by hashing  (classical)
//     kmeanspp : k-means++  (Arthur & Vassilvitskii 2007) - spreads the starting
//                centroids, gives lower SSE and fewer iterations (OPTIMISED).
//  Both work on points generated from their global index, so serial and hybrid
//  programs obtain IDENTICAL starting centroids.
// ----------------------------------------------------------------------------
static inline void init_random(const DataSpec& S, uint64_t N, int K, double* C) {
    std::vector<uint64_t> used;
    for (int j = 0; j < K; ++j) {
        uint64_t idx, tries = 0;
        do { idx = splitmix64(mix(S.seed + 555, (uint64_t)j * 1000 + tries++)) % N; }
        while (std::find(used.begin(), used.end(), idx) != used.end());
        used.push_back(idx);
        gen_point(S, idx, C + (size_t)j * S.D);
    }
}

// k-means++ on a sample Xs (M x D) with optional weights Ws.  The pick of centre j
// DEPENDS on centres 0..j-1  -> real loop-carried dependency, stays sequential.
static inline void kmeanspp(const double* Xs, const double* Ws, long M, int D, int K, uint64_t seed, double* C) {
    std::vector<double> minD((size_t)M, std::numeric_limits<double>::infinity());
    long first = (long)(u01(mix(seed, 4242)) * M);
    if (first >= M) first = M - 1;
    std::memcpy(C, Xs + (size_t)first * D, sizeof(double) * D);
    for (int j = 1; j < K; ++j) {
        const double* last = C + (size_t)(j - 1) * D;
        double total = 0.0;
        for (long i = 0; i < M; ++i) {                       // independent over i
            double d = dist2(Xs + (size_t)i * D, last, D);
            if (d < minD[i]) minD[i] = d;
            total += (Ws ? Ws[i] : 1.0) * minD[i];
        }
        double r = u01(mix(seed, 9000 + j)) * total, acc = 0.0;
        long pick = M - 1;
        for (long i = 0; i < M; ++i) {                       // prefix-sum scan (dependency)
            acc += (Ws ? Ws[i] : 1.0) * minD[i];
            if (acc >= r) { pick = i; break; }
        }
        std::memcpy(C + (size_t)j * D, Xs + (size_t)pick * D, sizeof(double) * D);
    }
}

// build a sample of M points spread evenly over the global index range
static inline void make_sample(const DataSpec& S, uint64_t N, long M, bool weighted,
                               std::vector<double>& Xs, std::vector<double>& Ws) {
    Xs.resize((size_t)M * S.D);
    if (weighted) Ws.resize((size_t)M); else Ws.clear();
    for (long j = 0; j < M; ++j) {
        uint64_t gi = (uint64_t)((double)j * (double)N / (double)M);
        gen_point(S, gi, &Xs[(size_t)j * S.D]);
        if (weighted) Ws[j] = gen_weight(S, gi);
    }
}

// choose initial centroids (called on rank 0 only in the hybrid program)
static inline void make_init(const DataSpec& S, uint64_t N, int K, bool weighted, const std::string& how, double* C) {
    if (how == "random") { init_random(S, N, K, C); return; }
    long M = (long)std::min<uint64_t>(N, 20000);
    std::vector<double> Xs, Ws;
    make_sample(S, N, M, weighted, Xs, Ws);
    kmeanspp(Xs.data(), weighted ? Ws.data() : nullptr, M, S.D, K, S.seed, C);
}

// ----------------------------------------------------------------------------
//  5. CHOICE OF K  -  small serial k-means + SSE + silhouette on a SAMPLE
//  (Called as ONE independent job per candidate K: the hybrid program runs each
//   job as an OpenMP *task* and spreads the jobs over MPI ranks.)
// ----------------------------------------------------------------------------
struct KEval { double sse = 0, sil = 0; };

static inline KEval eval_k(const double* X, const double* W, long M, int D, int K, uint64_t seed) {
    std::vector<double> C((size_t)K * D), sums((size_t)K * D), cnt(K);
    std::vector<int> lab(M, -1);
    kmeanspp(X, W, M, D, K, seed + K, C.data());
    for (int it = 0; it < 40; ++it) {
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(cnt.begin(), cnt.end(), 0.0);
        long changed = 0;
        for (long i = 0; i < M; ++i) {
            int b; double bd, sd;
            nearest2(X + (size_t)i * D, C.data(), K, D, b, bd, sd);
            if (b != lab[i]) { lab[i] = b; ++changed; }
            double w = W ? W[i] : 1.0;
            for (int d = 0; d < D; ++d) sums[(size_t)b * D + d] += w * X[(size_t)i * D + d];
            cnt[b] += w;
        }
        for (int k = 0; k < K; ++k)
            if (cnt[k] > 0) for (int d = 0; d < D; ++d) C[(size_t)k * D + d] = sums[(size_t)k * D + d] / cnt[k];
        if (changed == 0) break;
    }
    KEval r;
    for (long i = 0; i < M; ++i) r.sse += (W ? W[i] : 1.0) * dist2(X + (size_t)i * D, &C[(size_t)lab[i] * D], D);
    // silhouette  s(i) = (b-a)/max(a,b); a = mean dist to own cluster, b = mean dist to nearest other cluster
    std::vector<double> sumd(K);
    std::vector<long> sz(K, 0);
    for (long i = 0; i < M; ++i) ++sz[lab[i]];
    double tot = 0;
    for (long i = 0; i < M; ++i) {
        std::fill(sumd.begin(), sumd.end(), 0.0);
        for (long j = 0; j < M; ++j)
            if (j != i) sumd[lab[j]] += std::sqrt(dist2(X + (size_t)i * D, X + (size_t)j * D, D));
        int li = lab[i];
        if (sz[li] <= 1) continue;                     // s(i)=0 for singleton clusters
        double a = sumd[li] / (double)(sz[li] - 1), b = std::numeric_limits<double>::infinity();
        for (int k = 0; k < K; ++k)
            if (k != li && sz[k] > 0) b = std::min(b, sumd[k] / (double)sz[k]);
        if (std::isfinite(b)) tot += (b - a) / std::max(a, b);
    }
    r.sil = tot / (double)M;
    return r;
}

// "elbow" = the K whose (normalised) SSE point is farthest from the straight line
// joining the first and last point of the SSE curve  (Kneedle-style rule).
static inline int pick_elbow(const std::vector<int>& Ks, const std::vector<double>& sse) {
    int n = (int)Ks.size();
    if (n < 3) return Ks[0];
    double x0 = Ks[0], x1 = Ks[n - 1], y0 = sse[0], y1 = sse[n - 1];
    int best = 0; double bestd = -1;
    for (int i = 0; i < n; ++i) {
        double x = (Ks[i] - x0) / (x1 - x0), y = (sse[i] - y1) / (y0 - y1);   // both in [0,1]
        double dd = std::fabs(y - (1.0 - x)) / std::sqrt(2.0);                // distance to line y=1-x
        if (dd > bestd) { bestd = dd; best = i; }
    }
    return Ks[best];
}

// ----------------------------------------------------------------------------
//  6. tiny command-line helper  ("--name value")
// ----------------------------------------------------------------------------
struct Args {
    int argc; char** argv;
    Args(int c, char** v) : argc(c), argv(v) {}
    bool has(const char* k) const { for (int i = 1; i < argc; ++i) if (!std::strcmp(argv[i], k)) return true; return false; }
    std::string str(const char* k, const std::string& def) const {
        for (int i = 1; i + 1 < argc; ++i) if (!std::strcmp(argv[i], k)) return argv[i + 1];
        return def;
    }
    long num(const char* k, long def) const { return std::stol(str(k, std::to_string(def))); }
    double dbl(const char* k, double def) const { return std::stod(str(k, std::to_string(def))); }
};
