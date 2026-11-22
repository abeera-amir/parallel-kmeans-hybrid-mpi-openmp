// ============================================================================
//  kmeans_serial.cpp  -  SEQUENTIAL BASELINE  (no MPI, no OpenMP)
//
//  build : g++ -O3 -std=c++17 -o kmeans_serial kmeans_serial.cpp
//  run   : ./kmeans_serial --n 4000000 --d 16 --k 16 --iters 50
//          ./kmeans_serial --algo hamerly --init kmeanspp --weighted
//          ./kmeans_serial --mode choosek --ktrue 6 --kmax 12
//
//  algo  lloyd   = CLASSICAL k-means (every iteration: N*K distances)
//        hamerly = OPTIMISED exact k-means: keeps for every point an upper bound u on
//                  the distance to its own centroid and a lower bound l to the 2nd
//                  nearest; most points can be skipped without any distance calc.
//  --weighted     = WEIGHTED k-means (point i counts w_i times)
// ============================================================================
#include <chrono>
#include "common.h"

int main(int argc, char** argv) {
    Args A(argc, argv);
    const long N = A.num("--n", 4000000);
    const int D = (int)A.num("--d", 16), K = (int)A.num("--k", 16), iters = (int)A.num("--iters", 50);
    const bool weighted = A.has("--weighted");
    const std::string algo = A.str("--algo", "lloyd"), init = A.str("--init", "kmeanspp"), mode = A.str("--mode", "run");
    const bool ham = (algo == "hamerly");

    DataSpec S;
    S.D = D; S.Ktrue = (int)A.num("--ktrue", K); S.seed = (uint64_t)A.num("--seed", 12345); S.sigma = A.dbl("--sigma", 4.0);
    S.init();
    auto now = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };

    // ------------------------------------------------------------------ choose K
    if (mode == "choosek") {
        const long M = A.num("--sample", 2000);
        const int kmin = (int)A.num("--kmin", 2), kmax = (int)A.num("--kmax", 12);
        std::vector<double> Xs, Ws;
        make_sample(S, (uint64_t)N, M, weighted, Xs, Ws);
        std::vector<int> Ks; std::vector<double> sse, sil;
        double t0 = now();
        for (int k = kmin; k <= kmax; ++k) {
            KEval e = eval_k(Xs.data(), weighted ? Ws.data() : nullptr, M, D, k, S.seed);
            Ks.push_back(k); sse.push_back(e.sse); sil.push_back(e.sil);
            std::printf("K=%2d  SSE=%14.2f  silhouette=%.4f\n", k, e.sse, e.sil);
        }
        double t1 = now();
        int bs = 0; for (size_t i = 1; i < Ks.size(); ++i) if (sil[i] > sil[bs]) bs = (int)i;
        std::printf("RESULT mode=choosek prog=serial sample=%ld time=%.6f elbowK=%d silhouetteK=%d\n", M, t1 - t0, pick_elbow(Ks, sse), Ks[bs]);
        return 0;
    }

    // ------------------------------------------------------------------ data
    std::vector<double> X((size_t)N * D), W;
    if (weighted) W.resize(N);
    for (long i = 0; i < N; ++i) {
        gen_point(S, (uint64_t)i, &X[(size_t)i * D]);
        if (weighted) W[i] = gen_weight(S, (uint64_t)i);
    }
    std::vector<double> C((size_t)K * D), Cn((size_t)K * D);
    make_init(S, (uint64_t)N, K, weighted, init, C.data());

    std::vector<int> lab(N, -1);
    std::vector<double> up, lo, sh(K, 0.0), delta(K, 0.0), acc((size_t)K * D + K);
    if (ham) { up.assign(N, 0.0); lo.assign(N, 0.0); }
    double maxdelta = 0;
    long skipped_total = 0;

    // ------------------------------------------------------------------ main loop (timed)
    double t0 = now();
    for (int it = 0; it < iters; ++it) {
        std::fill(acc.begin(), acc.end(), 0.0);
        long changed = 0, skipped = 0;
        double sse = 0;
        for (long i = 0; i < N; ++i) {
            const double* x = &X[(size_t)i * D];
            const double w = weighted ? W[i] : 1.0;
            int a = lab[i], best; double bd = 0, sd = 0;
            if (!ham || it == 0) {
                nearest2(x, C.data(), K, D, best, bd, sd);
                if (ham) { up[i] = std::sqrt(bd); lo[i] = std::sqrt(sd); }
            } else {
                double u = up[i] + delta[a], l = lo[i] - maxdelta;      // bounds drift with centroid movement
                double m = std::max(sh[a], l);                         // sh[a] = half distance to closest other centroid
                best = a;
                if (u <= m) ++skipped;                                 // 1st test: skip with NO distance calc
                else {
                    u = std::sqrt(dist2(x, &C[(size_t)a * D], D));     // tighten u (1 distance)
                    if (u <= m) ++skipped;                             // 2nd test
                    else { nearest2(x, C.data(), K, D, best, bd, sd); u = std::sqrt(bd); l = std::sqrt(sd); }
                }
                up[i] = u; lo[i] = l;
            }
            if (best != a) { lab[i] = best; ++changed; }
            if (!ham) sse += w * bd;
            for (int d = 0; d < D; ++d) acc[(size_t)best * D + d] += w * x[d];
            acc[(size_t)K * D + best] += w;
        }
        // update step
        maxdelta = 0;
        for (int k = 0; k < K; ++k) {
            double cnt = acc[(size_t)K * D + k];
            for (int d = 0; d < D; ++d) Cn[(size_t)k * D + d] = cnt > 0 ? acc[(size_t)k * D + d] / cnt : C[(size_t)k * D + d];  // empty cluster: keep old
            delta[k] = std::sqrt(dist2(&C[(size_t)k * D], &Cn[(size_t)k * D], D));
            maxdelta = std::max(maxdelta, delta[k]);
        }
        C = Cn;
        if (ham)
            for (int j = 0; j < K; ++j) {
                double m = std::numeric_limits<double>::infinity();
                for (int j2 = 0; j2 < K; ++j2) if (j2 != j) m = std::min(m, std::sqrt(dist2(&C[(size_t)j * D], &C[(size_t)j2 * D], D)));
                sh[j] = 0.5 * m;
            }
        skipped_total += skipped;
        if (it < 3 || it == iters - 1) std::printf("iter %3d  changed=%ld  skipped=%ld%s\n", it, changed, skipped, ham ? "" : "");
    }
    double t1 = now();

    // ------------------------------------------------------------------ final exact SSE (not timed)
    double sse = 0, worst = -1; long worstI = 0;
    for (long i = 0; i < N; ++i) {
        int b; double bd, sd;
        nearest2(&X[(size_t)i * D], C.data(), K, D, b, bd, sd);
        sse += (weighted ? W[i] : 1.0) * bd;
        if (bd > worst) { worst = bd; worstI = i; }
    }
    std::printf("RESULT mode=run prog=serial algo=%s init=%s weighted=%d N=%ld D=%d K=%d iters=%d procs=1 threads=1 time=%.6f sse=%.6f outlier=%ld skipped=%ld\n",
                algo.c_str(), init.c_str(), (int)weighted, N, D, K, iters, t1 - t0, sse, worstI, skipped_total);
    std::string out = A.str("--out", "");
    if (!out.empty()) { FILE* f = std::fopen(out.c_str(), "w"); for (int k = 0; k < K; ++k) { for (int d = 0; d < D; ++d) std::fprintf(f, "%.10g ", C[(size_t)k * D + d]); std::fprintf(f, "\n"); } std::fclose(f); }
    return 0;
}
