// =============================================================================
//  kmeans_serial.cpp  -  SERIAL baseline (no MPI, no OpenMP)
//
//  Three algorithms (choose with --algo):
//    classic   : random starting centroids  + Lloyd's k-means
//    kmeanspp  : k-means++ starting centroids + Lloyd's k-means
//    hamerly   : k-means++ starting centroids + Hamerly's k-means
//                (same answer as Lloyd, but skips most distance calculations)
//
//  Run:   ./kmeans_serial --algo hamerly --n 4000000 --k 16 --iters 50
//  Output line (read by the benchmark script):
//    RESULT version=serial algo=... procs=1 threads=1 time=... sse=...
// =============================================================================
#include "common.h"
#include <ctime>

static double now_seconds() {
    timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

// -----------------------------------------------------------------------------
// LLOYD (classic k-means), fixed number of iterations
// -----------------------------------------------------------------------------
static void lloyd(const double* points, long num_points, int num_clusters, int num_iters,
                  double* centroids, int* labels) {
    double* sums   = new double[num_clusters * NUM_DIMS];
    double* counts = new double[num_clusters];

    // LOOP-CARRIED DEPENDENCY: iteration "iter" needs the centroids from
    // iteration "iter-1".  This outer loop can NEVER run in parallel.
    for (int iter = 0; iter < num_iters; iter++) {
        memset(sums, 0, sizeof(double) * num_clusters * NUM_DIMS);
        memset(counts, 0, sizeof(double) * num_clusters);

        for (long i = 0; i < num_points; i++) {
            const double* point = &points[i * NUM_DIMS];

            // ASSIGNMENT STEP: find the nearest centroid
            int best_cluster = 0;
            double best_dist = squared_distance(point, &centroids[0]);
            for (int j = 1; j < num_clusters; j++) {
                double dist = squared_distance(point, &centroids[j * NUM_DIMS]);
                if (dist < best_dist) { best_dist = dist; best_cluster = j; }
            }
            labels[i] = best_cluster;

            // add the point to its cluster's sum (used by the UPDATE STEP)
            // (in the parallel code this "+=" is the dangerous part: two threads
            //  can add into the same cluster at the same time)
            for (int d = 0; d < NUM_DIMS; d++) sums[best_cluster * NUM_DIMS + d] += point[d];
            counts[best_cluster] += 1.0;
        }
        // UPDATE STEP
        compute_new_centroids(num_clusters, sums, counts, centroids);
    }
    delete[] sums;
    delete[] counts;
}

// -----------------------------------------------------------------------------
// HAMERLY k-means, fixed number of iterations
//
// Idea: keep for every point
//   upper[i] = an upper bound of the distance to its OWN centroid
//   lower[i] = a lower bound of the distance to the SECOND nearest centroid
// If upper[i] <= lower[i]  the point surely did not change cluster, so we
// skip the K distance calculations for this point.
// half_gap[j] = half the distance from centroid j to its nearest other centroid.
// If upper[i] <= half_gap[label] the point also surely stays.
// -----------------------------------------------------------------------------
static void hamerly(const double* points, long num_points, int num_clusters, int num_iters,
                    double* centroids, int* labels) {
    double* upper     = new double[num_points];
    double* lower     = new double[num_points];
    double* sums      = new double[num_clusters * NUM_DIMS];
    double* counts    = new double[num_clusters];
    double* half_gap  = new double[num_clusters];
    double* old_centroids = new double[num_clusters * NUM_DIMS];
    double* movement  = new double[num_clusters];

    // start: upper = infinity, lower = 0, so every point is checked fully
    for (long i = 0; i < num_points; i++) { upper[i] = INFINITY; lower[i] = 0.0; labels[i] = 0; }

    for (int iter = 0; iter < num_iters; iter++) {   // sequential (loop-carried dependency)

        // (1) half_gap of each centroid  (only K*K distances - small)
        for (int j = 0; j < num_clusters; j++) {
            double closest = INFINITY;
            for (int other = 0; other < num_clusters; other++) {
                if (other == j) continue;
                double dist = squared_distance(&centroids[j * NUM_DIMS], &centroids[other * NUM_DIMS]);
                if (dist < closest) closest = dist;
            }
            half_gap[j] = 0.5 * sqrt(closest);
        }

        // (2) assignment with bounds
        memset(sums, 0, sizeof(double) * num_clusters * NUM_DIMS);
        memset(counts, 0, sizeof(double) * num_clusters);
        for (long i = 0; i < num_points; i++) {
            const double* point = &points[i * NUM_DIMS];
            int a = labels[i];
            double bound = (half_gap[a] > lower[i]) ? half_gap[a] : lower[i];

            if (upper[i] > bound) {
                // make the upper bound exact and test again
                upper[i] = sqrt(squared_distance(point, &centroids[a * NUM_DIMS]));
                if (upper[i] > bound) {
                    // test failed: full search for nearest and second nearest
                    double best = INFINITY, second = INFINITY;
                    int best_cluster = 0;
                    for (int j = 0; j < num_clusters; j++) {
                        double dist = squared_distance(point, &centroids[j * NUM_DIMS]);
                        if (dist < best) { second = best; best = dist; best_cluster = j; }
                        else if (dist < second) { second = dist; }
                    }
                    labels[i] = best_cluster;
                    upper[i] = sqrt(best);
                    lower[i] = sqrt(second);
                }
            }
            int c = labels[i];
            for (int d = 0; d < NUM_DIMS; d++) sums[c * NUM_DIMS + d] += point[d];
            counts[c] += 1.0;
        }

        // (3) update centroids and measure how far each one moved
        memcpy(old_centroids, centroids, sizeof(double) * num_clusters * NUM_DIMS);
        compute_new_centroids(num_clusters, sums, counts, centroids);
        double max_move = 0.0;
        for (int j = 0; j < num_clusters; j++) {
            movement[j] = sqrt(squared_distance(&old_centroids[j * NUM_DIMS], &centroids[j * NUM_DIMS]));
            if (movement[j] > max_move) max_move = movement[j];
        }

        // (4) move the bounds: own centroid moved by movement[a] -> upper grows;
        //     any other centroid moved at most max_move -> lower shrinks.
        for (long i = 0; i < num_points; i++) {
            upper[i] += movement[labels[i]];
            lower[i] -= max_move;
        }
    }
    delete[] upper; delete[] lower; delete[] sums; delete[] counts;
    delete[] half_gap; delete[] old_centroids; delete[] movement;
}

// SSE = sum of squared distances of every point to its centroid (quality)
static double compute_sse(const double* points, long num_points, const double* centroids,
                          const int* labels) {
    double sse = 0.0;
    for (long i = 0; i < num_points; i++)
        sse += squared_distance(&points[i * NUM_DIMS], &centroids[labels[i] * NUM_DIMS]);
    return sse;
}

int main(int argc, char** argv) {
    const char* algo  = get_arg(argc, argv, "--algo", "kmeanspp");
    long num_points   = atol(get_arg(argc, argv, "--n", "4000000"));
    int num_clusters  = atoi(get_arg(argc, argv, "--k", "16"));
    int num_iters     = atoi(get_arg(argc, argv, "--iters", "50"));

    double true_centers[TRUE_GROUPS * NUM_DIMS];
    make_true_centers(true_centers);

    // generate all points (not timed)
    double* points = new double[(size_t)num_points * NUM_DIMS];
    for (long i = 0; i < num_points; i++) make_point(i, true_centers, &points[i * NUM_DIMS]);
    int* labels = new int[num_points];
    double* centroids = new double[num_clusters * NUM_DIMS];

    double start = now_seconds();                       // ---- timer starts
    if (strcmp(algo, "classic") == 0)
        random_init(num_points, num_clusters, true_centers, centroids);
    else
        kmeanspp_init(num_points, num_clusters, true_centers, centroids, 0);

    if (strcmp(algo, "hamerly") == 0)
        hamerly(points, num_points, num_clusters, num_iters, centroids, labels);
    else
        lloyd(points, num_points, num_clusters, num_iters, centroids, labels);
    double elapsed = now_seconds() - start;             // ---- timer stops

    double sse = compute_sse(points, num_points, centroids, labels);
    printf("RESULT version=serial algo=%s procs=1 threads=1 n=%ld k=%d iters=%d time=%.4f sse=%.10e\n",
           algo, num_points, num_clusters, num_iters, elapsed, sse);

    delete[] points; delete[] labels; delete[] centroids;
    return 0;
}
