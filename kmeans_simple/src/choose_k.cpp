// =============================================================================
//  choose_k.cpp  -  find the best number of clusters K
//
//  Runs k-means (k-means++ start + Lloyd) for K = 2 .. 20 on a SAMPLE of the
//  same 4M-point data set and measures two things:
//
//  1. ELBOW METHOD
//     SSE (sum of squared distances to the centroids) always gets smaller when
//     K grows.  At the right K the curve bends like an elbow: after it, adding
//     clusters hardly helps.  We find the elbow automatically: the K where
//     the drop in SSE before K is much bigger than the drop after K.
//
//  2. SILHOUETTE METHOD
//     For one point:  a = mean distance to points of its OWN cluster
//                     b = mean distance to points of the NEAREST OTHER cluster
//                     s = (b - a) / max(a, b)        (between -1 and 1)
//     Average s over the points.  Higher = better separated clusters.
//     Best K = the K with the highest average silhouette.
//
//  Why a sample?  Silhouette compares every point with every other point
//  (N x N).  For 4M points that is 16 * 10^12 distances - impossible.
//  A sample of 20,000 points (spread evenly over the data) is enough.
//
//  Output: results/choose_k.csv and results/chosen_k.txt
//  Run:    ./choose_k --n 4000000
// =============================================================================
#include "common.h"

#define K_MIN 2
#define K_MAX 20
#define SILHOUETTE_POINTS 2000     // silhouette uses this many of the sample points
#define NUM_ATTEMPTS 10            // k-means++ starts tried for every K (best one kept)

int main(int argc, char** argv) {
    long total_points = atol(get_arg(argc, argv, "--n", "4000000"));

    double true_centers[TRUE_GROUPS * NUM_DIMS];
    make_true_centers(true_centers);

    // the sample: the same points k-means++ uses
    long sample_size = SAMPLE_SIZE;
    if (sample_size > total_points) sample_size = total_points;
    double* sample = new double[sample_size * NUM_DIMS];
    for (long s = 0; s < sample_size; s++)
        make_point((uint64_t)s * total_points / sample_size, true_centers, &sample[s * NUM_DIMS]);

    int* labels         = new int[sample_size];   // labels of the best attempt
    int* attempt_labels = new int[sample_size];   // labels of the current attempt
    double* centroids = new double[K_MAX * NUM_DIMS];
    double* sums      = new double[K_MAX * NUM_DIMS];
    double* counts    = new double[K_MAX];
    double sse_of_k[K_MAX + 1];
    double silhouette_of_k[K_MAX + 1];

    FILE* csv = fopen("results/choose_k.csv", "w");
    if (!csv) { printf("Please create the folder 'results' first.\n"); return 1; }
    fprintf(csv, "k,sse,silhouette\n");
    printf("  K          SSE   silhouette\n");

    for (int k = K_MIN; k <= K_MAX; k++) {
        // k-means can get stuck in a bad answer (local minimum), so we try
        // NUM_ATTEMPTS different k-means++ starts and keep the lowest SSE.
        double sse = INFINITY;
        for (int attempt = 0; attempt < NUM_ATTEMPTS; attempt++) {
            // ---- k-means on the sample (k-means++ start, Lloyd until stable) ----
            kmeanspp_init(total_points, k, true_centers, centroids, attempt);
            for (int iter = 0; iter < 100; iter++) {
                memset(sums, 0, sizeof(double) * k * NUM_DIMS);
                memset(counts, 0, sizeof(double) * k);
                int changed = 0;
                for (long i = 0; i < sample_size; i++) {
                    int best_cluster = 0;
                    double best_dist = squared_distance(&sample[i * NUM_DIMS], &centroids[0]);
                    for (int j = 1; j < k; j++) {
                        double dist = squared_distance(&sample[i * NUM_DIMS], &centroids[j * NUM_DIMS]);
                        if (dist < best_dist) { best_dist = dist; best_cluster = j; }
                    }
                    if (iter == 0 || attempt_labels[i] != best_cluster) changed++;
                    attempt_labels[i] = best_cluster;
                    for (int d = 0; d < NUM_DIMS; d++) sums[best_cluster * NUM_DIMS + d] += sample[i * NUM_DIMS + d];
                    counts[best_cluster] += 1.0;
                }
                compute_new_centroids(k, sums, counts, centroids);
                if (changed == 0) break;     // converged
            }
            // ---- SSE of this attempt (elbow) ----
            double attempt_sse = 0.0;
            for (long i = 0; i < sample_size; i++)
                attempt_sse += squared_distance(&sample[i * NUM_DIMS], &centroids[attempt_labels[i] * NUM_DIMS]);
            if (attempt_sse < sse) {         // best so far -> keep its labels
                sse = attempt_sse;
                memcpy(labels, attempt_labels, sizeof(int) * sample_size);
            }
        }

        // ---- silhouette on SILHOUETTE_POINTS evenly spaced sample points ----
        long step = sample_size / SILHOUETTE_POINTS;
        if (step < 1) step = 1;
        double silhouette_total = 0.0;
        long silhouette_count = 0;
        double* dist_sum = new double[k];     // sum of distances to each cluster
        long*   in_cluster = new long[k];     // how many points of each cluster
        for (long i = 0; i < sample_size; i += step) {
            memset(dist_sum, 0, sizeof(double) * k);
            memset(in_cluster, 0, sizeof(long) * k);
            for (long other = 0; other < sample_size; other += step) {
                if (other == i) continue;
                dist_sum[labels[other]] += sqrt(squared_distance(&sample[i * NUM_DIMS], &sample[other * NUM_DIMS]));
                in_cluster[labels[other]]++;
            }
            int own = labels[i];
            if (in_cluster[own] == 0) { silhouette_count++; continue; }   // alone in cluster: s = 0
            double a = dist_sum[own] / in_cluster[own];
            double b = INFINITY;
            for (int j = 0; j < k; j++)
                if (j != own && in_cluster[j] > 0 && dist_sum[j] / in_cluster[j] < b)
                    b = dist_sum[j] / in_cluster[j];
            double bigger = (a > b) ? a : b;
            silhouette_total += (b - a) / bigger;
            silhouette_count++;
        }
        delete[] dist_sum;
        delete[] in_cluster;

        sse_of_k[k] = sse;
        silhouette_of_k[k] = silhouette_total / silhouette_count;
        printf("%3d %12.4e %12.4f\n", k, sse, silhouette_of_k[k]);
        fprintf(csv, "%d,%.6e,%.6f\n", k, sse, silhouette_of_k[k]);
    }
    fclose(csv);

    // ---- elbow: the K where the curve suddenly becomes flat.
    //      drop_before = SSE(K-1) - SSE(K)   (how much K helped)
    //      drop_after  = SSE(K) - SSE(K+1)   (how much K+1 would still help)
    //      elbow = the K with the biggest  drop_before / drop_after
    int elbow_k = K_MIN + 1;
    double biggest_ratio = -1.0;
    for (int k = K_MIN + 1; k < K_MAX; k++) {
        double drop_before = sse_of_k[k - 1] - sse_of_k[k];
        double drop_after  = sse_of_k[k] - sse_of_k[k + 1];
        if (drop_after < 1e-12) drop_after = 1e-12;     // avoid dividing by zero
        double ratio = drop_before / drop_after;
        if (ratio > biggest_ratio) { biggest_ratio = ratio; elbow_k = k; }
    }
    // ---- silhouette: highest average
    int silhouette_k = K_MIN;
    for (int k = K_MIN; k <= K_MAX; k++)
        if (silhouette_of_k[k] > silhouette_of_k[silhouette_k]) silhouette_k = k;

    printf("\nElbow method      -> K = %d\n", elbow_k);
    printf("Silhouette method -> K = %d\n", silhouette_k);
    if (elbow_k == silhouette_k) printf("Both methods agree. Chosen K = %d\n", silhouette_k);
    else printf("Methods disagree; using the silhouette answer K = %d\n", silhouette_k);

    FILE* out = fopen("results/chosen_k.txt", "w");
    fprintf(out, "%d %d %d\n", silhouette_k, elbow_k, silhouette_k);   // chosen elbow silhouette
    fclose(out);

    delete[] sample; delete[] labels; delete[] attempt_labels; delete[] centroids; delete[] sums; delete[] counts;
    return 0;
}
