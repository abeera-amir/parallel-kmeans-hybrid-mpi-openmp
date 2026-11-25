// =============================================================================
//  common.h  -  code shared by ALL programs (serial, hybrid, choose_k)
//
//  No MPI and no OpenMP here.  Because the serial and the hybrid program use
//  the SAME data, the SAME distance function and the SAME starting centroids,
//  their final answers can be compared to prove the parallel code is correct.
//
//  The data is EXACTLY the same as in the old project:
//      4,000,000 points, 16 dimensions, 16 natural groups, seed 12345, noise 4.0
// =============================================================================
#ifndef COMMON_H
#define COMMON_H

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define NUM_DIMS        16        // features per point
#define TRUE_GROUPS     16        // natural groups hidden in the data
#define DATA_SEED       12345     // random seed
#define NOISE           4.0       // spread of points around each group centre
#define SAMPLE_SIZE     20000     // points used to pick k-means++ start centroids
#define CACHE_LINE_DOUBLES 8      // 64-byte cache line = 8 doubles

// -----------------------------------------------------------------------------
// 1. RANDOM NUMBERS WITHOUT A LOOP-CARRIED DEPENDENCY
//
//    LOOP-CARRIED DEPENDENCY (removed here):
//    Normal rand() keeps a hidden state: random number i can only be made
//    AFTER random number i-1.  So "for i: point[i] = rand()" has a dependency
//    from one loop step to the next and cannot be split between threads.
//
//    Fix: random number = hash(seed, point index, coordinate).
//    Point i always gets the same value, no matter which process or thread
//    makes it.  So every point can be generated independently (in parallel).
// -----------------------------------------------------------------------------
static inline uint64_t hash64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
static inline uint64_t hash_two(uint64_t a, uint64_t b) {
    return hash64(a ^ hash64(b + 0x1234567ULL));
}
// random number between 0 and 1
static inline double random01(uint64_t key) {
    return ((hash64(key) >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

// -----------------------------------------------------------------------------
// 2. THE DATA SET
//    true_centers holds TRUE_GROUPS x NUM_DIMS numbers (centres of the groups).
//    Each point = centre of a random group + Gaussian noise.
// -----------------------------------------------------------------------------
static void make_true_centers(double* true_centers) {
    for (int g = 0; g < TRUE_GROUPS; g++)
        for (int d = 0; d < NUM_DIMS; d++)
            true_centers[g * NUM_DIMS + d] =
                100.0 * random01(hash_two(hash_two(DATA_SEED, 99991), (uint64_t)g * 4096 + d));
}

// writes the NUM_DIMS coordinates of point number "index" into "out"
static void make_point(uint64_t index, const double* true_centers, double* out) {
    int group = (int)(hash64(hash_two(DATA_SEED, index)) % TRUE_GROUPS);
    const double* center = &true_centers[group * NUM_DIMS];
    uint64_t base = hash_two(DATA_SEED + 7, index);
    for (int d = 0; d < NUM_DIMS; d++) {
        double u1 = random01(hash_two(base, 2 * (uint64_t)d));
        double u2 = random01(hash_two(base, 2 * (uint64_t)d + 1));
        double gauss = sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);  // Box-Muller
        out[d] = center[d] + NOISE * gauss;
    }
}

// -----------------------------------------------------------------------------
// 3. SQUARED DISTANCE between two points
// -----------------------------------------------------------------------------
static inline double squared_distance(const double* a, const double* b) {
    double sum = 0.0;
    for (int d = 0; d < NUM_DIMS; d++) {
        double diff = a[d] - b[d];
        sum += diff * diff;
    }
    return sum;
}

// -----------------------------------------------------------------------------
// 4. STARTING CENTROIDS
// -----------------------------------------------------------------------------

// (a) CLASSIC K-MEANS: pick num_clusters random points as the starting centroids
static void random_init(uint64_t num_points, int num_clusters, const double* true_centers,
                        double* centroids) {
    uint64_t* chosen = new uint64_t[num_clusters];
    for (int j = 0; j < num_clusters; j++) {
        uint64_t tries = 0;
        bool already_used;
        do {   // pick a random index that was not picked before
            chosen[j] = hash64(hash_two(DATA_SEED + 555, (uint64_t)j * 1000 + tries)) % num_points;
            tries++;
            already_used = false;
            for (int t = 0; t < j; t++)
                if (chosen[t] == chosen[j]) already_used = true;
        } while (already_used);
        make_point(chosen[j], true_centers, &centroids[j * NUM_DIMS]);
    }
    delete[] chosen;
}

// (b) K-MEANS++: the first centroid is random; every next centroid is chosen
//     with probability proportional to (distance to nearest chosen centroid)^2.
//     So the starting centroids are spread out -> better and faster result.
//
//     LOOP-CARRIED DEPENDENCY: centroid j depends on centroids 0..j-1, so the
//     loop over j MUST stay sequential.  It is run on a sample of SAMPLE_SIZE
//     points, which is very cheap (about 0.01 s), so it is done by one process.
//
//     The sample = points spread evenly over the whole data set.
//     "attempt" changes the random choices (0 = default; choose_k tries several).
static void kmeanspp_init(uint64_t num_points, int num_clusters, const double* true_centers,
                          double* centroids, int attempt) {
    long sample_size = SAMPLE_SIZE;
    if ((uint64_t)sample_size > num_points) sample_size = (long)num_points;
    if (sample_size < 1) sample_size = 1;
    uint64_t pick_seed = DATA_SEED + (uint64_t)attempt * 1000003;

    double* sample = new double[sample_size * NUM_DIMS];
    for (long s = 0; s < sample_size; s++)
        make_point((uint64_t)s * num_points / sample_size, true_centers, &sample[s * NUM_DIMS]);

    double* nearest_dist = new double[sample_size];   // distance^2 to nearest chosen centroid
    long first = (long)(random01(hash_two(pick_seed, 4242)) * sample_size);
    memcpy(&centroids[0], &sample[first * NUM_DIMS], sizeof(double) * NUM_DIMS);
    for (long s = 0; s < sample_size; s++)
        nearest_dist[s] = squared_distance(&sample[s * NUM_DIMS], &centroids[0]);

    for (int j = 1; j < num_clusters; j++) {
        double total = 0.0;
        for (long s = 0; s < sample_size; s++) total += nearest_dist[s];

        // pick a point: bigger distance -> bigger chance
        double target = random01(hash_two(pick_seed, 9000 + j)) * total;
        double running = 0.0;
        long picked = sample_size - 1;
        for (long s = 0; s < sample_size; s++) {
            running += nearest_dist[s];
            if (running >= target) { picked = s; break; }
        }
        memcpy(&centroids[j * NUM_DIMS], &sample[picked * NUM_DIMS], sizeof(double) * NUM_DIMS);

        // update nearest distance with the new centroid
        for (long s = 0; s < sample_size; s++) {
            double dist = squared_distance(&sample[s * NUM_DIMS], &centroids[j * NUM_DIMS]);
            if (dist < nearest_dist[s]) nearest_dist[s] = dist;
        }
    }
    delete[] nearest_dist;
    delete[] sample;
}

// -----------------------------------------------------------------------------
// 5. SMALL HELPERS
// -----------------------------------------------------------------------------
// After the sums are added up: new centroid = sum / count.
// An empty cluster keeps its old centroid.
static void compute_new_centroids(int num_clusters, const double* sums, const double* counts,
                                  double* centroids) {
    for (int j = 0; j < num_clusters; j++) {
        if (counts[j] == 0.0) continue;
        for (int d = 0; d < NUM_DIMS; d++)
            centroids[j * NUM_DIMS + d] = sums[j * NUM_DIMS + d] / counts[j];
    }
}

// read "--name value" from the command line
static const char* get_arg(int argc, char** argv, const char* name, const char* default_value) {
    for (int a = 1; a + 1 < argc; a++)
        if (strcmp(argv[a], name) == 0) return argv[a + 1];
    return default_value;
}

#endif
