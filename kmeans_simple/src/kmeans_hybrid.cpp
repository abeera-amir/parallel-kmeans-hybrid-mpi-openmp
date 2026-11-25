// =============================================================================
//  kmeans_hybrid.cpp  -  PARALLEL version: MPI (between processes) + OpenMP
//                        (between threads inside one process)
//
//  Same three algorithms as the serial program:
//    classic   : random start      + Lloyd
//    kmeanspp  : k-means++ start   + Lloyd
//    hamerly   : k-means++ start   + Hamerly
//
//  How the work is split:
//    MPI    : the 4M points are cut into P equal blocks, one block per process.
//             Every process only stores and works on ITS OWN block.
//    OpenMP : inside a process, the points of the block are shared by T threads.
//    After each iteration, the partial sums of all processes are added up with
//    MPI_Allreduce, so every process gets the same new centroids.
//
//  Run:   mpirun -np 2 --bind-to none ./kmeans_hybrid --algo hamerly --threads 2
//
//  ---------------------------------------------------------------------------
//  LIST OF MPI STATEMENTS USED
//    MPI_Init_thread  start MPI; FUNNELED = only the main thread calls MPI
//    MPI_Comm_rank    my process number (0 .. P-1)
//    MPI_Comm_size    number of processes P
//    MPI_Bcast        process 0 sends the starting centroids to everyone
//    MPI_Allreduce    add the partial sums / counts of all processes; every
//                     process receives the total
//    MPI_Reduce       add the SSE of all processes on process 0 only
//    MPI_Barrier      wait for all processes (so the timing is fair)
//    MPI_Wtime        wall-clock timer
//    MPI_Finalize     stop MPI
//
//  LIST OF OPENMP STATEMENTS USED
//    #pragma omp parallel              create a team of threads
//    #pragma omp for schedule(static)  split the loop over the threads in
//                                      equal contiguous blocks
//    #pragma omp parallel for          the two above in one line
//    reduction(+:sse)                  each thread keeps its own copy of sse,
//                                      OpenMP adds the copies at the end
//    omp_get_thread_num()              my thread number (0 .. T-1)
//    omp_set_num_threads()             set number of threads T
//  ---------------------------------------------------------------------------
// =============================================================================
#include "common.h"
#include <mpi.h>
#include <omp.h>

// -----------------------------------------------------------------------------
// FALSE SHARING AND PADDING
//
// Every thread needs its OWN copy of "sums" (K x D) and "counts" (K), because
// if all threads added into ONE shared array, two threads could update the
// same number at the same time (race condition).
//
// But even with separate copies there is a hidden problem: FALSE SHARING.
// The CPU moves memory in blocks of 64 bytes (a "cache line" = 8 doubles).
// If the end of thread 0's array and the start of thread 1's array lie in the
// SAME cache line, then every write of thread 0 forces thread 1's core to
// reload that line (and the other way round), although they never touch the
// same number.  This makes the program much slower.
//
// PADDING fix:
//   - each thread's block is rounded UP to a multiple of 8 doubles (64 bytes)
//   - the whole array starts on a 64-byte boundary (aligned_alloc)
//   => every thread's block starts on its own cache line; no two threads ever
//      share a cache line.
//
//   block of thread t:  [ sums K*D | counts K | padding up to multiple of 8 ]
// -----------------------------------------------------------------------------
static int padded_block_size(int num_clusters) {
    int needed = num_clusters * NUM_DIMS + num_clusters;          // sums + counts
    int padded = ((needed + CACHE_LINE_DOUBLES - 1) / CACHE_LINE_DOUBLES) * CACHE_LINE_DOUBLES;
    return padded;
}

// Adds the blocks of all threads into "process_sums", then MPI_Allreduce adds
// the results of all processes into "global_sums" (same on every process).
static void combine_sums(const double* thread_blocks, int block_size, int num_threads,
                         int num_clusters, double* process_sums, double* global_sums) {
    int values = num_clusters * NUM_DIMS + num_clusters;
    memset(process_sums, 0, sizeof(double) * values);
    for (int t = 0; t < num_threads; t++)            // small loop (272 numbers per thread)
        for (int v = 0; v < values; v++)
            process_sums[v] += thread_blocks[t * block_size + v];

    // MPI: sum over all processes, every process gets the total
    MPI_Allreduce(process_sums, global_sums, values, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

// -----------------------------------------------------------------------------
// LLOYD (classic k-means) - parallel
// -----------------------------------------------------------------------------
static void lloyd_parallel(const double* my_points, long my_num_points, int num_clusters,
                           int num_iters, int num_threads, double* centroids, int* labels) {
    int block_size = padded_block_size(num_clusters);
    int values = num_clusters * NUM_DIMS + num_clusters;
    // 64-byte aligned array, one padded block per thread (see FALSE SHARING above)
    double* thread_blocks = (double*)aligned_alloc(64, sizeof(double) * block_size * num_threads);
    double* process_sums  = new double[values];
    double* global_sums   = new double[values];

    // LOOP-CARRIED DEPENDENCY: iteration "iter" needs the centroids of
    // iteration "iter-1" -> this loop stays sequential.  Only the loop over
    // the POINTS (inside) is parallel.
    for (int iter = 0; iter < num_iters; iter++) {
        memset(thread_blocks, 0, sizeof(double) * block_size * num_threads);

        #pragma omp parallel                      // OpenMP: start threads
        {
            int my_thread = omp_get_thread_num();
            double* my_sums   = &thread_blocks[my_thread * block_size];
            double* my_counts = &my_sums[num_clusters * NUM_DIMS];

            // OpenMP: each thread gets one contiguous block of points.
            // NO loop-carried dependency between points: point i only reads
            // the centroids and writes labels[i] -> safe to run in parallel.
            // The "+=" into sums WOULD be a dependency (many points add into the
            // same cluster) -> removed by giving each thread its own my_sums.
            #pragma omp for schedule(static)
            for (long i = 0; i < my_num_points; i++) {
                const double* point = &my_points[i * NUM_DIMS];
                int best_cluster = 0;
                double best_dist = squared_distance(point, &centroids[0]);
                for (int j = 1; j < num_clusters; j++) {
                    double dist = squared_distance(point, &centroids[j * NUM_DIMS]);
                    if (dist < best_dist) { best_dist = dist; best_cluster = j; }
                }
                // labels: with schedule(static) each thread writes one big
                // contiguous piece, so threads only meet at the 2 edges of their
                // piece -> almost no false sharing on labels.
                labels[i] = best_cluster;
                for (int d = 0; d < NUM_DIMS; d++) my_sums[best_cluster * NUM_DIMS + d] += point[d];
                my_counts[best_cluster] += 1.0;
            }
        }                                           // OpenMP: threads end (implicit barrier)

        combine_sums(thread_blocks, block_size, num_threads, num_clusters, process_sums, global_sums);
        compute_new_centroids(num_clusters, global_sums, &global_sums[num_clusters * NUM_DIMS], centroids);
    }
    free(thread_blocks);
    delete[] process_sums;
    delete[] global_sums;
}

// -----------------------------------------------------------------------------
// HAMERLY - parallel  (same idea as in the serial file; see comments there)
// -----------------------------------------------------------------------------
static void hamerly_parallel(const double* my_points, long my_num_points, int num_clusters,
                             int num_iters, int num_threads, double* centroids, int* labels) {
    int block_size = padded_block_size(num_clusters);
    int values = num_clusters * NUM_DIMS + num_clusters;
    double* thread_blocks = (double*)aligned_alloc(64, sizeof(double) * block_size * num_threads);
    double* process_sums  = new double[values];
    double* global_sums   = new double[values];
    double* upper    = new double[my_num_points];
    double* lower    = new double[my_num_points];
    double* half_gap = new double[num_clusters];
    double* movement = new double[num_clusters];
    double* old_centroids = new double[num_clusters * NUM_DIMS];

    #pragma omp parallel for schedule(static)
    for (long i = 0; i < my_num_points; i++) { upper[i] = INFINITY; lower[i] = 0.0; labels[i] = 0; }

    for (int iter = 0; iter < num_iters; iter++) {   // sequential (loop-carried dependency)

        // (1) half_gap: only K*K = 256 distances, every process computes the
        //     same values itself (cheaper than sending them).
        for (int j = 0; j < num_clusters; j++) {
            double closest = INFINITY;
            for (int other = 0; other < num_clusters; other++) {
                if (other == j) continue;
                double dist = squared_distance(&centroids[j * NUM_DIMS], &centroids[other * NUM_DIMS]);
                if (dist < closest) closest = dist;
            }
            half_gap[j] = 0.5 * sqrt(closest);
        }

        // (2) assignment with bounds - parallel over points
        memset(thread_blocks, 0, sizeof(double) * block_size * num_threads);
        #pragma omp parallel
        {
            int my_thread = omp_get_thread_num();
            double* my_sums   = &thread_blocks[my_thread * block_size];
            double* my_counts = &my_sums[num_clusters * NUM_DIMS];

            // NO loop-carried dependency: upper[i], lower[i], labels[i] belong to
            // point i only.  Note: some points need a full search and others do
            // not, so the work per point is uneven.  "static" still works well
            // here because skipped points are spread evenly over the data.
            #pragma omp for schedule(static)
            for (long i = 0; i < my_num_points; i++) {
                const double* point = &my_points[i * NUM_DIMS];
                int a = labels[i];
                double bound = (half_gap[a] > lower[i]) ? half_gap[a] : lower[i];
                if (upper[i] > bound) {
                    upper[i] = sqrt(squared_distance(point, &centroids[a * NUM_DIMS]));
                    if (upper[i] > bound) {
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
                for (int d = 0; d < NUM_DIMS; d++) my_sums[c * NUM_DIMS + d] += point[d];
                my_counts[c] += 1.0;
            }
        }

        // (3) new centroids (MPI_Allreduce inside combine_sums) + movement
        memcpy(old_centroids, centroids, sizeof(double) * num_clusters * NUM_DIMS);
        combine_sums(thread_blocks, block_size, num_threads, num_clusters, process_sums, global_sums);
        compute_new_centroids(num_clusters, global_sums, &global_sums[num_clusters * NUM_DIMS], centroids);
        double max_move = 0.0;
        for (int j = 0; j < num_clusters; j++) {
            movement[j] = sqrt(squared_distance(&old_centroids[j * NUM_DIMS], &centroids[j * NUM_DIMS]));
            if (movement[j] > max_move) max_move = movement[j];
        }

        // (4) move the bounds - every point independent -> parallel
        #pragma omp parallel for schedule(static)
        for (long i = 0; i < my_num_points; i++) {
            upper[i] += movement[labels[i]];
            lower[i] -= max_move;
        }
    }
    free(thread_blocks);
    delete[] process_sums; delete[] global_sums; delete[] upper; delete[] lower;
    delete[] half_gap; delete[] movement; delete[] old_centroids;
}

int main(int argc, char** argv) {
    // MPI: start.  FUNNELED = threads exist, but only the main thread calls MPI
    // (all MPI calls here are outside the "omp parallel" regions).
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int my_rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    const char* algo   = get_arg(argc, argv, "--algo", "kmeanspp");
    long total_points  = atol(get_arg(argc, argv, "--n", "4000000"));
    int num_clusters   = atoi(get_arg(argc, argv, "--k", "16"));
    int num_iters      = atoi(get_arg(argc, argv, "--iters", "50"));
    int num_threads    = atoi(get_arg(argc, argv, "--threads", "1"));
    omp_set_num_threads(num_threads);

    // MPI: my block of points = [first_point, first_point + my_num_points)
    long first_point   = (long)((double)total_points * my_rank / num_procs);
    long end_point     = (long)((double)total_points * (my_rank + 1) / num_procs);
    long my_num_points = end_point - first_point;

    double true_centers[TRUE_GROUPS * NUM_DIMS];
    make_true_centers(true_centers);

    // generate only MY points (not timed).  No loop-carried dependency thanks to
    // the hash-based random numbers -> parallel with OpenMP.
    double* my_points = new double[(size_t)my_num_points * NUM_DIMS];
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < my_num_points; i++)
        make_point(first_point + i, true_centers, &my_points[i * NUM_DIMS]);
    int* labels = new int[my_num_points];
    double* centroids = new double[num_clusters * NUM_DIMS];

    MPI_Barrier(MPI_COMM_WORLD);                // everyone ready
    double start = MPI_Wtime();                 // ---- timer starts

    // process 0 chooses the starting centroids, then sends them to everyone
    if (my_rank == 0) {
        if (strcmp(algo, "classic") == 0)
            random_init(total_points, num_clusters, true_centers, centroids);
        else
            kmeanspp_init(total_points, num_clusters, true_centers, centroids, 0);
    }
    MPI_Bcast(centroids, num_clusters * NUM_DIMS, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (strcmp(algo, "hamerly") == 0)
        hamerly_parallel(my_points, my_num_points, num_clusters, num_iters, num_threads, centroids, labels);
    else
        lloyd_parallel(my_points, my_num_points, num_clusters, num_iters, num_threads, centroids, labels);

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - start;       // ---- timer stops

    // SSE: each thread adds its part (reduction), then each process sends its
    // part to process 0 (MPI_Reduce)
    double my_sse = 0.0, total_sse = 0.0;
    #pragma omp parallel for schedule(static) reduction(+:my_sse)
    for (long i = 0; i < my_num_points; i++)
        my_sse += squared_distance(&my_points[i * NUM_DIMS], &centroids[labels[i] * NUM_DIMS]);
    MPI_Reduce(&my_sse, &total_sse, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (my_rank == 0)
        printf("RESULT version=hybrid algo=%s procs=%d threads=%d n=%ld k=%d iters=%d time=%.4f sse=%.10e\n",
               algo, num_procs, num_threads, total_points, num_clusters, num_iters, elapsed, total_sse);

    delete[] my_points; delete[] labels; delete[] centroids;
    MPI_Finalize();                             // MPI: stop
    return 0;
}
