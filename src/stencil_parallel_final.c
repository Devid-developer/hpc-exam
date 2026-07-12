/* --------------------------------------------------------------------------
 * MPI implementation of the two-dimensional five-point stencil.
 *
 * This file derives from stencil_template_parallel.c and carries over the
 * completed serial/OpenMP implementation.  The domain is decomposed manually
 * and halo columns are explicitly packed, so no Cartesian topology or derived
 * MPI datatype is required.
 * -------------------------------------------------------------------------- */

#define _XOPEN_SOURCE 700
#define _POSIX_C_SOURCE 200809L

#include "stencil_parallel_final.h"

#include <getopt.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define MEM_ALIGN 64

enum {
    TAG_TO_NORTH = 100,
    TAG_TO_SOUTH = 101,
    TAG_TO_EAST = 102,
    TAG_TO_WEST = 103
};

typedef struct {
    int niterations;
    int periodic;
    int nsources;
    int output_energy_at_steps;
    int injection_frequency;
    int fixed_sources;
    int verbose;
    long seed;
    double energy_per_source;
} options_t;

static int parse_options(int argc, char **argv, int rank,
                         vec2_t global_size, options_t *options);
static int choose_process_grid(const vec2_t global_size, int ntasks,
                               vec2_t process_grid);
static uint block_size(uint global_size, uint nblocks, uint coordinate);
static uint block_offset(uint global_size, uint nblocks, uint coordinate);
static void find_neighbours(int rank, const vec2_t process_grid, int periodic,
                            int neighbours[4]);
static int memory_allocate(plane_t planes[2], buffers_t buffers[2]);
static int initialize_sources(int rank, MPI_Comm *comm,
                              const vec2_t global_size,
                              const vec2_t local_offset,
                              const plane_t planes[2], int nsources,
                              int fixed_sources, long seed,
                              int *nsources_local, vec2_t **sources_local);
static int exchange_halos(const int neighbours[4], MPI_Comm *comm,
                          plane_t *plane, buffers_t buffers[2]);
static int global_max_int(int local_value, MPI_Comm *comm);
static int memory_release(plane_t planes[2], buffers_t buffers[2],
                          vec2_t *sources_local);
static int output_energy_stat(int step, const plane_t *plane, double budget,
                              double global_points, int rank, MPI_Comm *comm,
                              double *global_energy);
static void print_decomposition(int rank, int ntasks,
                                const vec2_t process_grid,
                                const vec2_t coordinates,
                                const vec2_t local_offset,
                                const int neighbours[4],
                                const plane_t planes[2], MPI_Comm *comm);

int main(int argc, char **argv)
{
    MPI_Comm my_comm_world = MPI_COMM_WORLD;
    int rank;
    int ntasks;
    int level_obtained;
    int neighbours[4];
    vec2_t global_size = {0, 0};
    vec2_t process_grid = {0, 0};
    vec2_t coordinates = {0, 0};
    vec2_t local_offset = {0, 0};
    plane_t planes[2] = {{NULL, {0, 0}}, {NULL, {0, 0}}};
    buffers_t buffers[2] = {
        {NULL, NULL, NULL, NULL},
        {NULL, NULL, NULL, NULL}
    };
    options_t options;
    int nsources_local = 0;
    vec2_t *sources_local = NULL;
    int current = OLD;
    int nthreads = 1;
    double injected_heat = 0.0;
    double t_update = 0.0;
    double t_inject = 0.0;
    double t_comm = 0.0;
    double t_energy = 0.0;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &level_obtained);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ntasks);

    if (level_obtained < MPI_THREAD_FUNNELED) {
        if (rank == 0)
            fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Finalize();
        return 1;
    }

    int ret = parse_options(argc, argv, rank, global_size, &options);
    if (ret != 0) {
        MPI_Finalize();
        return ret > 0 ? ret : 0;
    }

    ret = choose_process_grid(global_size, ntasks, process_grid);
    if (ret != 0) {
        if (rank == 0)
            fprintf(stderr, "cannot decompose a %u x %u domain over %d ranks\n",
                    global_size[_x_], global_size[_y_], ntasks);
        MPI_Finalize();
        return 3;
    }

    coordinates[_x_] = (uint)rank % process_grid[_x_];
    coordinates[_y_] = (uint)rank / process_grid[_x_];
    local_offset[_x_] = block_offset(global_size[_x_], process_grid[_x_],
                                     coordinates[_x_]);
    local_offset[_y_] = block_offset(global_size[_y_], process_grid[_y_],
                                     coordinates[_y_]);

    planes[OLD].size[_x_] = block_size(global_size[_x_], process_grid[_x_],
                                      coordinates[_x_]);
    planes[OLD].size[_y_] = block_size(global_size[_y_], process_grid[_y_],
                                      coordinates[_y_]);
    planes[NEW].size[_x_] = planes[OLD].size[_x_];
    planes[NEW].size[_y_] = planes[OLD].size[_y_];

    find_neighbours(rank, process_grid, options.periodic, neighbours);

    ret = memory_allocate(planes, buffers);
    {
        ret = global_max_int(ret, &my_comm_world);
    }
    if (ret != 0) {
        if (rank == 0)
            fprintf(stderr, "distributed plane/buffer allocation failed\n");
        memory_release(planes, buffers, NULL);
        MPI_Finalize();
        return 4;
    }

    ret = initialize_sources(rank, &my_comm_world, global_size,
                             local_offset, planes,
                             options.nsources, options.fixed_sources,
                             options.seed, &nsources_local, &sources_local);
    if (ret != 0) {
        if (rank == 0)
            fprintf(stderr, "distributed source initialization failed\n");
        memory_release(planes, buffers, sources_local);
        MPI_Finalize();
        return 5;
    }

#ifdef _OPENMP
#pragma omp parallel
    {
#pragma omp single
        nthreads = omp_get_num_threads();
    }
#endif

    if (options.verbose)
        print_decomposition(rank, ntasks, process_grid, coordinates,
                            local_offset, neighbours, planes, &my_comm_world);

    MPI_Barrier(my_comm_world);
    const double t_wall_start = MPI_Wtime();

    for (int iter = 0; iter < options.niterations; iter++) {
        double t0;

        if (iter % options.injection_frequency == 0) {
            t0 = MPI_Wtime();
            inject_energy(options.periodic, nsources_local, sources_local,
                          options.energy_per_source, &planes[current],
                          process_grid);
            t_inject += MPI_Wtime() - t0;
            injected_heat += options.nsources * options.energy_per_source;
        }

        t0 = MPI_Wtime();
        exchange_halos(neighbours, &my_comm_world, &planes[current], buffers);
        t_comm += MPI_Wtime() - t0;

        t0 = MPI_Wtime();
        update_plane(options.periodic, process_grid,
                     &planes[current], &planes[!current]);
        t_update += MPI_Wtime() - t0;

        if (options.output_energy_at_steps) {
            double global_energy;
            t0 = MPI_Wtime();
            output_energy_stat(iter, &planes[!current], injected_heat,
                               (double)global_size[_x_] * global_size[_y_], rank,
                               &my_comm_world, &global_energy);
            t_energy += MPI_Wtime() - t0;
        }

        current = !current;
    }

    double global_energy = 0.0;
    double t0 = MPI_Wtime();
    output_energy_stat(-1, &planes[current], injected_heat,
                       (double)global_size[_x_] * global_size[_y_], rank,
                       &my_comm_world, &global_energy);
    t_energy += MPI_Wtime() - t0;
    const double t_wall = MPI_Wtime() - t_wall_start;

    double local_times[5] = {
        t_update, t_inject, t_comm, t_energy, t_wall
    };
    double max_times[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    MPI_Reduce(local_times, max_times, 5, MPI_DOUBLE, MPI_MAX, 0,
               my_comm_world);

    if (rank == 0) {
        const double updates = (double)global_size[_x_] *
                               (double)global_size[_y_] *
                               (double)options.niterations;
        const double glups = updates / max_times[0] / 1e9;
        const double gflops = 6.0 * updates / max_times[0] / 1e9;
        const double gbs = 24.0 * updates / max_times[0] / 1e9;

        printf("\n============================================================\n");
        printf("global grid      : %u x %u (+ local halos)\n",
               global_size[_x_], global_size[_y_]);
        printf("process grid     : %u x %u   MPI tasks: %d\n",
               process_grid[_x_], process_grid[_y_], ntasks);
        printf("OpenMP threads   : %d per MPI task\n", nthreads);
        printf("iterations       : %d   sources: %d   periodic: %d\n",
               options.niterations, options.nsources, options.periodic);
        printf("source mode      : %s",
               options.fixed_sources ? "fixed quarter positions" : "random");
        if (!options.fixed_sources)
            printf(" (seed %ld)", options.seed);
        printf("\n------------------------------------------------------------\n");
        printf("t_update max     : %.6f s   (%.6e s/iter)\n",
               max_times[0], max_times[0] / options.niterations);
        printf("t_comm max       : %.6f s\n", max_times[2]);
        printf("t_inject max     : %.6f s\n", max_times[1]);
        printf("t_energy max     : %.6f s\n", max_times[3]);
        printf("t_wall max       : %.6f s\n", max_times[4]);
        printf("------------------------------------------------------------\n");
        printf("performance      : %.6f GLUP/s   %.6f GFLOP/s\n", glups, gflops);
        printf("bandwidth model  : %.6f GB/s [24 B/update]\n", gbs);
        printf("------------------------------------------------------------\n");
        printf("injected energy  : %g\n", injected_heat);
        printf("system energy    : %g\n", global_energy);
        printf("============================================================\n");
        printf("CSV,%d,%d,%u,%u,%d,%.6e,%.6e,%.6e,%.6f,%.6f,%.6f\n",
               ntasks, nthreads, global_size[_x_], global_size[_y_],
               options.niterations, max_times[4], max_times[0], max_times[2],
               glups, gflops, gbs);
    }

    memory_release(planes, buffers, sources_local);
    MPI_Finalize();
    return 0;
}

static int parse_options(int argc, char **argv, int rank,
                         vec2_t global_size, options_t *options)
{
    int opt;
    int parsed_x = 1000;
    int parsed_y = 1000;
    double frequency = 0.0;

    options->periodic = 0;
    options->nsources = 1;
    options->niterations = 99;
    options->output_energy_at_steps = 0;
    options->energy_per_source = 1.0;
    options->injection_frequency = 1;
    options->fixed_sources = 0;
    options->seed = 0;
    options->verbose = 0;

    while ((opt = getopt(argc, argv, ":x:y:e:E:f:n:p:o:Fs:v:h")) != -1) {
        switch (opt) {
        case 'x': parsed_x = atoi(optarg); break;
        case 'y': parsed_y = atoi(optarg); break;
        case 'e': options->nsources = atoi(optarg); break;
        case 'E': options->energy_per_source = atof(optarg); break;
        case 'f': frequency = atof(optarg); break;
        case 'n': options->niterations = atoi(optarg); break;
        case 'p': options->periodic = (atoi(optarg) > 0); break;
        case 'o': options->output_energy_at_steps = (atoi(optarg) > 0); break;
        case 'F': options->fixed_sources = 1; break;
        case 's': options->seed = atol(optarg); break;
        case 'v': options->verbose = (atoi(optarg) > 0); break;
        case 'h':
            if (rank == 0)
                printf("valid options are:\n"
                       "-x    x size of the plate [1000]\n"
                       "-y    y size of the plate [1000]\n"
                       "-e    number of energy sources [1]\n"
                       "-E    energy per source [1.0]\n"
                       "-f    injection frequency as fraction of iterations [0.0]\n"
                       "-n    number of iterations [99]\n"
                       "-p    periodic boundaries [0]\n"
                       "-o    output global energy at every step [0]\n"
                       "-F    use fixed quarter-position sources\n"
                       "-s    seed for random sources [0]\n"
                       "-v    print MPI domain decomposition [0]\n");
            return -1;
        case ':':
            if (rank == 0)
                fprintf(stderr, "option -%c requires an argument\n", optopt);
            return 1;
        case '?':
            if (rank == 0)
                fprintf(stderr, "unknown option -%c\n", optopt);
            return 1;
        }
    }

    if (parsed_x <= 0 || parsed_y <= 0 || options->niterations <= 0 ||
        options->nsources < 0 || options->energy_per_source < 0.0 ||
        frequency < 0.0) {
        if (rank == 0)
            fprintf(stderr, "invalid parameters\n");
        return 2;
    }

    global_size[_x_] = (uint)parsed_x;
    global_size[_y_] = (uint)parsed_y;

    if (frequency == 0.0) {
        options->injection_frequency = 1;
    } else {
        if (frequency > 1.0)
            frequency = 1.0;
        options->injection_frequency =
            (int)(frequency * options->niterations);
        if (options->injection_frequency < 1)
            options->injection_frequency = 1;
    }

    return 0;
}

static int choose_process_grid(const vec2_t global_size, int ntasks,
                               vec2_t process_grid)
{
    int found = 0;
    double best_score = 0.0;

    for (int nx = 1; nx <= ntasks; nx++) {
        int ny;
        double local_x;
        double local_y;
        double score;

        if (ntasks % nx != 0)
            continue;
        ny = ntasks / nx;
        if ((uint)nx > global_size[_x_] || (uint)ny > global_size[_y_])
            continue;

        local_x = (double)global_size[_x_] / nx;
        local_y = (double)global_size[_y_] / ny;
        score = local_x > local_y ? local_x / local_y : local_y / local_x;

        if (!found || score < best_score) {
            process_grid[_x_] = (uint)nx;
            process_grid[_y_] = (uint)ny;
            best_score = score;
            found = 1;
        }
    }

    return found ? 0 : 1;
}

static uint block_size(uint global_size, uint nblocks, uint coordinate)
{
    return global_size / nblocks + (coordinate < global_size % nblocks);
}

static uint block_offset(uint global_size, uint nblocks, uint coordinate)
{
    const uint base = global_size / nblocks;
    const uint remainder = global_size % nblocks;
    return coordinate * base +
           (coordinate < remainder ? coordinate : remainder);
}

static void find_neighbours(int rank, const vec2_t process_grid, int periodic,
                            int neighbours[4])
{
    const int nx = (int)process_grid[_x_];
    const int ny = (int)process_grid[_y_];
    const int x = rank % nx;
    const int y = rank / nx;

    for (int direction = 0; direction < 4; direction++)
        neighbours[direction] = MPI_PROC_NULL;

    if (nx > 1) {
        neighbours[EAST] = x + 1 < nx ? rank + 1 :
                           (periodic ? y * nx : MPI_PROC_NULL);
        neighbours[WEST] = x > 0 ? rank - 1 :
                           (periodic ? y * nx + nx - 1 : MPI_PROC_NULL);
    }

    if (ny > 1) {
        neighbours[NORTH] = y > 0 ? rank - nx :
                            (periodic ? (ny - 1) * nx + x : MPI_PROC_NULL);
        neighbours[SOUTH] = y + 1 < ny ? rank + nx :
                            (periodic ? x : MPI_PROC_NULL);
    }
}

static int memory_allocate(plane_t planes[2], buffers_t buffers[2])
{
    const size_t fxsize = (size_t)planes[OLD].size[_x_] + 2;
    const size_t fysize = (size_t)planes[OLD].size[_y_] + 2;
    const size_t frame = fxsize * fysize;
    const size_t column_size = (size_t)planes[OLD].size[_y_];
    double *old_plane = NULL;
    double *new_plane = NULL;

    for (int kind = SEND; kind <= RECV; kind++)
        for (int direction = 0; direction < 4; direction++)
            buffers[kind][direction] = NULL;

    if (posix_memalign((void **)&old_plane, MEM_ALIGN,
                       frame * sizeof(double)) != 0)
        return 1;
    if (posix_memalign((void **)&new_plane, MEM_ALIGN,
                       frame * sizeof(double)) != 0) {
        free(old_plane);
        return 1;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (size_t index = 0; index < frame; index++) {
        old_plane[index] = 0.0;
        new_plane[index] = 0.0;
    }

    planes[OLD].data = old_plane;
    planes[NEW].data = new_plane;

    for (int kind = SEND; kind <= RECV; kind++) {
        if (posix_memalign((void **)&buffers[kind][EAST], MEM_ALIGN,
                           column_size * sizeof(double)) != 0 ||
            posix_memalign((void **)&buffers[kind][WEST], MEM_ALIGN,
                           column_size * sizeof(double)) != 0) {
            for (int cleanup_kind = SEND; cleanup_kind <= RECV;
                 cleanup_kind++) {
                free(buffers[cleanup_kind][EAST]);
                free(buffers[cleanup_kind][WEST]);
                buffers[cleanup_kind][EAST] = NULL;
                buffers[cleanup_kind][WEST] = NULL;
            }
            free(planes[OLD].data);
            free(planes[NEW].data);
            planes[OLD].data = NULL;
            planes[NEW].data = NULL;
            return 1;
        }
    }

    return 0;
}

static int initialize_sources(int rank, MPI_Comm *comm,
                              const vec2_t global_size,
                              const vec2_t local_offset,
                              const plane_t planes[2], int nsources,
                              int fixed_sources, long seed,
                              int *nsources_local, vec2_t **sources_local)
{
    vec2_t *global_sources = NULL;
    int nlocal = 0;

    *nsources_local = 0;
    *sources_local = NULL;

    if (nsources == 0)
        return 0;

    global_sources = malloc((size_t)nsources * sizeof(*global_sources));
    int local_failure = (global_sources == NULL);
    int any_failure = global_max_int(local_failure, comm);
    if (any_failure) {
        free(global_sources);
        return 1;
    }

    if (rank == 0) {
        if (fixed_sources) {
            const uint fixed[4][2] = {
                {global_size[_x_] / 4, global_size[_y_] / 4},
                {3 * global_size[_x_] / 4, global_size[_y_] / 4},
                {global_size[_x_] / 4, 3 * global_size[_y_] / 4},
                {3 * global_size[_x_] / 4, 3 * global_size[_y_] / 4}
            };

            for (int source = 0; source < nsources; source++) {
                global_sources[source][_x_] = fixed[source % 4][_x_];
                global_sources[source][_y_] = fixed[source % 4][_y_];
                if (global_sources[source][_x_] < 1)
                    global_sources[source][_x_] = 1;
                if (global_sources[source][_y_] < 1)
                    global_sources[source][_y_] = 1;
            }
        } else {
            srand48(seed);
            for (int source = 0; source < nsources; source++) {
                global_sources[source][_x_] =
                    1 + (uint)(lrand48() % global_size[_x_]);
                global_sources[source][_y_] =
                    1 + (uint)(lrand48() % global_size[_y_]);
            }
        }
    }

    MPI_Bcast(global_sources, nsources * 2, MPI_UNSIGNED, 0, *comm);

    for (int source = 0; source < nsources; source++) {
        const uint global_x = global_sources[source][_x_] - 1;
        const uint global_y = global_sources[source][_y_] - 1;
        if (global_x >= local_offset[_x_] &&
            global_x < local_offset[_x_] + planes[OLD].size[_x_] &&
            global_y >= local_offset[_y_] &&
            global_y < local_offset[_y_] + planes[OLD].size[_y_])
            nlocal++;
    }

    if (nlocal > 0) {
        *sources_local = malloc((size_t)nlocal * sizeof(**sources_local));
    }

    local_failure = (nlocal > 0 && *sources_local == NULL);
    any_failure = global_max_int(local_failure, comm);
    if (any_failure) {
        free(*sources_local);
        *sources_local = NULL;
        free(global_sources);
        return 1;
    }

    if (nlocal > 0) {
        int local_index = 0;
        for (int source = 0; source < nsources; source++) {
            const uint global_x = global_sources[source][_x_] - 1;
            const uint global_y = global_sources[source][_y_] - 1;
            if (global_x >= local_offset[_x_] &&
                global_x < local_offset[_x_] + planes[OLD].size[_x_] &&
                global_y >= local_offset[_y_] &&
                global_y < local_offset[_y_] + planes[OLD].size[_y_]) {
                (*sources_local)[local_index][_x_] =
                    global_x - local_offset[_x_] + 1;
                (*sources_local)[local_index][_y_] =
                    global_y - local_offset[_y_] + 1;
                local_index++;
            }
        }
    }

    *nsources_local = nlocal;
    free(global_sources);
    return 0;
}

static int exchange_halos(const int neighbours[4], MPI_Comm *comm,
                          plane_t *plane, buffers_t buffers[2])
{
    const size_t fxsize = (size_t)plane->size[_x_] + 2;
    const uint xsize = plane->size[_x_];
    const uint ysize = plane->size[_y_];
    double *restrict data = plane->data;
    MPI_Request requests[8];
    int nrequests = 0;

#define IDX(i, j) ((size_t)(j) * fxsize + (size_t)(i))

    if (neighbours[WEST] != MPI_PROC_NULL ||
        neighbours[EAST] != MPI_PROC_NULL) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (uint j = 1; j <= ysize; j++) {
            buffers[SEND][WEST][j - 1] = data[IDX(1, j)];
            buffers[SEND][EAST][j - 1] = data[IDX(xsize, j)];
        }
    }

    if (neighbours[NORTH] != MPI_PROC_NULL)
        MPI_Irecv(&data[IDX(1, 0)], (int)xsize, MPI_DOUBLE,
                  neighbours[NORTH], TAG_TO_SOUTH, *comm,
                  &requests[nrequests++]);
    if (neighbours[SOUTH] != MPI_PROC_NULL)
        MPI_Irecv(&data[IDX(1, ysize + 1)], (int)xsize, MPI_DOUBLE,
                  neighbours[SOUTH], TAG_TO_NORTH, *comm,
                  &requests[nrequests++]);
    if (neighbours[WEST] != MPI_PROC_NULL)
        MPI_Irecv(buffers[RECV][WEST], (int)ysize, MPI_DOUBLE,
                  neighbours[WEST], TAG_TO_EAST, *comm,
                  &requests[nrequests++]);
    if (neighbours[EAST] != MPI_PROC_NULL)
        MPI_Irecv(buffers[RECV][EAST], (int)ysize, MPI_DOUBLE,
                  neighbours[EAST], TAG_TO_WEST, *comm,
                  &requests[nrequests++]);

    if (neighbours[NORTH] != MPI_PROC_NULL)
        MPI_Isend(&data[IDX(1, 1)], (int)xsize, MPI_DOUBLE,
                  neighbours[NORTH], TAG_TO_NORTH, *comm,
                  &requests[nrequests++]);
    if (neighbours[SOUTH] != MPI_PROC_NULL)
        MPI_Isend(&data[IDX(1, ysize)], (int)xsize, MPI_DOUBLE,
                  neighbours[SOUTH], TAG_TO_SOUTH, *comm,
                  &requests[nrequests++]);
    if (neighbours[WEST] != MPI_PROC_NULL)
        MPI_Isend(buffers[SEND][WEST], (int)ysize, MPI_DOUBLE,
                  neighbours[WEST], TAG_TO_WEST, *comm,
                  &requests[nrequests++]);
    if (neighbours[EAST] != MPI_PROC_NULL)
        MPI_Isend(buffers[SEND][EAST], (int)ysize, MPI_DOUBLE,
                  neighbours[EAST], TAG_TO_EAST, *comm,
                  &requests[nrequests++]);

    if (nrequests > 0)
        MPI_Waitall(nrequests, requests, MPI_STATUSES_IGNORE);

    if (neighbours[WEST] != MPI_PROC_NULL ||
        neighbours[EAST] != MPI_PROC_NULL) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (uint j = 1; j <= ysize; j++) {
            if (neighbours[WEST] != MPI_PROC_NULL)
                data[IDX(0, j)] = buffers[RECV][WEST][j - 1];
            if (neighbours[EAST] != MPI_PROC_NULL)
                data[IDX(xsize + 1, j)] = buffers[RECV][EAST][j - 1];
        }
    }

#undef IDX
    return 0;
}

static int global_max_int(int local_value, MPI_Comm *comm)
{
    int global_value = 0;

    MPI_Reduce(&local_value, &global_value, 1, MPI_INT, MPI_MAX, 0, *comm);
    MPI_Bcast(&global_value, 1, MPI_INT, 0, *comm);
    return global_value;
}

static int memory_release(plane_t planes[2], buffers_t buffers[2],
                          vec2_t *sources_local)
{
    free(planes[OLD].data);
    free(planes[NEW].data);
    for (int kind = SEND; kind <= RECV; kind++) {
        for (int direction = 0; direction < 4; direction++) {
            free(buffers[kind][direction]);
            buffers[kind][direction] = NULL;
        }
    }
    free(sources_local);
    planes[OLD].data = NULL;
    planes[NEW].data = NULL;
    return 0;
}

static int output_energy_stat(int step, const plane_t *plane, double budget,
                              double global_points, int rank, MPI_Comm *comm,
                              double *global_energy)
{
    double local_energy = 0.0;
    double total_energy = 0.0;

    get_total_energy(plane, &local_energy);
    MPI_Reduce(&local_energy, &total_energy, 1, MPI_DOUBLE, MPI_SUM, 0, *comm);

    if (rank == 0) {
        if (step >= 0)
            printf("step %d :: ", step);
        printf("injected energy is %g, system energy is %g "
               "(average %g per grid point)\n",
               budget, total_energy, total_energy / global_points);
        if (global_energy != NULL)
            *global_energy = total_energy;
    }

    return 0;
}

static void print_decomposition(int rank, int ntasks,
                                const vec2_t process_grid,
                                const vec2_t coordinates,
                                const vec2_t local_offset,
                                const int neighbours[4],
                                const plane_t planes[2], MPI_Comm *comm)
{
    if (rank == 0) {
        printf("MPI process grid: %u x %u\n",
               process_grid[_x_], process_grid[_y_]);
        fflush(stdout);
    }

    MPI_Barrier(*comm);
    for (int task = 0; task < ntasks; task++) {
        if (task == rank) {
            printf("rank %4d coords=(%u,%u) offset=(%u,%u) size=(%u,%u) "
                   "N=%d S=%d E=%d W=%d\n",
                   rank, coordinates[_x_], coordinates[_y_],
                   local_offset[_x_], local_offset[_y_],
                   planes[OLD].size[_x_], planes[OLD].size[_y_],
                   neighbours[NORTH], neighbours[SOUTH],
                   neighbours[EAST], neighbours[WEST]);
            fflush(stdout);
        }
        MPI_Barrier(*comm);
    }
}
