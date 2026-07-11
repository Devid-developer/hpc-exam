/* --------------------------------------------------------------------------
 * MPI + OpenMP implementation of the two-dimensional five-point stencil.
 * This file derives from stencil_template_parallel.c and retains the
 * correctness and single-core optimizations of stencil_serial_final.c.
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
    TAG_TO_EAST  = 102,
    TAG_TO_WEST  = 103
};

typedef struct {
    MPI_Comm cart;
    int rank;
    int ntasks;
    int neighbours[4];
    int coords[2];                 /* MPI order: y, x */
    vec2_t global_size;
    vec2_t process_grid;
    vec2_t local_offset;           /* zero-based global offset */
    MPI_Datatype column_type;
} domain_t;

typedef struct {
    int niterations;
    int periodic;
    int nsources;
    double energy_per_source;
    int output_energy_at_steps;
    int injection_frequency;
    int fixed_sources;
    long seed;
    int verbose;
} options_t;

static void print_help(void);
static int parse_options(int argc, char **argv, int rank, vec2_t size,
                         options_t *options);
static int choose_process_grid(const vec2_t size, int ntasks, vec2_t grid);
static uint block_size(uint global, uint blocks, uint coordinate);
static uint block_offset(uint global, uint blocks, uint coordinate);
static int domain_initialize(MPI_Comm world, const vec2_t size, int periodic,
                             domain_t *domain, plane_t planes[2]);
static int memory_allocate(plane_t planes[2]);
static void memory_release(plane_t planes[2]);
static int initialize_sources(const domain_t *domain, int nsources,
                              int fixed_sources, long seed,
                              int *nsources_local, vec2_t **sources_local);
static int exchange_halos(const domain_t *domain, plane_t *plane);
static int output_energy_stat(int step, const plane_t *plane, double budget,
                              const domain_t *domain);
static void print_domain(const domain_t *domain, const plane_t planes[2]);

int main(int argc, char **argv)
{
    int level_obtained;
    int world_rank;
    int ret;
    vec2_t global_size;
    options_t options;
    domain_t domain;
    plane_t planes[2] = {{NULL, {0, 0}}, {NULL, {0, 0}}};
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
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (level_obtained < MPI_THREAD_FUNNELED) {
        if (world_rank == 0)
            fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Finalize();
        return 1;
    }

    ret = parse_options(argc, argv, world_rank, global_size, &options);
    if (ret != 0) {
        if (world_rank == 0 && ret > 0)
            fprintf(stderr, "initialization failed with code %d\n", ret);
        MPI_Finalize();
        return ret > 0 ? ret : 0;
    }

    ret = domain_initialize(MPI_COMM_WORLD, global_size, options.periodic,
                            &domain, planes);
    if (ret != 0) {
        if (world_rank == 0)
            fprintf(stderr, "domain initialization failed with code %d\n", ret);
        MPI_Finalize();
        return ret;
    }

    ret = memory_allocate(planes);
    {
        int any_failure;
        MPI_Allreduce(&ret, &any_failure, 1, MPI_INT, MPI_MAX, domain.cart);
        if (any_failure != 0) {
            if (domain.rank == 0)
                fprintf(stderr, "distributed memory allocation failed\n");
            memory_release(planes);
            MPI_Comm_free(&domain.cart);
            MPI_Finalize();
            return 4;
        }
    }

    {
        const int stride = (int)planes[OLD].size[_x_] + 2;
        MPI_Type_vector((int)planes[OLD].size[_y_], 1, stride,
                        MPI_DOUBLE, &domain.column_type);
        MPI_Type_commit(&domain.column_type);
    }

    ret = initialize_sources(&domain, options.nsources, options.fixed_sources,
                             options.seed, &nsources_local, &sources_local);
    {
        int any_failure;
        MPI_Allreduce(&ret, &any_failure, 1, MPI_INT, MPI_MAX, domain.cart);
        if (any_failure != 0) {
            if (domain.rank == 0)
                fprintf(stderr, "source initialization failed\n");
            free(sources_local);
            memory_release(planes);
            MPI_Type_free(&domain.column_type);
            MPI_Comm_free(&domain.cart);
            MPI_Finalize();
            return 5;
        }
    }

#ifdef _OPENMP
#pragma omp parallel
    {
#pragma omp single
        nthreads = omp_get_num_threads();
    }
#endif

    if (options.verbose)
        print_domain(&domain, planes);

    MPI_Barrier(domain.cart);
    const double t_wall_start = MPI_Wtime();

    for (int iter = 0; iter < options.niterations; iter++) {
        double t0;

        if (iter % options.injection_frequency == 0) {
            t0 = MPI_Wtime();
            inject_energy(options.periodic, nsources_local, sources_local,
                          options.energy_per_source, &planes[current],
                          domain.process_grid);
            t_inject += MPI_Wtime() - t0;
            injected_heat += options.nsources * options.energy_per_source;
        }

        t0 = MPI_Wtime();
        exchange_halos(&domain, &planes[current]);
        t_comm += MPI_Wtime() - t0;

        t0 = MPI_Wtime();
        update_plane(options.periodic, domain.process_grid,
                     &planes[current], &planes[!current]);
        t_update += MPI_Wtime() - t0;

        if (options.output_energy_at_steps) {
            t0 = MPI_Wtime();
            output_energy_stat(iter, &planes[!current], injected_heat, &domain);
            t_energy += MPI_Wtime() - t0;
        }

        current = !current;
    }

    {
        double local_energy;
        double global_energy = 0.0;
        double local_times[5];
        double max_times[5];
        const double t0 = MPI_Wtime();

        get_total_energy(&planes[current], &local_energy);
        MPI_Reduce(&local_energy, &global_energy, 1, MPI_DOUBLE, MPI_SUM,
                   0, domain.cart);
        t_energy += MPI_Wtime() - t0;

        local_times[0] = t_update;
        local_times[1] = t_inject;
        local_times[2] = t_comm;
        local_times[3] = t_energy;
        local_times[4] = MPI_Wtime() - t_wall_start;
        MPI_Reduce(local_times, max_times, 5, MPI_DOUBLE, MPI_MAX,
                   0, domain.cart);

        if (domain.rank == 0) {
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
                   domain.process_grid[_x_], domain.process_grid[_y_],
                   domain.ntasks);
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
            printf("performance      : %.6f GLUP/s   %.6f GFLOP/s\n",
                   glups, gflops);
            printf("bandwidth model  : %.6f GB/s [24 B/update]\n", gbs);
            printf("------------------------------------------------------------\n");
            printf("injected energy  : %g\n", injected_heat);
            printf("system energy    : %g\n", global_energy);
            printf("============================================================\n");
            printf("CSV,%d,%d,%u,%u,%d,%.6e,%.6e,%.6f,%.6f,%.6f\n",
                   domain.ntasks, nthreads, global_size[_x_], global_size[_y_],
                   options.niterations, max_times[0], max_times[2],
                   glups, gflops, gbs);
        }
    }

    free(sources_local);
    memory_release(planes);
    MPI_Type_free(&domain.column_type);
    MPI_Comm_free(&domain.cart);
    MPI_Finalize();
    return 0;
}

static void print_help(void)
{
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
}

static int parse_options(int argc, char **argv, int rank, vec2_t size,
                         options_t *options)
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
    options->injection_frequency = options->niterations;
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
                print_help();
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
        options->nsources < 0 || options->energy_per_source < 0.0) {
        if (rank == 0)
            fprintf(stderr, "invalid parameters\n");
        return 2;
    }

    size[_x_] = (uint)parsed_x;
    size[_y_] = (uint)parsed_y;

    if (frequency == 0.0) {
        options->injection_frequency = 1;
    } else {
        frequency = frequency > 1.0 ? 1.0 : frequency;
        options->injection_frequency =
            (int)(frequency * options->niterations);
        if (options->injection_frequency < 1)
            options->injection_frequency = 1;
    }

    return 0;
}

static int choose_process_grid(const vec2_t size, int ntasks, vec2_t grid)
{
    int found = 0;
    double best_score = 0.0;

    for (int px = 1; px <= ntasks; px++) {
        int py;
        double dx;
        double dy;
        double score;

        if (ntasks % px != 0)
            continue;
        py = ntasks / px;
        if ((uint)px > size[_x_] || (uint)py > size[_y_])
            continue;

        dx = (double)size[_x_] / px;
        dy = (double)size[_y_] / py;
        score = dx > dy ? dx / dy : dy / dx;
        if (!found || score < best_score) {
            grid[_x_] = (uint)px;
            grid[_y_] = (uint)py;
            best_score = score;
            found = 1;
        }
    }

    return found ? 0 : 1;
}

static uint block_size(uint global, uint blocks, uint coordinate)
{
    return global / blocks + (coordinate < global % blocks);
}

static uint block_offset(uint global, uint blocks, uint coordinate)
{
    const uint base = global / blocks;
    const uint remainder = global % blocks;
    return coordinate * base + (coordinate < remainder ? coordinate : remainder);
}

static int domain_initialize(MPI_Comm world, const vec2_t size, int periodic,
                             domain_t *domain, plane_t planes[2])
{
    int cart_dims[2];
    int periods[2] = {periodic, periodic};
    int reorder = 0;
    vec2_t grid;

    MPI_Comm_size(world, &domain->ntasks);
    if (choose_process_grid(size, domain->ntasks, grid) != 0)
        return 3;

    domain->global_size[_x_] = size[_x_];
    domain->global_size[_y_] = size[_y_];
    domain->process_grid[_x_] = grid[_x_];
    domain->process_grid[_y_] = grid[_y_];

    cart_dims[0] = (int)grid[_y_];
    cart_dims[1] = (int)grid[_x_];
    MPI_Cart_create(world, 2, cart_dims, periods, reorder, &domain->cart);
    if (domain->cart == MPI_COMM_NULL)
        return 3;

    MPI_Comm_rank(domain->cart, &domain->rank);
    MPI_Cart_coords(domain->cart, domain->rank, 2, domain->coords);
    MPI_Cart_shift(domain->cart, 0, 1,
                   &domain->neighbours[NORTH], &domain->neighbours[SOUTH]);
    MPI_Cart_shift(domain->cart, 1, 1,
                   &domain->neighbours[WEST], &domain->neighbours[EAST]);

    planes[OLD].size[_x_] = block_size(size[_x_], grid[_x_],
                                      (uint)domain->coords[1]);
    planes[OLD].size[_y_] = block_size(size[_y_], grid[_y_],
                                      (uint)domain->coords[0]);
    planes[NEW].size[_x_] = planes[OLD].size[_x_];
    planes[NEW].size[_y_] = planes[OLD].size[_y_];
    domain->local_offset[_x_] = block_offset(size[_x_], grid[_x_],
                                             (uint)domain->coords[1]);
    domain->local_offset[_y_] = block_offset(size[_y_], grid[_y_],
                                             (uint)domain->coords[0]);
    domain->column_type = MPI_DATATYPE_NULL;

    return 0;
}

static int memory_allocate(plane_t planes[2])
{
    const size_t fxsize = (size_t)planes[OLD].size[_x_] + 2;
    const size_t fysize = (size_t)planes[OLD].size[_y_] + 2;
    const size_t frame = fxsize * fysize;
    double *old_plane = NULL;
    double *new_plane = NULL;

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
    for (size_t idx = 0; idx < frame; idx++) {
        old_plane[idx] = 0.0;
        new_plane[idx] = 0.0;
    }

    planes[OLD].data = old_plane;
    planes[NEW].data = new_plane;
    return 0;
}

static void memory_release(plane_t planes[2])
{
    free(planes[OLD].data);
    free(planes[NEW].data);
    planes[OLD].data = NULL;
    planes[NEW].data = NULL;
}

static int initialize_sources(const domain_t *domain, int nsources,
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
    {
        const int local_failure = (global_sources == NULL);
        int any_failure;
        MPI_Allreduce(&local_failure, &any_failure, 1, MPI_INT, MPI_MAX,
                      domain->cart);
        if (any_failure) {
            free(global_sources);
            return 1;
        }
    }

    if (domain->rank == 0) {
        if (fixed_sources) {
            const uint fixed[4][2] = {
                {domain->global_size[_x_] / 4,
                 domain->global_size[_y_] / 4},
                {3 * domain->global_size[_x_] / 4,
                 domain->global_size[_y_] / 4},
                {domain->global_size[_x_] / 4,
                 3 * domain->global_size[_y_] / 4},
                {3 * domain->global_size[_x_] / 4,
                 3 * domain->global_size[_y_] / 4}
            };

            for (int s = 0; s < nsources; s++) {
                global_sources[s][_x_] = fixed[s % 4][_x_];
                global_sources[s][_y_] = fixed[s % 4][_y_];
                if (global_sources[s][_x_] < 1)
                    global_sources[s][_x_] = 1;
                if (global_sources[s][_y_] < 1)
                    global_sources[s][_y_] = 1;
            }
        } else {
            srand48(seed);
            for (int s = 0; s < nsources; s++) {
                global_sources[s][_x_] =
                    1 + (uint)(lrand48() % domain->global_size[_x_]);
                global_sources[s][_y_] =
                    1 + (uint)(lrand48() % domain->global_size[_y_]);
            }
        }
    }

    MPI_Bcast(global_sources, nsources * 2, MPI_UNSIGNED, 0, domain->cart);

    for (int s = 0; s < nsources; s++) {
        const uint zero_x = global_sources[s][_x_] - 1;
        const uint zero_y = global_sources[s][_y_] - 1;
        if (zero_x >= domain->local_offset[_x_] &&
            zero_x < domain->local_offset[_x_] +
                     (uint)block_size(domain->global_size[_x_],
                                      domain->process_grid[_x_],
                                      (uint)domain->coords[1]) &&
            zero_y >= domain->local_offset[_y_] &&
            zero_y < domain->local_offset[_y_] +
                     (uint)block_size(domain->global_size[_y_],
                                      domain->process_grid[_y_],
                                      (uint)domain->coords[0]))
            nlocal++;
    }

    if (nlocal > 0) {
        int local_index = 0;
        *sources_local = malloc((size_t)nlocal * sizeof(**sources_local));
        if (*sources_local == NULL) {
            free(global_sources);
            return 1;
        }

        for (int s = 0; s < nsources; s++) {
            const uint zero_x = global_sources[s][_x_] - 1;
            const uint zero_y = global_sources[s][_y_] - 1;
            if (zero_x >= domain->local_offset[_x_] &&
                zero_x < domain->local_offset[_x_] +
                         block_size(domain->global_size[_x_],
                                    domain->process_grid[_x_],
                                    (uint)domain->coords[1]) &&
                zero_y >= domain->local_offset[_y_] &&
                zero_y < domain->local_offset[_y_] +
                         block_size(domain->global_size[_y_],
                                    domain->process_grid[_y_],
                                    (uint)domain->coords[0])) {
                (*sources_local)[local_index][_x_] =
                    zero_x - domain->local_offset[_x_] + 1;
                (*sources_local)[local_index][_y_] =
                    zero_y - domain->local_offset[_y_] + 1;
                local_index++;
            }
        }
    }

    *nsources_local = nlocal;
    free(global_sources);
    return 0;
}

static int exchange_halos(const domain_t *domain, plane_t *plane)
{
    const size_t fxsize = (size_t)plane->size[_x_] + 2;
    const uint xsize = plane->size[_x_];
    const uint ysize = plane->size[_y_];
    double *data = plane->data;
    MPI_Request requests[8];
    int count = 0;

#define IDX(i, j) ((size_t)(j) * fxsize + (size_t)(i))
    MPI_Irecv(&data[IDX(1, 0)], (int)xsize, MPI_DOUBLE,
              domain->neighbours[NORTH], TAG_TO_SOUTH,
              domain->cart, &requests[count++]);
    MPI_Irecv(&data[IDX(1, ysize + 1)], (int)xsize, MPI_DOUBLE,
              domain->neighbours[SOUTH], TAG_TO_NORTH,
              domain->cart, &requests[count++]);
    MPI_Irecv(&data[IDX(0, 1)], 1, domain->column_type,
              domain->neighbours[WEST], TAG_TO_EAST,
              domain->cart, &requests[count++]);
    MPI_Irecv(&data[IDX(xsize + 1, 1)], 1, domain->column_type,
              domain->neighbours[EAST], TAG_TO_WEST,
              domain->cart, &requests[count++]);

    MPI_Isend(&data[IDX(1, 1)], (int)xsize, MPI_DOUBLE,
              domain->neighbours[NORTH], TAG_TO_NORTH,
              domain->cart, &requests[count++]);
    MPI_Isend(&data[IDX(1, ysize)], (int)xsize, MPI_DOUBLE,
              domain->neighbours[SOUTH], TAG_TO_SOUTH,
              domain->cart, &requests[count++]);
    MPI_Isend(&data[IDX(1, 1)], 1, domain->column_type,
              domain->neighbours[WEST], TAG_TO_WEST,
              domain->cart, &requests[count++]);
    MPI_Isend(&data[IDX(xsize, 1)], 1, domain->column_type,
              domain->neighbours[EAST], TAG_TO_EAST,
              domain->cart, &requests[count++]);

    MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
#undef IDX
    return 0;
}

static int output_energy_stat(int step, const plane_t *plane, double budget,
                              const domain_t *domain)
{
    double local_energy;
    double global_energy = 0.0;

    get_total_energy(plane, &local_energy);
    MPI_Reduce(&local_energy, &global_energy, 1, MPI_DOUBLE, MPI_SUM,
               0, domain->cart);

    if (domain->rank == 0) {
        const double points = (double)domain->global_size[_x_] *
                              (double)domain->global_size[_y_];
        printf("step %d :: injected energy is %g, system energy is %g "
               "(average %g per grid point)\n",
               step, budget, global_energy, global_energy / points);
    }
    return 0;
}

static void print_domain(const domain_t *domain, const plane_t planes[2])
{
    if (domain->rank == 0) {
        printf("MPI process grid: %u x %u\n",
               domain->process_grid[_x_], domain->process_grid[_y_]);
        fflush(stdout);
    }

    MPI_Barrier(domain->cart);
    for (int task = 0; task < domain->ntasks; task++) {
        if (task == domain->rank) {
            printf("rank %4d coords=(%d,%d) offset=(%u,%u) size=(%u,%u) "
                   "N=%d S=%d E=%d W=%d\n",
                   domain->rank, domain->coords[1], domain->coords[0],
                   domain->local_offset[_x_], domain->local_offset[_y_],
                   planes[OLD].size[_x_], planes[OLD].size[_y_],
                   domain->neighbours[NORTH], domain->neighbours[SOUTH],
                   domain->neighbours[EAST], domain->neighbours[WEST]);
            fflush(stdout);
        }
        MPI_Barrier(domain->cart);
    }
}
