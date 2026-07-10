/* -----------------------------------------------------------------------------
 * stencil_serial_final.c -- serial driver for the 5-point stencil.
 *
 * Final serial implementation, with:
 * - aligned allocation with posix_memalign;
 * - separate OLD/NEW allocations, released separately;
 * - wall-clock timing for update, injection, and energy reduction;
 * - final performance report and CSV line;
 * - fixed dump stride for the halo layout;
 * - no initial uncounted energy injection before the iteration loop.
 * --------------------------------------------------------------------------- */

#define _XOPEN_SOURCE 700
#define _POSIX_C_SOURCE 200809L

#include "stencil_serial_final.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#define MEM_ALIGN 64

static double wtime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int memory_allocate(const int size[2], double **planes);
static int initialize_sources(const int size[2],
                              int nsources,
                              int fixed_sources,
                              long seed,
                              int **sources);
static int memory_release_proto(double **planes, int *sources);
static int dump_proto(const double *data, const int size[2], const char *filename);

static int initialize_proto(int argc,
                            char **argv,
                            int *size,
                            int *periodic,
                            int *niterations,
                            int *nsources,
                            int **sources,
                            double *energy_per_source,
                            double **planes,
                            int *output_energy_at_steps,
                            int *injection_frequency,
                            int *fixed_sources,
                            long *seed)
{
    int opt;
    double freq = 0.0;

    size[_x_] = 1000;
    size[_y_] = 1000;
    *periodic = 0;
    *nsources = 1;
    *niterations = 99;
    *output_energy_at_steps = 0;
    *energy_per_source = 1.0;
    *injection_frequency = *niterations;
    *fixed_sources = 0;
    *seed = 0;

    while ((opt = getopt(argc, argv, ":x:y:e:E:f:n:p:o:Fs:h")) != -1) {
        switch (opt) {
        case 'x':
            size[_x_] = atoi(optarg);
            break;
        case 'y':
            size[_y_] = atoi(optarg);
            break;
        case 'e':
            *nsources = atoi(optarg);
            break;
        case 'E':
            *energy_per_source = atof(optarg);
            break;
        case 'f':
            freq = atof(optarg);
            break;
        case 'n':
            *niterations = atoi(optarg);
            break;
        case 'p':
            *periodic = (atoi(optarg) > 0);
            break;
        case 'o':
            *output_energy_at_steps = (atoi(optarg) > 0);
            break;
        case 'F':
            *fixed_sources = 1;
            break;
        case 's':
            *seed = atol(optarg);
            break;
        case 'h':
            printf("valid options are:\n"
                   "-x    x size of the plate [1000]\n"
                   "-y    y size of the plate [1000]\n"
                   "-e    number of energy sources [1]\n"
                   "-E    energy per source [1.0]\n"
                   "-f    injection frequency as fraction of iterations [0.0]\n"
                   "-n    number of iterations [99]\n"
                   "-p    periodic boundaries [0]\n"
                   "-o    output energy at every step [0]\n"
                   "-F    use fixed quarter-position sources for validation\n"
                   "-s    seed for random sources [0]\n");
            return 1;
        case ':':
            fprintf(stderr, "option -%c requires an argument\n", optopt);
            return 1;
        case '?':
            fprintf(stderr, "unknown option -%c\n", optopt);
            return 1;
        }
    }

    if (freq == 0.0) {
        *injection_frequency = 1;
    } else {
        freq = (freq > 1.0 ? 1.0 : freq);
        *injection_frequency = (int)(freq * *niterations);
        if (*injection_frequency < 1)
            *injection_frequency = 1;
    }

    if (size[_x_] <= 0 || size[_y_] <= 0 || *niterations <= 0 ||
        *nsources < 0 || *energy_per_source < 0.0) {
        fprintf(stderr, "invalid parameters\n");
        return 2;
    }

    if (memory_allocate(size, planes) != 0)
        return 3;

    if (initialize_sources(size, *nsources, *fixed_sources, *seed, sources) != 0) {
        memory_release_proto(planes, NULL);
        return 4;
    }

    return 0;
}

int main(int argc, char **argv)
{
    int niterations;
    int periodic;
    int size[2];
    int nsources;
    int *sources = NULL;
    double energy_per_source;
    double *planes[2] = {NULL, NULL};
    double injected_heat = 0.0;
    int injection_frequency;
    int output_energy_at_steps = 0;
    int fixed_sources = 0;
    long seed = 0;
    int current = OLD;

    double t_update = 0.0;
    double t_inject = 0.0;
    double t_energy = 0.0;
    double t0;

    int nthreads = 1;

    int ret = initialize_proto(argc, argv, size, &periodic, &niterations,
                               &nsources, &sources, &energy_per_source,
                               planes, &output_energy_at_steps,
                               &injection_frequency, &fixed_sources, &seed);
    if (ret != 0) {
        fprintf(stderr, "initialization failed with code %d\n", ret);
        return ret;
    }

#ifdef _OPENMP
#pragma omp parallel
    {
#pragma omp single
        nthreads = omp_get_num_threads();
    }
#endif

    const double t_total0 = wtime();

    for (int iter = 0; iter < niterations; iter++) {
        if (iter % injection_frequency == 0) {
            t0 = wtime();
            inject_energy(periodic, nsources, sources, energy_per_source,
                          size, planes[current]);
            t_inject += wtime() - t0;
            injected_heat += nsources * energy_per_source;
        }

        t0 = wtime();
        update_plane(periodic, size, planes[current], planes[!current]);
        t_update += wtime() - t0;

        if (output_energy_at_steps) {
            double system_heat;
            char filename[100];

            t0 = wtime();
            get_total_energy(size, planes[!current], &system_heat);
            t_energy += wtime() - t0;

            printf("step %d :: injected energy is %g, updated system energy is %g\n",
                   iter, injected_heat, system_heat);

            sprintf(filename, "plane_%05d.bin", iter);
            dump_proto(planes[!current], size, filename);
        }

        current = !current;
    }

    double system_heat;
    t0 = wtime();
    get_total_energy(size, planes[current], &system_heat);
    t_energy += wtime() - t0;

    const double t_total = wtime() - t_total0;
    const double updates = (double)size[_x_] * (double)size[_y_] *
                           (double)niterations;
    const double glups = updates / t_update / 1e9;
    const double gflops = 6.0 * updates / t_update / 1e9;
    const double bytes_per_update = 24.0;
    const double gbs = bytes_per_update * updates / t_update / 1e9;

    printf("\n============================================================\n");
    printf("grid             : %d x %d (+2 halo)\n", size[_x_], size[_y_]);
    printf("iterations       : %d   sources: %d   periodic: %d\n",
           niterations, nsources, periodic);
    printf("source mode      : %s", fixed_sources ? "fixed quarter positions" : "random");
    if (!fixed_sources)
        printf(" (seed %ld)", seed);
    printf("\n");
    printf("OpenMP threads   : %d\n", nthreads);
    printf("------------------------------------------------------------\n");
    printf("t_update total   : %.6f s   (%.6e s/iter)\n",
           t_update, t_update / niterations);
    printf("t_inject total   : %.6f s\n", t_inject);
    printf("t_energy total   : %.6f s\n", t_energy);
    printf("t_wall total     : %.6f s\n", t_total);
    printf("------------------------------------------------------------\n");
    printf("performance      : %.6f GLUP/s   %.6f GFLOP/s\n", glups, gflops);
    printf("bandwidth model  : %.6f GB/s [standard 24 B/update]\n", gbs);
    printf("------------------------------------------------------------\n");
    printf("injected energy  : %g\n", injected_heat);
    printf("system energy    : %g\n", system_heat);
    printf("============================================================\n");
    printf("CSV,%d,%d,%d,%d,%.6e,%.6f,%.6f,%.6f\n",
           nthreads, size[_x_], size[_y_], niterations,
           t_update, glups, gflops, gbs);

    memory_release_proto(planes, sources);
    return 0;
}

static int memory_allocate(const int size[2], double **planes)
{
    const size_t fxsize = (size_t)size[_x_] + 2;
    const size_t fysize = (size_t)size[_y_] + 2;
    const size_t frame = fxsize * fysize;
    double *old_plane = NULL;
    double *new_plane = NULL;

    if (planes == NULL)
        return 1;

    if (posix_memalign((void **)&old_plane, MEM_ALIGN,
                       frame * sizeof(double)) != 0)
        return 2;

    if (posix_memalign((void **)&new_plane, MEM_ALIGN,
                       frame * sizeof(double)) != 0) {
        free(old_plane);
        return 2;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int j = 0; j < (int)fysize; j++) {
        for (int i = 0; i < (int)fxsize; i++) {
            const size_t idx = (size_t)j * fxsize + (size_t)i;
            old_plane[idx] = 0.0;
            new_plane[idx] = 0.0;
        }
    }

    planes[OLD] = old_plane;
    planes[NEW] = new_plane;
    return 0;
}

static int initialize_sources(const int size[2],
                              int nsources,
                              int fixed_sources,
                              long seed,
                              int **sources)
{
    if (sources == NULL)
        return 1;

    *sources = NULL;
    if (nsources == 0)
        return 0;

    *sources = (int *)malloc((size_t)nsources * 2 * sizeof(int));
    if (*sources == NULL)
        return 2;

    if (fixed_sources) {
        const int fixed[4][2] = {
            {size[_x_] / 4,     size[_y_] / 4},
            {3 * size[_x_] / 4, size[_y_] / 4},
            {size[_x_] / 4,     3 * size[_y_] / 4},
            {3 * size[_x_] / 4, 3 * size[_y_] / 4}
        };

        for (int s = 0; s < nsources; s++) {
            (*sources)[2 * s] = fixed[s % 4][_x_];
            (*sources)[2 * s + 1] = fixed[s % 4][_y_];

            if ((*sources)[2 * s] < 1)
                (*sources)[2 * s] = 1;
            if ((*sources)[2 * s + 1] < 1)
                (*sources)[2 * s + 1] = 1;
        }
    } else {
        srand48(seed);
        for (int s = 0; s < nsources; s++) {
            (*sources)[2 * s] = 1 + (int)(lrand48() % size[_x_]);
            (*sources)[2 * s + 1] = 1 + (int)(lrand48() % size[_y_]);
        }
    }

    return 0;
}

static int memory_release_proto(double **planes, int *sources)
{
    if (planes != NULL) {
        free(planes[OLD]);
        free(planes[NEW]);
    }

    free(sources);
    return 0;
}

static int dump_proto(const double *data, const int size[2], const char *filename)
{
    const int fxsize = size[_x_] + 2;
    float *row;
    FILE *outfile;

    if (data == NULL || filename == NULL || filename[0] == '\0')
        return 1;

    outfile = fopen(filename, "wb");
    if (outfile == NULL)
        return 2;

    row = (float *)malloc((size_t)size[_x_] * sizeof(float));
    if (row == NULL) {
        fclose(outfile);
        return 3;
    }

    for (int j = 1; j <= size[_y_]; j++) {
        const double *line = data + (size_t)j * fxsize + 1;
        for (int i = 0; i < size[_x_]; i++)
            row[i] = (float)line[i];

        fwrite(row, sizeof(float), (size_t)size[_x_], outfile);
    }

    free(row);
    fclose(outfile);
    return 0;
}
