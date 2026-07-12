/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */

#ifndef STENCIL_PARALLEL_FINAL_H
#define STENCIL_PARALLEL_FINAL_H

#include <stddef.h>

#define NORTH 0
#define SOUTH 1
#define EAST  2
#define WEST  3

#define SEND 0
#define RECV 1

#define OLD 0
#define NEW 1

#define _x_ 0
#define _y_ 1

#define ALPHA     0.6
#define C_CENTER  (ALPHA)
#define C_NEIGH   (0.25 * (1.0 - ALPHA))

typedef unsigned int uint;
typedef uint vec2_t[2];
typedef double *restrict buffers_t[4];

typedef struct {
    double *restrict data;
    vec2_t size;
} plane_t;

static inline int inject_energy(const int periodic,
                                const int nsources,
                                vec2_t *sources,
                                const double energy,
                                plane_t *plane,
                                const vec2_t process_grid)
{
    const size_t fxsize = (size_t)plane->size[_x_] + 2;
    double *restrict data = plane->data;

#define IDX(i, j) ((size_t)(j) * fxsize + (size_t)(i))
    for (int s = 0; s < nsources; s++) {
        const uint x = sources[s][_x_];
        const uint y = sources[s][_y_];
        data[IDX(x, y)] += energy;

        /* A direction represented by one rank applies its periodic boundary
           locally.  Multi-rank directions are propagated by halo exchange. */
        if (periodic) {
            if (process_grid[_x_] == 1) {
                if (x == 1)
                    data[IDX(plane->size[_x_] + 1, y)] += energy;
                if (x == plane->size[_x_])
                    data[IDX(0, y)] += energy;
            }
            if (process_grid[_y_] == 1) {
                if (y == 1)
                    data[IDX(x, plane->size[_y_] + 1)] += energy;
                if (y == plane->size[_y_])
                    data[IDX(x, 0)] += energy;
            }
        }
    }
#undef IDX

    return 0;
}

static inline int update_plane(const int periodic,
                               const vec2_t process_grid,
                               const plane_t *oldplane,
                               plane_t *newplane)
{
    const size_t fxsize = (size_t)oldplane->size[_x_] + 2;
    const uint xsize = oldplane->size[_x_];
    const uint ysize = oldplane->size[_y_];
    const double cc = C_CENTER;
    const double cn = C_NEIGH;
    const double *restrict old = oldplane->data;
    double *restrict new = newplane->data;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (uint j = 1; j <= ysize; j++) {
        const double *restrict center = old + (size_t)j * fxsize;
        const double *restrict north = old + (size_t)(j - 1) * fxsize;
        const double *restrict south = old + (size_t)(j + 1) * fxsize;
        double *restrict result = new + (size_t)j * fxsize;

        for (uint i = 1; i <= xsize; i++)
            result[i] = cc * center[i] +
                        cn * (north[i] + south[i] +
                              center[i - 1] + center[i + 1]);
    }

    /* Multi-rank directions are refreshed by halo exchange at the beginning
       of the next iteration.  One-rank directions are completed locally. */
    if (periodic) {
#define IDX(i, j) ((size_t)(j) * fxsize + (size_t)(i))
        if (process_grid[_y_] == 1) {
            for (uint i = 1; i <= xsize; i++) {
                new[IDX(i, 0)] = new[IDX(i, ysize)];
                new[IDX(i, ysize + 1)] = new[IDX(i, 1)];
            }
        }

        if (process_grid[_x_] == 1) {
            for (uint j = 1; j <= ysize; j++) {
                new[IDX(0, j)] = new[IDX(xsize, j)];
                new[IDX(xsize + 1, j)] = new[IDX(1, j)];
            }
        }
#undef IDX
    }

    return 0;
}

static inline int get_total_energy(const plane_t *plane, double *energy)
{
    const size_t fxsize = (size_t)plane->size[_x_] + 2;
    const uint xsize = plane->size[_x_];
    const uint ysize = plane->size[_y_];

#if defined(LONG_ACCURACY)
    long double total = 0.0L;
#else
    double total = 0.0;
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:total)
#endif
    for (uint j = 1; j <= ysize; j++) {
        const double *line = plane->data + (size_t)j * fxsize;
        for (uint i = 1; i <= xsize; i++)
            total += line[i];
    }

    *energy = (double)total;
    return 0;
}

#endif /* STENCIL_PARALLEL_FINAL_H */
