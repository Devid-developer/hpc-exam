/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */

#ifndef STENCIL_TEMPLATE_SERIAL_H
#define STENCIL_TEMPLATE_SERIAL_H

#include <stddef.h>

#define OLD 0
#define NEW 1

#define _x_ 0
#define _y_ 1

/* Baseline kernel from the original serial template.  The static linkage is
   only a C linkage fix; the arithmetic and indexing remain unoptimized. */
static inline int inject_energy(const int periodic,
                                const int nsources,
                                const int *sources,
                                const double energy,
                                const int size[2],
                                double *plane)
{
#define IDX(i, j) ((j) * (size[_x_] + 2) + (i))
    for (int s = 0; s < nsources; s++) {
        const int x = sources[2 * s];
        const int y = sources[2 * s + 1];
        plane[IDX(x, y)] += energy;

        if (periodic) {
            if (x == 1)
                plane[IDX(size[_x_] + 1, y)] += energy;
            if (x == size[_x_])
                plane[IDX(0, y)] += energy;
            if (y == 1)
                plane[IDX(x, size[_y_] + 1)] += energy;
            if (y == size[_y_])
                plane[IDX(x, 0)] += energy;
        }
    }
#undef IDX
    return 0;
}

static inline int update_plane(const int periodic,
                               const int size[2],
                               const double *old,
                               double *new)
{
    const int fxsize = size[_x_] + 2;
    const int xsize = size[_x_];
    const int ysize = size[_y_];

#define IDX(i, j) ((j) * fxsize + (i))
    for (int j = 1; j <= ysize; j++) {
        for (int i = 1; i <= xsize; i++) {
            const double alpha = 0.6;
            double result = old[IDX(i, j)] * alpha;
            const double sum_i =
                (old[IDX(i - 1, j)] + old[IDX(i + 1, j)]) /
                4.0 * (1.0 - alpha);
            const double sum_j =
                (old[IDX(i, j - 1)] + old[IDX(i, j + 1)]) /
                4.0 * (1.0 - alpha);
            result += sum_i + sum_j;
            new[IDX(i, j)] = result;
        }
    }

    if (periodic) {
        for (int i = 1; i <= xsize; i++) {
            new[IDX(i, 0)] = new[IDX(i, ysize)];
            new[IDX(i, ysize + 1)] = new[IDX(i, 1)];
        }
        for (int j = 1; j <= ysize; j++) {
            new[IDX(0, j)] = new[IDX(xsize, j)];
            new[IDX(xsize + 1, j)] = new[IDX(1, j)];
        }
    }
#undef IDX
    return 0;
}

static inline int get_total_energy(const int size[2],
                                   const double *plane,
                                   double *energy)
{
    const int xsize = size[_x_];
    double total = 0.0;

#define IDX(i, j) ((j) * (xsize + 2) + (i))
    for (int j = 1; j <= size[_y_]; j++)
        for (int i = 1; i <= size[_x_]; i++)
            total += plane[IDX(i, j)];
#undef IDX

    *energy = total;
    return 0;
}

#endif /* STENCIL_TEMPLATE_SERIAL_H */
