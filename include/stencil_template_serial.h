/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>
#include <float.h>
#include <math.h>



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
#define C_NEIGH   (0.25*(1.0-ALPHA))   // = 0.1


// ============================================================
//
// function prototypes

int initialize ( int      ,
		 char   **,
		 int    *,
		 int     *,
		 int     *,
		 int     *,
		 int   **,
		 double  *,
		 double **,
                 int     *,
                 int     *
		 );

int memory_release ( double *, int * );


static inline int inject_energy ( const  int,
                           const int    ,
			   const int   *,
			   const double  ,
			   const int    [2],
                                 double * );

static inline int update_plane ( const int       ,
			  const int    [2],
			  const double   *,
		                double   * );


static inline int get_total_energy( const int     [2],
                             const double *,
                             double * );


// ============================================================
//
// function definition for inline functions

static inline int inject_energy(const int     periodic,
                                const int     Nsources,
                                const int    *Sources,
                                const double  energy,
                                const int     size[2],
                                double       *plane)
{
    const int fxsize = size[_x_] + 2;

#define IDX(i,j) ( (size_t)(j)*fxsize + (i) )

    for (int s = 0; s < Nsources; s++) {
        const int x = Sources[2*s];
        const int y = Sources[2*s+1];
        plane[IDX(x,y)] += energy;

        if (periodic) {
            if (x == 1)             plane[IDX(size[_x_]+1, y)] += energy;
            if (x == size[_x_])     plane[IDX(0, y)]           += energy;
            if (y == 1)             plane[IDX(x, size[_y_]+1)] += energy;
            if (y == size[_y_])     plane[IDX(x, 0)]           += energy;
        }
    }
#undef IDX
    return 0;
}


static inline int update_plane ( const int     periodic, 
                          const int     size[2],
			              const double *restrict old    ,
                          double       *restrict new    )

{
    const int  fxsize = size[_x_]+2;
    const int  xsize = size[_x_];
    const int  ysize = size[_y_];
    const double cc = C_CENTER;
    const double cn = C_NEIGH;
    


    for (int j = 1; j <= ysize; j++) {
        const double *restrict center = old + (size_t)j      *fxsize;
        const double *restrict north  = old + (size_t)(j-1)  *fxsize;
        const double *restrict south  = old + (size_t)(j+1)  *fxsize;
        double *restrict result       = new + (size_t)j      *fxsize;

        for ( int i = 1; i <= xsize; i++)
            result[i] = cc*center[i] + cn*(north[i]+south[i]+center[i-1]+center[i+1]);
    }

    if ( periodic ) {

       #define IDX( i, j ) ( (size_t)(j)*fxsize + (i) )

            for ( int i = 1; i <= xsize; i++ ) {
                    new[ IDX(i, 0) ]        = new[ IDX(i, ysize) ];
                    new[ IDX(i, ysize+1) ]  = new[ IDX(i, 1) ];
                }

            for ( int j = 1; j <= ysize; j++ ) {
                    new[ IDX( 0, j) ]       = new[ IDX(xsize, j) ];
                    new[ IDX( xsize+1, j) ] = new[ IDX(1, j) ];
                }
                
   #undef IDX
        }
    
    return 0;
}


static inline int get_total_energy( const int     size[2],
                             const double *restrict plane,
                                   double *energy )

{
    const int fxsize = size[_x_] + 2;
    const int xsize  = size[_x_];
    const int ysize  = size[_y_];

#if defined(LONG_ACCURACY)
    long double tot = 0.0L;
#else
    double tot = 0.0;
#endif

    for ( int j = 1; j <= ysize; j++ ) {
        const double *line = plane + (size_t)j*fxsize;
        for ( int i = 1; i <= xsize; i++ )
            tot += line[i];
    }


    *energy = (double)tot;
    return 0;
}
                            
