# HPC 5-Point Stencil

Optimization project for a two-dimensional five-point stencil developed for a
High Performance Computing course. The program models energy diffusion over a
grid and is progressively optimized through serial, OpenMP, MPI, and hybrid
MPI+OpenMP implementations.

The main benchmarks were run on the Booster module of Leonardo (CINECA),
covering serial performance, cache behavior, strong and weak scaling, and
different MPI rank/OpenMP thread configurations.

## Requirements

- a C compiler with OpenMP support, such as GCC;
- GNU Make;
- an MPI implementation providing `mpicc` and `mpirun`;
- Python 3 with `numpy`, `pandas`, and `matplotlib` for plotting;
- Slurm to run the benchmark scripts on Leonardo.

Compilation uses `-O3 -march=native -fopenmp`; the resulting binaries are not
necessarily portable across different CPU architectures.

## Repository layout

```text
src/                     serial and MPI source files
include/                 kernels and associated headers
sbatch/                  Slurm jobs for serial, OpenMP, MPI, and hybrid runs
results/                 CSV files collected on Leonardo
plots/                   figures generated from the results
Makefile                 build targets and local run helpers
plots.py                 plot-generation script
documentazione_opt.md     detailed optimization and benchmark report
```

The `stencil_template_*` files are the project baselines. The optimized
implementations are `stencil_serial_final.c` and `stencil_parallel_final.c`.

## Building

```bash
make all             # baseline and final serial/OpenMP builds
make final-serial    # final serial version with -O1 and -O3
make omp-serial      # final OpenMP version
make mpi             # final MPI+OpenMP version
make clean
```

Executables are written to `build/`.

## Local execution

A small grid is sufficient for a functional check:

```bash
./build/stencil_serial_final_O3 -x 100 -y 100 -n 100 -o 0

OMP_NUM_THREADS=4 OMP_PLACES=cores OMP_PROC_BIND=spread \
    ./build/stencil_serial_final_omp_O3 -x 100 -y 100 -n 100 -o 0

OMP_NUM_THREADS=2 mpirun -np 4 \
    ./build/stencil_parallel_final_mpi_O3 -x 100 -y 100 -n 100 -o 0
```

The main options are `-x` and `-y` for the grid dimensions, `-n` for the
number of iterations, and `-o` for intermediate energy output.

## Slurm benchmarks

Submit the scripts from the repository root:

```bash
sbatch sbatch/go_omp_serial.sh
sbatch sbatch/go_mpi_smoke.sh
sbatch sbatch/go_mpi_node_scaling.sh
```

Each job creates a dedicated directory under `results/`, containing its summary
CSV and, when run on Leonardo, raw output, logs, Slurm configuration, and
environment information. The account, partition, and node count in the
`#SBATCH` headers may need to be adapted to the available CINECA project.

## Plots

Regenerate all figures from the collected CSV files with:

```bash
python3 plots.py --results-dir results --output-dir plots
```

See `documentazione_opt.md` for the complete discussion of implementation
choices, benchmark methodology, and results.
