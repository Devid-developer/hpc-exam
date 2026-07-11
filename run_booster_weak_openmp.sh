#!/bin/bash
#SBATCH --job-name=hpc_stencil_weak
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --account=IscrB_SPIESMD
#SBATCH --partition=boost_usr_prod
#SBATCH --output=hpc_stencil_weak_%j.bootstrap.out
#SBATCH --error=hpc_stencil_weak_%j.bootstrap.err

set -euo pipefail

# Submit from the repository root: DEV/hpc-exam.
PROJECT_DIR=${SLURM_SUBMIT_DIR:?Submit the job from the hpc-exam directory}
RESULTS_ROOT=${RESULTS_ROOT:-${PROJECT_DIR}/results}
RUN_DIR=${RESULTS_ROOT}/${SLURM_JOB_ID}_weak_openmp
RAW_DIR=${RUN_DIR}/raw
MEASUREMENTS=${RUN_DIR}/weak_scaling_measurements.csv
SUMMARY=${RUN_DIR}/weak_scaling_summary.csv

mkdir -p "${RAW_DIR}"
cd "${PROJECT_DIR}"

exec > >(tee "${RUN_DIR}/job.out") 2> >(tee "${RUN_DIR}/job.err" >&2)

# BASE_SIDE^2 is the number of physical grid cells assigned to each thread.
# With BASE_SIDE=5000, every thread updates 25 million cells per iteration.
BASE_SIDE=${BASE_SIDE:-5000}
ITERATIONS=${ITERATIONS:-100}
SOURCES=${SOURCES:-4}
PERIODIC=${PERIODIC:-1}
OMP_THREAD_LIST=${OMP_THREAD_LIST:-"1 2 4 8 16 32"}

OPENMP_BIN=${PROJECT_DIR}/build/stencil_serial_final_omp

grid_for_threads()
{
    local threads=$1

    # Alternate growth along x and y to keep the global domain square or 2:1.
    case "${threads}" in
        1)  GRID_X=${BASE_SIDE};       GRID_Y=${BASE_SIDE} ;;
        2)  GRID_X=$((2*BASE_SIDE));   GRID_Y=${BASE_SIDE} ;;
        4)  GRID_X=$((2*BASE_SIDE));   GRID_Y=$((2*BASE_SIDE)) ;;
        8)  GRID_X=$((4*BASE_SIDE));   GRID_Y=$((2*BASE_SIDE)) ;;
        16) GRID_X=$((4*BASE_SIDE));   GRID_Y=$((4*BASE_SIDE)) ;;
        32) GRID_X=$((8*BASE_SIDE));   GRID_Y=$((4*BASE_SIDE)) ;;
        *)
            echo "Unsupported thread count for weak scaling: ${threads}" >&2
            return 1
            ;;
    esac
}

save_run()
{
    local threads=$1
    local grid_x=$2
    local grid_y=$3
    shift 3

    local run_name=weak_t$(printf '%03d' "${threads}")
    local logfile=${RAW_DIR}/${run_name}.log
    local wall_time
    local glups
    local total_cells=$((grid_x * grid_y))
    local cells_per_thread=$((total_cells / threads))

    echo "RUN ${run_name}: grid=${grid_x}x${grid_y}, cells/thread=${cells_per_thread}"
    "$@" >"${logfile}" 2>&1

    wall_time=$(awk '/t_wall total/{print $4}' "${logfile}")
    glups=$(awk -F, '/^CSV,/{print $7}' "${logfile}")

    if [[ -z "${wall_time}" || -z "${glups}" ]]; then
        echo "Unable to extract t_wall or GLUP/s from ${logfile}" >&2
        return 1
    fi

    printf '%s,%d,%d,%d,%d,%d,%s,%s\n' \
        "${run_name}" "${threads}" "${grid_x}" "${grid_y}" \
        "${total_cells}" "${cells_per_thread}" "${wall_time}" "${glups}" \
        >>"${MEASUREMENTS}"
}

echo "Job ID       : ${SLURM_JOB_ID}"
echo "Partition    : ${SLURM_JOB_PARTITION:-unknown}"
echo "Node list    : ${SLURM_JOB_NODELIST:-unknown}"
echo "Project dir  : ${PROJECT_DIR}"
echo "Results dir  : ${RUN_DIR}"
echo "Base side    : ${BASE_SIDE}"
echo "Cells/thread : $((BASE_SIDE * BASE_SIDE))"
echo "Iterations   : ${ITERATIONS}"
echo "Threads      : ${OMP_THREAD_LIST}"
echo "Binding      : spread"
echo "MPI          : disabled"

{
    echo "date=$(date --iso-8601=seconds)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "partition=${SLURM_JOB_PARTITION:-unknown}"
    echo "nodes=${SLURM_JOB_NODELIST:-unknown}"
    echo "submit_dir=${PROJECT_DIR}"
    echo "base_side=${BASE_SIDE}"
    echo "cells_per_thread=$((BASE_SIDE * BASE_SIDE))"
    echo "iterations=${ITERATIONS}"
    echo "sources=${SOURCES}"
    echo "periodic=${PERIODIC}"
    echo "omp_threads=${OMP_THREAD_LIST}"
    echo "omp_places=cores"
    echo "omp_proc_bind=spread"
    echo "mpi=disabled"
    echo "runs_per_configuration=1"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "cc=$(command -v gcc || true)"
    gcc --version 2>/dev/null | head -n 1 || true
} >"${RUN_DIR}/environment.txt"

module -t list >"${RUN_DIR}/modules.txt" 2>&1 || true
scontrol show job "${SLURM_JOB_ID}" >"${RUN_DIR}/slurm_job.txt"
srun --nodes=1 --ntasks=1 --cpus-per-task=1 lscpu >"${RUN_DIR}/lscpu.txt"
git diff >"${RUN_DIR}/source_changes.patch" || true

echo "Building the optimized OpenMP executable"
make clean
make openmp 2>&1 | tee "${RUN_DIR}/build.log"

printf 'run,threads,grid_x,grid_y,total_cells,cells_per_thread,t_wall_s,glups_kernel\n' \
    >"${MEASUREMENTS}"

for threads in ${OMP_THREAD_LIST}; do
    grid_for_threads "${threads}"

    RUN_ARGS=(
        -x "${GRID_X}"
        -y "${GRID_Y}"
        -n "${ITERATIONS}"
        -e "${SOURCES}"
        -p "${PERIODIC}"
        -F
        -o 0
    )

    save_run "${threads}" "${GRID_X}" "${GRID_Y}" \
        env OMP_NUM_THREADS="${threads}" OMP_PLACES=cores OMP_PROC_BIND=spread \
        srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=32 \
        --cpu-bind=cores "${OPENMP_BIN}" "${RUN_ARGS[@]}"
done

# Weak-scaling efficiency is T(1)/T(P); ideal weak scaling is 1.0 (100%).
awk -F, 'BEGIN {OFS=","}
    NR==1 {print $0,"weak_efficiency","weak_efficiency_percent"; next}
    NR==2 {reference=$7}
    {efficiency=reference/$7; print $0,efficiency,100*efficiency}' \
    "${MEASUREMENTS}" >"${SUMMARY}"

sacct -j "${SLURM_JOB_ID}" \
    --format=JobID,JobName,Partition,State,Elapsed,NNodes,NTasks,AllocCPUS,MaxRSS \
    >"${RUN_DIR}/sacct.txt" || true

echo "All weak-scaling runs completed"
echo "Summary: ${SUMMARY}"
cat "${SUMMARY}"
