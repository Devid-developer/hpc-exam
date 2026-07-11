#!/bin/bash
#SBATCH --job-name=hpc_stencil_complete
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --account=IscrB_SPIESMD
#SBATCH --partition=boost_usr_prod
#SBATCH --output=hpc_stencil_complete_%j.bootstrap.out
#SBATCH --error=hpc_stencil_complete_%j.bootstrap.err

set -euo pipefail

# Submit from the repository root: DEV/hpc-exam.
PROJECT_DIR=${SLURM_SUBMIT_DIR:?Submit the job from the hpc-exam directory}
RESULTS_ROOT=${RESULTS_ROOT:-${PROJECT_DIR}/results}
RUN_DIR=${RESULTS_ROOT}/${SLURM_JOB_ID}_complete_no_mpi
RAW_DIR=${RUN_DIR}/raw
SUMMARY=${RUN_DIR}/benchmark_summary.csv

mkdir -p "${RAW_DIR}"
cd "${PROJECT_DIR}"

exec > >(tee "${RUN_DIR}/job.out") 2> >(tee "${RUN_DIR}/job.err" >&2)

GRID_X=${GRID_X:-20000}
GRID_Y=${GRID_Y:-20000}
ITERATIONS=${ITERATIONS:-100}
SOURCES=${SOURCES:-4}
PERIODIC=${PERIODIC:-1}
OMP_THREAD_LIST=${OMP_THREAD_LIST:-"2 4 8 16 24 32"}

RUN_ARGS=(
    -x "${GRID_X}"
    -y "${GRID_Y}"
    -n "${ITERATIONS}"
    -e "${SOURCES}"
    -p "${PERIODIC}"
    -F
    -o 0
)

BASE_BIN=${PROJECT_DIR}/build/stencil_serial_base
SERIAL_BIN=${PROJECT_DIR}/build/stencil_serial_final
OPENMP_BIN=${PROJECT_DIR}/build/stencil_serial_final_omp

save_run()
{
    local run_name=$1
    shift

    local logfile=${RAW_DIR}/${run_name}.log
    local wall_time
    local glups

    echo "RUN ${run_name}"
    "$@" >"${logfile}" 2>&1

    wall_time=$(awk '/t_wall total/{print $4}' "${logfile}")
    glups=$(awk -F, '/^CSV,/{print $7}' "${logfile}")

    if [[ -z "${wall_time}" || -z "${glups}" ]]; then
        echo "Unable to extract t_wall or GLUP/s from ${logfile}" >&2
        return 1
    fi

    printf '%s,%s,%s\n' "${run_name}" "${wall_time}" "${glups}" >>"${SUMMARY}"
}

echo "Job ID       : ${SLURM_JOB_ID}"
echo "Partition    : ${SLURM_JOB_PARTITION:-unknown}"
echo "Node list    : ${SLURM_JOB_NODELIST:-unknown}"
echo "Project dir  : ${PROJECT_DIR}"
echo "Results dir  : ${RUN_DIR}"
echo "Summary file : ${SUMMARY}"
echo "Run args     : ${RUN_ARGS[*]}"
echo "MPI          : disabled"
echo "Runs/config  : 1"

{
    echo "date=$(date --iso-8601=seconds)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "partition=${SLURM_JOB_PARTITION:-unknown}"
    echo "nodes=${SLURM_JOB_NODELIST:-unknown}"
    echo "submit_dir=${PROJECT_DIR}"
    echo "run_args=${RUN_ARGS[*]}"
    echo "omp_threads=1 ${OMP_THREAD_LIST}"
    echo "omp_bindings=close spread"
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

echo "Building baseline, optimized serial and OpenMP executables"
make clean
make base serial openmp 2>&1 | tee "${RUN_DIR}/build.log"

printf 'run,t_wall_s,glups_kernel\n' >"${SUMMARY}"

echo "Running baseline"
save_run baseline \
    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --cpu-bind=cores "${BASE_BIN}" "${RUN_ARGS[@]}"

echo "Running optimized serial code"
save_run serial_optimized \
    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --cpu-bind=cores "${SERIAL_BIN}" "${RUN_ARGS[@]}"

echo "Running OpenMP with one thread"
save_run omp_t001 \
    env OMP_NUM_THREADS=1 OMP_PLACES=cores OMP_PROC_BIND=spread \
    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=32 \
    --cpu-bind=cores "${OPENMP_BIN}" "${RUN_ARGS[@]}"

echo "Running OpenMP close/spread comparison"
for binding in close spread; do
    for threads in ${OMP_THREAD_LIST}; do
        save_run omp_${binding}_t$(printf '%03d' "${threads}") \
            env OMP_NUM_THREADS="${threads}" OMP_PLACES=cores \
            OMP_PROC_BIND="${binding}" \
            srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=32 \
            --cpu-bind=cores "${OPENMP_BIN}" "${RUN_ARGS[@]}"
    done
done

sacct -j "${SLURM_JOB_ID}" \
    --format=JobID,JobName,Partition,State,Elapsed,NNodes,NTasks,AllocCPUS,MaxRSS \
    >"${RUN_DIR}/sacct.txt" || true

echo "All runs completed"
echo "Summary: ${SUMMARY}"
cat "${SUMMARY}"
