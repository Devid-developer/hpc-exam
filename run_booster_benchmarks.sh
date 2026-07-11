#!/bin/bash
#SBATCH --job-name=hpc_stencil
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --account=IscrB_SPIESMD
#SBATCH --partition=boost_usr_prod
#SBATCH --output=hpc_stencil_%j.bootstrap.out
#SBATCH --error=hpc_stencil_%j.bootstrap.err

set -euo pipefail

# Submit this script from the repository root (DEV/hpc-exam). Slurm sets
# SLURM_SUBMIT_DIR to that directory and SLURM_JOB_ID to the current job.
PROJECT_DIR=${SLURM_SUBMIT_DIR:?Submit the job from the hpc-exam directory}
RESULTS_ROOT=${RESULTS_ROOT:-${PROJECT_DIR}/results}
RUN_DIR=${RESULTS_ROOT}/${SLURM_JOB_ID}
RAW_DIR=${RUN_DIR}/raw

mkdir -p "${RAW_DIR}"
cd "${PROJECT_DIR}"

# From this point onward, keep a complete job log inside the result directory.
exec > >(tee "${RUN_DIR}/job.out") 2> >(tee "${RUN_DIR}/job.err" >&2)

REPEATS=${REPEATS:-5}
GRID_X=${GRID_X:-10000}
GRID_Y=${GRID_Y:-10000}
ITERATIONS=${ITERATIONS:-100}
SOURCES=${SOURCES:-4}
PERIODIC=${PERIODIC:-1}
OMP_THREAD_LIST=${OMP_THREAD_LIST:-"1 2 4 8 16 24 32"}
MPI_TASK_LIST=${MPI_TASK_LIST:-"1 2 4 8 16 32"}

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
MPI_BIN=${PROJECT_DIR}/build/stencil_parallel_final

save_run()
{
    local family=$1
    local label=$2
    local repetition=$3
    shift 3

    local logfile=${RAW_DIR}/${family}_${label}_rep$(printf '%02d' "${repetition}").log
    echo "RUN family=${family} label=${label} repetition=${repetition}"
    "$@" >"${logfile}" 2>&1

    # Preserve the program CSV verbatim and prepend configuration information.
    awk -v label="${label}" -v rep="${repetition}" \
        '/^CSV,/{print label "," rep "," $0}' "${logfile}" \
        >>"${RUN_DIR}/${family}.csv"
}

echo "Job ID       : ${SLURM_JOB_ID}"
echo "Partition    : ${SLURM_JOB_PARTITION:-unknown}"
echo "Node list    : ${SLURM_JOB_NODELIST:-unknown}"
echo "Project dir  : ${PROJECT_DIR}"
echo "Results dir  : ${RUN_DIR}"
echo "Run args     : ${RUN_ARGS[*]}"
echo "Repeats      : ${REPEATS}"

{
    echo "date=$(date --iso-8601=seconds)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "partition=${SLURM_JOB_PARTITION:-unknown}"
    echo "nodes=${SLURM_JOB_NODELIST:-unknown}"
    echo "submit_dir=${PROJECT_DIR}"
    echo "run_args=${RUN_ARGS[*]}"
    echo "repeats=${REPEATS}"
    echo "omp_threads=${OMP_THREAD_LIST}"
    echo "mpi_tasks=${MPI_TASK_LIST}"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "cc=$(command -v gcc || true)"
    gcc --version 2>/dev/null | head -n 1 || true
    echo "mpicc=$(command -v mpicc || true)"
    mpicc --version 2>/dev/null | head -n 1 || true
} >"${RUN_DIR}/environment.txt"

module -t list >"${RUN_DIR}/modules.txt" 2>&1 || true
scontrol show job "${SLURM_JOB_ID}" >"${RUN_DIR}/slurm_job.txt"
srun --nodes=1 --ntasks=1 --cpus-per-task=1 lscpu >"${RUN_DIR}/lscpu.txt"
git diff >"${RUN_DIR}/source_changes.patch" || true

echo "Building baseline, serial, OpenMP and MPI executables"
make clean
make base serial openmp mpi 2>&1 | tee "${RUN_DIR}/build.log"

echo "label,repeat,program_csv" >"${RUN_DIR}/serial.csv"
echo "label,repeat,program_csv" >"${RUN_DIR}/openmp.csv"
echo "label,repeat,program_csv" >"${RUN_DIR}/mpi.csv"

echo "Running single-core baseline and optimized serial code"
for repetition in $(seq 1 "${REPEATS}"); do
    save_run serial base "${repetition}" \
        srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=1 \
        --cpu-bind=cores "${BASE_BIN}" "${RUN_ARGS[@]}"

    save_run serial optimized "${repetition}" \
        srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=1 \
        --cpu-bind=cores "${SERIAL_BIN}" "${RUN_ARGS[@]}"
done

echo "Running OpenMP scaling"
for threads in ${OMP_THREAD_LIST}; do
    for repetition in $(seq 1 "${REPEATS}"); do
        save_run openmp t$(printf '%03d' "${threads}") "${repetition}" \
            env OMP_NUM_THREADS="${threads}" OMP_PLACES=cores OMP_PROC_BIND=close \
            srun --exclusive --nodes=1 --ntasks=1 \
            --cpus-per-task="${threads}" --cpu-bind=cores \
            "${OPENMP_BIN}" "${RUN_ARGS[@]}"
    done
done

echo "Running single-node MPI scaling"
for tasks in ${MPI_TASK_LIST}; do
    for repetition in $(seq 1 "${REPEATS}"); do
        save_run mpi r$(printf '%03d' "${tasks}") "${repetition}" \
            env OMP_NUM_THREADS=1 \
            srun --exclusive --nodes=1 --ntasks="${tasks}" \
            --ntasks-per-node="${tasks}" --cpus-per-task=1 --cpu-bind=cores \
            "${MPI_BIN}" "${RUN_ARGS[@]}"
    done
done

sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Partition,State,Elapsed,NNodes,NTasks,AllocCPUS,MaxRSS \
    >"${RUN_DIR}/sacct.txt" || true

echo "All runs completed. Results are in ${RUN_DIR}"
