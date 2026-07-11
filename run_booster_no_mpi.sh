#!/bin/bash
#SBATCH --job-name=hpc_stencil_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --account=IscrB_SPIESMD
#SBATCH --partition=boost_usr_prod
#SBATCH --output=hpc_stencil_cpu_%j.bootstrap.out
#SBATCH --error=hpc_stencil_cpu_%j.bootstrap.err

set -euo pipefail

# Submit from the repository root: DEV/hpc-exam.
PROJECT_DIR=${SLURM_SUBMIT_DIR:?Submit the job from the hpc-exam directory}
RESULTS_ROOT=${RESULTS_ROOT:-${PROJECT_DIR}/results}
RUN_DIR=${RESULTS_ROOT}/${SLURM_JOB_ID}_no_mpi
RAW_DIR=${RUN_DIR}/raw

mkdir -p "${RAW_DIR}"
cd "${PROJECT_DIR}"

exec > >(tee "${RUN_DIR}/job.out") 2> >(tee "${RUN_DIR}/job.err" >&2)

GRID_X=20000
GRID_Y=20000
ITERATIONS=${ITERATIONS:-100}
SOURCES=${SOURCES:-4}
PERIODIC=${PERIODIC:-1}
OMP_THREAD_LIST=${OMP_THREAD_LIST:-"1 2 4 8 16 24 32"}

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
    local family=$1
    local label=$2
    shift 2

    local logfile=${RAW_DIR}/${family}_${label}.log
    echo "RUN family=${family} label=${label}"
    "$@" >"${logfile}" 2>&1

    awk -v label="${label}" '/^CSV,/{print label "," $0}' "${logfile}" \
        >>"${RUN_DIR}/${family}.csv"
}

echo "Job ID       : ${SLURM_JOB_ID}"
echo "Partition    : ${SLURM_JOB_PARTITION:-unknown}"
echo "Node list    : ${SLURM_JOB_NODELIST:-unknown}"
echo "Project dir  : ${PROJECT_DIR}"
echo "Results dir  : ${RUN_DIR}"
echo "Run args     : ${RUN_ARGS[*]}"
echo "MPI          : disabled"
echo "Repetitions  : none (one run per configuration)"

{
    echo "date=$(date --iso-8601=seconds)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "partition=${SLURM_JOB_PARTITION:-unknown}"
    echo "nodes=${SLURM_JOB_NODELIST:-unknown}"
    echo "submit_dir=${PROJECT_DIR}"
    echo "run_args=${RUN_ARGS[*]}"
    echo "omp_threads=${OMP_THREAD_LIST}"
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

echo "label,program_csv" >"${RUN_DIR}/serial.csv"
echo "label,program_csv" >"${RUN_DIR}/openmp.csv"

echo "Running the single-core baseline"
save_run serial base \
    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --cpu-bind=cores "${BASE_BIN}" "${RUN_ARGS[@]}"

echo "Running the optimized serial code"
save_run serial optimized \
    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --cpu-bind=cores "${SERIAL_BIN}" "${RUN_ARGS[@]}"

echo "Running OpenMP scaling"
for threads in ${OMP_THREAD_LIST}; do
    save_run openmp t$(printf '%03d' "${threads}") \
        env OMP_NUM_THREADS="${threads}" OMP_PLACES=cores OMP_PROC_BIND=close \
        srun --exclusive --nodes=1 --ntasks=1 \
        --cpus-per-task="${threads}" --cpu-bind=cores \
        "${OPENMP_BIN}" "${RUN_ARGS[@]}"
done

sacct -j "${SLURM_JOB_ID}" --format=JobID,JobName,Partition,State,Elapsed,NNodes,NTasks,AllocCPUS,MaxRSS \
    >"${RUN_DIR}/sacct.txt" || true

echo "All non-MPI runs completed. Results are in ${RUN_DIR}"
