#!/bin/bash
#SBATCH --job-name=stencil_omp_serial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --account=uTS26_Tornator
#SBATCH --partition=boost_usr_prod
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR=${SLURM_SUBMIT_DIR:?Submit this job from the repository root}
RESULTS_ROOT=${RESULTS_ROOT:-${PROJECT_DIR}/results}
RUN_DIR=${RESULTS_ROOT}/${SLURM_JOB_ID}_omp_serial
RAW_DIR=${RUN_DIR}/raw
JOB_BUILD_DIR=${RUN_DIR}/build
SUMMARY=${RUN_DIR}/go_omp_serial.csv

ARGS=${ARGS:-"-x 25000 -y 25000 -n 200 -o 0"}
REPEATS=${REPEATS:-1}
STRONG_THREADS=${STRONG_THREADS:-"1 2 4 8 16 24 32"}
WEAK_THREADS=${WEAK_THREADS:-"1 2 4 8 16 32"}
OMP_BINDINGS=${OMP_BINDINGS:-"close spread"}
BASE_SIDE=${BASE_SIDE:-5000}
ITERATIONS=${ITERATIONS:-200}
read -r -a STRONG_ARGS <<< "${ARGS}"

mkdir -p "${RAW_DIR}"
cd "${PROJECT_DIR}"
exec > >(tee "${RUN_DIR}/job.out") 2> >(tee "${RUN_DIR}/job.err" >&2)

module load gcc/12.2.0

extract_metrics()
{
    awk '
        $1 == "t_wall" { wall = $4 }
        $1 == "t_get_total_energy" || $1 == "t_energy" { energy = $4 }
        $1 == "t_update_plane" || $1 == "t_update" { update = $4 }
        $1 == "t_inject_energy" || $1 == "t_inject" { inject = $4 }
        $1 == "performance" { glups = $3 }
        END {
            if (wall == "" || energy == "" || update == "" ||
                inject == "" || glups == "")
                exit 1
            printf "%s,%s,%s,%s,%s\n", wall, energy, update, inject, glups
        }
    ' "$1"
}

save_run()
{
    local run_name=$1
    shift
    local logfile=${RAW_DIR}/${run_name}.log
    local metrics

    echo "RUN ${run_name}"
    "$@" >"${logfile}" 2>&1
    if ! metrics=$(extract_metrics "${logfile}"); then
        echo "Unable to extract all metrics from ${logfile}" >&2
        return 1
    fi
    printf '%s,%s\n' "${run_name}" "${metrics}" >>"${SUMMARY}"
}

grid_for_threads()
{
    case "$1" in
        1)  GRID_X=${BASE_SIDE};       GRID_Y=${BASE_SIDE} ;;
        2)  GRID_X=$((2*BASE_SIDE));   GRID_Y=${BASE_SIDE} ;;
        4)  GRID_X=$((2*BASE_SIDE));   GRID_Y=$((2*BASE_SIDE)) ;;
        8)  GRID_X=$((4*BASE_SIDE));   GRID_Y=$((2*BASE_SIDE)) ;;
        16) GRID_X=$((4*BASE_SIDE));   GRID_Y=$((4*BASE_SIDE)) ;;
        32) GRID_X=$((8*BASE_SIDE));   GRID_Y=$((4*BASE_SIDE)) ;;
        *)
            echo "Unsupported weak-scaling thread count: $1" >&2
            return 1
            ;;
    esac
}

echo "Job ID          : ${SLURM_JOB_ID}"
echo "Node list       : ${SLURM_JOB_NODELIST:-unknown}"
echo "Strong args     : ${ARGS}"
echo "Strong threads  : ${STRONG_THREADS}"
echo "Bindings        : ${OMP_BINDINGS}"
echo "Weak threads    : ${WEAK_THREADS}"
echo "Weak base side  : ${BASE_SIDE}"
echo "Repetitions     : ${REPEATS}"

{
    echo "date=$(date --iso-8601=seconds)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "partition=${SLURM_JOB_PARTITION:-unknown}"
    echo "nodes=${SLURM_JOB_NODELIST:-unknown}"
    echo "strong_arguments=${ARGS}"
    echo "strong_threads=${STRONG_THREADS}"
    echo "omp_places=cores"
    echo "omp_dynamic=false"
    echo "omp_bindings=${OMP_BINDINGS}"
    echo "weak_threads=${WEAK_THREADS}"
    echo "weak_base_side=${BASE_SIDE}"
    echo "weak_iterations=${ITERATIONS}"
    echo "source_mode=random"
    echo "repetitions=${REPEATS}"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "git_status:"
    git status --short || true
    gcc --version | head -n 1 || true
    module -t list 2>&1 || true
} >"${RUN_DIR}/environment.txt"

scontrol show job "${SLURM_JOB_ID}" >"${RUN_DIR}/slurm_job.txt"
srun --nodes=1 --ntasks=1 --cpus-per-task=1 lscpu >"${RUN_DIR}/lscpu.txt"
git diff >"${RUN_DIR}/source_changes.patch" || true

make BUILD_DIR="${JOB_BUILD_DIR}" omp-serial 2>&1 | tee "${RUN_DIR}/build.log"
OPENMP_BIN=${JOB_BUILD_DIR}/stencil_serial_final_omp_O3

printf 'run_name,t_wall,t_get_total_energy,t_update_plane,t_inject_energy,glups\n' >"${SUMMARY}"

# Strong scaling and close/spread comparison use the same fixed problem.
for binding in ${OMP_BINDINGS}; do
    for threads in ${STRONG_THREADS}; do
        for repetition in $(seq 1 "${REPEATS}"); do
            run_name=strong_${binding}_t$(printf '%03d' "${threads}")_rep$(printf '%02d' "${repetition}")
            save_run "${run_name}" \
                env OMP_NUM_THREADS="${threads}" OMP_DYNAMIC=false \
                OMP_PLACES=cores OMP_PROC_BIND="${binding}" \
                srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=32 \
                --cpu-bind=cores "${OPENMP_BIN}" "${STRONG_ARGS[@]}"
        done
    done
done

# Weak scaling keeps BASE_SIDE^2 cells per thread and uses random sources.
for threads in ${WEAK_THREADS}; do
    grid_for_threads "${threads}"
    WEAK_ARGS=(-x "${GRID_X}" -y "${GRID_Y}" -n "${ITERATIONS}" -o 0)
    for repetition in $(seq 1 "${REPEATS}"); do
        run_name=weak_spread_t$(printf '%03d' "${threads}")_x${GRID_X}_y${GRID_Y}_rep$(printf '%02d' "${repetition}")
        save_run "${run_name}" \
            env OMP_NUM_THREADS="${threads}" OMP_DYNAMIC=false \
            OMP_PLACES=cores OMP_PROC_BIND=spread \
            srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=32 \
            --cpu-bind=cores "${OPENMP_BIN}" "${WEAK_ARGS[@]}"
    done
done

sacct -j "${SLURM_JOB_ID}" \
    --format=JobID,JobName,Partition,State,Elapsed,NNodes,NTasks,AllocCPUS,MaxRSS \
    >"${RUN_DIR}/sacct.txt" || true

echo "Completed. Summary: ${SUMMARY}"
cat "${SUMMARY}"
