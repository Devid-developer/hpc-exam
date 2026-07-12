#!/bin/bash
#SBATCH --job-name=stencil_mpi_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --account=IscrB_SPIESMD
#SBATCH --partition=boost_usr_prod
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR=${SLURM_SUBMIT_DIR:?Submit this job from the repository root}
RESULTS_ROOT=${RESULTS_ROOT:-${PROJECT_DIR}/results}
RUN_DIR=${RESULTS_ROOT}/${SLURM_JOB_ID}_mpi_smoke
RAW_DIR=${RUN_DIR}/raw
JOB_BUILD_DIR=${RUN_DIR}/build
SUMMARY=${RUN_DIR}/go_mpi_smoke.csv

MPI_TASKS=${MPI_TASKS:-"1 2 4 8"}
SMOKE_X=${SMOKE_X:-101}
SMOKE_Y=${SMOKE_Y:-97}
ITERATIONS=${ITERATIONS:-20}
SOURCES=${SOURCES:-4}
SEED=${SEED:-7}

mkdir -p "${RAW_DIR}"
cd "${PROJECT_DIR}"
exec > >(tee "${RUN_DIR}/job.out") 2> >(tee "${RUN_DIR}/job.err" >&2)

module purge
module load openmpi/4.1.6--gcc--12.2.0

extract_metrics()
{
    awk '
        $1 == "t_wall" && $2 == "max" { wall = $4 }
        $1 == "t_energy" && $2 == "max" { energy = $4 }
        $1 == "t_update" && $2 == "max" { update = $4 }
        $1 == "t_inject" && $2 == "max" { inject = $4 }
        $1 == "t_comm" && $2 == "max" { comm = $4 }
        $1 == "performance" { glups = $3 }
        $1 == "injected" && $2 == "energy" && $3 == ":" { injected = $4 }
        $1 == "system" && $2 == "energy" && $3 == ":" { system_energy = $4 }
        END {
            if (wall == "" || energy == "" || update == "" ||
                inject == "" || comm == "" || glups == "" ||
                injected == "" || system_energy == "")
                exit 1
            printf "%s,%s,%s,%s,%s,%s,%s,%s\n", wall, energy, update,
                   inject, comm, glups, injected, system_energy
        }
    ' "$1"
}

save_run()
{
    local run_name=$1
    local tasks=$2
    local periodic=$3
    shift 3
    local logfile=${RAW_DIR}/${run_name}.log
    local metrics

    echo "RUN ${run_name}: tasks=${tasks}, grid=${SMOKE_X}x${SMOKE_Y}, periodic=${periodic}"
    "$@" >"${logfile}" 2>&1
    if ! metrics=$(extract_metrics "${logfile}"); then
        echo "Unable to extract all metrics from ${logfile}" >&2
        return 1
    fi
    printf '%s,%s,1,%s,%s,%s,%s\n' \
        "${run_name}" "${tasks}" "${SMOKE_X}" "${SMOKE_Y}" \
        "${periodic}" "${metrics}" >>"${SUMMARY}"
}

{
    echo "date=$(date --iso-8601=seconds)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "partition=${SLURM_JOB_PARTITION:-unknown}"
    echo "nodes=${SLURM_JOB_NODELIST:-unknown}"
    echo "mpi_tasks=${MPI_TASKS}"
    echo "grid=${SMOKE_X}x${SMOKE_Y}"
    echo "iterations=${ITERATIONS}"
    echo "sources=${SOURCES}"
    echo "seed=${SEED}"
    echo "openmp_threads=1"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "git_status:"
    git status --short || true
    mpicc --version | head -n 1 || true
    module -t list 2>&1 || true
} >"${RUN_DIR}/environment.txt"

scontrol show job "${SLURM_JOB_ID}" >"${RUN_DIR}/slurm_job.txt"
srun --nodes=1 --ntasks=1 --cpus-per-task=1 lscpu >"${RUN_DIR}/lscpu.txt"
git diff >"${RUN_DIR}/source_changes.patch" || true

make BUILD_DIR="${JOB_BUILD_DIR}" mpi 2>&1 | tee "${RUN_DIR}/build.log"
MPI_BIN=${JOB_BUILD_DIR}/stencil_parallel_final_mpi_O3

printf 'run_name,tasks,threads,grid_x,grid_y,periodic,t_wall,t_get_total_energy,t_update_plane,t_inject_energy,t_exchange_halos,glups,injected_energy,system_energy\n' >"${SUMMARY}"

for periodic in 0 1; do
    if [[ "${periodic}" == "0" ]]; then boundary=np; else boundary=p; fi
    for tasks in ${MPI_TASKS}; do
        run_name=smoke_${boundary}_r$(printf '%03d' "${tasks}")
        save_run "${run_name}" "${tasks}" "${periodic}" \
            env OMP_NUM_THREADS=1 OMP_DYNAMIC=false \
            srun --exclusive --kill-on-bad-exit=1 --nodes=1 \
            --ntasks="${tasks}" --ntasks-per-node="${tasks}" \
            --cpus-per-task=1 --cpu-bind=cores \
            "${MPI_BIN}" -x "${SMOKE_X}" -y "${SMOKE_Y}" \
            -n "${ITERATIONS}" -e "${SOURCES}" -s "${SEED}" \
            -p "${periodic}" -o 0
    done
done

awk -F, '
    NR == 1 { next }
    {
        group = $6
        if (!(group in reference))
            reference[group] = $14
        else {
            difference = $14 - reference[group]
            if (difference < 0)
                difference = -difference
            scale = reference[group]
            if (scale < 0)
                scale = -scale
            if (difference <= 1e-5 * (scale + 1))
                next
            printf "energy mismatch for %s: %s instead of %s\n", $1, $14,
                   reference[group] > "/dev/stderr"
            failed = 1
        }
    }
    END { exit failed }
' "${SUMMARY}"

sacct -j "${SLURM_JOB_ID}" \
    --format=JobID,JobName,Partition,State,Elapsed,NNodes,NTasks,AllocCPUS,MaxRSS \
    >"${RUN_DIR}/sacct.txt" || true

echo "Smoke test completed: all rank counts produced the same final energy."
echo "Summary: ${SUMMARY}"
cat "${SUMMARY}"
