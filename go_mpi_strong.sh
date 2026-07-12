#!/bin/bash
#SBATCH --job-name=stencil_mpi_strong
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
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
RUN_DIR=${RESULTS_ROOT}/${SLURM_JOB_ID}_mpi_strong
RAW_DIR=${RUN_DIR}/raw
JOB_BUILD_DIR=${RUN_DIR}/build
SUMMARY=${RUN_DIR}/go_mpi_strong.csv

ARGS=${ARGS:-"-x 25000 -y 25000 -n 200 -o 0"}
MPI_TASKS=${MPI_TASKS:-"1 2 4 8 16 32 64"}
REPEATS=${REPEATS:-1}
read -r -a RUN_ARGS <<< "${ARGS}"

mkdir -p "${RAW_DIR}"
cd "${PROJECT_DIR}"
exec > >(tee "${RUN_DIR}/job.out") 2> >(tee "${RUN_DIR}/job.err" >&2)

module load gcc/12.2.0
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
    local nodes=$3
    shift 3
    local logfile=${RAW_DIR}/${run_name}.log
    local metrics

    echo "RUN ${run_name}: tasks=${tasks}, nodes=${nodes}"
    "$@" >"${logfile}" 2>&1
    if ! metrics=$(extract_metrics "${logfile}"); then
        echo "Unable to extract all metrics from ${logfile}" >&2
        return 1
    fi
    printf '%s,%s,1,%s,%s\n' "${run_name}" "${tasks}" "${nodes}" \
        "${metrics}" >>"${SUMMARY}"
}

{
    echo "date=$(date --iso-8601=seconds)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "partition=${SLURM_JOB_PARTITION:-unknown}"
    echo "nodes=${SLURM_JOB_NODELIST:-unknown}"
    echo "arguments=${ARGS}"
    echo "mpi_tasks=${MPI_TASKS}"
    echo "openmp_threads=1"
    echo "repetitions=${REPEATS}"
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

printf 'run_name,tasks,threads,nodes,t_wall,t_get_total_energy,t_update_plane,t_inject_energy,t_exchange_halos,glups,injected_energy,system_energy\n' >"${SUMMARY}"

for tasks in ${MPI_TASKS}; do
    nodes=$(( (tasks + 31) / 32 ))
    if (( nodes > 2 )); then
        echo "${tasks} tasks require ${nodes} nodes, but this job reserves only 2" >&2
        exit 1
    fi
    tasks_per_node=$(( tasks < 32 ? tasks : 32 ))

    for repetition in $(seq 1 "${REPEATS}"); do
        run_name=strong_r$(printf '%03d' "${tasks}")_n$(printf '%02d' "${nodes}")_rep$(printf '%02d' "${repetition}")
        save_run "${run_name}" "${tasks}" "${nodes}" \
            env OMP_NUM_THREADS=1 OMP_DYNAMIC=false \
            srun --exclusive --kill-on-bad-exit=1 --nodes="${nodes}" \
            --ntasks="${tasks}" --ntasks-per-node="${tasks_per_node}" \
            --cpus-per-task=1 --cpu-bind=cores \
            "${MPI_BIN}" "${RUN_ARGS[@]}"
    done
done

sacct -j "${SLURM_JOB_ID}" \
    --format=JobID,JobName,Partition,State,Elapsed,NNodes,NTasks,AllocCPUS,MaxRSS \
    >"${RUN_DIR}/sacct.txt" || true

echo "Strong-scaling runs completed. Summary: ${SUMMARY}"
cat "${SUMMARY}"
