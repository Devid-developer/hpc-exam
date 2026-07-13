#!/bin/bash
#SBATCH --job-name=stencil_mpi_hybrid
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
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
RUN_DIR=${RESULTS_ROOT}/${SLURM_JOB_ID}_mpi_hybrid
RAW_DIR=${RUN_DIR}/raw
JOB_BUILD_DIR=${RUN_DIR}/build
SUMMARY=${RUN_DIR}/go_mpi_hybrid.csv

ARGS=${ARGS:-"-x 25000 -y 25000 -n 200 -o 0"}
REPEATS=${REPEATS:-1}
ONE_NODE_CONFIGS=${ONE_NODE_CONFIGS:-"1:32 2:16 4:8 8:4 16:2 32:1"}
TWO_NODE_CONFIGS=${TWO_NODE_CONFIGS:-"2:32 4:16 8:8 16:4 32:2 64:1"}
read -r -a RUN_ARGS <<< "${ARGS}"

mkdir -p "${RAW_DIR}"
cd "${PROJECT_DIR}"
exec > >(tee "${RUN_DIR}/job.out") 2> >(tee "${RUN_DIR}/job.err" >&2)

module purge
module load openmpi/4.1.6--gcc--12.2.0

extract_metrics()
{
    awk '
        $1 == "process" && $2 == "grid" { process_x = $4; process_y = $6 }
        $1 == "t_wall" && $2 == "max" { wall = $4 }
        $1 == "t_energy" && $2 == "max" { energy = $4 }
        $1 == "t_update" && $2 == "max" { update = $4 }
        $1 == "t_inject" && $2 == "max" { inject = $4 }
        $1 == "t_comm" && $2 == "max" { comm = $4 }
        $1 == "performance" { glups = $3 }
        $1 == "injected" && $2 == "energy" && $3 == ":" { injected = $4 }
        $1 == "system" && $2 == "energy" && $3 == ":" { system_energy = $4 }
        END {
            if (process_x == "" || process_y == "" || wall == "" ||
                energy == "" || update == "" || inject == "" ||
                comm == "" || glups == "" || injected == "" ||
                system_energy == "")
                exit 1
            printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", process_x,
                   process_y, wall, energy, update, inject, comm, glups,
                   injected, system_energy
        }
    ' "$1"
}

save_run()
{
    local run_name=$1
    local nodes=$2
    local ranks=$3
    local threads=$4
    shift 4
    local total_cores=$((ranks * threads))
    local logfile=${RAW_DIR}/${run_name}.log
    local metrics

    echo "RUN ${run_name}: nodes=${nodes}, ranks=${ranks}, threads/rank=${threads}, cores=${total_cores}"
    "$@" >"${logfile}" 2>&1
    if ! metrics=$(extract_metrics "${logfile}"); then
        echo "Unable to extract all metrics from ${logfile}" >&2
        return 1
    fi
    printf '%s,%s,%s,%s,%s,%s\n' "${run_name}" "${nodes}" "${ranks}" \
        "${threads}" "${total_cores}" "${metrics}" >>"${SUMMARY}"
}

run_configuration()
{
    local nodes=$1
    local configuration=$2
    local ranks=${configuration%%:*}
    local threads=${configuration##*:}
    local tasks_per_node=$((ranks / nodes))

    if (( ranks % nodes != 0 )); then
        echo "Configuration ${configuration} cannot be distributed over ${nodes} nodes" >&2
        return 1
    fi
    if (( tasks_per_node * threads != 32 )); then
        echo "Configuration ${configuration} does not use 32 cores per node" >&2
        return 1
    fi

    for repetition in $(seq 1 "${REPEATS}"); do
        local run_name=hybrid_n$(printf '%02d' "${nodes}")_r$(printf '%03d' "${ranks}")_t$(printf '%03d' "${threads}")_rep$(printf '%02d' "${repetition}")
        save_run "${run_name}" "${nodes}" "${ranks}" "${threads}" \
            env OMP_NUM_THREADS="${threads}" OMP_DYNAMIC=false \
            OMP_PLACES=cores OMP_PROC_BIND=spread \
            srun --exclusive --kill-on-bad-exit=1 --nodes="${nodes}" \
            --ntasks="${ranks}" --ntasks-per-node="${tasks_per_node}" \
            --cpus-per-task="${threads}" --distribution=block:block \
            --cpu-bind=cores "${MPI_BIN}" "${RUN_ARGS[@]}"
    done
}

{
    echo "date=$(date --iso-8601=seconds)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "partition=${SLURM_JOB_PARTITION:-unknown}"
    echo "nodes=${SLURM_JOB_NODELIST:-unknown}"
    echo "arguments=${ARGS}"
    echo "one_node_configs=${ONE_NODE_CONFIGS}"
    echo "two_node_configs=${TWO_NODE_CONFIGS}"
    echo "omp_places=cores"
    echo "omp_proc_bind=spread"
    echo "omp_dynamic=false"
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

printf 'run_name,nodes,ranks,threads_per_rank,total_cores,process_grid_x,process_grid_y,t_wall,t_get_total_energy,t_update_plane,t_inject_energy,t_exchange_halos,glups,injected_energy,system_energy\n' >"${SUMMARY}"

for configuration in ${ONE_NODE_CONFIGS}; do
    run_configuration 1 "${configuration}"
done

for configuration in ${TWO_NODE_CONFIGS}; do
    run_configuration 2 "${configuration}"
done

sacct -j "${SLURM_JOB_ID}" \
    --format=JobID,JobName,Partition,State,Elapsed,NNodes,NTasks,AllocCPUS,MaxRSS \
    >"${RUN_DIR}/sacct.txt" || true

echo "Hybrid MPI+OpenMP sweep completed. Summary: ${SUMMARY}"
cat "${SUMMARY}"
