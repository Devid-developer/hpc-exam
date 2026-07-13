#!/bin/bash
#SBATCH --job-name=stencil_mpi_nodes
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
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
RUN_DIR=${RESULTS_ROOT}/${SLURM_JOB_ID}_mpi_node_scaling
RAW_DIR=${RUN_DIR}/raw
JOB_BUILD_DIR=${RUN_DIR}/build
SUMMARY=${RUN_DIR}/go_mpi_node_scaling.csv

NODE_COUNTS=${NODE_COUNTS:-"1 2 4 8 16"}
RANKS_PER_NODE=${RANKS_PER_NODE:-4}
THREADS_PER_RANK=${THREADS_PER_RANK:-8}
STRONG_X=${STRONG_X:-25000}
STRONG_Y=${STRONG_Y:-25000}
WEAK_MAX_SIDE=${WEAK_MAX_SIDE:-25000}
WEAK_REFERENCE_NODES=${WEAK_REFERENCE_NODES:-16}
ITERATIONS=${ITERATIONS:-100}
REPEATS=${REPEATS:-1}

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

weak_side_for_nodes()
{
    local nodes=$1

    awk -v max_side="${WEAK_MAX_SIDE}" \
        -v nodes="${nodes}" \
        -v reference_nodes="${WEAK_REFERENCE_NODES}" \
        'BEGIN {
            side = max_side * sqrt(nodes / reference_nodes)
            printf "%d\n", int(side + 0.5)
        }'
}

save_run()
{
    local run_name=$1
    local scaling=$2
    local nodes=$3
    local grid_x=$4
    local grid_y=$5
    shift 5
    local ranks=$((nodes * RANKS_PER_NODE))
    local total_cores=$((ranks * THREADS_PER_RANK))
    local total_cells=$((grid_x * grid_y))
    local cells_per_node=$((total_cells / nodes))
    local logfile=${RAW_DIR}/${run_name}.log
    local metrics

    echo "RUN ${run_name}: scaling=${scaling}, nodes=${nodes}, ranks=${ranks}, threads/rank=${THREADS_PER_RANK}, grid=${grid_x}x${grid_y}"
    "$@" >"${logfile}" 2>&1
    if ! metrics=$(extract_metrics "${logfile}"); then
        echo "Unable to extract all metrics from ${logfile}" >&2
        return 1
    fi

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${run_name}" "${scaling}" "${nodes}" "${ranks}" \
        "${RANKS_PER_NODE}" "${THREADS_PER_RANK}" "${total_cores}" \
        "${grid_x}" "${grid_y}" "${ITERATIONS}" "${total_cells}" \
        "${cells_per_node}" "${metrics}" >>"${SUMMARY}"
}

run_case()
{
    local scaling=$1
    local nodes=$2
    local repetition=$3
    local grid_x
    local grid_y
    local ranks=$((nodes * RANKS_PER_NODE))
    local side
    local run_name

    if [[ ${scaling} == strong ]]; then
        grid_x=${STRONG_X}
        grid_y=${STRONG_Y}
        run_name=node_strong_n$(printf '%02d' "${nodes}")_r$(printf '%03d' "${ranks}")_t$(printf '%03d' "${THREADS_PER_RANK}")_rep$(printf '%02d' "${repetition}")
    else
        side=$(weak_side_for_nodes "${nodes}")
        grid_x=${side}
        grid_y=${side}
        run_name=node_weak_n$(printf '%02d' "${nodes}")_r$(printf '%03d' "${ranks}")_t$(printf '%03d' "${THREADS_PER_RANK}")_x${grid_x}_y${grid_y}_rep$(printf '%02d' "${repetition}")
    fi

    save_run "${run_name}" "${scaling}" "${nodes}" "${grid_x}" "${grid_y}" \
        env OMP_NUM_THREADS="${THREADS_PER_RANK}" OMP_DYNAMIC=false \
        OMP_PLACES=cores OMP_PROC_BIND=spread \
        srun --exclusive --kill-on-bad-exit=1 --nodes="${nodes}" \
        --ntasks="${ranks}" --ntasks-per-node="${RANKS_PER_NODE}" \
        --cpus-per-task="${THREADS_PER_RANK}" --distribution=block:block \
        --cpu-bind=cores "${MPI_BIN}" -x "${grid_x}" -y "${grid_y}" \
        -n "${ITERATIONS}" -o 0
}

if (( RANKS_PER_NODE * THREADS_PER_RANK != 32 )); then
    echo "RANKS_PER_NODE * THREADS_PER_RANK must equal the 32 physical cores of a Booster node" >&2
    exit 1
fi

for nodes in ${NODE_COUNTS}; do
    if (( nodes < 1 || nodes > ${SLURM_JOB_NUM_NODES:-16} )); then
        echo "Node count ${nodes} is incompatible with the ${SLURM_JOB_NUM_NODES:-16}-node allocation" >&2
        exit 1
    fi
done

{
    echo "date=$(date --iso-8601=seconds)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "partition=${SLURM_JOB_PARTITION:-unknown}"
    echo "nodes=${SLURM_JOB_NODELIST:-unknown}"
    echo "node_counts=${NODE_COUNTS}"
    echo "ranks_per_node=${RANKS_PER_NODE}"
    echo "threads_per_rank=${THREADS_PER_RANK}"
    echo "strong_grid=${STRONG_X}x${STRONG_Y}"
    echo "weak_max_side=${WEAK_MAX_SIDE}"
    echo "weak_reference_nodes=${WEAK_REFERENCE_NODES}"
    echo "weak_formula=N_side=weak_max_side*sqrt(nodes/weak_reference_nodes)"
    echo "iterations=${ITERATIONS}"
    echo "omp_places=cores"
    echo "omp_proc_bind=spread"
    echo "omp_dynamic=false"
    echo "source_mode=random"
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

printf 'run_name,scaling,nodes,ranks,ranks_per_node,threads_per_rank,total_cores,grid_x,grid_y,iterations,total_cells,cells_per_node,process_grid_x,process_grid_y,t_wall,t_get_total_energy,t_update_plane,t_inject_energy,t_exchange_halos,glups,injected_energy,system_energy\n' \
    >"${SUMMARY}"

for scaling in strong weak; do
    for nodes in ${NODE_COUNTS}; do
        for repetition in $(seq 1 "${REPEATS}"); do
            run_case "${scaling}" "${nodes}" "${repetition}"
        done
    done
done

sacct -j "${SLURM_JOB_ID}" \
    --format=JobID,JobName,Partition,State,Elapsed,NNodes,NTasks,AllocCPUS,MaxRSS \
    >"${RUN_DIR}/sacct.txt" || true

echo "MPI+OpenMP node-scaling runs completed. Summary: ${SUMMARY}"
cat "${SUMMARY}"
