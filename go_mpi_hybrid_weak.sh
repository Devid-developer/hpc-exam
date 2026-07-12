#!/bin/bash
#SBATCH --job-name=stencil_hybrid_weak
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
RUN_DIR=${RESULTS_ROOT}/${SLURM_JOB_ID}_mpi_hybrid_weak
RAW_DIR=${RUN_DIR}/raw
JOB_BUILD_DIR=${RUN_DIR}/build
SUMMARY=${RUN_DIR}/go_mpi_hybrid_weak.csv

BASE_X=${BASE_X:-25000}
BASE_Y=${BASE_Y:-25000}
ITERATIONS=${ITERATIONS:-200}
REPEATS=${REPEATS:-1}
PER_NODE_CONFIGS=${PER_NODE_CONFIGS:-"1:32 2:16 4:8 8:4 16:2 32:1"}

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
    local layout=$2
    local nodes=$3
    local ranks=$4
    local ranks_per_node=$5
    local threads=$6
    local grid_x=$7
    local grid_y=$8
    shift 8
    local total_cores=$((ranks * threads))
    local total_cells=$((grid_x * grid_y))
    local cells_per_core=$((total_cells / total_cores))
    local logfile=${RAW_DIR}/${run_name}.log
    local metrics

    echo "RUN ${run_name}: layout=${layout}, nodes=${nodes}, ranks=${ranks}, threads/rank=${threads}, grid=${grid_x}x${grid_y}"
    "$@" >"${logfile}" 2>&1
    if ! metrics=$(extract_metrics "${logfile}"); then
        echo "Unable to extract all metrics from ${logfile}" >&2
        return 1
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${run_name}" "${layout}" "${nodes}" "${ranks}" \
        "${ranks_per_node}" "${threads}" "${total_cores}" \
        "${grid_x}" "${grid_y}" "${total_cells}" "${cells_per_core}" \
        "${metrics}" >>"${SUMMARY}"
}

run_configuration()
{
    local nodes=$1
    local ranks_per_node=$2
    local threads=$3
    local repetition=$4
    local ranks=$((nodes * ranks_per_node))
    local grid_x=$((nodes * BASE_X))
    local grid_y=${BASE_Y}
    local layout=rpn$(printf '%02d' "${ranks_per_node}")_t$(printf '%03d' "${threads}")
    local run_name=weak_${layout}_n$(printf '%02d' "${nodes}")_r$(printf '%03d' "${ranks}")_rep$(printf '%02d' "${repetition}")

    save_run "${run_name}" "${layout}" "${nodes}" "${ranks}" \
        "${ranks_per_node}" "${threads}" "${grid_x}" "${grid_y}" \
        env OMP_NUM_THREADS="${threads}" OMP_DYNAMIC=false \
        OMP_PLACES=cores OMP_PROC_BIND=spread \
        srun --exclusive --kill-on-bad-exit=1 --nodes="${nodes}" \
        --ntasks="${ranks}" --ntasks-per-node="${ranks_per_node}" \
        --cpus-per-task="${threads}" --distribution=block:block \
        --cpu-bind=cores "${MPI_BIN}" -x "${grid_x}" -y "${grid_y}" \
        -n "${ITERATIONS}" -o 0
}

{
    echo "date=$(date --iso-8601=seconds)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "partition=${SLURM_JOB_PARTITION:-unknown}"
    echo "nodes=${SLURM_JOB_NODELIST:-unknown}"
    echo "base_grid=${BASE_X}x${BASE_Y}_per_node"
    echo "iterations=${ITERATIONS}"
    echo "per_node_configs=${PER_NODE_CONFIGS}"
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

printf 'run_name,layout,nodes,ranks,ranks_per_node,threads_per_rank,total_cores,grid_x,grid_y,total_cells,cells_per_core,process_grid_x,process_grid_y,t_wall,t_get_total_energy,t_update_plane,t_inject_energy,t_exchange_halos,glups,injected_energy,system_energy\n' >"${SUMMARY}"

for configuration in ${PER_NODE_CONFIGS}; do
    ranks_per_node=${configuration%%:*}
    threads=${configuration##*:}
    if (( ranks_per_node * threads != 32 )); then
        echo "Configuration ${configuration} does not use 32 cores per node" >&2
        exit 1
    fi

    for repetition in $(seq 1 "${REPEATS}"); do
        run_configuration 1 "${ranks_per_node}" "${threads}" "${repetition}"
        run_configuration 2 "${ranks_per_node}" "${threads}" "${repetition}"
    done
done

sacct -j "${SLURM_JOB_ID}" \
    --format=JobID,JobName,Partition,State,Elapsed,NNodes,NTasks,AllocCPUS,MaxRSS \
    >"${RUN_DIR}/sacct.txt" || true

echo "Hybrid weak-scaling sweep completed. Summary: ${SUMMARY}"
cat "${SUMMARY}"
