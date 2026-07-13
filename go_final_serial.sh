#!/bin/bash
#SBATCH --job-name=stencil_final_serial
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
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
RUN_DIR=${RESULTS_ROOT}/${SLURM_JOB_ID}_final_serial
RAW_DIR=${RUN_DIR}/raw
JOB_BUILD_DIR=${RUN_DIR}/build
SUMMARY=${RUN_DIR}/go_final_serial.csv

ARGS=${ARGS:-"-x 25000 -y 25000 -n 200 -o 0"}
REPEATS=${REPEATS:-1}
read -r -a RUN_ARGS <<< "${ARGS}"

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

echo "Job ID      : ${SLURM_JOB_ID}"
echo "Node list   : ${SLURM_JOB_NODELIST:-unknown}"
echo "Arguments   : ${ARGS}"
echo "Repetitions : ${REPEATS}"

{
    echo "date=$(date --iso-8601=seconds)"
    echo "job_id=${SLURM_JOB_ID}"
    echo "partition=${SLURM_JOB_PARTITION:-unknown}"
    echo "nodes=${SLURM_JOB_NODELIST:-unknown}"
    echo "arguments=${ARGS}"
    echo "openmp=disabled"
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

make BUILD_DIR="${JOB_BUILD_DIR}" final-serial 2>&1 | tee "${RUN_DIR}/build.log"

printf 'run_name,t_wall,t_get_total_energy,t_update_plane,t_inject_energy,glups\n' >"${SUMMARY}"

for optimization in O1 O3; do
    executable=${JOB_BUILD_DIR}/stencil_serial_final_${optimization}
    for repetition in $(seq 1 "${REPEATS}"); do
        run_name=final_${optimization}_rep$(printf '%02d' "${repetition}")
        save_run "${run_name}" \
            srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=1 \
            --cpu-bind=cores "${executable}" "${RUN_ARGS[@]}"
    done
done

sacct -j "${SLURM_JOB_ID}" \
    --format=JobID,JobName,Partition,State,Elapsed,NNodes,NTasks,AllocCPUS,MaxRSS \
    >"${RUN_DIR}/sacct.txt" || true

echo "Completed. Summary: ${SUMMARY}"
cat "${SUMMARY}"
