#!/bin/bash 
#SBATCH -C gpu
#SBATCH -A dune_g
#SBATCH -q regular
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH --time=01:00:00
#SBATCH --image=docker:infophysics/nersc:latest
#SBATCH --module=gpu
#SBATCH -J vit-era5-mp
#SBATCH -o %x-%j.out

LOCAL_SCRATCH=/pscratch/sd/${USER:0:1}/${USER}
LOCAL_DATA=/global/cfs/cdirs/dune/users/${USER}

export FI_MR_CACHE_MONITOR=userfaultfd
export HDF5_USE_FILE_LOCKING=FALSE

# Profiling
# if [ "${ENABLE_PROFILING:-0}" -eq 1 ]; then
#     echo "Enabling profiling..."
#     NSYS_ARGS="--trace=cuda,cublas,nvtx --kill none -c cudaProfilerApi -f true"
#     NSYS_OUTPUT=${LOGDIR}/${PROFILE_OUTPUT:-"profile"}
#     export PROFILE_CMD="nsys profile $NSYS_ARGS -o $NSYS_OUTPUT"
# fi

export MASTER_ADDR=$(hostname)

# Reversing order of GPUs to match default CPU affinities from Slurm
export CUDA_VISIBLE_DEVICES=3,2,1,0

set -x
srun --mpi=pmi2 -u shifter --image=docker:infophysics/nersc:latest \
                           -V "${LOCAL_SCRATCH}:/local_scratch;${LOCAL_DATA}:/local_data" \
    bash -c "
    source /global/common/software/nersc9/nccl/2.19/env_nccl.sh
    source export_DDP_vars.sh
    ${PROFILE_CMD} blip segmentation_blip_config.yaml
    "