#!/bin/bash

LOCAL_SCRATCH=/pscratch/sd/${USER:0:1}/${USER}
LOCAL_DATA=/global/cfs/cdirs/dune/users/${USER}

# Allocate an interactive session
salloc --nodes=1 --constraint gpu --account dune_g --qos interactive --ntasks-per-node=1 \
       --cpus-per-task=32 --gpus-per-node=1 --time=02:00:00 \
       srun --image=docker:infophysics/nersc:latest shifter --volume="${LOCAL_SCRATCH}:/local_scratch;${LOCAL_DATA}:/local_data" \
        -- bash