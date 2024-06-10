#!/bin/bash

LOCAL_SCRATCH=/pscratch/sd/${USER:0:1}/${USER}
LOCAL_ARRAKIS=/global/cfs/cdirs/dune/users/${USER}
LOCAL_DATA=/global/cfs/cdirs/dune/users/${USER}

shifter --image=docker:infophysics/nersc:latest \
        --volume="${LOCAL_SCRATCH}:/local_scratch;${LOCAL_ARRAKIS}:/local_arrakis;${LOCAL_DATA}:/local_data" \
        bash