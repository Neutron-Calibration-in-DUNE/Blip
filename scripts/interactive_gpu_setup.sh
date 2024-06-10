#!/bin/bash

salloc --nodes=1 --constraint gpu --account dune_g --qos interactive --ntasks-per-node=1 \
       --cpus-per-task=32 --gpus-per-node=1 --time=04:00:00 