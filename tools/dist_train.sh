#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
GPUS=$2

#$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=11002\
#    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

#CUDA_VISIBLE_DEVICES='4,5' 
$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=11003\
    $(dirname "$0")/train.py $CONFIG --seed 0 --launcher pytorch ${@:3}
