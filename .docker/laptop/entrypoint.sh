#!/bin/bash

args=("$@")

set --

# activate conda
source ~/miniconda3/bin/activate
conda activate robot

export PYTHONUNBUFFERED=1

# run user command
exec $args
