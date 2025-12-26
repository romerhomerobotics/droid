#!/bin/bash

args=("$@")

set --

# activate conda
source ~/miniconda3/bin/activate
conda activate robot

# run user command
exec $args
