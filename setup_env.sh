#!/bin/bash
ENV_NAME=qwenfix
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $ENV_NAME
python main.py "$@"
