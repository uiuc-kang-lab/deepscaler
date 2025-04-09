#!/bin/bash

#SBATCH --job-name=rllm-8k-coding
#SBATCH --output=log.%x.job_%j
#SBATCH --time=2-00:00:00
#SBATCH --partition=schmidt_sciences
#SBATCH --exclusive

#SBATCH --mail-type=ALL
#SBATCH --mail-user=slack:U08LFGDSU8P

#SBATCH --nodes=1
#SBATCH --gpus=8

set -ex

cd /data/antony_kellermann/rllm

. .venv/bin/activate

# pip install -e ./verl
# pip install -e .

# python scripts/data/deepscaler_dataset.py
# python scripts/data/eurus_dataset.py

./scripts/deepcoder/train/deepcoder_1.5b_coding_8k.sh
