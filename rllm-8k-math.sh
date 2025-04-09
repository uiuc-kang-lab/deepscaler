#!/bin/bash

#SBATCH --job-name=rllm-8k-math
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

pip install -e ./verl
pip install -e .

python scripts/data/deepscaler_dataset.py
python scripts/data/eurus_dataset.py

./scripts/deepscaler/train/deepscaler_1.5b_8k_eurus_2_math.sh
