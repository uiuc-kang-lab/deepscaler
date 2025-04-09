install:
	pip install --upgrade pip setuptools wheel
	pip install -e ./verl
	pip install -e .

dataset:
	python scripts/data/deepscaler_dataset.py
	python scripts/data/eurus_dataset.py

train-math:
	./scripts/train/run_deepscaler_1.5b_8k_eurus_2_math.sh

deploy:
	rsync -vzr --delete --size-only --no-owner --no-group --exclude=".venv" --exclude="wandb" --exclude="checkpoints"  --exclude="outputs" --exclude="prometheus-3.2.1.linux-amd64*" --exclude="data" --exclude="*.egg-info" --exclude="*.pyc" --exclude="__pycache__" --exclude="*.git" --exclude="*.DS_Store" . lambda:/home/ubuntu/rl-gen/deepscaler
