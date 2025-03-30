sync:
	ansible-playbook -v -i inventory.ini setup.yml

dataset:
	python3 scripts/data/eurus_dataset.py

train-math:
	./scripts/train/run_deepscaler_1.5b_8k_eurus_2_math.sh
