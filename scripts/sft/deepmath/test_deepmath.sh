set -x

hdfs_path="/home/ubuntu/rl-gen/models/sft/qwen2.5-1.5b/"
model_path="Qwen/Qwen2.5-1.5B"

# Use torchrun for distributed training
torchrun --nnodes=1 --nproc_per_node=8 -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/workspace/data/train.parquet \
    data.val_files=/workspace/data/test.parquet \
    data.response_key=label \
    data.prompt_key=prompt \
    data.micro_batch_size=32 \
    data.train_batch_size=256 \
    data.max_length=4096 \
    data.truncation=right \
    model.partial_pretrain=$model_path \
    +model.system_prompt=RLVR_PROMPT \
    trainer.default_hdfs_dir=$hdfs_path \
    trainer.project_name=deepmath-sft \
    trainer.experiment_name=deepmath-sft-qwen2.5-1.5b \
    trainer.total_training_steps=4 \
    trainer.logger=['console','wandb'] \
    +trainer.n_gpus_per_node=8 \
    +trainer.nnodes=1 \
    +trainer.test_freq=4
