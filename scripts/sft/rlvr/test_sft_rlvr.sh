set -x

hdfs_path="/home/ubuntu/rl-gen/models/sft/qwen2.5-1.5b/"
model_path="Qwen/Qwen2.5-1.5B"

# Use torchrun for distributed training
torchrun --nnodes=1 --nproc_per_node=8 -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/rl-gen/data/rlvr/math_rlvr_train.parquet \
    data.val_files=$HOME/rl-gen/data/rlvr/math_rlvr_validation.parquet \
    data.prompt_key=query \
    data.response_key=label \
    data.micro_batch_size=8 \
    model.partial_pretrain=$model_path \
    trainer.default_hdfs_dir=$hdfs_path \
    trainer.project_name=rlvr-sft \
    trainer.experiment_name=rlvr-sft-qwen2.5-1.5b \
    trainer.total_training_steps=4 \
    trainer.logger=['console','wandb'] \
    +trainer.n_gpus_per_node=8 \
    +trainer.nnodes=1 \
    +trainer.test_freq=4
