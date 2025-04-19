set -x

# hdfs_path="/home/ubuntu/rl-gen/models/sft/qwen2.5-1.5b/"
# model_path="Qwen/Qwen2.5-1.5B"

# # Use torchrun for distributed training
# torchrun --nnodes=1 --nproc_per_node=8 -m verl.trainer.fsdp_sft_trainer \
#     data.train_files=/workspace/data/train.parquet \
#     data.val_files=/workspace/data/test.parquet \
#     data.response_key=label \
#     data.prompt_key=prompt \
#     data.micro_batch_size=32 \
#     data.train_batch_size=256 \
#     data.max_length=4096 \
#     data.truncation=right \
#     model.partial_pretrain=$model_path \
#     +model.system_prompt=RLVR_PROMPT \
#     trainer.default_hdfs_dir=$hdfs_path \
#     trainer.project_name=deepmath-sft \
#     trainer.experiment_name=deepmath-sft-qwen2.5-1.5b \
#     trainer.total_training_steps=4 \
#     trainer.logger=['console','wandb'] \
#     +trainer.n_gpus_per_node=8 \
#     +trainer.nnodes=1 \
#     +trainer.test_freq=4


nproc_per_node=1
HOME=/workspace
hdfs_path=/workspace/hdfs

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/train.parquet \
    data.val_files=$HOME/data/test.parquet \
    data.prompt_key=prompt \
    data.response_key=answer \
    data.max_length=4096 \
    data.truncation=right \
    data.micro_batch_size=1 \
    optim.lr=1e-4 \
    model.partial_pretrain=Qwen/Qwen2.5-1.5B \
    +model.system_prompt=DEEPSEEK_MATH_SYSTEM_PROMPT \
    trainer.project_name=deepmath-sft \
    trainer.experiment_name=deepmath-sft-qwen-2.5-1.5b \
    trainer.logger=['console','wandb'] \
    trainer.total_training_steps=4 \
    trainer.default_hdfs_dir=$hdfs_path \
    +trainer.n_gpus_per_node=1 \
    +trainer.nnodes=1 \
    +trainer.test_freq=4 \
    +trainer.entity_name=ddkang-uiuc-rl-generalization \
    # +data.response_dict_keys=['answer'] \
    # +data.prompt_dict_keys=['content'] \
    # ulysses_sequence_parallel_size=2 \
    # use_remove_padding=true
    # trainer.default_local_dir=$save_path \
