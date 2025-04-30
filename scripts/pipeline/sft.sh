#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Set default model path if not provided
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="Qwen/Qwen2.5-1.5B"
fi

rllm_root=$(realpath "$(dirname "${BASH_SOURCE[0]}")"/../../)


torchrun --standalone --nnodes=1 --nproc_per_node=4 -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$rllm_root/data/math_code_mlvr_scs_train.parquet \
    data.val_files=$rllm_root/data/math_code_mlvr_scs_test.parquet \
    data.prompt_key=prompt \
    data.response_key=answer \
    data.max_length=2048 \
    data.truncation=right \
    data.micro_batch_size_per_gpu=2 \
    data.train_batch_size=64 \
    model.partial_pretrain=Qwen/Qwen2.5-1.5B \
    use_remove_padding=true \
    trainer.project_name=deepmath-sft \
    trainer.logger=['console','wandb'] \
    trainer.project_name='sft' \
    trainer.experiment_name='sft-all' \
    trainer.experiment_name=deepmath-sft-qwen-2.5-1.5b \
    trainer.total_training_steps=30000 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$rllm_root/checkpoints/sft-all \
    +trainer.n_gpus_per_node=4 \
    +trainer.nnodes=1 \
    +trainer.test_freq=100 \
    +trainer.save_freq=2000 \
    +trainer.entity_name=ddkang-uiuc-rl-generalization
        # ulysses_sequence_parallel_size=2 \

    # trainer.resume_from_path=/workspace/checkpoints/deepseek-sft \
    # optim.lr=1e-4 \
    # trainer.resume_mode=auto \
    # +data.response_dict_keys=['answer'] \
    # +data.prompt_dict_keys=['content'] \
    # ulysses_sequence_parallel_size=2 \
    # use_remove_padding=true
    # trainer.default_local_dir=$save_path \
