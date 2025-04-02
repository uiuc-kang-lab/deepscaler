from huggingface_hub import HfApi

api = HfApi()

# Create a new repository (if it doesn't exist)
api.create_repo(
    repo_id="aokellermann/deepscaler_1.5b_8k_eurus_2_math",
    repo_type="model",
    exist_ok=True,
)

# Push your model files
api.upload_folder(
    folder_path="/home/ubuntu/deepscaler/checkpoints/deepscaler-eurus-2/deepscaler-1.5b-8k-eurus-2-math/actor/global_step_20",
    repo_id="aokellermann/deepscaler_1.5b_8k_eurus_2_math",
    repo_type="model",
)
