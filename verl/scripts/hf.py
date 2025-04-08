from huggingface_hub import HfApi

api = HfApi()

# Create a new repository (if it doesn't exist)
api.create_repo(
    repo_id="aokellermann/deepscaler_1.5b_16k_eurus_2_math",
    repo_type="model",
    exist_ok=True,
)

# Push your model files
api.upload_folder(
    folder_path="/tmp/ray/session_2025-04-03_14-54-19_806363_3017896/runtime_resources/working_dir_files/_ray_pkg_2e9fb47dc5824757/checkpoints/deepscaler-eurus-2/deepscaler-1.5b-16k-eurus-2-math/actor/global_step_540",
    repo_id="aokellermann/deepscaler_1.5b_16k_eurus_2_math",
    repo_type="model",
)
