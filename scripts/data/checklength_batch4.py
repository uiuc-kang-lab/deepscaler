import json
import os
import pandas as pd

# Load the JSON file
file_path = "/home/ubuntu/chuxuan3/rllm/data/chuxuan_coding_train_taco.json"
with open(file_path, "r") as f:
    data = json.load(f)

# Select the 4th batch (indexes 32*3 to 32*4, i.e., 96 to 127)
start_idx = 32 * 10
end_idx = 32 * 11
batch_4 = data[start_idx:end_idx]

# Create output directory
output_dir = "/home/ubuntu/chuxuan3/rllm/data"
os.makedirs(output_dir, exist_ok=True)

# Save as JSON
json_path = os.path.join(output_dir, "coding_train_taco_batch17.json")
with open(json_path, "w") as f:
    json.dump(batch_4, f, indent=2)

# Save as Parquet
df = pd.DataFrame(batch_4)
parquet_path = os.path.join(output_dir, "coding_train_taco_batch17.parquet")
df.to_parquet(parquet_path, index=False)

print(f"✅ Saved {len(batch_4)} examples to:")
print(f"    JSON: {json_path}")
print(f"    Parquet: {parquet_path}")
