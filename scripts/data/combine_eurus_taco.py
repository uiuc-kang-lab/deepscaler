import json
import pandas as pd
import os
import random

# Paths
dir_path = "/home/ubuntu/chuxuan3/rllm/data/"
eurus_path = os.path.join(dir_path, "chuxuan_coding_train_eurus.json")
taco_path = os.path.join(dir_path, "chuxuan_coding_train_taco.json")

# Load the JSON files
with open(eurus_path, 'r') as f:
    eurus_data = json.load(f)

with open(taco_path, 'r') as f:
    taco_data = json.load(f)

# Concatenate the data
combined_data = eurus_data + taco_data

# Shuffle the combined data
random.seed(42)  # for reproducibility
random.shuffle(combined_data)

# Output filenames
json_output_path = os.path.join(dir_path, "chuxuan_coding_train_eurus_taco.json")
parquet_output_path = os.path.join(dir_path, "chuxuan_coding_train_eurus_taco.parquet")

# Save combined JSON
with open(json_output_path, 'w') as f:
    json.dump(combined_data, f, indent=2)

# Save combined Parquet
df = pd.DataFrame(combined_data)
df.to_parquet(parquet_output_path, index=False)

print(f"✅ Saved shuffled combined JSON to {json_output_path}")
print(f"✅ Saved shuffled combined Parquet to {parquet_output_path}")
