import json
import os
import pandas as pd
import copy

# Load the JSON file
file_path = "/home/ubuntu/chuxuan3/rllm/data/chuxuan_coding_train.json"
with open(file_path, "r") as f:
    data = json.load(f)

# Select the 4th batch (indexes 32*3 to 32*4, i.e., 96 to 127)
start_idx = 32 * 3
end_idx = 32 * 4
batch_4 = data[start_idx:end_idx]

# Step 1: Filter out the instance with index == 119
batch_4_filtered = [item for item in batch_4 if item.get("extra_info", {}).get("index") != 119]

# Step 2: Duplicate one instance to make the count back to 32
# (Here, we simply duplicate the first example. You can change to another.)
if len(batch_4_filtered) == 31:
    duplicate_instance = copy.deepcopy(batch_4_filtered[0])
    # Optionally, update its index to a new unused one to avoid exact duplicate
    duplicate_instance["extra_info"]["index"] = 9999  # or any number not used
    batch_4_filtered.append(duplicate_instance)

# Step 3: Confirm size
assert len(batch_4_filtered) == 32, f"Batch size after processing is {len(batch_4_filtered)}, expected 32."

# Create output directory
output_dir = "/home/ubuntu/chuxuan3/rllm/data/coding_train_batches"
os.makedirs(output_dir, exist_ok=True)

# Save as JSON
json_path = os.path.join(output_dir, "excluding_119.json")
with open(json_path, "w") as f:
    json.dump(batch_4_filtered, f, indent=2)

# Save as Parquet
df = pd.DataFrame(batch_4_filtered)
parquet_path = os.path.join(output_dir, "excluding_119.parquet")
df.to_parquet(parquet_path, index=False)

print(f"✅ Saved {len(batch_4_filtered)} examples (after filtering and duplicating) to:")
print(f"    JSON: {json_path}")
print(f"    Parquet: {parquet_path}")
