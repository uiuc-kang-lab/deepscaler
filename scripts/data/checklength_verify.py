import json
import os
import pandas as pd

# Load the JSON file
file_path = "/home/ubuntu/chuxuan3/rllm/data/chuxuan_coding_train.json"
with open(file_path, "r") as f:
    data = json.load(f)

# Compute lengths
lengths = [(idx, len(example.get("reward_model", {}).get("ground_truth", ""))) for idx, example in enumerate(data)]

# Sort by length descending
sorted_by_length = sorted(lengths, key=lambda x: x[1], reverse=True)

# Take the top 32 examples (no reformatting)
top_32 = [data[idx] for idx, _ in sorted_by_length[:32]]

# Create output directory
output_dir = "/home/ubuntu/chuxuan3/rllm/data/longest_ground_truth_subset"
os.makedirs(output_dir, exist_ok=True)

# Save as JSON
json_path = os.path.join(output_dir, "top32_longest_ground_truth.json")
with open(json_path, "w") as f:
    json.dump(top_32, f, indent=2)

# Save as Parquet
df = pd.DataFrame(top_32)
parquet_path = os.path.join(output_dir, "top32_longest_ground_truth.parquet")
df.to_parquet(parquet_path, index=False)

print(f"✅ Saved {len(top_32)} examples to:")
print(f"    JSON: {json_path}")
print(f"    Parquet: {parquet_path}")
