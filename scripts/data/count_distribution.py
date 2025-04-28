import json
from collections import defaultdict

# Load the JSON file
file_path = "/home/ubuntu/chuxuan3/rllm/data/chuxuan_coding_train.json"
with open(file_path, "r") as f:
    data = json.load(f)

# Initialize counters
count_fn_name = 0
source_counter = defaultdict(int)

# Process each example
for example in data:
    ground_truth = example.get("reward_model", {}).get("ground_truth", "")
    if "fn_name" in ground_truth:
        count_fn_name += 1
        source = example.get("data_source", "unknown")
        source_counter[source] += 1

# Print results
print(f"Total examples with 'fn_name' in ground_truth: {count_fn_name}")
print("\nBreakdown by data_source:")
for source, count in source_counter.items():
    print(f"  {source}: {count}")
