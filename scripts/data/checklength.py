import json

# Load the JSON file
file_path = "/home/ubuntu/chuxuan3/rllm/data/chuxuan_coding_train.json"

with open(file_path, "r") as f:
    data = json.load(f)

# Define the keyword to search for
keyword = "test_rotate_matrix_3x3"

# Find all matching examples
matches = []
for idx, example in enumerate(data):
    if idx == 112:
        print(example)
        assert False
    # ground_truth = example.get("reward_model", {}).get("ground_truth", "")
    # if keyword in ground_truth:
    #     print(example)
    #     assert False
    #     matches.append((idx, example))

# Print results
print(f"Found {len(matches)} matching examples containing '{keyword}':\n")
for idx, match in matches:
    print(f"--- Example index {idx} ---")
    print(json.dumps(match, indent=2))
    print("\n")

# Optionally, save matches to a file
output_path = "/home/ubuntu/chuxuan3/rllm/data/found_make_palindrome_examples.json"
with open(output_path, "w") as f:
    json.dump([ex for _, ex in matches], f, indent=2)

print(f"\n✅ Saved matching examples to {output_path}")
