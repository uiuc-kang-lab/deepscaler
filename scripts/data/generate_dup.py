import json
import os
import pandas as pd

# Define the single instance
single_instance = {
    "data_source": "eurus",
    "prompt": [
        {
            "role": "user",
            "content": "Given a binary string, that is it contains only 0s and 1s. We need to make this string a sequence of alternate characters by flipping some of the bits, our goal is to minimize the number of bits to be flipped.\nExample 1:\nInput:\nS = \"001\"\nOutput: 1\nExplanation: \nWe can flip the 0th bit to 1 to have\n101.\nExample 2:\nInput:\nS = \"0001010111\" \nOutput: 2\nExplanation: We can flip the 1st and 8th bit \nbit to have \"0101010101\"\nYour Task:\nYou don't need to read input or print anything. Your task is to complete the function minFlips() which takes the string S as input and returns the minimum number of flips required.\nExpected Time Complexity: O(|S|).\nExpected Auxiliary Space: O(1).\nConstraints:\n1<=|S|<=10^{5}\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n``` at the end."
        }
    ],
    "ability": "code",
    "reward_model": {
        "style": "rule",
        "ground_truth": "{\"fn_name\": \"minFlips\", \"inputs\": [[\"01010101\"], [\"11111111\"], [\"00000000\"], [\"10101010\"], [\"00110101\"], [\"0\"], [\"1\"], [\"10\"], [\"11\"], [\"00\"]], \"outputs\": [[0], [4], [4], [0], [2], [0], [0], [0], [1], [1]]}"
    },
    "extra_info": {
        "split": "train",
        "index": 119,
        "reference": None
    }
}

# Generate 32 copies
batch_32 = [single_instance for _ in range(32)]

# Create output directory
output_dir = "/home/ubuntu/chuxuan3/rllm/data/coding_train_batches"
os.makedirs(output_dir, exist_ok=True)

# Save as JSON
json_path = os.path.join(output_dir, "fn_dup.json")
with open(json_path, "w") as f:
    json.dump(batch_32, f, indent=2)

# Save as Parquet
df = pd.DataFrame(batch_32)
parquet_path = os.path.join(output_dir, "fn_dup.parquet")
df.to_parquet(parquet_path, index=False)

print(f"✅ Saved {len(batch_32)} examples to:")
print(f"    JSON: {json_path}")
print(f"    Parquet: {parquet_path}")