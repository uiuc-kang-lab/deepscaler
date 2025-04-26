from datasets import load_dataset
import os
import pandas as pd
import json

CODING_SOURCES = ["apps"]
CODING_SOURCES_MORE = ["apps", "lcbv5", "primeintellect"]

def filter_by_sources(dataset, sources, include=True):
    if include:
        return dataset.filter(lambda x: x["data_source"] in sources)
    else:
        return dataset.filter(lambda x: x["data_source"] not in sources)

def format_example(example, split, idx, max_test_length=10000, max_single_number_digits=1000):
    if example["tests"] is None:
        return 1
    
    try:
        # Serialize tests to JSON
        ground_truth_str = json.dumps(example["tests"])
    except (TypeError, ValueError) as e:
        print(f"Error serializing tests at index {idx}: {e}")
        return 1
    
    # Check inside the tests for any single giant number
    def has_giant_number(obj):
        if isinstance(obj, int) and len(str(abs(obj))) > max_single_number_digits:
            return True
        if isinstance(obj, list):
            return any(has_giant_number(x) for x in obj)
        if isinstance(obj, dict):
            return any(has_giant_number(v) for v in obj.values())
        return False
    
    if has_giant_number(example["tests"]):
        print(f"Skipping example {idx}: contains giant number")
        return 1

    return {
        "data_source": example["data_source"],
        "prompt": [{
            "role": "user",
            "content": example["problem"]
        }],
        "ability": "code",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth_str
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "reference": example.get("completion", None)
        }
    }


def process_and_save(dataset, split, output_basename):
    cnt = 0
    formatted_data = []
    for idx, example in enumerate(dataset):
        formatted_example = format_example(example, split, idx)
        if formatted_example == 1:
            cnt += 1
            continue
        formatted_data.append(formatted_example)
    print(cnt)

    df = pd.DataFrame(formatted_data)

    # Save to both Parquet and JSON (🔥 match their script exactly)
    df.to_parquet(f"{output_basename}.parquet")
    df.to_json(f"{output_basename}.json", orient="records")

def main():
    # Load dataset
    ds = load_dataset("chuxuan/RL-gen-code-train-rl")

    # Create output directory
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    # Filter out apps, lcbv5, primeintellect
    exclude_sources = ["apps", "lcbv5", "primeintellect"]
    filtered_data = ds["train"].filter(lambda x: x["data_source"] not in exclude_sources)

    # Shuffle the filtered dataset
    full_data = filtered_data.shuffle(seed=42)

    # Take 128 examples for validation
    coding_data_validation = full_data.select(range(128))

    # Take the rest for training
    coding_data_train = full_data.select(range(128, len(full_data)))

    # Save datasets
    process_and_save(coding_data_validation, split="validation", output_basename=f"{output_dir}/chuxuan_coding_validation")
    process_and_save(coding_data_train, split="train", output_basename=f"{output_dir}/chuxuan_coding_train")

if __name__ == "__main__":
    main()
