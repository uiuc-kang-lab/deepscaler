from datasets import load_dataset
import os
import pandas as pd
import json

CODING_SOURCES = ["apps"]

def filter_by_sources(dataset, sources, include=True):
    if include:
        return dataset.filter(lambda x: x["data_source"] in sources)
    else:
        return dataset.filter(lambda x: x["data_source"] not in sources)

def safe_dumps(x):
    """Safely serialize if x is dict or list; otherwise leave as-is."""
    if isinstance(x, (dict, list)):
        return json.dumps(x)
    return x

def format_example(example, split, idx):
    return {
        "data_source": example["data_source"],
        "prompt": safe_dumps([{
            "role": "user",
            "content": example["problem"]
        }]),
        "ability": "code",
        "reward_model": safe_dumps({
            "style": "rule",
            "ground_truth": example["tests"]
        }),
        "extra_info": safe_dumps({
            "split": split,
            "index": idx,
            "reference": example.get("completion", None)
        })
    }

def process_and_save(dataset, split, output_path):
    formatted_data = []
    for idx, example in enumerate(dataset):
        formatted_example = format_example(example, split, idx)
        formatted_data.append(formatted_example)

    df = pd.DataFrame(formatted_data)
    df.to_parquet(output_path)

def main():
    # Load dataset
    ds = load_dataset("chuxuan/RL-gen-code-train-rl")

    # Create output directory
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    # Validation = only from apps
    coding_data_validation = filter_by_sources(ds["train"], CODING_SOURCES, include=True)

    # Training = everything else
    coding_data_train = filter_by_sources(ds["train"], CODING_SOURCES, include=False)

    # Save datasets
    process_and_save(coding_data_validation, split="validation", output_path=f"{output_dir}/chuxuan_coding_validation.parquet")
    process_and_save(coding_data_train, split="train", output_path=f"{output_dir}/chuxuan_coding_train.parquet")

if __name__ == "__main__":
    main()
