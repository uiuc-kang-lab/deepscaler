from datasets import load_dataset
import os
import json


def extract_prompt_from_rlvr(query):
    """
    Extract the actual math problem from the RLVR format.
    
    Format:
    query: [ 
        { "content": "system prompt", "role": "system" }, 
        { "content": "actual math problem", "role": "user" } 
    ]
    """
    return next((message for message in query if message.get("role") == "user"))["content"]

def process_example(example):
    return {
        "prompt": extract_prompt_from_rlvr(example["query"]),
        "label": example["label"]
    }


def main():
    # Load dataset
    cache_dir = os.path.expanduser("~/rl-gen/data/rlvr")
    os.makedirs(cache_dir, exist_ok=True)
    ds = load_dataset("virtuoussy/Math-RLVR", cache_dir=cache_dir)

    # Process each split
    # Shuffle the dataset with a fixed seed for reproducibility
    shuffled_ds = ds["train"].shuffle(seed=42)

    # Calculate the size for validation (15% of train)
    val_size = int(len(shuffled_ds) * 0.15)

    # Split the dataset
    train_ds = shuffled_ds.select(range(len(shuffled_ds) - val_size))
    val_ds = shuffled_ds.select(range(len(shuffled_ds) - val_size, len(shuffled_ds)))

    train_ds = train_ds.map(process_example)
    val_ds = val_ds.map(process_example)
    test_ds = ds["test"].map(process_example)

    # Save validation set
    val_ds.to_parquet(f"{cache_dir}/math_rlvr_validation.parquet")

    # Replace the train split with the reduced version
    train_ds.to_parquet(f"{cache_dir}/math_rlvr_train.parquet")
    test_ds.to_parquet(f"{cache_dir}/math_rlvr_test.parquet")

    print(f"Processed train split: {len(train_ds)} rows")
    print(f"Created validation split: {len(val_ds)} rows")
    print(f"Processed test split: {len(test_ds)} rows")


if __name__ == "__main__":
    main()
