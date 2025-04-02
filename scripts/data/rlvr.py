from datasets import load_dataset
import os


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

    # Save validation set
    val_ds.to_parquet(f"{cache_dir}/math_rlvr_validation.parquet")

    # Replace the train split with the reduced version
    train_ds.to_parquet(f"{cache_dir}/math_rlvr_train.parquet")
    ds["test"].to_parquet(f"{cache_dir}/math_rlvr_test.parquet")

    print(f"Processed train split: {len(train_ds)} rows")
    print(f"Created validation split: {len(val_ds)} rows")
    print(f"Processed test split: {len(ds['test'])} rows")


if __name__ == "__main__":
    main()
