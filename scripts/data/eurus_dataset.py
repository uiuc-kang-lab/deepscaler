from datasets import load_dataset
import os


def filter_by_ability(dataset, ability):
    return dataset.filter(lambda x: x["ability"] == ability)


def main():
    # Load dataset
    ds = load_dataset("PRIME-RL/Eurus-2-RL-Data")

    # Create output directory if it doesn't exist
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    # Process each split
    for split in ["train", "validation"]:
        # Filter math and coding data
        math_data = filter_by_ability(ds[split], "math")
        coding_data = filter_by_ability(ds[split], "code")

        # Save as parquet files
        math_data.to_parquet(f"{output_dir}/eurus_math_{split}.parquet")
        coding_data.to_parquet(f"{output_dir}/eurus_coding_{split}.parquet")

        print(f"Processed {split} split:")
        print(f"Math data: {len(math_data)} rows")
        print(f"Coding data: {len(coding_data)} rows")


if __name__ == "__main__":
    main()
