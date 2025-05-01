from datasets import load_dataset
import os


def filter_by_ability(dataset, ability):
    return dataset.filter(lambda x: x["ability"] == ability)


def main():
    # Load dataset
    ds = load_dataset("uiuc-kang-lab/eurus-2-math-100k")

    # Create output directory if it doesn't exist
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    # Process each split
    for split in ["train", "validation"]:
        # Filter math and coding data
        math_data = filter_by_ability(ds[split], "math")
        # coding_data = filter_by_ability(ds[split], "code").filter(lambda x: x["data_source"] != "codeforces")

        # Save as parquet files
        math_data.to_parquet(f"{output_dir}/eurus_math_{split}_100k.parquet")
        # coding_data.to_parquet(f"{output_dir}/eurus_coding_{split}_100k.parquet")

        print(f"Processed {split} split:")
        print(f"Math data: {len(math_data)} rows")
        # print(f"Coding data: {len(coding_data)} rows")


if __name__ == "__main__":
    main()
