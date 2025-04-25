from datasets import load_dataset
import numpy as np
import os


def filter_by_ability(dataset, ability):
    return dataset.filter(lambda x: x["ability"] == ability)


def main():
    # Load dataset
    original_ds = load_dataset("virtuoussy/Multi-subject-RLVR")
    annotated_ds = load_dataset("yuxuan18/Multi-subject-RLVR-annotated")

    # Create output directory if it doesn't exist
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    # Process train split
    general_data = annotated_ds["others"].to_pandas()
    
    general_data["prompt"] = general_data["query"].apply(lambda x: np.array(eval(x.replace("}\n {", "}, {"))))
    general_data["data_source"] = "general"
    general_data["reward_model"] = general_data["label"].apply(lambda x: {'ground_truth': x, 'style': 'rule'})
    general_data["extra_info"] = general_data["label"].apply(lambda x: {'index': 0, 'split': 'dummy'})
    general_data["ability"] = "general"
    general_data.to_parquet(f"{output_dir}/mlvr_general_train.parquet")
    general_data.iloc[:1000].to_parquet(f"{output_dir}/mlvr_general_train_1000.parquet")
    
    # Process validation split
    validation_data = original_ds["test"]
    validation_data = validation_data.filter(lambda x: x["subset"] not in ["Computer Science and Technology", "Mathematics"]).to_pandas()
    validation_data = validation_data.rename(columns={"query": "prompt"})
    validation_data["data_source"] = "general"
    validation_data["reward_model"] = validation_data["label"].apply(lambda x: {'ground_truth': x, 'style': 'rule'})
    validation_data["extra_info"] = validation_data["label"].apply(lambda x: {'index': 0, 'split': 'dummy'})
    validation_data["ability"] = "general"
    validation_data.to_parquet(f"{output_dir}/mlvr_general_validation.parquet")


if __name__ == "__main__":
    main()

