import os
import numpy as np
import argparse
import datasets

from datasets import load_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/deepmath')
    args = parser.parse_args()

    # train data
    train_data = load_dataset("zwhe99/DeepMath-103K", split='train')
    def process_fn_train(example, idx):
        data = {
            "data_source": "deepmath-103k",
            "prompt":  example["question"],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": example["final_answer"]
            },
            "extra_info": {
                'split': 'train',
                'index': idx,
                'answer': example["r1_solution_1"],
                "question": example["question"],
            },
            "answer": example["r1_solution_1"]
        }
        return data

    # test data
    test_dataset = datasets.load_dataset('zwhe99/MATH', split='math500')
    def process_fn_test(example, idx):
        data = {
            "data_source": "math500",
            "prompt": example["problem"],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": example["expected_answer"],
            },
            "extra_info": {
                "split": "math500",
                "index": idx,
                "answer": example["expected_answer"],
                "question": example["problem"],
            },
            "answer": example["expected_answer"]
        }
        return data

    train_dataset = train_data.map(function=process_fn_train, with_indices=True)
    test_dataset = test_dataset.map(function=process_fn_test, with_indices=True)
    train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
