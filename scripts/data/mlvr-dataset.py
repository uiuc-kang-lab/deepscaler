from datasets import load_dataset
import numpy as np
import os
from typing import Dict, List, Optional, Any
import pandas as pd


def make_map_fn_train(split: str):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that processes individual dataset examples
    """
    def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        question = eval(example.pop('query').replace("}\n {", "}, {"))
        instruction = "Let's think step by step and output the final answer within \\boxed{}."
        question = f"{question[1]['content']} {instruction}"
        answer = example.pop('label')

        data = {
            "data_source": "",
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "general",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }
        return data
    return process_fn


def make_map_fn_test(split: str):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that processes individual dataset examples
    """
    def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        question = example.pop('query')
        instruction = "Let's think step by step and output the final answer within \\boxed{}."
        question = f"{question[1]['content']} {instruction}"
        answer = example.pop('label')
        subject = example.pop('subject')

        data = {
            "data_source": subject,
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "general",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }
        return data
    return process_fn


if __name__ == '__main__':

    # Load dataset
    original_ds = load_dataset("virtuoussy/Multi-subject-RLVR")['test']
    annotated_ds = load_dataset("yuxuan18/Multi-subject-RLVR-annotated")['others']

    # Process training data
    train_data: List[Dict[str, Any]] = []
    process_fn = make_map_fn_train('others')
    for idx, example in enumerate(annotated_ds):
        processed_example = process_fn(example, idx)
        if processed_example is not None:
            train_data.append(processed_example)

    # Process and save each test dataset separately
    test_data: List[Dict[str, Any]] = []
    process_fn = make_map_fn_test('test')
    for idx, example in enumerate(original_ds):
        processed_example = process_fn(example, idx)
        if processed_example is not None:
            test_data.append(processed_example)

    test_df = pd.DataFrame(test_data)
    test_df.to_parquet(os.path.join('data', f'mlvr_test.parquet'))
    print(f"mlvr test data size:", len(test_data))

    # Save training dataset
    print("train data size:", len(train_data))
    train_df = pd.DataFrame(train_data)
    train_df.to_parquet(os.path.join('data', 'mlvr_train.parquet'))
