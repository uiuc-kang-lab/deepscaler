from datasets import load_dataset
from rllm.data.utils import fetch_live_code_bench_system_prompt
import os
import pandas as pd
import json
import ast

def filter_by_sources(dataset, sources, include=True):
    if include:
        return dataset.filter(lambda x: x["data_source"] in sources)
    else:
        return dataset.filter(lambda x: x["data_source"] not in sources)

def safe_parse(x):
    """
    Safely parse an input into a Python object (dict, list, etc.).
    
    - If already a dict or list, return as-is.
    - If a string, try json.loads first.
    - If json.loads fails, fallback to ast.literal_eval.
    - Otherwise, raise an error.
    """
    if isinstance(x, (dict, list)):
        return x
    elif isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return ast.literal_eval(x)
    else:
        raise ValueError(f"Unsupported type: {type(x)}")

def format_example(example, split, idx, max_test_length=10000, max_single_number_digits=1000):
    question = example['problem']
    tests = example['tests']
    if example["tests"] is None:
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
    
    if example.get('metadata', {}):
        print(example['metadata'])
        assert False
        tests =  safe_parse(tests)
        assert 'func_name' in example['metadata'], f"Function name is not found, check if your LCB data is preprocessed correctly: {example['metadata']}"
        if isinstance(tests, dict):
            tests['metadata'] = example['metadata']
        else:
            for test in tests:
                assert isinstance(test, dict), "Test is not a dict"
                test['metadata'] = example['metadata']
        
    ground_truth_str = json.dumps(tests)

    if split == "validation" and example["source"] == "lcbv5":
        starter_code = example.get("starter_code", None)
        question = fetch_live_code_bench_system_prompt(question, starter_code)
    if isinstance(question, dict):
        question = json.dumps(question)

    return {
        "data_source": example["data_source"] if split == "train" else example["source"],
        "prompt": [{
            "role": "user",
            "content": question
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
    # # Load dataset
    ds = load_dataset("chuxuan/RL-gen-code-train-rl")

    # # Create output directory
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    # # Filter out apps, lcbv5, primeintellect
    exclude_sources = ["apps", "lcbv5", "primeintellect"]
    filtered_data = ds["train"].filter(lambda x: x["data_source"] not in exclude_sources)

    # # Shuffle the filtered dataset
    coding_data_train = filtered_data.shuffle(seed=42)

    # # Save datasets
    process_and_save(coding_data_train, split="train", output_basename=f"{output_dir}/chuxuan_coding_train")

    ds_val = load_dataset("chuxuan/RL-gen-code-test")
    exclude_sources = ["lcbv5"]
    coding_data_validation = ds_val["train"].filter(lambda x: x["source"] not in exclude_sources)
    process_and_save(coding_data_validation, split="validation", output_basename=f"{output_dir}/chuxuan_coding_validation")

if __name__ == "__main__":
    main()
