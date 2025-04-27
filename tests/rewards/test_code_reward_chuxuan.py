from rllm.rewards import RewardConfig, RewardInput, RewardType
from rllm.rewards.code_reward import RewardCodeFn
import os
import json


def test_reward_taco(model_code, tests):
    """
    Test the reward function on the taco dataset.
    """
    model_response = f"""
    ```python
    {model_code}
    ```
    """
    metadata = tests
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=metadata, data_source="taco")
    output = reward(input)
    print(output.is_correct)


def test_reward_kodcode(model_code, tests):
    model_response = f"""
    ```python
    {model_code}
    ```
    """
    reward = RewardCodeFn(RewardConfig)
    input = RewardInput(problem="", problem_type=RewardType.CODE, model_response=model_response, metadata=tests, data_source="kodcode")
    output = reward(input)
    print(output.is_correct)


if __name__ == "__main__":
    output_dir = "/home/ubuntu/chuxuan3/rllm/debug_traces/"
    for filename in os.listdir(output_dir):
        if filename.endswith(".json"):
            print("checking json file: ", filename)
            file_path = os.path.join(output_dir, filename)
            with open(file_path, "r") as f:
                data = json.load(f)

            # Extract required fields
            data_source = data.get("data_source", None)
            model_code = data.get("model_code", None)
            tests = data.get("tests", None)

            if data_source == "kodcode":
                test_reward_kodcode(model_code, tests)
            else:
                test_reward_taco(model_code, tests)
