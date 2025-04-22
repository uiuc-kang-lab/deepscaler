grading_examples = """\
Question 1: How long should one fast before collecting samples for triglycerides (TG) and other lipoproteins?
Ground truth answer: 12 hours
Candidate answers:
1. 12 hours
2. 12 hrs
3. 8 hours
4. 12 h
5. a day
6. 12 hours fasting
7. 1.2
8. 12
9. half of a day
10. 12.0
Response: [True, True, False, True, False, True, False, True, True, True]

Question 2: The subjective aspect of the crime of traffic accident ( )
Ground truth answer 2: Can only be negligent, cannot be intentional
Candidate answers:
1. Can only be negligent, cannot be intentional
2. Can be negligent or intentional
3. Can be intentional
4. Can be negligent
5. Can be negligent but not intentional
6. only negligent not intentional
7. Can be intentional but not negligent
8. negligent not intentional
9. should be negligent but not intentional
10. Can be negligent or intentional but not both
Response: [True, False, False, True, True, True, False, True, True, False]

Question 3: The opening degree refers to the distance between the cutting edges of the upper and lower central incisors when the patient opens their mouth wide. The normal opening degree for a person is approximately
Ground truth answer: 3.7~4.5 cm
Candidate answers:
1. 3.7~4.5 cm
2. between 3.7 and 4.5 cm
3. 3.7 to 4.5 cm
4. 3～4cm
5. 3.7 cm
6. 4.5 cm
7. 3.7cm - 4.5 cm
8. 3.7 cm to 4.5 cm
9. 3.7~4.5
10. 3.7-4.5 cm
Response: [True, True, True, False, False, False, True, True, False, True]
"""

grading_prompt = """\
Given a question, the ground truth answer, and a list of candidate answwers, determine the correctness of each candidate answer. 

Here are examples:
{grading_examples}

The question is: {question}
The ground truth answer is: {ground_truth_answer}
The candidate answers are: 
{candidate_answers}
Please respond with a list of boolean values, where each value corresponds to the correctness of the candidate answer at the same index. Respond with "True" if the candidate answer is correct, and "False" if it is incorrect. Respond only with the list of boolean values, without any additional text or explanation.

Your response is:
"""

grading_code_prompt = """\
Given a question, and the ground truth answer, write a Python function that takes a candidate answer and returns a boolean value indicating whether the candidate answer is correct or not. The function has the following requirements:
1. The function should be named `is_answer_correct` and should take one argument: `candidate_answer` (a string). 
2. The function should return `True` if the candidate answer is correct, and `False` if it is incorrect. 
3. To test the function, you are provided with 10 example candidate answers and their correctness. But the function should be able to handle any candidate answer, not just the examples provided.
4. The function should be able to handle different formats of the candidate answer, including but not limited to:
- Different units (e.g., "12 hours", "12 hrs", "12 h")
- Different representations of the same value (e.g., "3.7~4.5 cm", "between 3.7 and 4.5 cm", "3.7 to 4.5 cm")
5. The function should be able to handle variations in wording and phrasing of the candidate answer.
6. The function should be able to handle different languages and character sets, if applicable.

The question is: {question}
The ground truth answer is: {ground_truth_answer}
The candidate answers are:
{candidate_answers}
The correctness of the candidate answers is:
{grading_results}

Write the function `is_answer_correct` below:
```python
def is_answer_correct(candidate_answer):
    # Your code here
    pass
```

Respond only with the function definition and the code inside the function. Do not include any additional text or explanation.

Your response:
"""

import tiktoken
import openai
import os
from datasets import load_dataset
import json
import argparse
import multiprocessing
import threading

def get_llm_response(prompt, client):
    n_input_tokens = len(tiktoken.encoding_for_model("o3-mini").encode(prompt))
    response = client.chat.completions.create(
        model="o3-mini-2025-01-31",
        messages=[
            {"role": "user", "content": prompt}
        ])
    n_output_tokens = len(tiktoken.encoding_for_model("o3-mini").encode(response.choices[0].message.content))
    cost = n_input_tokens / 1e6 * 1.1 + n_output_tokens / 1e6 * 4.4
    return response.choices[0].message.content, n_input_tokens, n_output_tokens, cost


def gen_answer(data, n_try=10):
    print("Generating answer...")
    question = data["query"][0]["content"] + " " + data["query"][1]["content"]
    client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])
    answers = []
    total_cost = 0
    for _ in range(n_try):
        response, _, _, cost = get_llm_response(question, client)
        try:
            answer = response.split("Therefore, the answer is")[1].split(".")[0].strip()
        except IndexError:
            try:
                answer = response.split("The answer is")[1].split(".")[0].strip()
            except IndexError:
                try:
                    answer = response.split("The answer is")[1].strip()
                except IndexError:
                    answer = "I don't know."
        answers.append(answer)
        total_cost += cost
    return answers, total_cost

def gen_grading_code(data, answers, grading_results):
    print("Generating grading code...")
    question = data["query"][1]["content"]
    ground_truth_answer = data["label"]
    candidate_answers = ""
    for i, answer in enumerate(answers):
        candidate_answers += f"{i + 1}. {answer}\n"
    candidate_answers = candidate_answers.strip()
    prompt = grading_code_prompt.format(
        question=question,
        ground_truth_answer=ground_truth_answer,
        candidate_answers=candidate_answers,
        grading_results=grading_results
    )
    client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])
    response, _, _, cost = get_llm_response(prompt, client)
    return response, cost
    

def gen_grading(data, answers):
    print("Grading answer...")
    question = data["query"][1]["content"]
    ground_truth_answer = data["label"]
    candidate_answers = ""
    for i, answer in enumerate(answers):
        candidate_answers += f"{i + 1}. {answer}\n"
    candidate_answers = candidate_answers.strip()
    prompt = grading_prompt.format(
        grading_examples=grading_examples,
        question=question,
        ground_truth_answer=ground_truth_answer,
        candidate_answers=candidate_answers
    )
    client = openai.Client(api_key=os.environ["OPENAI_API_KEY"])
    response, _, _, cost = get_llm_response(prompt, client)
    return response, cost

def execute_code(function: str, input_str: str):
    # Create a local scope for the function
    local_scope = {}
    exec(function, local_scope)
    # Call the function with the input string
    result = local_scope["is_answer_correct"](input_str)
    return result


def verify_code(code, candidate_answers, grading_results):
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]
    grading_results = eval(grading_results)
    matching = []
    for candidate_answer, grading_result in zip(candidate_answers, grading_results):
        # run code in a separate process
        with multiprocessing.Pool(processes=1) as pool:
            result = pool.apply_async(execute_code, (code, candidate_answer))
            try:
                output = result.get(timeout=5)
            except multiprocessing.TimeoutError:
                print("Timeout error")
                matching.append(-1)
                continue
            except Exception as e:
                print(f"Error executing code: {e}")
                matching.append(-1)
                continue
            if output != grading_result:
                print(f"Grading for answer '{candidate_answer}' is incorrect. Expected {grading_result}, got {output}")
                matching.append(0)
            else: 
                matching.append(1)
    return matching

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate verifiable answers and grading.")
    parser.add_argument("--answers", action="store_true", help="Generate answers")
    parser.add_argument("--limit", type=int, default=10, help="Limit the number of samples to process")
    parser.add_argument("--n_try", type=int, default=10, help="Number of attempts to generate answers")
    parser.add_argument("--grading", action="store_true", help="Generate grading")
    parser.add_argument("--code", action="store_true", help="Generate grading code")
    parser.add_argument("--verify", action="store_true", help="Verify grading code")
    args = parser.parse_args()
    
    if args.answers:
        dataset = load_dataset("virtuoussy/Multi-subject-RLVR")["train"].to_pandas()
        total_cost = 0
        for i, data in dataset.iterrows():
            answers, gen_cost = gen_answer(data, n_try=args.n_try)
            print(f"Answers: {answers}, Generation cost: {gen_cost}")
            print("-" * 50)
            data["answers"] = list(answers)
            data["query"] = data["query"].tolist()
            data["generation_cost"] = gen_cost
            total_cost += gen_cost
            with open("answers-gpt4o.jsonl", "a") as f:
                f.write(json.dumps(data.to_dict()) + "\n")
            if i == args.limit:
                break
        print(f"Total generation cost: {total_cost}")
    if args.grading:
        with open("answers-gpt4o.jsonl", "r") as f:
            total_cost = 0
            for i, line in enumerate(f):
                data = json.loads(line)
                answers = data["answers"]
                grading_response, grading_cost = gen_grading(data, answers)
                total_cost += grading_cost
                print(f"Grading response: {grading_response}, Grading cost: {grading_cost}")
                print("-" * 50)
                data["grading"] = grading_response
                data["grading_cost"] = grading_cost
                with open("grading-gpt4o.jsonl", "a") as f:
                    f.write(json.dumps(data) + "\n")
            print(f"Total grading cost: {total_cost}")
    if args.code:
        with open("grading.jsonl") as f:
            total_cost = 0
            for i, line in enumerate(f):
                data = json.loads(line)
                answers = data["answers"]
                grading_results = data["grading"]
                grading_code_response, grading_code_cost = gen_grading_code(data, answers, grading_results)
                total_cost += grading_code_cost
                print(f"Grading code response:\n{grading_code_response}\nGrading code cost: {grading_code_cost}")
                print("-" * 50)
                data["grading_code"] = grading_code_response
                data["grading_code_cost"] = grading_code_cost
                with open("grading_code.jsonl", "a") as f:
                    f.write(json.dumps(data) + "\n")
            print(f"Total grading code cost: {total_cost}")
    if args.verify:
        with open("grading_code.jsonl") as f:
            code_data = f.readlines()
        with open("grading.jsonl") as f:
            grading_data = f.readlines()
        all_matching_results = []
        for code, grading in zip(code_data, grading_data):
            code = json.loads(code)
            grading = json.loads(grading)
            answers = grading["answers"]
            grading_results = grading["grading"]
            grading_code = code["grading_code"]
            matching = verify_code(grading_code, answers, grading_results)
            all_matching_results += matching
        n_matching = all_matching_results.count(1)
        n_mismatching = all_matching_results.count(0)
        n_invalid = all_matching_results.count(-1)
        print(f"Number of matching results: {n_matching}")
        print(f"Number of mismatching results: {n_mismatching}")
        print(f"Number of invalid results: {n_invalid}")
