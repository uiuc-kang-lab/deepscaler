knowledge_questions = """\
Question 1: The poet who is known as one of the 'Two Ans of Jinan' along with Li Qingzhao is ( )
Answer 1: Xin Qiji
Question 2: There are several commonly used network platform software, but ______ is not a network management platform software.
Answer 2: NetManager
Question 3: The restrictions on marriage during the Western Zhou period are ( )
Answer 3: No marriage between the same surname
"""

low_reasoning_questions = """\
Question 1: In patients with mitral stenosis, the auscultation sign that indicates the valve still has a certain degree of elasticity is
Answer 1: Mitral valve opening snap
Question 2: Custom type conversion is performed by converting from lower-level data to higher-level data according to priority, the order of priority is ( ).
Answer 2: char-int long-float-double
Question 3: () is a method of selecting a certain number of properties that have been traded and meet certain conditions, which are similar to the property being appraised, and then comparing them with the appraised property, appropriately adjusting their transaction prices to determine the value of the appraised property.
Answer 3: Market Approach
"""

reasoning_questions = """\
Question 1: The minimum concentration of Ag+ that can be detected using K2CrO4 reagent is 40μg/L, and the detection limit is 2μg. During the experiment, at least ( ) mL of the test solution should be taken.
Answer 1: 0.05
Question 2: Female, 17 years old, suddenly felt pain in the left knee joint accompanied by high fever half a month ago, with a history of injury to the left knee. Physical examination shows swelling of the left knee joint, elevated skin temperature, tenderness, and positive patellar float test. The diagnosis is
Answer 2: Pyogenic arthritis
Question 3: A year ago, due to the country's tightening of monetary policy, the stock market experienced a significant decline. Over the year, the Shanghai stock index dropped from around 3500 points to about 2000 points. Company A, based on the advice of its certified public accountant, prepared 30 million yuan in cash to purchase a certain amount of stocks at the appropriate time. The purpose of the company holding this cash is to meet ______.
Answer 3: speculative demand
"""

prompt_template = """\
Given a question-answer pair, your task is to determine the level of reasoning required to answer the question:
- No reasoning: The answer can be found directly from general world knowledge, without any analysis or processing of the information in the question.
- One-step reasoning: Answering the question requires performing a single logical or inferential step based on the information provided in the question.
- Multistep reasoning: Answering the question requires completing two or more logical or inferential steps, often involving the synthesis or combination of information from multiple parts of the question.

Here are examples of questions that requires no reasoning.
{knowledge_examples}

Here are examples of questions that require one-step reasoning.
{low_reasoning_questions}

Here are examples of questions that require multistep reasoning.
{reasoning_examples}

Here are the question-answer pairs that you need to classify.
Question: {question}
Answer: {answer}

Your response should only contain one of the following phrases and nothing else.
1. no reasoning
2. one-step reasoning
3. multistep reasoning

Your response is:
"""

from datasets import load_dataset
from openai import OpenAI
import pandas as pd
import os, json, time, glob
import argparse

def get_llm_response(prompt, client):
    response = client.chat.completions.create(
        model="o3-mini-2025-01-31",
        messages=[
            {"role": "user", "content": prompt}
        ])
    n_input_tokens = response.usage.prompt_tokens
    n_output_tokens = response.usage.completion_tokens
    cost = n_input_tokens / 1e6 * 1.1 + n_output_tokens / 1e6 * 4.4
    return response.choices[0].message.content, n_input_tokens, n_output_tokens, cost

def gen_annotations(dataset: pd.DataFrame, limit):
    running_cost = 0
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    recent_10_durations = []
    for index, row in dataset.iterrows():
        start = time.time()
        question = row["query"][1]['content']
        answer = row["label"]
        prompt = prompt_template.format(
            knowledge_examples=knowledge_questions,
            reasoning_examples=reasoning_questions,
            low_reasoning_questions=low_reasoning_questions,
            question=question,
            answer=answer
        )
        # print(prompt)
        response, n_input_tokens, n_output_tokens, cost = get_llm_response(prompt, client)
        # print(response, n_input_tokens, n_output_tokens, cost)
        result = {
            "question": question,
            "answer": answer,
            "response": response,
            "cost": cost
        }
        running_cost += cost
        with open("mlvr_annotations.jsonl", "a+") as f:
            output_str = json.dumps(result)
            f.write(output_str + "\n")
        if index == limit:
            break
        duration = time.time() - start
        if len(recent_10_durations) == 10:
            recent_10_durations = recent_10_durations[1:] + [duration]
        else:
            recent_10_durations.append(duration)
        if (index+1) % 10 == 0:
            n_data_to_run = limit - index - 1 if limit > 0 else len(dataset) - index - 1
            eta = n_data_to_run * sum(recent_10_durations) / len(recent_10_durations) / 3600
            print(f"Sample #{index+1}, running cost ${running_cost:.2f}, eta {eta:.1f} hrs")

def submit_batch_job_files(dataset: pd.DataFrame, batch_size=10000):
    batches = []
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size] if i+batch_size < len(dataset) else dataset[i:]
        batches.append(batch)
    for i, batch in enumerate(batches):
        if os.path.exists(f"data/mlvr_batch_{i}.jsonl"):
            print(f"Batch {i} already exists, skipping...")
            continue
        for index, row in batch.iterrows():
            question = row["query"][1]['content']
            answer = row["label"]
            prompt = prompt_template.format(
                knowledge_examples=knowledge_questions,
                reasoning_examples=reasoning_questions,
                low_reasoning_questions=low_reasoning_questions,
                question=question,
                answer=answer
            )
            request = {
                "custom_id": f"request-{i}-{index}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "o3-mini-2025-01-31",
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                }
            }
            with open(f"data/mlvr_batch_{i}.jsonl", "a+") as f:
                output_str = json.dumps(request)
                f.write(output_str + "\n")
        print(f"Batch processing done {i}/{len(batches)}")
    for i in range(len(batches)):
        batch_input_file = client.files.create(
            file=open(f"data/mlvr_batch_{i}.jsonl", "rb"),
            purpose="batch"
        )
        batch_input_file_id = batch_input_file.id
        with open("data/file_ids.txt", "a+") as f:
            f.write(f"Batch {i}: {batch_input_file_id}\n")
        print(f"Batch {i} file ID: {batch_input_file_id}")
    return

def create_batch_job(file_id):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": f"Batch job {file_id} for MLVR dataset"
        }
    )
    print(batch)
    with open("data/batch_job_ids.txt", "a+") as f:
        f.write(f"{file_id}: {batch.id}\n")
        
def download_all():
    with open("data/batch_job_ids_o4.txt", "r") as f:
        batch_ids = f.readlines()
        batch_ids = [batch_id.strip().split(": ")[1] for batch_id in batch_ids]
    for i, batch_id in enumerate(batch_ids):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        batch = client.batches.retrieve(batch_id)
        if batch.status == "completed" and batch.output_file_id:
            file_id = batch.output_file_id
            file_response = client.files.content(file_id)
            file_content = file_response.text
            with open(f"data/mlvr_batch_results_{i}.jsonl", "a+") as f:
                f.write(file_content)
        elif batch.status == "failed":
            print(f"Batch job {batch_id} failed")
        else:
            print(f"Batch job {batch_id} is still running")
            
def merge():
    dataset = load_dataset("virtuoussy/Multi-subject-RLVR")["train"].to_pandas()
    n_results = len(glob.glob("data/mlvr_batch_results_*.jsonl"))
    labels = []
    for i in range(n_results):
        with open(f"data/mlvr_batch_results_{i}.jsonl", "r") as f:
            lines = f.readlines()
            print(i, len(lines))
            for line in lines:
                result = json.loads(line)
                if result["custom_id"] in ["request-16-329779", "request-21-439097"]:
                    labels.append("error")
                labels.append(result["response"]["body"]["choices"][0]["message"]["content"])
    print(f"Total {len(labels)} labels")
    labels = [label.strip() for label in labels]
    print(f"Dataset size: {len(dataset)}")
    dataset["type"] = labels
    dataset.to_csv("data/mlvr_knowledge_annotated.csv", index=False)

# {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": [{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Hello world!"}],"max_tokens": 1000}}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate annotations for MLVR dataset")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of samples to process")
    parser.add_argument("--real_time", action="store_true", help="Use real-time mode")
    parser.add_argument("--batch_size", type=int, default=20000, help="Batch size for batch mode")
    parser.add_argument("--submit_batch", action="store_true", help="Submit batch job files")
    parser.add_argument("--create_job", action="store_true", help="Create batch job")
    parser.add_argument("--check_status", action="store_true", help="Check batch job status")
    parser.add_argument("--file_id", type=str, help="File ID for batch job")
    parser.add_argument("--download_all", action="store_true", help="Download all files")
    parser.add_argument("--merge", action="store_true", help="Merge all files")
    args = parser.parse_args()

    if args.real_time:
        dataset = load_dataset("virtuoussy/Multi-subject-RLVR")["train"].to_pandas()
        gen_annotations(dataset, args.limit)
    if args.submit_batch:
        dataset = load_dataset("virtuoussy/Multi-subject-RLVR")["train"].to_pandas()
        submit_batch_job_files(dataset, args.batch_size)
    if args.create_job:
        if args.file_id is None:
            raise ValueError("File ID is required for creating batch job")
        create_batch_job(args.file_id)
    if args.check_status:   
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        batch_job = client.batches.retrieve(args.file_id)
        print(f"Batch job status: {batch_job}")
    if args.download_all:
        download_all()
    if args.merge:
        merge()