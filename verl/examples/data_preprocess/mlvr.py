math_def = """\
Mathematics is a field of study that discovers and organizes methods, theories and theorems that are developed and proved for the needs of empirical sciences and mathematics itself. There are many areas of mathematics, which include number theory (the study of numbers), algebra (the study of formulas and related structures), geometry (the study of shapes and spaces that contain them), analysis (the study of continuous changes), and set theory (presently used as a foundation for all mathematics).

Mathematics involves the description and manipulation of abstract objects that consist of either abstractions from nature or—in modern mathematics—purely abstract entities that are stipulated to have certain properties, called axioms. Mathematics uses pure reason to prove properties of objects, a proof consisting of a succession of applications of deductive rules to already established results. These results include previously proved theorems, axioms, and—in case of abstraction from nature—some basic properties that are considered true starting points of the theory under consideration.
"""

cs_def = """\
Computer science is the study of computation, information, and automation. Computer science spans theoretical disciplines (such as algorithms, theory of computation, and information theory) to applied disciplines (including the design and implementation of hardware and software).

Algorithms and data structures are central to computer science. The theory of computation concerns abstract models of computation and general classes of problems that can be solved using them. The fields of cryptography and computer security involve studying the means for secure communication and preventing security vulnerabilities. Computer graphics and computational geometry address the generation of images. Programming language theory considers different ways to describe computational processes, and database theory concerns the management of repositories of data. Human–computer interaction investigates the interfaces through which humans and computers interact, and software engineering focuses on the design and principles behind developing software. Areas such as operating systems, networks and embedded systems investigate the principles and design behind complex systems. Computer architecture describes the construction of computer components and computer-operated equipment. Artificial intelligence and machine learning aim to synthesize goal-orientated processes such as problem-solving, decision-making, environmental adaptation, planning and learning found in humans and animals. Within artificial intelligence, computer vision aims to understand and process image and video data, while natural language processing aims to understand and process textual and linguistic data.
"""

math_examples = """\
Question 1: If in a grouped frequency distribution, the midpoint of each group decreases by 10 while the frequency of each group remains unchanged, the arithmetic mean will ______.
Answer 1: decrease by 10
Question 2: If a certain convertible bond has a face value of 3000 yuan and a conversion price of 20 yuan, when the stock price is 15 yuan, the current conversion value of the bond is ( ) yuan.
Answer 2: 2250
Question 3: A square vegetable garden with an area of 625 square meters is enclosed by a fence. Now, using a total length of 100 meters of fence, the garden is divided into smaller vegetable gardens of equal area. What is the maximum number of smaller gardens that can be created?______.
Answer 3: 9
"""

cs_examples = """\
Question 1: A binary tree has a total of 25 nodes, of which 5 are leaf nodes. The number of nodes with degree 1 is (  ).
Answer 2: 16
Question 2: The SORT command physically sorts the currently opened database file by the specified field name, and the sorted result is placed into ( ).
Answer 2: specific database file
Question 3: If the level of the root node is 0, then the maximum number of nodes in a binary tree of height k is (    ).    A. 2k    B. 2k-1    C. 2k+1  D. 2k+1-1
Answer 3: 2k+1-1
"""

missing_info_examples = """\
Question 1: There are three independent plans: A, B, and C, and all three plans have the same structural type. Their investment amounts and net present values are shown in Table 9-7-1. Due to a funding limit of 7.5 million yuan, the optimal combination plan is ____. Table 9-7-1 (in ten thousand yuan) Plan Investment Amount Net Present Value A 200 50.25 B 350 60.85 C 400 55.45
Answer 1: Combination of B and C
"""
incomplete_question_examples = """\
Question 1: The members of the two-dimensional array a are strings composed of 6 characters, with row index i ranging from 0 to 8 and column index j ranging from 1 to 10. Therefore, at least (44) bytes are needed to store a.
Answer 1: 540
"""

other_domain_examples = """\
Question 1: On November 30, 2002, a certain county court issued a notice to find a missing person, stating: If the respondent has not appeared or their whereabouts are confirmed after 3 months from the announcement, the people's court may lawfully declare the respondent missing. What should be the expiration date of the announcement period in this case? ( ).
Answer 1: February 28, 2003
Question 2: The founder of China's first public kindergarten teacher training school - Jiangxi Experimental Kindergarten Teacher School is (  ).
Answer 2: Chen Heqin
Question 3: CT enhanced scan of the axial position, cerebellar tentorium notch:
Answer 3: Presenting a 'V' shape above the level of the confluence of sinuses.
Question 4: Without the permission of ( ) and the project design leader, the construction unit has no right to modify the design.
Answer 4: Design Unit
Question 5: A group composed of several archives with a certain connection is ( )
Answer 5: Archive Group
Question 6: TSP and PM10 are conventional atmospheric monitoring projects in our country, which refer to particles suspended in the air with aerodynamic equivalent diameters less than or equal to ( ) and ( ) respectively.
Answer 6: 100μm 10μm
Question 6: What management principle do you think this case best illustrates? ( )
Answer 6: Principle of Combining Centralization and Decentralization.
Question 7: The sentence 'On the top of that persimmon tree, there still hangs a small fire persimmon. The small fire persimmon shines even brighter red under the winter sun' is from
Answer 7: Zhang Jie’s 'Picking Wheat Ears'
Question 8: The product process design activities carried out by machinery manufacturing enterprises belong to ______.
Answer 8: Production technology preparation process
"""

prompt_template = """\
Given a question-answer pair, you need to detect whether the question is valid and answerable. The question should contain the complete question sentence(s) and all the relevant information, such as referenced Tables. If the question is valid an answerable, you need to classify the the question into mathematics, computer science, and others. Please only response with one of the following phrases:
1. invalid due to missing information
2. invalid due to incomplete questions
3. mathematics
4. computer science
5. others

Here is the definition of mathematics:
{math_def}

Here is the definition of computer science:
{cs_def}

Here is an example of a invalid question with a missing table:
{missing_info_examples}

Here is an example of a invalid question with an incomplete question:
{incomplete_question_examples}

Here are examples of mathematics questions:
{math_examples}

Here are examples of computer science questions:
{cs_examples}

Here are examples of other questions:
{other_domain_examples}

The given question-answer pair is: 
Question: {question}
Answer: {answer}

Your response should only contain one of the following phrases and nothing else.
1. invalid due to missing information
2. invalid due to incomplete questions
3. mathematics
4. computer science
5. others

Your response is:
"""

from datasets import load_dataset
from openai import OpenAI
import pandas as pd
import tiktoken
import os, json, time, sys
import argparse

def get_llm_response(prompt, client):
    n_input_tokens = len(tiktoken.encoding_for_model("o3-mini-2025-01-31").encode(prompt))
    response = client.chat.completions.create(
        model="o3-mini-2025-01-31",
        messages=[
            {"role": "user", "content": prompt}
        ])
    n_output_tokens = len(tiktoken.encoding_for_model("o3-mini-2025-01-31").encode(response.choices[0].message.content))
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
            math_def=math_def,
            cs_def=cs_def,
            missing_info_examples=missing_info_examples,
            incomplete_question_examples=incomplete_question_examples,
            math_examples=math_examples,
            cs_examples=cs_examples,
            other_domain_examples=other_domain_examples,
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
                math_def=math_def,
                cs_def=cs_def,
                missing_info_examples=missing_info_examples,
                incomplete_question_examples=incomplete_question_examples,
                math_examples=math_examples,
                cs_examples=cs_examples,
                other_domain_examples=other_domain_examples,
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

def create_batch_job(file_id, batch_size=50000):
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
        create_batch_job(args.file_id, args.batch_size)
    if args.check_status:   
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        batch_job = client.batches.retrieve(args.file_id)
        print(f"Batch job status: {batch_job}")