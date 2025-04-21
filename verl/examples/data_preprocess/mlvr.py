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
import os, json

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
    for index, row in dataset.iterrows():
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
        print(prompt)
        response, n_input_tokens, n_output_tokens, cost = get_llm_response(prompt, client)
        print(response, n_input_tokens, n_output_tokens, cost)
        result = {
            "question": question,
            "answer": answer,
            "response": response,
            "cost": cost
        }
        with open("mlvr_annotations.jsonl", "a+") as f:
            output_str = json.dumps(result)
            f.write(output_str + "\n")
        if index >= limit:
            break

dataset = load_dataset("virtuoussy/Multi-subject-RLVR")
gen_annotations(dataset["train"].to_pandas(), limit=100)