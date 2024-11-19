import json
import collections
import numpy as np
from openai import AzureOpenAI
import os
from tqdm import tqdm
import ast


client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version="2024-09-01-preview",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
)

with open("concepts.json", "r") as f:
    data = json.loads(f.read())


for level, areas in data.items():
    # if level != "primary school":
    #     continue

    for area, qs in tqdm(areas.items()):
        # if area != "accounting":
        #     continue
        data[level][area]["quiz"] = []
        
        for question, concepts in zip(qs["questions"], qs["concepts"]):

                
            prompt = f"""You are a quiz generator. The students are currently studying {area} at the {level} level and have asked a question. Your task is to create 3 quizzes that help the student better understand the question. The quiz should consist of one question, one correct answer, and three incorrect options. The correct answer must always be placed in option A.

Example:

Student Question: Where is Beijing located?
[Quiz]
Quiz: What is the capital city of China?
A. Beijing
B. Chengdu
C. Shanghai
D. Hangzhou

[Quiz]
Quiz: What continent is Beijing located?
A. Asia
B. Europe
C. Africa
D. North America

Now, please generate 3 quizzes following the format, each quiz should follow thw sign of [Quiz]:
Student Question: {question}"""

            response = client.chat.completions.create(
                model="gpt-4o-mini-240718",
                messages=[
                    {"role": "user", "content": prompt},
                ]
            )

            _quiz = [i.strip() for i in response.choices[0].message.content.split('[Quiz]') if i.strip()]

            data[level][area]["quiz"].append(_quiz)
            # data[level][area]["score"] = _scores

with open("quiz_baseline.json", "w") as f:
    f.write(json.dumps(data, indent=4))