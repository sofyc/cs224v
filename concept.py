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

with open("questions.json", "r") as f:
    questions = json.loads(f.read())

for level, areas in questions.items():
    for area, qs in tqdm(areas.items()):

        areas[area] = {}
        areas[area]["questions"] = qs
        areas[area]["concepts"] = []

        for question in qs:
            prompt = f"""Please identify key concepts in the following question. Each concept should be a noun, listed in singular form if countable. Provide the concepts in a list, separated by commas, without bullet points.

Here is an example:
Question: What is aldose?
Education Level: university
Area: chemistry
Concept: [aldose, carbohydrate, sugar, organic chemistry]

Here is the question:
Question: {question}
Education Level: {level}
Area: {area}
Concept:"""

            response = client.chat.completions.create(
                model="gpt-4o-mini-240718",
                messages=[
                    {"role": "user", "content": prompt},
                ]
            )
            answer = response.choices[0].message.content.strip('[]').split(", ")
            concepts = [i.strip() for i in answer if i.strip()]
            areas[area]["concepts"].append(concepts)

with open("concepts.json", "w") as f:
    f.write(json.dumps(questions, indent=4))