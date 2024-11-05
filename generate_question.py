# from datasets import load_dataset

# ds = load_dataset("MMMU/MMMU", "Accounting")

from openai import AzureOpenAI
import os
import json
from tqdm import tqdm

with open("areas.txt", "r") as f:
    areas = f.read().splitlines()


client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version="2024-09-01-preview",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
)

levels = ["primary school", "high school", "university"]
out = {}
for level in levels:
    out[level] = {}
    for area in tqdm(areas):
        area = area.replace("_", " ").lower()

        prompt = f"""You are a curious student at a specified education level and are learning about a particular area of study. Your goal is to generate 5 diverse questions you would want to ask while learning about this subject. Directly output the questions in the format below without bullet points.

Here is an example:
Area: Biology
Education level: primary school
Question:
What are the different parts of a plant, and how do they help it grow?
Why do animals need food, water, and air to survive?
Why do some animals sleep during the day and are awake at night?

Here is the area and education level:
Area: {area}
Education level: {level}
Question:
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt},
            ]
        )

        questions = response.choices[0].message.content.split('\n')
        out[level][area] = [i.strip() for i in questions if i.strip()]

with open("questions.json", "w") as f:
    f.write(json.dumps(out, indent=4))