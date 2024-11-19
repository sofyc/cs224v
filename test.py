from openai import AzureOpenAI
import os

client = AzureOpenAI(
    api_key="4c2a6ab7a11b4e309812d98a399ce9e0",  
    api_version="2023-12-01-preview",
    azure_endpoint = "https://diyigroupgpt4.openai.azure.com/"
)

response = client.chat.completions.create(
  model="gpt-4-1106-preview", # repalce with "gpt-4-1106-preview"/ "gpt-35-turbo-0613" / "gpt-4-turbo-2024-04-09" / "gpt-4o-240513" / "gpt-4o-mini-240718"
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"}
  ]
)

prompt = """我想要根据如下的python代码搭建一个网站，要求如下：
1. 用户输入area, level, question
2. 网页展示concepts
3. 网页展示wiki_docs
4. 网页依次展示每一个quiz, 用户输入答案，网页告知用户是否正确以及正确答案
5. 用户作答完毕后，网页展示feedback
import json
import collections
import numpy as np
from openai import AzureOpenAI
import os
from tqdm import tqdm
import ast
import wikipediaapi
wiki_wiki = wikipediaapi.Wikipedia('easonfu@stanford.edu', 'en')
import re
import random


client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version="2024-09-01-preview",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
)

# area = Input("Enter the area: ")
# level = Input("Enter your education level: ")
# question = Input("Enter the question: ")
area = "geography"
level = "university"
question = "Where is Stanford located?"
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

wiki_docs = []
for concept in concepts:
    try:
        page_py = wiki_wiki.page(concept)
        page_content = page_py.summary
        # print(type(page_content))
        wiki_docs.append(page_content)
    except Exception as e:
        print(f"Error loading page {concept}: {e}")

wiki = "\n\n".join(wiki_docs)
prompt = f"""You are a quiz generator. The students are currently studying {area} at the {level} level and have asked a question. Your task is to create 3 quizzes that helps the student better understand the question. Use relevant information from the Wikipedia page on various related concepts to craft the quiz. The quiz should consist of one question, one correct answer, and three incorrect options. The correct answer must always be placed in option A. Note that the primary focus should be on addressing the student's question rather than the Wikipedia page, and the difficulty level should align with the knowledge and reasoning complexity appropriate for {level} education.

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

Here is relevant information from the Wikipedia:

{wiki}

Now, please generate 3 quizzes following the format, each quiz should follow thw sign of [Quiz]:
Student Question: {question}"""

response = client.chat.completions.create(
    model="gpt-4o-mini-240718",
    messages=[
        {"role": "user", "content": prompt},
    ]
)


quiz_pattern = re.compile(r'Quiz:\s*(.*?)\s*A\.\s*(.*?)\s*B\.\s*(.*?)\s*C\.\s*(.*?)\s*D\.\s*(.*?)\s*(?:\[Quiz\]|$)', re.DOTALL)

quiz_list = []
for match in quiz_pattern.findall(response.choices[0].message.content):
    quiz, optionA, optionB, optionC, optionD = match
    options = [optionA.strip(), optionB.strip(), optionC.strip(), optionD.strip()]
    
    original_a = optionA.strip()
    
    random.shuffle(options)
    idx = options.index(original_a)
    
    quiz_dict = {
        "quiz": quiz.strip(),
        "options": {
            "A": options[0],
            "B": options[1],
            "C": options[2],
            "D": options[3]
        },
        "correct_letter": chr(ord("A") + idx),
    }
    quiz_list.append(quiz_dict)

performance = """"""
for quiz in quiz_list:
    print(f"Quiz: {quiz['quiz']}")
    performance += f"Quiz: {quiz['quiz']}\n"
    for label, option in quiz['options'].items():
        performance += f"{label}. {option}\n"
        print(f"{label}. {option}")
    
    user_answer = Input()
    if user_answer.lower() == quiz['correct_letter'].lower():
        print("Correct!")
        performance += f"Student chooses {user_answer}, which is correct\n"
    else:
        print("Incorrect!")
        performance += f"Student chooses {user_answer}, which is incorrect, the correct answer is {quiz['correct_letter']}\n"
    
    performance += "\n"

prompt = f"""You are a student mentor. Your students are currently studying {area} at the {level} level and have asked a question. You have provided them with three quizzes to work on. Your task is to give personalized feedback to the student, highlighting both their strengths and areas for improvement, as well as offering a study plan or additional advice to help them better understand the material. Your feedback should be based on the relevant information from the Wikipedia page provided.

Relevant Wikipedia Information:

{wiki}

Student Question: 

{question}

Student's Performance on Quizzes:

{performance}

Please generate constructive feedback for the student, emphasizing their strengths, identifying weaknesses that could be improved, and suggesting possible directions for future study."""

feedback = client.chat.completions.create(
    model="gpt-4o-mini-240718",
    messages=[
        {"role": "user", "content": prompt},
    ]
)

print(feedback.choices[0].message.content)"""