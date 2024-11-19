import json
import re
from openai import AzureOpenAI
import os
from tqdm import tqdm


client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version="2024-09-01-preview",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
)

file_name1 = "baseline"
file_name2 = "concept_wiki"

# file_name1 = "concept_wiki"
# file_name2 = "baseline"

# file_name1 = "concept_wiki"
# file_name2 = "concept_conceptnet"

# file_name1 = "concept_conceptnet"
# file_name2 = "concept_wiki"

# file_name1 = "concept_conceptnet"
# file_name2 = "baseline"

file_name1 = "baseline"
file_name2 = "concept_conceptnet"

for file_name1, file_name2 in [("baseline", "concept_wiki"), ("baseline", "concept_conceptnet"), ("concept_wiki", "baseline"), ("concept_wiki", "concept_conceptnet"), ("concept_conceptnet", "baseline"), ("concept_conceptnet", "concept_wiki")]:

    with open(f"quiz_{file_name1}.json", "r") as f:
        quiz1 = json.loads(f.read())

    with open(f"quiz_{file_name2}.json", "r") as f:
        quiz2 = json.loads(f.read())

    out = []
    for level, areas in quiz1.items():
        for area, questions_concepts_quizs_score in tqdm(areas.items()):
            if "quiz" in questions_concepts_quizs_score:
                questions = questions_concepts_quizs_score["questions"]

                llm_evaluation_metric = []
                for idx, question in enumerate(questions):

                    quizs1 = questions_concepts_quizs_score["quiz"][idx]
                    quizs2 = quiz2[level][area]["quiz"][idx]     
                    aggregated_quiz1 = "\n\n".join([f"{i+1}: {quiz}" for i, quiz in enumerate(quizs1)])
                    aggregated_quiz2 = "\n\n".join([f"{i+1}: {quiz}" for i, quiz in enumerate(quizs2)])


                    # Revised prompt with complete scoring scale
                    prompt = f"""A student studying {area} at the {level} level has asked the following question: "{question}". You are given two quiz sets that aim to help the student better understand the question. Please choose the quiz set that best address this question. Please evaluate and compare the educational quality of these quiz sets based on the criteria listed below. For each criterion, select the quiz set that performs better by outputting 1 or 2.

1. Educational Value: Which quiz set offers greater learning potential? Which set will help students gain a deeper understanding of the topic?
2. Diverseness: Which quiz set covers a broader range of topics? Does it explore a variety of concepts or focus narrowly on a single idea?
3. Area Relevance: Which quiz set is more aligned with the student's question and the key concepts they are studying? How well is it tailored to the specific subject area?
4. Difficulty Appropriateness: Which quiz set is better suited to the student's current educational level, neither too simple nor too advanced?
5. Comprehensiveness: Which quiz set provides greater depth and breadth? Which one is more thorough in addressing key concepts and details?

Here is the quiz set 1:
{aggregated_quiz1}

Here is the quiz set 2:
{aggregated_quiz2}

Please start by providing a step-by-step reasoning analysis of the quiz sets, then return your evaluation as a JSON object in the following format:
```json
{{
"Educational Value": choice,
"Diverseness": choice,
"Area Relevance": choice,
"Difficulty Appropriateness": choice,
"Comprehensiveness": choice
}}
```
"""
                    # You can use this prompt for generating a response or making further decisions based on the score
                                    # Send the prompt to the OpenAI API (using Azure OpenAI)
                    retries = 0
                    max_retries = 3
                    while retries < max_retries:
                        try:
                            response = client.chat.completions.create(
                                model="gpt-4o-mini-240718",
                                messages=[
                                    {"role": "user", "content": prompt},
                                ]
                            )

                            response_content = response.choices[0].message.content
                            # Use regex to extract JSON part (ignore extra explanation text)
                            match = re.search(r'```json\n(.*?)\n```', response_content, re.DOTALL)
                            json_data = match.group(1)
                            choices = json.loads(json_data)

                            # Example: Save the choices for easy use
                            evaluation_dict = {
                                "Educational Value": choices["Educational Value"],
                                "Diverseness": choices["Diverseness"],
                                "Area Relevance": choices["Area Relevance"],
                                "Difficulty Appropriateness": choices["Difficulty Appropriateness"],
                                "Comprehensiveness": choices["Comprehensiveness"]
                            }
                            break
                            # print("Evaluation Dictionary:", evaluation_dict)
                        
                        except Exception as e:
                            print(f"Error: {e}")
                            if retries < max_retries:
                                print(f"Retrying... (Attempt {retries + 1}/{max_retries})")
                            else:
                                print("Max retries reached. Skipping this question.")
                                print(response_content)
                                evaluation_dict = {
                                    "Educational Value": -1,
                                    "Diverseness": -1,
                                    "Area Relevance": -1,
                                    "Difficulty Appropriateness": -1,
                                    "Comprehensiveness": -1
                                }
                            
                            retries += 1
                    
                    out.append(evaluation_dict)

    with open(f"{file_name1}_{file_name2}_pairwise_evaluation.json", "w") as f:
        f.write(json.dumps(out, indent=4))