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

for file_name in ["baseline", "concept_wiki", "concept_conceptnet", "word_wiki"]:
    if file_name == "baseline":
        model_name = "gpt-4o-240513"
    else:
        model_name = "gpt-4o-mini-240718"

    with open(f"quiz_{file_name}.json", "r") as f:
        quizs = json.loads(f.read())

    for level, areas in quizs.items():
        for area, questions_concepts_quizs_score in tqdm(areas.items()):
            if "quiz" in questions_concepts_quizs_score:
                questions = questions_concepts_quizs_score["questions"]
                _quizs = questions_concepts_quizs_score["quiz"]
                
                # Aggregate all quizzes into a single string for clarity
                # aggregated_quiz = "\n".join([f"Quiz {i+1}: {quiz}" for i, quiz in enumerate(_quizs)])

                llm_evaluation_metric = []
                for question, _quiz_list in zip(questions, _quizs):
                    # Revised prompt with complete scoring scale

                    if not _quizs:
                        evaluation_dict = {
                            "Educational Value": 0,
                            "Diverseness": 0,
                            "Area Relevance": 0,
                            "Difficulty Appropriateness": 0,
                            "Comprehensiveness": 0
                        }
                        llm_evaluation_metric.append(evaluation_dict)
                        continue

                    aggregated_quiz = "\n\n".join([f"{i+1}: {quiz}" for i, quiz in enumerate(_quiz_list)])

                    prompt = f"""A student studying {area} at the {level} level is asking a question: "{question}". Based on the following quiz set related to the question, I need you to evaluate the educational quality of the quiz set.  For each of the following criteria, assign a score from 1 to 5 for the entire quiz set:

1. Educational Value: Do you think these quizzes are educational? Will students learn more by taking these quizzes?
    - 1: Not educational at all, no learning value.
    - 2: Minimally educational, little learning value.
    - 3: Moderately educational, some learning value.
    - 4: Very educational, strong learning value.
    - 5: Highly educational, great learning value.

2. Diverseness: Do you think these quizzes are diverse? Are the quizzes covering a broad range of topics, or do they all focus on the same concept?
    - 1: Very repetitive, covers a narrow area.
    - 2: Some diversity, but mostly focuses on one concept.
    - 3: Fairly diverse, covers a few different topics.
    - 4: Quite diverse, covers multiple relevant topics.
    - 5: Extremely diverse, covers a broad range of topics.

3. Area Relevance: Are these quizzes relevant to the student's question and the concepts they're trying to learn? Are the quizzes tailored to the subject area being studied?
    - 1: Not relevant to the question or subject at all.
    - 2: Minimally relevant, some connection to the question/subject.
    - 3: Moderately relevant, fairly aligned with the question/subject.
    - 4: Highly relevant, strongly aligned with the question/subject.
    - 5: Perfectly relevant, directly tied to the question/subject.

4. Difficulty Appropriateness: Do you think these quizzes match the student's current education level? Would these quizzes be too easy or too difficult for a student at this level?
    - 1: Too easy or too difficult, not appropriate for the level.
    - 2: Slightly mismatched, quizzes may be too easy or too hard.
    - 3: Moderately appropriate, quizzes are somewhat aligned with the level.
    - 4: Mostly appropriate, quizzes are well-suited for the level.
    - 5: Perfectly suited to the student's education level.

5. Comprehensiveness: Do these quizzes cover the depth and breadth of the topic? Are they thorough in addressing key concepts and details?
    - 1: Very superficial, only scratches the surface of the topic.
    - 2: Somewhat incomplete, misses important aspects.
    - 3: Moderately comprehensive, covers the basics but lacks depth.
    - 4: Quite comprehensive, addresses most key aspects with reasonable depth.
    - 5: Highly comprehensive, thoroughly covers the topic in great depth and detail.

Here is the quiz set related to the question:
{aggregated_quiz}

Please start by providing a step-by-step reasoning analysis of the quiz set, then return your evaluation as a JSON object in the following format:
```json
{{
"Educational Value": score,
"Diverseness": score,
"Area Relevance": score,
"Difficulty Appropriateness": score,
"Comprehensiveness": score
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
                                model=model_name,
                                messages=[
                                    {"role": "user", "content": prompt},
                                ]
                            )

                            response_content = response.choices[0].message.content
                            # Use regex to extract JSON part (ignore extra explanation text)
                            match = re.search(r'```json\n(.*?)\n```', response_content, re.DOTALL)
                            json_data = match.group(1)
                            scores = json.loads(json_data)

                            # Example: Save the scores for easy use
                            evaluation_dict = {
                                "Educational Value": scores["Educational Value"],
                                "Diverseness": scores["Diverseness"],
                                "Area Relevance": scores["Area Relevance"],
                                "Difficulty Appropriateness": scores["Difficulty Appropriateness"],
                                "Comprehensiveness": scores["Comprehensiveness"]
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

                    llm_evaluation_metric.append(evaluation_dict)
                                
                questions_concepts_quizs_score["llm_score"] = llm_evaluation_metric

    with open(f"{file_name}_evaluation.json", "w") as f:
        f.write(json.dumps(quizs, indent=4))