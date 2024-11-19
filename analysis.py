import json
import collections
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import tiktoken
import numpy as np
from openai import AzureOpenAI
import os
from tqdm import tqdm

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")  # or other model you're using

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version="2024-09-01-preview",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
)

with open("questions.json", "r") as f:
    questions = json.loads(f.read())

data1 = collections.defaultdict(list)
data2 = collections.defaultdict(list)
scores1 = collections.defaultdict(list)
scores2 = collections.defaultdict(list)
for level, areas in questions.items():
    for area, questions in tqdm(areas.items()):
        data1[area].extend(questions)
        data2[level].extend(questions)

        for question in questions:
            prompt = f"Please assess the difficulty of the following question based on the required knowledge and reasoning complexity: {question}. Provide an integer score from 1 to 5, where 1 indicates an easy question and 5 indicates a difficult one. Please output only the score."

            response = client.chat.completions.create(
                model="gpt-4o-mini-240718",
                messages=[
                    {"role": "user", "content": prompt},
                ]
            )
            score = response.choices[0].message.content
            scores1[area].append(int(score))
            scores2[level].append(int(score))


# count = {}
# # for level, text in data2.items():
# #     texts = ' '.join(text)
# #     count[level] = len(tokenizer.encode(texts))
# for area, text in data1.items():
#     texts = ' '.join(text)
#     count[area] = len(tokenizer.encode(texts)) / len(text)

# keys = list(count.keys())
# values = list(count.values())
# mean_value = np.mean(values)

# # Create a bar plot
# plt.bar(keys, values, color='skyblue')
# plt.xlabel('Question Area')
# plt.ylabel('Token Count')
# plt.title('Token Count Vs. Question Area')
# plt.xticks(rotation=90)
# plt.tight_layout()
# plt.axhline(mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')
# plt.legend(loc='lower right')
# plt.show()

# ------------
scores = {}
for area, dif in scores1.items():
    scores[area] = np.mean(dif)

keys = list(scores.keys())
values = list(scores.values())
mean_value = np.mean(values)

# Create a bar plot
plt.bar(keys, values, color='skyblue')
plt.xlabel('Question Area')
plt.ylabel('Question Difficulty')
plt.title('Question Difficulty Vs. Question Area')
plt.xticks(rotation=90)
plt.tight_layout()
plt.axhline(mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')
plt.legend(loc='lower right')
plt.show()

scores = {}
for level, dif in scores2.items():
    scores[level] = np.mean(dif)

keys = list(scores.keys())
values = list(scores.values())
mean_value = np.mean(values)

# Create a bar plot
plt.bar(keys, values, color='skyblue')
plt.xlabel('Question Area')
plt.ylabel('Question Difficulty')
plt.title('Question Difficulty Vs. Question Area')
plt.xticks(rotation=90)
plt.tight_layout()
plt.axhline(mean_value, color='red', linestyle='--', label=f'Mean: {mean_value:.2f}')
plt.legend(loc='lower right')
plt.show()
