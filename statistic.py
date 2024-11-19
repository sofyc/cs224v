import json
import collections
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# file_names = ["baseline", "concept_wiki", "concept_conceptnet", "word_wiki"]
file_names = ["concept_wiki"]
for file_name in file_names:
    with open(f"{file_name}_evaluation.json", "r") as f:
        data = json.loads(f.read())

    dimsnsions = ["Educational Value", "Diverseness","Area Relevance", "Difficulty Appropriateness", "Comprehensiveness"]

    scores = collections.defaultdict(list)
    for level, areas in data.items():
        for area, qs in areas.items():
            for score in qs["llm_score"]:
                for dim in dimsnsions:
                    if dim == "Diverseness":
                        _dim = "Diversity"
                    else:
                        _dim = dim
                    if score[dim] >= 0:
                        scores[_dim].append(score[dim])

    print(file_name)
    print('-----------------------')
    for key, value in scores.items():
        print(key, np.round(np.mean(value), 2))
    print('-----------------------')

    if file_name != "baseline":
        # dimsnsions = ["Groundedness", "Answer Relevance", "Context Relevance", "Comprehensiveness"]
        dimsnsions = ["Groundedness", "Answer Relevance", "Context Relevance"]

        # scores = collections.defaultdict(list)
        for level, areas in data.items():
            for area, qs in areas.items():
                for score in qs["score"]:
                    for dim in dimsnsions:
                        if score[dim] != None:
                            scores[dim].append(score[dim])
                        else:
                            scores[dim].append(0)

        print(file_name)
        print('-----------------------')
        for key, value in scores.items():
            print(key, np.round(np.mean(value), 2))
        print('-----------------------')

        df = pd.DataFrame(scores)
        corr_matrix = df.corr()
        plt.figure(figsize=(12, 8))
        sns.set(style='white') 

        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt='.2f', 
        center=0, square=True, linewidths=.5, cbar_kws={"shrink": .75})
        plt.title("Correlation Matrix", fontsize=16)
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.tight_layout()
        plt.show()

# file_names = ["baseline_concept_wiki_pairwise","concept_wiki_baseline_pairwise", "concept_wiki_concept_conceptnet_pairwise", "concept_conceptnet_concept_wiki_pairwise", "concept_conceptnet_baseline_pairwise", "baseline_concept_conceptnet_pairwise"]
# for file_name in file_names:
#     with open(f"{file_name}_evaluation.json", "r") as f:
#         data = json.loads(f.read())

#     dimsnsions = ["Educational Value", "Diverseness","Area Relevance", "Difficulty Appropriateness", "Comprehensiveness"]
#     scores = collections.defaultdict(list)
#     for score in data:
#         for dim in dimsnsions:
#             if score[dim] == 2:
#                 scores[dim].append(1)
#             elif score[dim] == 1:
#                 scores[dim].append(0)

#     print(file_name)
#     print('-----------------------')
#     for dim in dimsnsions:
#         print(dim, np.round(1 -  np.mean(scores[dim]), 2))
#     print('-----------------------')