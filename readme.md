# Tailoring Interactive Learning: Personalized Quiz Generation Based on Key Concepts

This repository contains the implementation and evaluation framework for a **concept-based quiz generation method** that leverages **Large Language Models (LLMs)** and corpora such as Wikipedia to generate high-quality, subject-aligned quizzes. The codebase includes tools for analyzing the performance differences between concept-based and non-concept-based strategies. Additionally, it explores alternative approaches, such as using different corpora sources like ConceptNet, to demonstrate the strengths and limitations of Wikipedia-based methodologies.

---

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
---



## Installation

### Requirements
- Python 3.7 or later
- Required libraries listed in `requirements.txt`

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/sofyc/cs224v.git
2. Navigate to the project directory
    ```bash
    cd cs224v
3. Install dependencies:
    ```bash
    pip install -r requirements.txt

## Usage
### Quiz Generaion App
1. Start quiz generation:
    ```bash
    python demo/app.py
![Demo](demo.gif)
### Method Analysis
#### Question dataset generation:
1. Question dataset generation:
    ```bash
    python generate_question.py
2. Question quality analysis:
    ```bash
    python analysis.py
3. Extract question concept:
    ```bash
    python concept.py
#### Methods implementation:
4. Quiz generation **Baseline**
    ```bash
    python baseline.py
5. Quiz generation **Concept+Wikipedia** Proposed Method
    ```bash
    python wiki.py
6. Quiz generation **Concept+ConceptNet**
    ```bash
    python conceptnet.py
7. Quiz generation **Word+Wikipedia**
    ```bash
    python wiki_word.py
8. Quiz generation **Word+ConceptNet**
    ```bash
    python conceptnet_word.py
#### Evaluation:
9. Evaluation **LLM as judge**
    ```bash
    python evaluation.py
10. Evaluation **Human as judge**
    ```bash
    python pairwise_evaluation.py
#### Analysis:
11. Metric correlation analysis
    ```bash
    python statistic.py
<!-- demo_python.py
test.py -->
