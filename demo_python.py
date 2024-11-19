import tkinter as tk
from tkinter import messagebox
import json
import collections
import numpy as np
from openai import AzureOpenAI
import os
from tqdm import tqdm
import ast
import wikipediaapi
import re
import random

wiki_wiki = wikipediaapi.Wikipedia('easonfu@stanford.edu', 'en')

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-09-01-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Function to generate the quizzes
def generate_quizzes(area, level, question):
    prompt = f"""Please identify key concepts in the following question. Each concept should be a noun, listed in singular form if countable. Provide the concepts in a list, separated by commas, without bullet points.
    Question: {question}
    Education Level: {level}
    Area: {area}
    Concept:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini-240718",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content.strip('[]').split(", ")
    concepts = [i.strip() for i in answer if i.strip()]

    wiki_docs = []
    for concept in concepts:
        try:
            page_py = wiki_wiki.page(concept)
            page_content = page_py.summary
            wiki_docs.append(page_content)
        except Exception as e:
            print(f"Error loading page {concept}: {e}")

    wiki = "\n\n".join(wiki_docs)

    prompt = f"""You are a quiz generator. The students are currently studying {area} at the {level} level and have asked a question. Your task is to create 3 quizzes that help the student better understand the question.
    Here is relevant information from the Wikipedia:

    {wiki}

    Student Question: {question}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini-240718",
        messages=[{"role": "user", "content": prompt}]
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

    return quiz_list, wiki

# Function to create the GUI
def create_gui():
    def start_quiz():
        area = area_entry.get()
        level = level_entry.get()
        question = question_entry.get()

        if not area or not level or not question:
            messagebox.showerror("Input Error", "Please fill in all fields!")
            return

        quizzes, wiki = generate_quizzes(area, level, question)

        concept_label.config(text="Concepts: " + ", ".join([quiz["quiz"] for quiz in quizzes]))
        wiki_label.config(text="Relevant Wikipedia Information: \n" + wiki)

        quiz_frame.pack_forget()
        feedback_frame.pack_forget()
        display_quiz(quizzes)

    def display_quiz(quizzes):
        quiz_frame.pack(fill=tk.BOTH, expand=True)

        def check_answer(user_answer, correct_answer):
            if user_answer.lower() == correct_answer.lower():
                messagebox.showinfo("Correct!", f"Correct! The right answer is {correct_answer}.")
            else:
                messagebox.showinfo("Incorrect!", f"Incorrect! The correct answer is {correct_answer}.")

        def next_quiz(quiz_list, index=0):
            if index >= len(quiz_list):
                return
            quiz = quiz_list[index]
            quiz_label.config(text=quiz["quiz"])
            option_A.config(text=quiz["options"]["A"])
            option_B.config(text=quiz["options"]["B"])
            option_C.config(text=quiz["options"]["C"])
            option_D.config(text=quiz["options"]["D"])

            def submit_answer():
                user_answer = user_answer_var.get()
                check_answer(user_answer, quiz["correct_letter"])
                next_quiz(quiz_list, index+1)

            submit_button.config(command=submit_answer)
            user_answer_var.set("")

        next_quiz(quizzes)

    root = tk.Tk()
    root.title("Quiz Generator")

    # Input frame
    input_frame = tk.Frame(root)
    input_frame.pack(fill=tk.BOTH, expand=True)

    area_label = tk.Label(input_frame, text="Enter the Area:")
    area_label.pack()
    area_entry = tk.Entry(input_frame)
    area_entry.pack()

    level_label = tk.Label(input_frame, text="Enter Your Education Level:")
    level_label.pack()
    level_entry = tk.Entry(input_frame)
    level_entry.pack()

    question_label = tk.Label(input_frame, text="Enter the Question:")
    question_label.pack()
    question_entry = tk.Entry(input_frame)
    question_entry.pack()

    start_button = tk.Button(input_frame, text="Start Quiz", command=start_quiz)
    start_button.pack()

    # Concepts and Wiki
    concept_label = tk.Label(root, text="Concepts: ")
    concept_label.pack()

    wiki_label = tk.Label(root, text="Relevant Wikipedia Information: ")
    wiki_label.pack()

    # Quiz frame
    quiz_frame = tk.Frame(root)

    quiz_label = tk.Label(quiz_frame, text="")
    quiz_label.pack()

    option_A = tk.Button(quiz_frame, text="", width=30)
    option_A.pack()

    option_B = tk.Button(quiz_frame, text="", width=30)
    option_B.pack()

    option_C = tk.Button(quiz_frame, text="", width=30)
    option_C.pack()

    option_D = tk.Button(quiz_frame, text="", width=30)
    option_D.pack()

    user_answer_var = tk.StringVar()
    user_answer_entry = tk.Entry(quiz_frame, textvariable=user_answer_var)
    user_answer_entry.pack()

    submit_button = tk.Button(quiz_frame, text="Submit", command=None)
    submit_button.pack()

    # Feedback frame
    feedback_frame = tk.Frame(root)

    root.mainloop()

create_gui()
