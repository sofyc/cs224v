import os
import re
import random
import json
import wikipediaapi
from openai import AzureOpenAI
from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
from flask_session import Session

app.config['SESSION_TYPE'] = 'redis'  # 使用 Redis 存储 session 数据
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
Session(app)

# Initialize Wikipedia and OpenAI clients
wiki_wiki = wikipediaapi.Wikipedia('easonfu@stanford.edu', 'en')
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version="2024-09-01-preview",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
)

def generate_concepts(area, level, question):
    """Generate concepts related to the question"""
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
    return [i.strip() for i in answer if i.strip()]

def fetch_wiki_docs(concepts):
    """Fetch Wikipedia summaries for given concepts"""
    wiki_docs = []
    for concept in concepts:
        try:
            page_py = wiki_wiki.page(concept)
            page_content = page_py.summary
            wiki_docs.append(page_content)
        except Exception as e:
            print(f"Error loading page {concept}: {e}")
    return "\n\n".join(wiki_docs)

def generate_quizzes(area, level, question, wiki):
    """Generate quizzes based on the question and wiki content"""
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

Now, please generate 3 quizzes following the format, each quiz should follow the sign of [Quiz]:
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
    
    return quiz_list

def generate_feedback(area, level, question, wiki, performance):
    """Generate personalized feedback based on quiz performance"""
    prompt = f"""You are a student mentor. Your students are currently studying {area} at the {level} level and have asked a question. You have provided them with three quizzes to work on. Your task is to give personalized feedback to the student, highlighting both their strengths and areas for improvement, as well as offering a study plan or additional advice to help them better understand the material. Your feedback should be based on the relevant information from the Wikipedia page provided.

Relevant Wikipedia Information:
{wiki}

Student Question: {question}

Student's Performance on Quizzes:
{performance}

Please generate constructive feedback for the student, emphasizing their strengths, identifying weaknesses that could be improved, and suggesting possible directions for future study."""

    feedback = client.chat.completions.create(
        model="gpt-4o-mini-240718",
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    
    return feedback.choices[0].message.content

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main page to input area, level, and question"""
    if request.method == 'POST':
        # Completely clear the session to ensure fresh start
        session.clear()
        
        # Collect input from form
        area = request.form['area']
        level = request.form['level']
        question = request.form['question']
        
        # Generate concepts - force regeneration each time
        concepts = generate_concepts(area, level, question)
        
        # Fetch wiki documents - force new fetch
        wiki_docs = fetch_wiki_docs(concepts)
        
        # Generate quizzes with fresh content
        quizzes = generate_quizzes(area, level, question, wiki_docs)
        
        # Store new data in session
        session['area'] = area
        session['level'] = level
        session['question'] = question
        session['concepts'] = concepts
        session['wiki_docs'] = wiki_docs
        session['quizzes'] = quizzes
        
        # Reset quiz tracking
        session['current_quiz_index'] = 0
        session['quiz_performance'] = ""
        
        return redirect(url_for('concepts'))
    
    return render_template('index.html')

@app.route('/concepts')
def concepts():
    """Display concepts page"""
    if 'concepts' not in session:
        return redirect(url_for('index'))
    
    return render_template('concepts.html', 
                           concepts=session['concepts'], 
                           wiki_docs=session['wiki_docs'])

@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    """Quiz page to display and check quiz questions"""
    if 'quizzes' not in session:
        return redirect(url_for('index'))
    
    quizzes = session['quizzes']
    current_quiz_index = session.get('current_quiz_index', 0)
    
    # Add default values to prevent undefined errors
    context = {
        'quiz': quizzes[current_quiz_index] if quizzes else {},
        'is_correct': None,
        'correct_answer': None
    }
    
    if request.method == 'POST':
        # Check the answer
        user_answer = request.form.get('answer')
        current_quiz = quizzes[current_quiz_index]
        
        # Record performance
        performance_text = f"Quiz: {current_quiz['quiz']}\n"
        for label, option in current_quiz['options'].items():
            performance_text += f"{label}. {option}\n"
        
        if user_answer == current_quiz['correct_letter']:
            performance_text += f"Student chooses {user_answer}, which is correct\n\n"
            context['is_correct'] = True
        else:
            performance_text += f"Student chooses {user_answer}, which is incorrect. The correct answer is {current_quiz['correct_letter']}\n\n"
            context['is_correct'] = False
            context['correct_answer'] = current_quiz['correct_letter']
        
        # Update session performance
        session['quiz_performance'] += performance_text
        session['current_quiz_index'] += 1
        
        # Check if all quizzes are done
        if session['current_quiz_index'] >= len(quizzes):
            return redirect(url_for('feedback'))
        
        # Update quiz in context for next quiz
        context['quiz'] = quizzes[session['current_quiz_index']]
        
        return render_template('quiz.html', **context)
    
    # First time loading quiz
    return render_template('quiz.html', **context)

@app.route('/feedback')
def feedback():
    """Display final feedback page"""
    if 'quiz_performance' not in session:
        return redirect(url_for('index'))
    
    # Generate feedback
    feedback_text = generate_feedback(
        session['area'], 
        session['level'], 
        session['question'], 
        session['wiki_docs'], 
        session['quiz_performance']
    )
    
    return render_template('feedback.html', feedback=feedback_text)

if __name__ == '__main__':
    app.run(debug=True)