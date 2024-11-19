import json
import collections
import numpy as np
from openai import AzureOpenAI
import os
from tqdm import tqdm
import ast
import chromadb
import wikipedia
import wikipediaapi
wiki_wiki = wikipediaapi.Wikipedia('easonfu@stanford.edu', 'en')

from llama_index.readers.wikipedia import WikipediaReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.schema import Document
from tenacity import retry, stop_after_attempt, wait_exponential
from trulens.core import Feedback
from trulens.apps.llamaindex import TruLlama
from trulens.providers.openai import OpenAI as _OpenAI

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)


client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version="2024-09-01-preview",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
)

with open("concepts.json", "r") as f:
    data = json.loads(f.read())


llm = OpenAI(model="gpt-4o-mini")
embed_model_name = "text-embedding-3-large"
embed_model = OpenAIEmbedding(model=embed_model_name)

# chroma_client = chromadb.EphemeralClient()
# try:
#     chroma_client.delete_collection("example_collection")
# except Exception:
#     pass  # If the collection does not exist, do nothing

# chroma_collection = chroma_client.create_collection("example_collection")
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
Settings.embed_model = embed_model
Settings.llm = llm
top_k = 3
chunk_size = 1024
chunk_overlap = 50
Settings.top_k = top_k
Settings.chunk_size = chunk_size
Settings.chunk_overlap = chunk_overlap
# storage_context = StorageContext.from_defaults(vector_store = vector_store)


for level, areas in data.items():
    # if level != "primary school":
    #     continue

    for area, qs in tqdm(areas.items()):
        # if area != "accounting":
        #     continue

        data[level][area]["quiz"] = []
        data[level][area]["score"] = []
        
        for question in qs["questions"]:
            i = 0
            wiki_docs = []
            words = word_tokenize(question)
            stop_words = set(stopwords.words('english'))
            concepts = [word for word in words if word.lower() not in stop_words and word.lower() not in punctuations]

            for concept in concepts:
                # concept = concept.capitalize()
                # results = []
                # wiki_page = wikipedia.page(concept)
                # pages = wikipedia.search(question)
                # if len(pages) == 0:
                #     print(f"No search results for {concept}")
                #     continue
                
                # for page in pages[:3]:
                try:
                    # print(f"Loading page {concept}")
                    page_py = wiki_wiki.page(concept)
                    # wiki_page = wikipedia.page(page)
                    page_content = page_py.text
                    page_id = str(page_py.pageid)
                    # print(type(page_content))
                    wiki_docs.append(Document(id_=page_id, text=page_content))
                except Exception as e:
                    print(f"Error loading page {concept}: {e}")
                    # doc = WikipediaReader().load_data(pages=[concept])
                
                # wiki_docs.extend(results)
            # print(len(wiki_docs))
            # exit()
                
                # print(wiki_docs)
                # exit()
                

                # chroma_client = chromadb.EphemeralClient()
                # try:
                #     chroma_client.delete_collection("example_collection")
                # except Exception:
                #     pass  # If the collection does not exist, do nothing

                # chroma_collection = chroma_client.create_collection("example_collection")
                # vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
 

                # Settings.embed_model = embed_model
                # Settings.llm = llm

                # storage_context = StorageContext.from_defaults(vector_store = vector_store)

                # wiki_docs = []
                
            index = VectorStoreIndex.from_documents(wiki_docs)
            # index.set_index_id("vector_index")
            # index.storage_context.persist("./storage")

            query_engine = index.as_query_engine(similarity_top_k= top_k)
            # @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=10))
            # def call_query_engine(prompt):
            #     return query_engine.query(prompt)
            
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

Now, please generate 3 quizzes following the format, each quiz should follow thw sign of [Quiz]:
Student Question: {question}"""

            # print(f"Prompt: {prompt}")
            # exit()
            # print(f"Response: {call_query_engine(prompt)}\n")

            provider = _OpenAI(model_engine="gpt-4o-mini")

            context = TruLlama.select_context(query_engine)
            
            f_groundedness = (
                Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
                .on(context.collect())
                .on_output()
            )

            f_answer_relevance  = (
                Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
                .on_input()
                .on_output()
            )

            f_context_relevance = (
                Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance")
                .on_input()
                .on(context)
                .aggregate(np.mean)
            )

            f_comprehensiveness = (
                Feedback(provider.comprehensiveness_with_cot_reasons, name="Comprehensiveness")
                .on(context.collect())
                .on_output()
            )


            # from trulens.core.session import TruSession
            # session = TruSession()
            # session.reset_database()
            
            # print(top_k, chunk_size)
            tru_query_engine = TruLlama(query_engine,
                app_name = "Wikipedia RAG",
                app_version = f"{embed_model_name}_{top_k}_{chunk_size}",
                feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance, f_comprehensiveness],
                metadata={
                    'embed_model':embed_model_name,
                    'top_k':top_k,
                    'chunk_size':chunk_size
                    })

            # you may want to uncomment exponential backoff when you're done debugging
            # @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=10))
            with tru_query_engine as recording:
                response = query_engine.query(prompt)
                rec = recording.get()
                score = {}
                for feedback, feedback_result in rec.wait_for_feedback_results().items():
                    score[feedback.name] = feedback_result.result
                    # print(feedback.name, feedback_result.result)
            
            quiz = [i.strip() for i in str(response).split('[Quiz]') if i.strip() and i.strip().startswith("Quiz")]

            data[level][area]["quiz"].append(quiz)
            data[level][area]["score"].append(score)

with open("quiz_word_wiki.json", "w") as f:
    f.write(json.dumps(data, indent=4))