import json
import collections
import numpy as np
from openai import AzureOpenAI
import os
from tqdm import tqdm
import ast
import chromadb

from llama_index.readers.wikipedia import WikipediaReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from tenacity import retry, stop_after_attempt, wait_exponential

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_KEY"),  
    api_version="2024-09-01-preview",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
)

with open("concepts.json", "r") as f:
    data = json.loads(f.read())

for level, areas in data.items():
    for area, qs in tqdm(areas.items()):
        wiki_docs = []
        for question, concepts in zip(qs["questions"], qs["concepts"]):

            for concept in concepts:
                wiki_docs = []
                try:
                    doc = WikipediaReader().load_data(pages=[concept])
                    wiki_docs.extend(doc)
                    # print(doc)
                except Exception as e:
                    print(f"Error loading page for concept {concept}: {e}")


                chroma_client = chromadb.EphemeralClient()
                chroma_collection = chroma_client.create_collection("example_collection")
                vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
 

                llm = OpenAI(model="gpt-4o-mini")
                embed_model_name = "text-embedding-3-large"
                embed_model = OpenAIEmbedding(model=embed_model_name)

                Settings.embed_model = embed_model
                Settings.llm = llm

                storage_context = StorageContext.from_defaults(vector_store = vector_store)

                # wiki_docs = []
                
                index = VectorStoreIndex.from_documents(wiki_docs, storage_context=storage_context)
                # index.set_index_id("vector_index")
                # index.storage_context.persist("./storage")

                top_k = 5
                chunk_size = 1024

                query_engine = index.as_query_engine(top_k = top_k)
                @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=10))
                def call_query_engine(prompt):
                    return query_engine.query(prompt)
                
                prompt = f"""You are a quiz generator. The student is currently studying {area} at the {level} level. Please create a quiz to help the student better understand the concept of {concept} by using information from the Wikipedia page on {concept}. The quiz should consist of one question, one correct answer, and three incorrect options. The correct answer must always be placed in option A.

    Example:

    Question: What is the capital city of China?
    A. Beijing
    B. Chengdu
    C. Shanghai
    D. Hangzhou

    Now, please generate the quiz:"""

                # print(f"Prompt: {prompt}")
                # print(f"Response: {call_query_engine(prompt)}\n")

                from trulens.providers.openai import OpenAI
                from trulens.core import Feedback
                from trulens.apps.llamaindex import TruLlama

                provider = OpenAI(model_engine="gpt-4o-mini")

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


                from trulens.core.session import TruSession
                session = TruSession()
                session.reset_database()

                tru_query_engine = TruLlama(query_engine,
                    app_name = "Wikipedia RAG",
                    app_version = f"{embed_model_name}_{top_k}_{chunk_size}",
                    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
                    metadata={
                        'embed_model':embed_model_name,
                        'top_k':top_k,
                        'chunk_size':chunk_size
                        })

                # you may want to uncomment exponential backoff when you're done debugging
                # @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=10))
                with tru_query_engine as recording:
                    respoonse = query_engine.query(prompt)
                    print(respoonse)
                    rec = recording.get()
                    for feedback, feedback_result in rec.wait_for_feedback_results().items():
                        print(feedback.name, feedback_result.result)

                                                            

                exit()