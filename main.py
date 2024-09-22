import gradio as gr
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from utilis import load_split_file, create_index, final_response
from langchain_mistralai.chat_models import ChatMistralAI

import os
import shutil
from dotenv import load_dotenv


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")



file_path = "data/last lesson.pdf"
docs = load_split_file(file_path)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

model = ChatMistralAI(mistral_api_key=MISTRAL_API_KEY)
pinecone_index = "index"
index_name = create_index(pinecone_index, PINECONE_API_KEY)

index = LangchainPinecone.from_documents(docs, embeddings, index_name=index_name)

question = "What data does google collects?"
matching_results=index.similarity_search(question,k=2)

answer = final_response(index, question, model)

print(f"{answer}\n\n{matching_results}")


