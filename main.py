from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain.vectorstores import Pinecone as LangchainPinecone

import os
from dotenv import load_dotenv
from utilis import load_split_file, create_index, final_response

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")



file_path = "data/fb.pdf"
docs = load_split_file(file_path)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
model=Ollama(model="llama3.1")

pinecone_index = "index2"

index_name = create_index(pinecone_index, PINECONE_API_KEY)

index = LangchainPinecone.from_documents(docs, embeddings, index_name=index_name)

question = "What data does facebook collects?"
answer = final_response(index, question, model)

print(answer)


