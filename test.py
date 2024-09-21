from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama

from langchain.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser # type: ignore
from operator import itemgetter
from langchain.vectorstores import Pinecone as LangchainPinecone

import os
from dotenv import load_dotenv

loader = PyPDFLoader("data/Final Submission.pdf")
pages = loader.load_and_split()

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
#COHERE_API_KEY = os.getenv("COHERE_API_KEY")
# MODEL = "gpt-3.5-turbo"

#splitting the docs
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
docs = text_splitter.split_documents(pages)

#pc = Pinecone(api_key=PINECONE_API_KEY)


embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


# pc.create_index(
#     name=index_name,
#     dimension=384, # Replace with your model dimensions
#     metric="cosine", # Replace with your model metric
#     spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     ) 
# )

model=Ollama(model="llama3.1")



### modificication -------------------------------------------------------------------------------------
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=PINECONE_API_KEY)


index_name = "index2"

if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name, 
            dimension=384, 
            metric='cosine',
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

# Use Pinecone from Langchain to create the vector store
index = LangchainPinecone.from_documents(docs, embeddings, index_name=index_name)



###-------------------------------------------------------------------------------------------------------------------



retriever = index.as_retriever()
query = "personal"
def retrieve_query(query,k=5):
    matching_results=index.similarity_search(query,k=k)
    return matching_results

retrieve_query(query)



parser = StrOutputParser()

chain = model | parser 
#chain.invoke("Who was Napoleon?")

template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

prompt = PromptTemplate.from_template(template)
prompt.format(context="Here is some context", question="Here is a question")

chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
    }
    | prompt
    | model
    | parser
)

question = """ What is the Meas Square Error for Naive forecast?
"""
print(f"Answer: {chain.invoke({'question': question})}")