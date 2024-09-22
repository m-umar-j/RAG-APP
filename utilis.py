from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser 
from operator import itemgetter

from pinecone import Pinecone, ServerlessSpec



def load_split_file(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
    docs = text_splitter.split_documents(pages)

    return docs


def create_index(index_name, PINECONE_API_KEY):
      
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name) # To avoid any conflicts in retrieval
    pc.create_index(
                name=index_name, 
                dimension=384, 
                metric='cosine',
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )

    return index_name


def final_response(index, question, model):
    retriever = index.as_retriever()

    parser = StrOutputParser()

    chain = model | parser 

    template = """
    You must provide an answer based strictly on the context below. 
    The answer is highly likely to be found within the given context, so analyze it thoroughly before responding. 
    Only if there is absolutely no relevant information, respond with "I don't know".
    Do not make things up.

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
    matching_results=index.similarity_search(question,k=2)

    return f"Answer: {chain.invoke({'question': question})}", matching_results

