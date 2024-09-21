from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.prompts import PromptTemplate 
from langchain_core.output_parsers import StrOutputParser 
from operator import itemgetter

from pinecone import Pinecone, ServerlessSpec



def load_split_file(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    docs = text_splitter.split_documents(pages)

    return docs


def create_index(index_name, PINECONE_API_KEY):
      
    pc = Pinecone(api_key=PINECONE_API_KEY)

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

    return index_name


def final_response(index, question, model):
    retriever = index.as_retriever()

    parser = StrOutputParser()

    chain = model | parser 

    template = """
    Answer the question based on the context below. Try to find the answer in the context.
    If you don't find the answer, reply "I don't know".

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

    return f"Answer: {chain.invoke({'question': question})}"

