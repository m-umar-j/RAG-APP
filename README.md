# PDF Question-Answering App using LangChain, Pinecone, and Mistral

This project is a RAG app designed to perform question-answering (QA) on PDF documents. It uses the `LangChain` framework for embedding, `Pinecone` for vector storage, and the `mistral` language model for generating responses to user queries.

## Features
- **PDF Handling**: Load and split PDF files into manageable chunks for processing.
- **Embeddings**: I am using the `SentenceTransformerEmbeddings` to create embeddings for document chunks.
- **Vector Storage**: Pinecone is used to store document embeddings and efficiently retrieve relevant chunks based on user questions.
- **LLM Integration**: I tried using LLMs locally using `Ollama`but due to lack of compute resources I used `mistral` for faster and better responses.
- **Environment Variables**: Secrets like API keys are securely managed using `.env` files.

## Requirements
- Python 3.12
- Run `pip install -r requirements.txt`
- The following teck stack is used:
  - `langchain`
  - `pinecone` Make sure to sign up and create Pinecone API key 
  - `Mistral API`
  

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/m-umar-j/RAG-APP
cd RAG-APP
```
### 2. install the requirements using 
 ```bash
 pip install -r requirements.txt`
```
### 3. create .env file in your root directory and add pinecone API key

``` makefile
PINECONE_API_KEY=your-pinecone-api-key
```
### 4. modify paths

`file_path = "/path/to/data.pdf"`