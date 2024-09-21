# PDF Question-Answering App using LangChain, Pinecone, and Ollama

This project is a RAG app designed to perform question-answering (QA) on PDF documents. It leverages the LangChain framework for embedding, Pinecone for vector storage, and the Ollama language model for generating responses to user queries.

## Features
- **PDF Handling**: Load and split PDF files into manageable chunks for processing.
- **Embeddings**: Use the `SentenceTransformerEmbeddings` to create embeddings for document chunks.
- **Vector Storage**: Pinecone is used to store document embeddings and efficiently retrieve relevant chunks based on user questions.
- **LLM Integration**: The `Ollama` model (Llama 3.1) is used to generate natural language responses based on retrieved document content.
- **Environment Variables**: Secrets like API keys are securely managed using `.env` files.

## Requirements
- Python 3.8+
- The following Python libraries:
  - `langchain_community`
  - `pinecone`
  - `dotenv`
  - `Ollama`
  - `SentenceTransformerEmbeddings`

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/m-umar-j/RAG-APP
cd your-repo-name
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