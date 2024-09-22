---
title: QA-bot
app_file: app.py
sdk: gradio
sdk_version: 4.44.0
---
# PDF Question-Answering App using LangChain, Pinecone, and Mistral
![image](https://github.com/user-attachments/assets/8c59f9ae-d1ef-4c35-a1ea-1a1d43815de4)
This project is a RAG app designed to perform question-answering (QA) on PDF documents. It uses the `LangChain` framework for embedding, `Pinecone` for vector storage, and the `mistral` language model for generating responses to user queries.
## Demo 
https://drive.google.com/file/d/1oRBGK7Y0gUcGnKUW2GgcqK5bZwNaMR_k/view?usp=sharing
## Huggingface space link
https://huggingface.co/spaces/umar-100/QA-bot

## Features
- **PDF Handling**: Load and split PDF files into manageable chunks for processing.
- **Embeddings**: I am using the `SentenceTransformerEmbeddings` to create embeddings for document chunks.
- **Vector Storage**: Pinecone is used to store document embeddings and efficiently retrieve relevant chunks based on user questions.
- **LLM Integration**: I tried using LLMs locally using `Ollama` but due to lack of compute resources I wnt with `mistral` for faster and better responses.
- **Environment Variables**: Secrets like API keys are securely managed using `.env` files.
- **UI/frontend**: Gradio 
## Deployement
- It is deployed on huggingface spaces with gradio interface
## Requirements
- Python 3.12
- Run `pip install -r requirements.txt`
- The following teck stack is used:
  - `langchain`
  - `pinecone` for vector database. Make sure to sign up and create Pinecone API key 
  - `Mistral API` Make sure to sign up and generate API key
  

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
