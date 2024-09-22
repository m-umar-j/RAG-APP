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



embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

model = ChatMistralAI(mistral_api_key=MISTRAL_API_KEY)
pinecone_index = "index"
index_name = create_index(pinecone_index, PINECONE_API_KEY)

def save_file(fileobj):
    
    upload_dir = "RAG-APP/data/"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    # Save the file to the disk
    file_path = os.path.join(upload_dir, os.path.basename(fileobj.name))
    shutil.copyfile(fileobj.name, file_path)
    
    return file_path

def process_pdf(fileobj):
    file_path = save_file(fileobj)
    docs = load_split_file(file_path)

    
    index = LangchainPinecone.from_documents(docs, embeddings, index_name=index_name)

    return index, "File Uploaded Successfully"



with gr.Blocks() as Iface:
    file_input = gr.File(label="Upload PDF")  # This will give you the uploaded file's tempfile object

    upload_file = gr.Button("Upload File")
    index_state = gr.State()

    message = gr.Textbox("Please wait while the file is processed!")
    upload_file.click(fn = lambda file:
                      process_pdf(file),
                      inputs = file_input,
                      outputs = [index_state, message])

                        
    question_input = gr.Textbox(label="Ask any question about your document")
    
    submit_button = gr.Button("Get Answer")

    with gr.Row():
        answer_output = gr.Textbox(label="Answer", scale=3)
        matching_results = gr.Textbox(label="Reference", scale=1)
    

    submit_button.click(
        fn=lambda index, q: final_response(index, q, model),
        inputs=[index_state, question_input],
        outputs=[answer_output, matching_results]
    )

Iface.launch()
