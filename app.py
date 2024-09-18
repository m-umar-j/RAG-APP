import pandas as pd
import pathway as pw
from pathway.stdlib.indexing import default_vector_document_index
from pathway.xpacks.llm import embedders
from pathway.xpacks.llm.llms import LiteLLMChat
from pathway.xpacks.llm.question_answering import (
    answer_with_geometric_rag_strategy_from_index,
)

class InputSchema(pw.Schema):
    doc: str

documents = pw.io.fs.read(
    "adaptive-rag-contexts.jsonl",
    format="json",
    schema=InputSchema,
    json_field_paths={"doc": "/context"},
    mode="static",
)

# Check if documents are correctly loaded
# print(documents)

df = pd.DataFrame(
    {
        "query": [
            "When it is burned what does hydrogen make?",
            "What was undertaken in 2010 to determine where dogs originated from?",
        ]
    }
)
query = pw.debug.table_from_pandas(df)

embedding_model = "avsolatorio/GIST-small-Embedding-v0"

embedder = embedders.SentenceTransformerEmbedder(
    embedding_model, call_kwargs={"show_progress_bar": False}
)  # disable verbose logs
embedding_dimension: int = embedder.get_embedding_dimension()
print("Embedding dimension:", embedding_dimension)

model = LiteLLMChat(
    model="ollama/mistral",
    temperature=0,
    top_p=1,
    api_base="http://localhost:11434",  # local deployment
    format="json",  # only available in Ollama local deploy, do not use in Mistral API
)

index = default_vector_document_index(
    documents.doc, documents, embedder=embedder, dimensions=embedding_dimension
)

result = query.select(
    question=query.query,
    result=answer_with_geometric_rag_strategy_from_index(
        query.query,
        index,
        documents.doc,
        model,
        n_starting_documents=2,
        factor=2,
        max_iterations=4,
        strict_prompt=True,  # needed for open source models, instructs LLM to give JSON output strictly
    ),
)

pw.debug.compute_and_print(result)

