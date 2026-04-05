from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
PERSIST_DIR = "./chroma_db"

def get_vectorstore(chunks: list | None = None, ) -> Chroma:
    embeddings = HuggingFaceEmbeddings()

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print("*" * 50)
        print("Existing Chroma DB found")
        return Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )

    if chunks is None:
        raise ValueError("No existing DB and no chunks provided")
    
    print("*" * 50)
    print("Building new Chroma DB")
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )