 # agent/rag_store.py
from typing import List
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_vectorstore(urls: List[str], persist_dir="vet_db",
                      chunk_size=600, chunk_overlap=80,
                      embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    docs = WebBaseLoader(urls).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    embedding = HuggingFaceEmbeddings(model_name=embedding_model)

    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_dir
    )
    return vs, chunks

def get_retriever(vectorstore, k=4):
    return vectorstore.as_retriever(search_kwargs={"k": k})