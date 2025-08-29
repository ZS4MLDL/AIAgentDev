from typing import List
from llama_index.embeddings.openai import OpenAIEmbedding
from fastapi import UploadFile
from sqlalchemy.orm import Session
from langchain_community.document_loaders import PyPDFLoader
import shutil, tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core import Document, StorageContext, VectorStoreIndex
from db import get_vector_store
from rag.agentic_rag.model_document import LiDocument

def ingest_pdf_to_li(file: UploadFile, embeddings: OpenAIEmbedding, db: Session) -> List[str]:
    """
    Ingest a PDF into the li_document table using LlamaIndex.
    Returns list of node_ids for inserted chunks.
    """
    # Save PDF to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        file.file.seek(0)
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # Load and split PDF
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    page_texts = [page.page_content for page in pages]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts: List[str] = []
    for page in page_texts:
        texts.extend(splitter.split_text(page))

    # Convert into LlamaIndex Document objects
    docs = [Document(text=t, metadata={"source_file": file.filename}) for t in texts]

    # Setup vector store
    store = get_vector_store("li_document")
    
    storage_context = StorageContext.from_defaults(vector_store=store)

    # Insert into vector store via LlamaIndex
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_context,
        embed_model=embeddings,
    )

    # Create embeddings for each chunk
    chunk_embeddings: List[List[float]] = embeddings.get_text_embedding_batch([d.text for d in docs])

    # Build ORM rows
    rows = []
    for d, vec in zip(docs, chunk_embeddings):
        rows.append(
            LiDocument(
                node_id=d.id_,            # LlamaIndex assigns a UUID-like id
                text=d.text,
                embedding=vec,            # pgvector expects a list[float]
                metadata_=(d.metadata or {})
            )
        )

    # Save
    db.add_all(rows)
    db.commit()
    # Optionally refresh if you need server defaults (not necessary for return ids)
    # for r in rows: db.refresh(r)

    # Each Document gets a node_id automatically
    return [doc.id_ for doc in docs]

   
    