from rag.models import Document
from rag.db.db import SessionLocal

def add_document_chunks(filename, chunks, embeddings, session):
    for chunk, embedding in zip(chunks, embeddings):
        doc = Document(
            filename=filename,
            content=chunk,
            embedding=embedding,
        )
        session.add(doc)
    session.commit()