from sqlalchemy import select
from rag.db import SessionLocal
from rag.models import Document
from rag.ingest import embed_text

def query_similar_documents(query, top_k=5):
    embedding = embed_text(query)
    session = SessionLocal()
    docs = session.execute(
        select(Document)
        .order_by(Document.embedding.l2_distance(embedding))
        .limit(top_k)
    ).scalars().all()
    session.close()
    return docs
