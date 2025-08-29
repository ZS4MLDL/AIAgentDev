import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from sqlalchemy import make_url

# SQLAlchemy setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5434/ragdb")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# FastAPI DB dependency
def get_db() -> Session:  # type: ignore
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Utility: expose engine and URL
def get_engine():
    return engine

def get_db_url():
    return DATABASE_URL


def get_vector_store(table_name: str) -> PGVectorStore:

    url = make_url("postgresql://postgres:postgres@localhost:5434/")
    vector_store = PGVectorStore.from_params(
    database="ragdb",
    host=url.host,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name=table_name,
    embed_dim=1536, 
    )

    return vector_store


# VectorStoreIndex over `li_document` table
def get_vector_store_index(table_name: str) -> VectorStoreIndex:

    url = make_url("postgresql://postgres:postgres@localhost:5434/")
    store = PGVectorStore.from_params(
    database="ragdb",
    host=url.host,
    password=url.password,
    port=url.port,
    user=url.username,
    table_name=table_name,
    embed_dim=1536, 
    )
   
    embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
    
    return VectorStoreIndex.from_vector_store(vector_store=store, embed_model=embed_model)
   