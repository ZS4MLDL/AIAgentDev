# AIAgentDev

# Simple RAG Backend for AI Agents with FastAPI and OpenAI

Python files and their purpose..

rag/app/main.py: FastAPI server and API routes (/upload/, /documents/, etc.)

rag/models/document.py: SQLAlchemy Document model; table for storing PDF chunks and embeddings.

rag/ingest/ingest.py: Functions to extract text from PDF, chunk, and embed via OpenAI API.

rag/operations/crud.py: Adds document chunks and embeddings to the DB.

rag/operations/vector_search.py: Adds document chunks and embeddings to the DB.

rag/db/db.py: Database session/connection logic.

alembic/: Database migrations for schema versioning.

pyproject.toml & poetry.lock: Python dependencies and exact version lock.

Add .env file for OPENAI_API_KEY or add key to environment variable (both should work)

Clone the code and run the application as below

# uvicorn rag.app.main:app --host 0.0.0.0 --port 8000 --reload


