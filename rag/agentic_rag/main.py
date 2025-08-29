from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List
from rag.agentic_rag.db import get_db, get_vector_store_index
from rag.agentic_rag.model_document import LiDocument, LiDocumentInDB, LiDocumentSummary
from rag.agentic_rag.agent import get_contextual_answer as get_answer
from llama_index.embeddings.openai import OpenAIEmbedding
from rag.agentic_rag.services import ingest_pdf_to_li
from rag.agentic_rag.agent import get_agent_instance
from dotenv import load_dotenv
import os
import logging

app = FastAPI()

logging.basicConfig(
    level=logging.INFO,  # or DEBUG, WARNING, ERROR
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

load_dotenv()

class QueryRequest(BaseModel):
    """Schema for the query endpoint."""
    question: str

@app.post("/upload_pdf", response_model=List[str])
async def upload_pdf(file: UploadFile = File(...),db: Session = Depends(get_db)) -> List[int]:
    """Upload a PDF, embed its content, and store it in `li_document`."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    try:
        embeddings = OpenAIEmbedding(model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))
        ids = ingest_pdf_to_li(file, embeddings, db)
        return ids
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/documents", response_model=List[LiDocumentSummary])
def list_documents(db: Session = Depends(get_db)) -> List[LiDocumentSummary]:
    """List all stored document chunks from li_document."""
    docs = db.query(LiDocument).all()
    return [LiDocumentSummary(node_id=doc.node_id, text=doc.text) for doc in docs]


@app.get("/documents/{node_id}", response_model=LiDocumentSummary)
def get_document(node_id: str, db: Session = Depends(get_db)) -> LiDocumentSummary:
    """Retrieve a document chunk from li_document by ID."""
    doc = db.query(LiDocument).filter(LiDocument.node_id == node_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return LiDocumentSummary(node_id=doc.node_id, text=doc.text) 


@app.delete("/documents/{node_id}")
def delete_document(node_id: str, db: Session = Depends(get_db)) -> JSONResponse:
    """Delete a document chunk from li_document by ID."""
    doc = db.query(LiDocument).filter(LiDocument.node_id == node_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    db.delete(doc)
    db.commit()
    return JSONResponse(content={"detail": "Deleted"})


@app.post("/query")
async def query_agent(request: QueryRequest, db: Session = Depends(get_db)) -> JSONResponse:
    """Query via agent (vector/sql/web)."""
    agent = get_agent_instance(db)
    answer = agent.run(request.question)
    return JSONResponse(content={"answer": answer})


@app.post("/get_contextual_answer")
async def get_contextual_answer(request: QueryRequest, db: Session = Depends(get_db)) -> JSONResponse:
    answer = await get_answer(request.question, db)
    return JSONResponse(content={"answer": answer})


class DebugNode(BaseModel):
    id: str
    text_snippet: str
    score: float = 0.0

@app.get("/debug/query_vectorstore", response_model=List[DebugNode])
def debug_query_vectorstore(question: str, db: Session = Depends(get_db)) -> List[DebugNode]:
    """Debug vector store retrieval via LlamaIndex."""
    try:
        index = get_vector_store_index("li_document")
        query_engine = index.as_query_engine(similarity_top_k=5, show_progress=True)
        response = query_engine.query(question)

        if not hasattr(response, "source_nodes") or not response.source_nodes:
            raise HTTPException(status_code=404, detail="No documents retrieved")

        nodes = response.source_nodes

        return [
            DebugNode(
                id=str(node.node.node_id),
                text_snippet=node.node.get_content()[:200],
                score=node.score or 0.0,
            )
            for node in nodes
        ]

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
