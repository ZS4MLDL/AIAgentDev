from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from rag.models import Document
from rag.db.db import SessionLocal
from rag.ingest.ingest import ingest_pdf
from sqlalchemy.orm import Session
import openai, os
from dotenv import load_dotenv
from fastapi.encoders import jsonable_encoder

app = FastAPI()

load_dotenv() 
openai.api_key = os.getenv("OPENAI_API_KEY")

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    path = f"/tmp/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    # ingest_pdf will need to accept a db session
    ingest_pdf(path, file.filename)
    return {"status": "uploaded"}

@app.get("/documents/", response_model=List[dict])
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).all()
    # Build a list of plain dicts
    payload = [
        {
            "id": d.id,
            "filename": d.filename,
            "uploaded_at": d.uploaded_at,
            "metadata": d.doc_metadata,
        }
        for d in docs
    ]
    # Use jsonable_encoder to turn datetimes into ISO strings, etc.
    return jsonable_encoder(payload)

@app.get("/documents/{doc_id}")
def get_document(doc_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "id": doc.id,
        "filename": doc.filename,
        "uploaded_at": doc.uploaded_at,
        "content": doc.content,
        "metadata": doc.doc_metadata,
    }

@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    db.delete(doc)
    db.commit()
    return {"status": "deleted"}
