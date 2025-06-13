import openai
from pypdf import PdfReader
from rag.operations.crud import add_document_chunks
from rag.db.db import SessionLocal

def extract_pdf_text(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() for page in reader.pages)

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    for i in range(0, len(words), chunk_size - overlap):
        yield " ".join(words[i:i + chunk_size])

def embed_text(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-3-small",
        
    )
    return response['data'][0]['embedding']

def ingest_pdf(path, filename):
    text = extract_pdf_text(path)
    chunks = list(chunk_text(text))
    embeddings = [embed_text(chunk) for chunk in chunks]
    session = SessionLocal()
    add_document_chunks(filename, chunks, embeddings, session)
    session.close()