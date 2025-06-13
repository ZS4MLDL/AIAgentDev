from fastapi import FastAPI, UploadFile, File
from rag.ingest.ingest  import ingest_pdf

app = FastAPI()

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    path = f"/tmp/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    ingest_pdf(path, file.filename)
    return {"status": "uploaded"}
