from sqlalchemy import Column, Integer, String, DateTime, LargeBinary, Text
from pgvector.sqlalchemy import VECTOR
from sqlalchemy.sql import func
from . import Base

class Document(Base):
    __tablename__ = "document"
    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    uploaded_at = Column(DateTime, server_default=func.now())
    content = Column(Text, nullable=False)  
    embedding = Column(VECTOR(1536), nullable=False)  
    doc_metadata = Column(String) 

    
