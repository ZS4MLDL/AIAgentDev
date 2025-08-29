import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel
from pydantic import ConfigDict
from sqlalchemy import Column, BigInteger, Text, DateTime,String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from rag.agentic_rag.db import Base  


class LiDocument(Base):
    __tablename__ = "li_document"
    node_id = Column(String, primary_key=True)    
    text    = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=False)
    metadata_ = Column("metadata",JSONB)    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    

class LiDocumentInDB(BaseModel):
    node_id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata_: Optional[Dict] = None
    created_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)


class LiDocumentSummary(BaseModel):
    node_id: str
    text: str

    model_config = {"from_attributes": True}
