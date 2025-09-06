from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ChatRequest(BaseModel):
    question: str
    max_results: int = 3
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    timestamp: datetime

class DocumentResponse(BaseModel):
    message: str
    document_count: int

class Document(BaseModel):
    content: str
    source: str
    timestamp: datetime

class DocumentPreview(BaseModel):
    source: str
    content_preview: str
    timestamp: datetime

class DocumentListResponse(BaseModel):
    document_count: int
    documents: List[DocumentPreview]

class SimilarDocument(BaseModel):
    content: str
    source: str
    similarity: float

class DeleteResponse(BaseModel):
    message: str