from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # 앱 설정
    app_name: str = "AI RAG Chatbot"
    app_description: str = "Simple RAG (Retrieval-Augmented Generation) chatbot with document ingestion"
    app_version: str = "1.0.0"
    
    # 서버 설정
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # RAG 설정
    max_results: int = 3
    similarity_threshold: float = 0.1
    max_features: int = 1000
    
    # 대화 히스토리 설정
    max_recent_conversations: int = 10  # RAG 대화가 없을 때 최근 대화 개수
    max_recent_with_rag: int = 3       # RAG 관련 대화가 있을 때 최근 대화 개수  
    max_related_conversations: int = 3  # 검색된 관련 대화 최대 개수
    
    # 파일 업로드 설정
    allowed_extensions: list = [".txt", ".md", ".pdf", ".docx"]
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    
    # OpenAI 설정 (선택사항)
    openai_api_key: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()