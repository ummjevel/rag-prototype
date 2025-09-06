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
    
    # 파일 업로드 설정
    allowed_extensions: list = [".txt", ".md", ".pdf"]
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    
    # OpenAI 설정 (선택사항)
    openai_api_key: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()