from typing import List, Dict, Optional
import logging
import os
import hashlib

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


from app.config import settings

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentStore:
    def __init__(self):
        self.documents: List[Document] = []
        self.embeddings = OllamaEmbeddings(model="llama3")
        
        # ChromaDB persistent storage
        self.chroma_dir = "chroma_db"
        self.vector_store = Chroma(
            persist_directory=self.chroma_dir,
            embedding_function=self.embeddings,
            collection_name="documents"
        )
        
        # 시작 시 기존 문서 개수 확인
        self._load_existing_documents()
        
    def _load_existing_documents(self):
        """ChromaDB에서 기존 문서 개수 확인"""
        try:
            collection = self.vector_store._collection
            count = collection.count()
            logger.info(f"ChromaDB에서 {count}개의 기존 문서 발견")
        except Exception as e:
            logger.info(f"ChromaDB 초기화: {e}")
    
    def _document_exists(self, filename: str, file_hash: str) -> bool:
        """문서가 이미 ChromaDB에 있는지 확인"""
        try:
            # 메타데이터로 검색 (AND 조건)
            results = self.vector_store.get(
                where={
                    "$and": [
                        {"filename": {"$eq": filename}},
                        {"file_hash": {"$eq": file_hash}}
                    ]
                }
            )
            return len(results['ids']) > 0
        except Exception as e:
            logger.error(f"문서 존재 확인 실패: {e}")
            return False
    
    def _get_file_hash(self, file_path: str) -> str:
        """파일의 해시값을 계산"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    
    def add_document(self, file_path: str, original_filename: str = None) -> None:
        """문서를 저장소에 추가"""
        filename = original_filename or os.path.basename(file_path)
        logger.info(f"문서 로딩 시작: {filename}")
        
        # 파일 해시 계산
        file_hash = self._get_file_hash(file_path)
        
        # ChromaDB에 이미 존재하는지 확인
        if self._document_exists(filename, file_hash):
            logger.info(f"문서가 이미 ChromaDB에 존재함: {filename}")
            return
        
        # 새 문서 처리
        logger.info("새 문서 처리 중...")
        loader = PyPDFLoader(file_path)
        pages = []
        
        logger.info("PDF 페이지 로딩 중...")
        for i, page in enumerate(loader.load()):
            # 메타데이터에 파일 정보 추가
            page.metadata.update({
                'filename': filename,
                'file_hash': file_hash,
                'page_number': i
            })
            pages.append(page)
            logger.info(f"페이지 {i+1} 로딩 완료")
            
        logger.info(f"총 {len(pages)}개 페이지 로딩 완료")
        
        logger.info("ChromaDB에 문서 추가 중...")
        self.vector_store.add_documents(documents=pages)
        self.documents.extend(pages)
        logger.info("문서 추가 완료")
    
    def search_similar(self, query: str, k: Optional[int] = 1):
        """유사한 문서 검색"""
        logger.info(f"'{query}' 검색 중... (k={k})")
        docs = self.vector_store.similarity_search(query=query, k=k)
        logger.info(f"{len(docs)}개 유사 문서 발견")
        
        return docs
    
    def get_all_documents(self) -> List[Document]:
        """모든 문서 반환"""
        return self.documents
    
    def get_document_count(self) -> int:
        """문서 개수 반환"""
        try:
            return self.vector_store._collection.count()
        except:
            return len(self.documents)
    
    def clear_documents(self) -> None:
        """모든 문서 삭제"""
        self.documents.clear()
        try:
            self.vector_store._collection.delete()
            logger.info("ChromaDB에서 모든 문서 삭제됨")
        except Exception as e:
            logger.error(f"ChromaDB 삭제 실패: {e}")

# 전역 문서 저장소 인스턴스
document_store = DocumentStore()