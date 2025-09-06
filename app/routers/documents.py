from fastapi import APIRouter, HTTPException, UploadFile, File
import os
import tempfile
from datetime import datetime

from app.models.schemas import DocumentResponse, DocumentListResponse, DocumentPreview, DeleteResponse
from app.services.document_store import document_store
from app.config import settings

router = APIRouter(
    prefix="/documents",
    tags=["documents"]
)

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """문서를 업로드하고 벡터 저장소에 추가합니다."""
    # 파일 확장자 검증
    file_ext = "." + file.filename.split(".")[-1] if "." in file.filename else ""
    if file_ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"지원되지 않는 파일 형식입니다. ({', '.join(settings.allowed_extensions)}만 지원)"
        )
    
    try:
        # 파일 크기 검증
        content = await file.read()
        if len(content) > settings.max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"파일 크기가 너무 큽니다. (최대 {settings.max_file_size // (1024*1024)}MB)"
            )
        
        # 문서 저장소에 추가 - 임시 파일 경로 전달
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            document_store.add_document(tmp_file_path, file.filename)
        finally:
            # 임시 파일 삭제
            os.unlink(tmp_file_path)
        
        return DocumentResponse(
            message=f"문서 '{file.filename}'이 성공적으로 업로드되었습니다.",
            document_count=document_store.get_document_count()
        )
        
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400, 
            detail="UTF-8로 디코딩할 수 없는 파일입니다."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"문서 처리 중 오류가 발생했습니다: {str(e)}"
        )

@router.get("", response_model=DocumentListResponse)
async def get_documents():
    """저장된 모든 문서 목록을 반환합니다."""
    documents = document_store.get_all_documents()
    
    document_previews = [
        DocumentPreview(
            source=doc.metadata.get('source', 'Unknown'),
            content_preview=(
                doc.page_content[:100] + "..." 
                if len(doc.page_content) > 100 
                else doc.page_content
            ),
            timestamp=datetime.now()  # 현재는 저장된 timestamp가 없으므로 현재 시간 사용
        )
        for doc in documents
    ]
    
    return DocumentListResponse(
        document_count=len(documents),
        documents=document_previews
    )

@router.delete("", response_model=DeleteResponse)
async def clear_documents():
    """모든 문서를 삭제합니다."""
    document_store.clear_documents()
    return DeleteResponse(message="모든 문서가 삭제되었습니다.")