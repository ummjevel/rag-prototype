from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from datetime import datetime

from app.models.schemas import ChatRequest, ChatResponse
from app.services.document_store import document_store
from app.services.ai_service import get_ai_service

router = APIRouter(
    prefix="/chat",
    tags=["chat"]
)

@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """사용자 질문에 대해 RAG 기반 답변을 제공합니다."""
    try:
        # 유사한 문서 검색
        similar_docs = document_store.search_similar(
            request.question, 
            request.max_results
        )
        
        # 검색 결과가 없을 때 처리
        if not similar_docs:
            return ChatResponse(
                answer="죄송합니다. 관련된 문서를 찾을 수 없어서 답변을 드릴 수 없습니다. 문서를 먼저 업로드해 주세요.",
                sources=[],
                timestamp=datetime.now()
            )
        
        # AI 답변 생성 (히스토리 포함)
        ai_service = get_ai_service()
        answer = ai_service.generate_answer_with_history(request.question, similar_docs, request.session_id)
        sources = [doc.metadata.get('source', 'Unknown') for doc in similar_docs]
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"답변 생성 중 오류가 발생했습니다: {str(e)}"
        )

@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """스트리밍 방식으로 답변을 제공합니다."""
    try:
        # 유사한 문서 검색
        similar_docs = document_store.search_similar(
            request.question, 
            request.max_results
        )
        
        # 검색 결과가 없을 때 처리
        if not similar_docs:
            def no_docs_generator():
                yield "data: " + "죄송합니다. 관련된 문서를 찾을 수 없어서 답변을 드릴 수 없습니다. 문서를 먼저 업로드해 주세요." + "\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                no_docs_generator(),
                media_type="text/plain"
            )
        
        # AI 스트리밍 답변 생성
        ai_service = get_ai_service()
        
        def stream_generator():
            for chunk in ai_service.generate_answer_stream_with_history(request.question, similar_docs, request.session_id):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type="text/plain"
        )
    
    except Exception as e:
        def error_generator():
            yield f"data: 답변 생성 중 오류가 발생했습니다: {str(e)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            error_generator(),
            media_type="text/plain"
        )