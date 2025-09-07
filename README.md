# AI RAG Chatbot

고급 RAG(Retrieval-Augmented Generation) 기반 AI 챗봇 시스템

## 🌟 주요 기능

### 📚 문서 처리
- **PDF 문서 업로드**: PDF 파일을 업로드하여 자동으로 텍스트 추출 및 벡터화
- **ChromaDB 영구 저장**: 문서 벡터를 ChromaDB에 영구 저장하여 재시작 시에도 유지
- **중복 방지**: 파일 해시를 이용한 중복 업로드 자동 감지
- **실시간 로그**: 문서 처리 과정을 실시간으로 확인 가능

### 🤖 AI 대화
- **Ollama Llama3 연동**: 로컬 Llama3 모델을 사용한 고품질 답변 생성
- **스트리밍 답변**: ChatGPT처럼 텍스트가 실시간으로 타이핑되어 표시
- **문맥 기반 답변**: 업로드된 문서 내용을 바탕으로 정확한 답변 생성

### 💭 대화 기록 관리
- **세션별 히스토리**: 각 사용자별로 독립적인 대화 세션 관리
- **RAG 대화 검색**: 과거 대화 중 현재 질문과 관련된 대화를 유사도 검색으로 자동 선별
- **효율적 저장**: 5번째 대화마다 ChromaDB에 영구 저장 (성능 최적화)
- **지능적 히스토리**: 관련 대화 3개 + 최근 대화 3개를 조합하여 맥락 제공

### 🎨 사용자 인터페이스
- **반응형 웹 UI**: 모던하고 직관적인 채팅 인터페이스
- **파일 드래그 앤 드롭**: 간편한 문서 업로드
- **세션 관리**: 새 대화 시작 버튼으로 언제든 새로운 세션 생성
- **실시간 타이핑 효과**: AI 답변이 실시간으로 표시

## 🏗️ 시스템 아키텍처

### 백엔드 구조
```
app/
├── main.py                 # FastAPI 애플리케이션 진입점
├── config.py              # 설정 관리 (파일 크기, 확장자 등)
├── models/
│   └── schemas.py         # Pydantic 모델 정의
├── routers/
│   ├── chat.py           # 채팅 API (일반/스트리밍)
│   ├── documents.py      # 문서 업로드/관리 API
│   └── ui.py             # 웹 UI 제공
└── services/
    ├── ai_service.py     # AI 답변 생성 및 대화 관리
    └── document_store.py # 문서 벡터 저장 관리
```

### 데이터 저장소
```
chroma_db/                 # 문서 벡터 저장소
conversation_db/           # 세션별 대화 저장소
├── session_{id}/          # 각 세션별 ChromaDB
```

## 🚀 설치 및 실행

### 1. 필요 조건
- Python 3.10+
- Ollama (Llama3 모델)
- ChromaDB

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. Ollama 설정
```bash
# Ollama 설치 후
ollama pull llama3
ollama serve  # 백그라운드에서 실행
```

### 4. 애플리케이션 실행
```bash
python main.py
```

브라우저에서 `http://localhost:8000` 접속

## 📋 requirements.txt
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
pydantic==2.5.0
pydantic-settings==2.1.0
numpy
scikit-learn
python-dotenv==1.0.0
langchain-community
langchain-core
langchain-ollama
pypdf
langchain-chroma
```

## 🔧 주요 설정

### config.py
```python
# 파일 업로드 설정
allowed_extensions = [".txt", ".md", ".pdf"]
max_file_size = 10 * 1024 * 1024  # 10MB

# RAG 설정
max_results = 3
similarity_threshold = 0.1
```

## 📡 API 엔드포인트

### 채팅
- `POST /chat` - 일반 채팅 (JSON 응답)
- `POST /chat/stream` - 스트리밍 채팅 (실시간 텍스트)

### 문서 관리
- `POST /documents/upload` - 문서 업로드
- `GET /documents` - 업로드된 문서 목록
- `DELETE /documents` - 모든 문서 삭제

### UI
- `GET /` - 웹 채팅 인터페이스

## 🧠 AI 기능 상세

### 1. 문서 RAG
- 업로드된 PDF에서 유사한 내용 검색
- 검색된 문서 조각을 컨텍스트로 제공
- Ollama 임베딩을 사용한 의미적 유사도 검색

### 2. 대화 RAG  
- 과거 대화 중 현재 질문과 관련된 대화 자동 검색
- 5개 이상의 대화가 있을 때 활성화
- 관련 대화 + 최근 대화를 조합하여 더 정확한 맥락 제공

### 3. 히스토리 관리
```
대화 1-4: 메모리에만 저장
대화 5: ChromaDB에 영구 저장 ✅
대화 6-9: 메모리에만 저장  
대화 10: ChromaDB에 영구 저장 ✅
```

## 🎯 사용 시나리오

1. **문서 업로드**: PDF 파일을 업로드하여 지식베이스 구성
2. **질문하기**: 업로드된 문서에 대해 자연어로 질문
3. **대화 이어가기**: AI가 이전 대화 맥락을 기억하여 연속적인 대화 가능
4. **관련 대화 활용**: 과거의 유사한 질문과 답변을 자동으로 참조

## 🔍 고급 기능

### 스트리밍 답변
- 실시간 텍스트 생성으로 사용자 경험 향상
- Server-Sent Events 방식 구현
- 타이핑 인디케이터 및 자동 스크롤

### 세션 관리
- localStorage 기반 세션 ID 자동 생성
- 세션별 독립적인 대화 히스토리
- 새 대화 시작 기능

### 성능 최적화
- 문서 벡터 영구 저장으로 재처리 방지
- 대화 저장 주기 최적화 (5번째마다)
- 임베딩 모델 재사용으로 초기화 시간 단축

## 🤝 기여하기

1. 이 저장소를 포크하세요
2. 새로운 기능 브랜치를 만드세요
3. 변경사항을 커밋하세요
4. 브랜치에 푸시하세요
5. Pull Request를 생성하세요

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다.
