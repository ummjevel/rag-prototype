from typing import List, Dict
from langchain_core.documents import Document
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
import logging
import requests
import json
import os

from app.config import settings

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        self.ollama_url = "http://localhost:11434"
        self.model = "llama3"
        
        # 세션별 채팅 히스토리 저장소 (메모리)
        self.store: Dict[str, InMemoryChatMessageHistory] = {}
        
        # 세션별 대화 ChromaDB 저장소 (영구)
        self.conversation_stores: Dict[str, Chroma] = {}
        self.conversation_dir = "conversation_db"
        self.embeddings = OllamaEmbeddings(model="llama3")
        
        # 대화 저장소 디렉토리 생성
        os.makedirs(self.conversation_dir, exist_ok=True)
        
        # Ollama 서버 연결 테스트
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                logger.info("Ollama 서버 연결 확인")
                self.available = True
            else:
                logger.error("Ollama 서버 연결 실패")
                self.available = False
        except Exception as e:
            logger.error(f"Ollama 서버 연결 실패: {e}")
            self.available = False
    
    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """세션 ID에 해당하는 채팅 히스토리 반환"""
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]
    
    def get_conversation_store(self, session_id: str) -> Chroma:
        """세션별 대화 ChromaDB 반환"""
        if session_id not in self.conversation_stores:
            session_dir = os.path.join(self.conversation_dir, f"session_{session_id}")
            self.conversation_stores[session_id] = Chroma(
                persist_directory=session_dir,
                embedding_function=self.embeddings,
                collection_name=f"conversations_{session_id}"
            )
        return self.conversation_stores[session_id]
    
    def store_conversation(self, session_id: str, question: str, answer: str, conversation_id: int):
        """대화를 ChromaDB에 저장"""
        try:
            conversation_store = self.get_conversation_store(session_id)
            
            # 대화 문서 생성 (질문과 답변을 하나로 결합)
            conversation_text = f"질문: {question}\n답변: {answer}"
            
            conversation_doc = Document(
                page_content=conversation_text,
                metadata={
                    'session_id': session_id,
                    'conversation_id': conversation_id,
                    'question': question,
                    'answer': answer,
                    'type': 'conversation'
                }
            )
            
            conversation_store.add_documents([conversation_doc])
            logger.info(f"대화 저장 완료: 세션 {session_id}, ID {conversation_id}")
            
        except Exception as e:
            logger.error(f"대화 저장 실패: {e}")
    
    def search_relevant_conversations(self, session_id: str, current_question: str, k: int = 3) -> List[Document]:
        """현재 질문과 관련된 이전 대화 검색"""
        try:
            conversation_store = self.get_conversation_store(session_id)
            
            # 해당 세션에 대화가 있는지 확인
            collection = conversation_store._collection
            if collection.count() < 5:  # 5개 미만이면 RAG 사용 안함
                return []
            
            # 유사한 대화 검색
            relevant_conversations = conversation_store.similarity_search(
                query=current_question,
                k=k
            )
            
            logger.info(f"관련 대화 {len(relevant_conversations)}개 검색됨")
            return relevant_conversations
            
        except Exception as e:
            logger.error(f"대화 검색 실패: {e}")
            return []
    
    def should_store_conversation(self, conversation_count: int) -> bool:
        """대화를 ChromaDB에 저장할지 판단 (5의 배수일 때만)"""
        return conversation_count > 0 and conversation_count % 5 == 0
    
    def _summarize_old_conversations(self, messages) -> str:
        """이전 대화들을 요약"""
        if len(messages) <= 10:
            return ""
        
        # 오래된 메시지들 (최근 10개 제외)
        old_messages = messages[:-10]
        
        # 대화 내용을 텍스트로 변환
        conversation_text = ""
        for msg in old_messages:
            if hasattr(msg, 'content'):
                role = "사용자" if msg.__class__.__name__ == "HumanMessage" else "AI"
                conversation_text += f"{role}: {msg.content}\n"
        
        if not conversation_text.strip():
            return ""
        
        # 요약 생성을 위한 프롬프트
        summary_prompt = f"""다음 대화를 간단하게 요약해주세요. 주요 주제와 중요한 정보만 포함하세요.

{conversation_text}

요약:"""
        
        try:
            if not self.available:
                return "이전 대화 요약 불가"
            
            payload = {
                "model": self.model,
                "prompt": summary_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                summary = result.get("response", "").strip()
                logger.info("대화 요약 생성 완료")
                return summary
            else:
                return "이전 대화 요약 실패"
                
        except Exception as e:
            logger.error(f"대화 요약 생성 실패: {e}")
            return "이전 대화 요약 중 오류 발생"
    
    def _format_history_with_rag(self, session_id: str, current_question: str, history: InMemoryChatMessageHistory) -> str:
        """RAG 방식으로 관련 대화 + 최근 대화를 포맷팅"""
        messages = history.messages
        
        if not messages:
            return ""
        
        history_text = "\n\n**대화 기록:**\n"
        
        # 관련 대화 검색 (5개 이상일 때만)
        relevant_conversations = self.search_relevant_conversations(session_id, current_question, k=settings.max_related_conversations)
        
        if relevant_conversations:
            history_text += "[관련된 이전 대화]:\n"
            for i, conv in enumerate(relevant_conversations):
                # 메타데이터에서 질문과 답변 추출
                question = conv.metadata.get('question', '')
                answer = conv.metadata.get('answer', '')
                if question and answer:
                    history_text += f"관련대화 {i+1}) 질문: {question[:100]}{'...' if len(question) > 100 else ''}\n"
                    history_text += f"            답변: {answer[:150]}{'...' if len(answer) > 150 else ''}\n\n"
        
        # 최근 대화 메시지 (RAG 관련 대화가 있으면 줄임)
        recent_count = settings.max_recent_with_rag if relevant_conversations else settings.max_recent_conversations
        recent_messages = messages[-recent_count:] if len(messages) > recent_count else messages
        
        if recent_messages:
            history_text += "[최근 대화]:\n"
            for msg in recent_messages:
                if hasattr(msg, 'content'):
                    role = "사용자" if msg.__class__.__name__ == "HumanMessage" else "AI"
                    content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                    history_text += f"{role}: {content}\n"
        
        return history_text
    
    def _create_prompt(self, context: str, question: str) -> str:
        """RAG 프롬프트 생성"""
        return f"""당신은 도움이 되는 AI 어시스턴트입니다. 
제공된 문서의 내용을 바탕으로 사용자의 질문에 정확하고 유용한 답변을 해주세요.

**지침:**
1. 제공된 문서의 정보만을 사용해서 답변하세요
2. 문서에 없는 내용은 추측하지 마세요  
3. 답변을 할 수 없다면 솔직히 말해주세요
4. 한국어로 자연스럽게 답변해주세요
5. 최대한 간단하고 쉽게 답변해주세요

**제공된 문서:**
{context}

**질문:** {question}

**답변:**"""
    
    def generate_answer(self, question: str, context_docs: List[Document]) -> str:
        """질문과 컨텍스트를 바탕으로 LLM 답변 생성"""
        if not context_docs:
            return "죄송합니다. 관련 문서를 찾을 수 없습니다. 더 많은 문서를 업로드해주세요."
        
        # 컨텍스트 문서들 결합
        context = "\n\n".join([
            f"[문서 {i+1}]\n{doc.page_content[:1000]}{'...' if len(doc.page_content) > 1000 else ''}"
            for i, doc in enumerate(context_docs)
        ])
        
        # Ollama 서버 확인
        if not self.available:
            return "죄송합니다. AI 서비스가 사용할 수 없습니다. Ollama가 실행 중인지 확인해주세요."
        
        try:
            # 프롬프트 생성
            prompt = self._create_prompt(context, question)
            
            # Ollama API 직접 호출
            logger.info(f"LLM에 질문 전송: {question[:50]}...")
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.1
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60,
                stream=True
            )
            
            if response.status_code == 200:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if 'response' in chunk:
                                full_response += chunk['response']
                            if chunk.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
                
                logger.info("LLM 답변 생성 완료")
                return full_response.strip()
            else:
                logger.error(f"Ollama API 오류: {response.status_code}")
                return f"죄송합니다. AI 서비스 오류가 발생했습니다. (상태코드: {response.status_code})"
            
        except Exception as e:
            logger.error(f"LLM 답변 생성 실패: {e}")
            return f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    def generate_answer_with_history(self, question: str, context_docs: List[Document], session_id: str = "default") -> str:
        """히스토리를 포함한 답변 생성 (요약 포함)"""
        if not context_docs:
            return "죄송합니다. 관련 문서를 찾을 수 없습니다. 더 많은 문서를 업로드해주세요."
        
        # 히스토리 기능이 사용 불가능한 경우 기본 답변 생성
        if not self.available:
            return self.generate_answer(question, context_docs)
        
        # 컨텍스트 문서들 결합
        context = "\n\n".join([
            f"[문서 {i+1}]\n{doc.page_content[:1000]}{'...' if len(doc.page_content) > 1000 else ''}"
            for i, doc in enumerate(context_docs)
        ])
        
        # 히스토리 가져오기 (RAG 관련 대화 + 최근 대화)
        history = self.get_session_history(session_id)
        history_text = self._format_history_with_rag(session_id, question, history)
        
        # 프롬프트 생성
        prompt = f"""당신은 도움이 되는 AI 어시스턴트입니다. 
제공된 문서의 내용을 바탕으로 사용자의 질문에 정확하고 유용한 답변을 해주세요.
이전 대화 내용을 참고해서 맥락을 이해하고 답변하세요.

**지침:**
1. 제공된 문서의 정보를 우선적으로 사용해서 답변하세요
2. 이전 대화 맥락을 고려해서 자연스럽게 답변하세요
3. 문서에 없고 대화에도 없는 내용은 추측하지 마세요  
4. 답변을 할 수 없다면 솔직히 말해주세요
5. 한국어로 자연스럽게 답변해주세요

**제공된 문서:**
{context}

{history_text}

**질문:** {question}

**답변:**"""
        
        try:
            logger.info(f"히스토리 포함 LLM에 질문 전송: {question[:50]}... (세션: {session_id})")
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
                
                # 대화를 히스토리에 저장
                history.add_user_message(question)
                history.add_ai_message(answer)
                
                # 대화를 ChromaDB에 저장 (5의 배수일 때만)
                conversation_id = len(history.messages) // 2  # 대화 쌍의 번호
                if self.should_store_conversation(conversation_id):
                    self.store_conversation(session_id, question, answer, conversation_id)
                    logger.info(f"대화 {conversation_id}번째: ChromaDB 저장됨")
                
                logger.info("히스토리 포함 LLM 답변 생성 완료")
                return answer
            else:
                logger.error(f"Ollama API 오류: {response.status_code}")
                return f"죄송합니다. AI 서비스 오류가 발생했습니다. (상태코드: {response.status_code})"
            
        except Exception as e:
            logger.error(f"히스토리 포함 LLM 답변 생성 실패: {e}")
            # 히스토리 기능 실패 시 기본 답변 생성으로 폴백
            return self.generate_answer(question, context_docs)
    
    def generate_answer_stream(self, question: str, context_docs: List[Document]):
        """스트리밍 방식으로 답변 생성"""
        if not context_docs:
            yield "죄송합니다. 관련 문서를 찾을 수 없습니다. 더 많은 문서를 업로드해주세요."
            return
        
        # 컨텍스트 문서들 결합
        context = "\n\n".join([
            f"[문서 {i+1}]\n{doc.page_content[:1000]}{'...' if len(doc.page_content) > 1000 else ''}"
            for i, doc in enumerate(context_docs)
        ])
        
        # Ollama 서버 확인
        if not self.available:
            yield "죄송합니다. AI 서비스가 사용할 수 없습니다. Ollama가 실행 중인지 확인해주세요."
            return
        
        try:
            # 프롬프트 생성
            prompt = self._create_prompt(context, question)
            
            # Ollama API 직접 호출
            logger.info(f"LLM 스트리밍 질문 전송: {question[:50]}...")
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.1
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60,
                stream=True
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if 'response' in chunk:
                                yield chunk['response']
                            if chunk.get('done', False):
                                logger.info("LLM 스트리밍 답변 생성 완료")
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                logger.error(f"Ollama API 오류: {response.status_code}")
                yield f"죄송합니다. AI 서비스 오류가 발생했습니다. (상태코드: {response.status_code})"
            
        except Exception as e:
            logger.error(f"LLM 스트리밍 답변 생성 실패: {e}")
            yield f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    def generate_answer_stream_with_history(self, question: str, context_docs: List[Document], session_id: str = "default"):
        """히스토리를 포함한 스트리밍 방식으로 답변 생성"""
        if not context_docs:
            yield "죄송합니다. 관련 문서를 찾을 수 없습니다. 더 많은 문서를 업로드해주세요."
            return
        
        # 히스토리 기능이 사용 불가능한 경우 기본 스트리밍 사용
        if not self.history_available:
            for chunk in self.generate_answer_stream(question, context_docs):
                yield chunk
            return
        
        # 컨텍스트 문서들 결합
        context = "\n\n".join([
            f"[문서 {i+1}]\n{doc.page_content[:1000]}{'...' if len(doc.page_content) > 1000 else ''}"
            for i, doc in enumerate(context_docs)
        ])
        
        # 히스토리 가져오기 (RAG 관련 대화 + 최근 대화)
        history = self.get_session_history(session_id)
        history_text = self._format_history_with_rag(session_id, question, history)
        
        prompt = f"""당신은 도움이 되는 AI 어시스턴트입니다. 
제공된 문서의 내용을 바탕으로 사용자의 질문에 정확하고 유용한 답변을 해주세요.
이전 대화 내용을 참고해서 맥락을 이해하고 답변하세요.

**지침:**
1. 제공된 문서의 정보를 우선적으로 사용해서 답변하세요
2. 이전 대화 맥락을 고려해서 자연스럽게 답변하세요
3. 문서에 없고 대화에도 없는 내용은 추측하지 마세요  
4. 답변을 할 수 없다면 솔직히 말해주세요
5. 한국어로 자연스럽게 답변해주세요

**제공된 문서:**
{context}

{history_text}

**질문:** {question}

**답변:**"""
        
        # Ollama 서버 확인
        if not self.available:
            yield "죄송합니다. AI 서비스가 사용할 수 없습니다. Ollama가 실행 중인지 확인해주세요."
            return
        
        try:
            # Ollama API 직접 호출
            logger.info(f"히스토리 포함 LLM 스트리밍 질문 전송: {question[:50]}... (세션: {session_id})")
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.1
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=60,
                stream=True
            )
            
            full_response = ""
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line.decode('utf-8'))
                            if 'response' in chunk:
                                chunk_text = chunk['response']
                                full_response += chunk_text
                                yield chunk_text
                            if chunk.get('done', False):
                                # 대화를 히스토리에 저장
                                history.add_user_message(question)
                                history.add_ai_message(full_response)
                                
                                # 대화를 ChromaDB에 저장 (5의 배수일 때만)
                                conversation_id = len(history.messages) // 2  # 대화 쌍의 번호
                                if self.should_store_conversation(conversation_id):
                                    self.store_conversation(session_id, question, full_response, conversation_id)
                                    logger.info(f"대화 {conversation_id}번째: ChromaDB 저장됨")
                                
                                logger.info("히스토리 포함 LLM 스트리밍 답변 생성 완료")
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                logger.error(f"Ollama API 오류: {response.status_code}")
                yield f"죄송합니다. AI 서비스 오류가 발생했습니다. (상태코드: {response.status_code})"
            
        except Exception as e:
            logger.error(f"히스토리 포함 LLM 스트리밍 답변 생성 실패: {e}")
            yield f"죄송합니다. 답변 생성 중 오류가 발생했습니다: {str(e)}"

# 전역 AI 서비스 인스턴스는 지연 생성
ai_service = None

def get_ai_service():
    global ai_service
    if ai_service is None:
        ai_service = AIService()
    return ai_service