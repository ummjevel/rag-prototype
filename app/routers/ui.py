from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from app.config import settings

router = APIRouter(tags=["ui"])

@router.get("/", response_class=HTMLResponse)
async def get_chatbot_ui():
    """챗봇 웹 UI를 반환합니다."""
    # 허용된 파일 확장자를 설정에서 가져오기
    allowed_extensions = ",".join(settings.allowed_extensions)
    
    html_content = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI RAG Chatbot</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f5f5f5;
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            
            .container {
                background: white;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                width: 90%;
                max-width: 800px;
                height: 80vh;
                display: flex;
                flex-direction: column;
            }
            
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px 10px 0 0;
                text-align: center;
            }
            
            .chat-area {
                flex: 1;
                padding: 20px;
                overflow-y: auto;
                display: flex;
                flex-direction: column;
                gap: 15px;
            }
            
            .message {
                max-width: 70%;
                padding: 12px 16px;
                border-radius: 18px;
                word-wrap: break-word;
            }
            
            .user-message {
                background: #007bff;
                color: white;
                align-self: flex-end;
                margin-left: auto;
            }
            
            .bot-message {
                background: #e9ecef;
                color: #333;
                align-self: flex-start;
            }
            
            .sources {
                font-size: 0.8em;
                color: #666;
                margin-top: 8px;
                padding-top: 8px;
                border-top: 1px solid #ddd;
            }
            
            .input-area {
                padding: 20px;
                border-top: 1px solid #e0e0e0;
                display: flex;
                gap: 10px;
            }
            
            .input-field {
                flex: 1;
                padding: 12px 16px;
                border: 1px solid #ddd;
                border-radius: 25px;
                outline: none;
                font-size: 14px;
            }
            
            .input-field:focus {
                border-color: #007bff;
            }
            
            .send-button {
                background: #007bff;
                color: white;
                border: none;
                border-radius: 50%;
                width: 45px;
                height: 45px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: background-color 0.2s;
            }
            
            .send-button:hover {
                background: #0056b3;
            }
            
            .send-button:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            
            .loading {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #007bff;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .upload-area {
                padding: 10px 20px;
                border-top: 1px solid #e0e0e0;
                display: flex;
                gap: 10px;
                align-items: center;
                background: #f8f9fa;
            }
            
            .file-input {
                display: none;
            }
            
            .upload-button {
                background: #28a745;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 12px;
            }
            
            .upload-button:hover {
                background: #218838;
            }
            
            .typing-indicator {
                display: flex;
                align-items: center;
                color: #666;
                font-style: italic;
            }
            
            .typing-indicator::after {
                content: '...';
                animation: dots 1.5s infinite;
            }
            
            @keyframes dots {
                0%, 20% { content: '.'; }
                40% { content: '..'; }
                60%, 100% { content: '...'; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 AI RAG Chatbot</h1>
                <p>문서를 업로드하고 질문해보세요!</p>
                <button onclick="newSession()" style="background: rgba(255,255,255,0.2); border: 1px solid rgba(255,255,255,0.3); color: white; padding: 5px 10px; border-radius: 15px; cursor: pointer; font-size: 12px; margin-top: 10px;">
                    🔄 새 대화 시작
                </button>
            </div>
            
            <div class="chat-area" id="chatArea">
                <div class="message bot-message">
                    안녕하세요! 저는 RAG 기반 AI 챗봇입니다. 문서를 업로드하거나 질문을 해보세요.
                </div>
            </div>
            
            <div class="upload-area">
                <input type="file" id="fileInput" class="file-input" accept="ALLOWED_EXTENSIONS_PLACEHOLDER" multiple>
                <button class="upload-button" onclick="document.getElementById('fileInput').click()">
                    📁 문서 업로드
                </button>
                <span id="uploadStatus"></span>
            </div>
            
            <div class="input-area">
                <input 
                    type="text" 
                    id="messageInput" 
                    class="input-field" 
                    placeholder="질문을 입력하세요..."
                    onkeypress="handleKeyPress(event)"
                >
                <button class="send-button" id="sendButton" onclick="sendMessage()">
                    ➤
                </button>
            </div>
        </div>
        
        <script>
            const chatArea = document.getElementById('chatArea');
            const messageInput = document.getElementById('messageInput');
            const sendButton = document.getElementById('sendButton');
            const fileInput = document.getElementById('fileInput');
            const uploadStatus = document.getElementById('uploadStatus');
            
            // 세션 ID 관리
            function getSessionId() {
                let sessionId = localStorage.getItem('chatSessionId');
                if (!sessionId) {
                    sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substring(2);
                    localStorage.setItem('chatSessionId', sessionId);
                }
                return sessionId;
            }
            
            function newSession() {
                localStorage.removeItem('chatSessionId');
                chatArea.innerHTML = `
                    <div class="message bot-message">
                        안녕하세요! 새로운 대화를 시작합니다. 문서를 업로드하거나 질문을 해보세요.
                    </div>
                `;
            }
            
            function addMessage(content, isUser = false, sources = []) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
                
                let messageHTML = content;
                if (!isUser && sources.length > 0) {
                    messageHTML += `<div class="sources">출처: ${sources.join(', ')}</div>`;
                }
                
                messageDiv.innerHTML = messageHTML;
                chatArea.appendChild(messageDiv);
                chatArea.scrollTop = chatArea.scrollHeight;
            }
            
            function showLoading() {
                const loadingDiv = document.createElement('div');
                loadingDiv.className = 'message bot-message';
                loadingDiv.innerHTML = '<div class="loading"></div>';
                loadingDiv.id = 'loading-message';
                chatArea.appendChild(loadingDiv);
                chatArea.scrollTop = chatArea.scrollHeight;
            }
            
            function hideLoading() {
                const loadingMessage = document.getElementById('loading-message');
                if (loadingMessage) {
                    loadingMessage.remove();
                }
            }
            
            async function sendMessage() {
                const message = messageInput.value.trim();
                if (!message) return;
                
                addMessage(message, true);
                messageInput.value = '';
                sendButton.disabled = true;
                
                // 스트리밍 답변을 위한 빈 메시지 추가
                const streamingMessageDiv = document.createElement('div');
                streamingMessageDiv.className = 'message bot-message';
                streamingMessageDiv.innerHTML = '<div class="typing-indicator">답변 생성 중...</div>';
                streamingMessageDiv.id = 'streaming-message';
                chatArea.appendChild(streamingMessageDiv);
                chatArea.scrollTop = chatArea.scrollHeight;
                
                try {
                    const response = await fetch('/chat/stream', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            question: message,
                            max_results: 3,
                            session_id: getSessionId()
                        })
                    });
                    
                    if (response.ok) {
                        const reader = response.body.getReader();
                        const decoder = new TextDecoder();
                        let streamedContent = '';
                        
                        // 타이핑 인디케이터 제거
                        const typingIndicator = streamingMessageDiv.querySelector('.typing-indicator');
                        if (typingIndicator) {
                            typingIndicator.remove();
                        }
                        
                        while (true) {
                            const { done, value } = await reader.read();
                            if (done) break;
                            
                            const chunk = decoder.decode(value, { stream: true });
                            const lines = chunk.split('\\n');
                            
                            for (const line of lines) {
                                if (line.startsWith('data: ')) {
                                    const data = line.slice(6);
                                    if (data === '[DONE]') {
                                        break;
                                    }
                                    streamedContent += data;
                                    streamingMessageDiv.innerHTML = streamedContent.replace(/\\n/g, '<br>');
                                    chatArea.scrollTop = chatArea.scrollHeight;
                                }
                            }
                        }
                    } else {
                        const data = await response.json();
                        streamingMessageDiv.innerHTML = `오류: ${data.detail}`;
                    }
                } catch (error) {
                    streamingMessageDiv.innerHTML = '네트워크 오류가 발생했습니다.';
                }
                
                sendButton.disabled = false;
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }
            
            fileInput.addEventListener('change', async function(event) {
                const files = event.target.files;
                if (files.length === 0) return;
                
                uploadStatus.textContent = '업로드 중...';
                
                for (let file of files) {
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    try {
                        const response = await fetch('/documents/upload', {
                            method: 'POST',
                            body: formData
                        });
                        
                        const data = await response.json();
                        if (response.ok) {
                            addMessage(`문서 "${file.name}"이 성공적으로 업로드되었습니다.`, false);
                        } else {
                            addMessage(`문서 업로드 실패: ${data.detail}`, false);
                        }
                    } catch (error) {
                        addMessage(`문서 업로드 오류: ${error.message}`, false);
                    }
                }
                
                uploadStatus.textContent = '';
                fileInput.value = '';
            });
        </script>
    </body>
    </html>
    """
    # 플레이스홀더를 실제 값으로 교체
    html_content = html_content.replace("ALLOWED_EXTENSIONS_PLACEHOLDER", allowed_extensions)
    return HTMLResponse(content=html_content)