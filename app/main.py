from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import ChatRequest, ChatResponse
from app.rag_chatbot import chatbot


# ============================================================
# CREATION APPLICATION FASTAPI
# ============================================================

app = FastAPI(
    title="EcoScan Chatbot API",
    description="API du chatbot intelligent EcoBot pour EcoScan",
    version="1.0.0"
)


# ============================================================
# CONFIGURATION CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROUTE PRINCIPALE
# ============================================================

@app.get("/")
def root():
    return {
        "message": "EcoScan Chatbot API fonctionne correctement."
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# ============================================================
# ROUTE CHATBOT
# ============================================================

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    history = [{"role": m.role, "content": m.content} for m in (request.history or [])]
    response = chatbot.ask(request.message, history=history)
    return ChatResponse(response=response)