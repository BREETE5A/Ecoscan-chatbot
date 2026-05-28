from pydantic import BaseModel


# REQUETE UTILISATEUR

class ChatRequest(BaseModel):
    message: str

# REPONSE CHATBOT

class ChatResponse(BaseModel):
    response: str