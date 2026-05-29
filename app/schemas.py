from typing import List, Optional
from pydantic import BaseModel


class MessageHistory(BaseModel):
    role: str      # "user" ou "bot"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[MessageHistory]] = []


class ChatResponse(BaseModel):
    response: str
