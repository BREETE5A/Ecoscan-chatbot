
import os
from typing import List

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


class EcoScanRAGChatbot:
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.chroma_path = os.getenv("CHROMA_PATH", "./chroma_ecoscan")
        self.collection_name = os.getenv("COLLECTION_NAME", "ecoscan_knowledge")
        self.embedding_model_name = os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.llm_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY est manquante dans le fichier .env")

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            encode_kwargs={"normalize_embeddings": True}
        )

        self.vector_db = Chroma(
            persist_directory=self.chroma_path,
            collection_name=self.collection_name,
            embedding_function=self.embedding_model
        )

        self.retriever = self.vector_db.as_retriever(
            search_kwargs={"k": 4}
        )

        self.llm = ChatGoogleGenerativeAI(
            model=self.llm_model,
            temperature=0.3
        )

        self.prompt = ChatPromptTemplate.from_template("""
Tu es EcoBot, l’assistant intelligent de l’application mobile EcoScan Maroc.

MISSION :
Aider les utilisateurs marocains à trier, recycler et gérer leurs déchets correctement.

RÈGLES IMPORTANTES :
1. Répondre de manière courte, claire et pratique.
2. Réponse idéale : maximum 3 à 6 lignes.
3. Ne jamais inventer une information.
4. Utiliser uniquement les informations du contexte fourni.
5. Si une information est incertaine ou absente, dire qu’il faut vérifier auprès de la commune ou de la source officielle.
6. Comprendre le français, la darija marocaine et le franco-arabe.
7. Toujours répondre en français.
8. Adapter les réponses au contexte marocain.
9. Donner des conseils concrets et utiles.
10. Utiliser des listes courtes si nécessaire.

STYLE :
- Ton bienveillant et pédagogique.
- Pas de longs paragraphes.
- Pas de jargon technique inutile.
- Réponse directe dès la première phrase.

CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE ECOBOT :
""")

        self.chain = (
            {
                "context": self.retriever | self._format_documents,
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_documents(self, docs: List[Document]) -> str:
        if not docs:
            return "Aucun contexte trouvé dans la base de connaissance."

        return "\n\n".join(doc.page_content for doc in docs)

    def ask(self, question: str) -> str:
        if not question or not question.strip():
            return "Veuillez écrire une question valide."

        try:
            return self.chain.invoke(question.strip())

        except Exception:
            return (
                "Désolé, je n’ai pas pu générer une réponse pour le moment. "
                "Veuillez réessayer."
            )


chatbot = EcoScanRAGChatbot()