
import os
from typing import List, Optional, Dict

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
            search_kwargs={"k": 6}
        )

        self.llm = ChatGoogleGenerativeAI(
            model=self.llm_model,
            temperature=0.3
        )

        self.prompt = ChatPromptTemplate.from_template("""
Tu es EcoBot, l'assistant intelligent de l'application mobile EcoScan Maroc.
HISTORIQUE DE LA CONVERSATION (pour contexte) :
{history}

MISSION :
Tu as DEUX rôles complémentaires :
1. ASSISTANT TRI & RECYCLAGE : aider les utilisateurs à trier, recycler et gérer leurs déchets.
2. ASSISTANT DE L'APPLICATION : guider les utilisateurs dans l'app (où trouver une fonctionnalité,
   comment changer le mot de passe, où scanner, etc.) et fournir des contacts d'opérateurs recyclage.

RÈGLES IMPORTANTES :
1. Répondre de manière courte, claire et pratique.
2. Réponse idéale : maximum 3 à 6 lignes.
3. Ne jamais inventer une information.
4. Utiliser uniquement les informations du contexte fourni.
5. Si une information est incertaine ou absente, dire qu'il faut vérifier auprès de la commune.
6. Comprendre 6 langues : français, anglais, arabe (MSA), espagnol, darija marocaine, franco-arabe.
7. LANGUE DE RÉPONSE : détecter automatiquement la langue de la question et répondre dans cette même langue.
   - Question en français → répondre en français
   - Question en anglais → répondre en anglais
   - Question en arabe (MSA) → répondre en arabe
   - Question en espagnol → répondre en espagnol
   - Question en darija marocaine → répondre en arabe
   - Question en franco-arabe → répondre en arabe
8. Adapter les réponses au contexte marocain.
9. Donner des conseils concrets et utiles.
10. Utiliser des listes courtes si nécessaire.

RÈGLES SPÉCIALES NAVIGATION APP :
- Si l'utilisateur demande où faire quelque chose dans l'app (changer mdp, voir stats, scanner...),
  donner le chemin exact depuis le contexte (ex: "Profil → Paramètres → Changer le mot de passe").
- Toujours indiquer le chemin de navigation de façon simple et claire.

RÈGLES SPÉCIALES CONTACTS ET NUMÉROS :
- Ne JAMAIS inventer un numéro de téléphone, une adresse email ou un site web.
- Utiliser UNIQUEMENT les numéros présents dans le contexte fourni.
- Si un numéro n'est pas dans le contexte, dire explicitement : "Je n'ai pas cette information dans ma base de données."
- Pour les numéros d'urgence au Maroc confirmés (sources gouvernementales) : 15 = SAMU médical, 150 = Protection Civile/Pompiers, 19 = Police nationale, 177 = Gendarmerie Royale, 112 = Urgence universelle mobile, 2511 = Enfants en danger (ONDE), 08000 8888 = Femmes victimes violence (Loi 103.13), 141 = Soutien psychologique.
- Ne JAMAIS contredire un numéro d'urgence officiel que l'utilisateur mentionne sans preuve dans le contexte.
- Toujours préciser que les numéros opérateurs doivent être vérifiés avant appel.

RÈGLES SPÉCIALES AMENDES ET LOIS :
- Les montants des amendes et références légales (lois, décrets) sont fournis à titre indicatif.
- En cas de doute, recommander de consulter le Bulletin Officiel du Royaume du Maroc ou un juriste.
- Ne JAMAIS inventer un numéro de loi ou de décret absent du contexte.

STYLE :
- Ton bienveillant et pédagogique.
- Pas de longs paragraphes.
- Pas de jargon technique inutile.
- Réponse directe dès la première phrase.

FORMAT OBLIGATOIRE :
- JAMAIS de markdown : interdit d'utiliser **, *, ##, __, ou tout autre symbole de mise en forme.
- Texte brut uniquement.
- Pour les listes, utiliser • ou - uniquement.
- Les mots importants s'écrivent normalement, sans gras ni italique.

CONTEXTE :
{context}

QUESTION :
{question}

RÉPONSE ECOBOT :
""")

        # Modèle multilingue → la question originale (FR/EN/AR/ES/Darija) est utilisée directement
        self.chain = (
            {
                "context": (lambda x: x["question"]) | self.retriever | self._format_documents,
                "question": lambda x: x["question"],
                "history": lambda x: x["history"],
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_documents(self, docs: List[Document]) -> str:
        if not docs:
            return "Aucun contexte trouvé dans la base de connaissance."

        return "\n\n".join(doc.page_content for doc in docs)

    def ask(self, question: str, history: Optional[List[Dict]] = None) -> str:
        if not question or not question.strip():
            return "Veuillez écrire une question valide."

        history_text = ""
        if history:
            for msg in history[-6:]:
                role = "Utilisateur" if msg.get("role") == "user" else "EcoBot"
                history_text += f"{role}: {msg.get('content', '')}\n"

        try:
            return self.chain.invoke({
                "question": question.strip(),
                "history":  history_text,
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return (
                "Désolé, je n'ai pas pu générer une réponse pour le moment. "
                "Veuillez réessayer."
            )


chatbot = EcoScanRAGChatbot()