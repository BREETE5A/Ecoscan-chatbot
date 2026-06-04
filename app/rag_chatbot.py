
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

RÈGLES SPÉCIALES CONTACTS :
- Si l'utilisateur demande un numéro de téléphone ou un contact d'opérateur de recyclage,
  donner l'information depuis le contexte (nom, téléphone, ville, spécialité).
- Toujours ajouter : "Veuillez vérifier ce numéro avant d'appeler, il peut avoir changé."
- Les numéros fournis sont indicatifs — inviter l'utilisateur à confirmer sur le site de l'opérateur.

RÈGLES SPÉCIALES AMENDES ET LOIS :
- Les montants des amendes et références légales (lois, décrets) sont fournis à titre indicatif.
- En cas de doute, recommander de consulter le Bulletin Officiel du Royaume du Maroc ou un juriste.

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

        self.chain = (
            {
                # retrieval_q = termes français purs (pas darija) → meilleure recherche vectorielle
                "context": (lambda x: x["retrieval_q"]) | self.retriever | self._format_documents,
                # question originale → Gemini répond dans la langue de l'utilisateur
                "question": lambda x: x["question"],
                "history": lambda x: x["history"],
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    # ── Mapping darija → termes de recherche enrichis ─────────────────────────
    _DARIJA_MAP = {
        # Plastique
        "قوارير": "bouteilles plastique PET",
        "بلاسطيك": "plastique recyclable",
        "كيسان": "sacs plastique interdits Loi 77-15",
        "كيسة": "sac plastique",
        "كيس": "sac plastique",
        "بلاستيكة": "emballage plastique",
        # Verre
        "قاروة": "bouteille verre colonne verte",
        "قواري زجاج": "bouteilles verre recyclage",
        # Métal
        "علبة": "canette métal bac jaune ECOVAL",
        "علب": "canettes métal aluminium recyclage",
        "قصدير": "métal recyclable bac jaune",
        "كانيطة": "canette aluminium",
        "صفيح": "métal ferraille recyclage",
        # Papier / Carton
        "كارطون": "carton papier ferracha recyclage",
        "كارتون": "carton recyclable ferracha",
        "ورقة": "papier recyclage ferracha",
        "جرنان": "journaux magazines recyclage",
        # Organique
        "قشر": "épluchures fruits légumes composteur",
        "خضرة": "déchets organiques composteur",
        "بقايا الكلة": "restes alimentaires organiques composteur",
        # Électronique
        "تيليفون": "téléphone smartphone DEEE Eco Zalagh",
        "تلفون": "téléphone portable DEEE",
        "كومبيوتر": "ordinateur DEEE recyclage électronique",
        "شاشة": "écran TV DEEE recyclage",
        "لابتوب": "laptop ordinateur DEEE",
        "تابليت": "tablette iPad DEEE",
        "جهاز خايب": "appareil électronique usagé DEEE",
        "موبايل خايب": "smartphone usagé DEEE Eco Zalagh",
        # Textile
        "حوايج": "vêtements textile borne collecte",
        "ليباس": "vêtements habits textile recyclage",
        "ثياب": "vêtements textile don association",
        "صبادر": "chaussures textile recyclage",
        "جلابة": "djellaba textile recyclage sadaqa",
        # Dangereux
        "دوا خايب": "médicaments périmés pharmacie agréée",
        "دوا": "médicaments dangereux pharmacie",
        "دواء": "médicaments périmés pharmacie",
        "صبغة": "peinture vernis déchetterie dangereux",
        "زيت المحرك": "huile moteur usagée garage agréé",
        "حقنة": "seringues DASRI pharmacie",
        # Piles
        "بيلة": "pile batterie collecteur Marjane",
        "بيلات": "piles batteries recyclage collecteur",
        "باطاري": "batterie Li-ion DEEE collecteur",
        # Mots génériques
        "زبل": "déchets tri recyclage Maroc",
        "نفاية": "déchets recyclage tri sélectif",
        "قمامة": "ordures déchets tri recyclage",
        "وين نحط": "où déposer recycler",
        "فين نحط": "où déposer recycler",
        "واش نرمي": "où jeter recycler",
        "وين نرمي": "où jeter déchets recyclage",
        "كيفاش نريسيكلي": "comment recycler trier déchets",
        "ريسيكلاج": "recyclage tri déchets Maroc",
        "فرز": "tri sélectif recyclage",
        # Navigation app
        "كلمة السر": "mot de passe changer paramètres profil",
        "باسوورد": "mot de passe changer paramètres",
        "بدل الاسم": "modifier profil nom photo",
        "بدل صورة": "modifier photo profil éditer",
        "شحال ريسيكليت": "statistiques recyclage CO2 économisé",
        "نقط ديالي": "récompenses points badges statistiques",
        "وين نلقى": "carte map points collecte",
        "شالنج": "défis challenges écologiques",
        "بيع": "marketplace vendre objets recyclés",
        "شرا": "marketplace acheter objets recyclés",
        # Salutations darija
        "آش حال": "bonjour assistant aide",
        "لاباس": "bonjour salut assistant",
        "كيداير": "bonjour salut assistant",
        "سلام عليكم": "bonjour salut assistant",
    }

    def _build_retrieval_query(self, question: str) -> str:
        """
        Construit la requête UNIQUEMENT pour ChromaDB.
        Si la question contient de la darija → utilise seulement les termes français mappés.
        Si pas de darija détectée → utilise la question originale.
        Séparer retrieval_q de la question originale permet au modèle de répondre
        dans la langue de l'utilisateur tout en trouvant les bons documents.
        """
        extra_terms = []
        for darija_term, french_term in self._DARIJA_MAP.items():
            if darija_term in question:
                extra_terms.append(french_term)

        if extra_terms:
            # Seulement les termes français → embedding de meilleure qualité
            return ' '.join(extra_terms)
        return question

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

        original_q  = question.strip()
        retrieval_q = self._build_retrieval_query(original_q)

        try:
            return self.chain.invoke({
                "question":   original_q,   # question originale → Gemini répond dans la bonne langue
                "retrieval_q": retrieval_q,  # termes français purs → ChromaDB trouve les bons docs
                "history":    history_text,
            })

        except Exception as e:
            import traceback
            traceback.print_exc()
            return (
                "Désolé, je n'ai pas pu générer une réponse pour le moment. "
                "Veuillez réessayer."
            )


chatbot = EcoScanRAGChatbot()