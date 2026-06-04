FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# PyTorch CPU-only (beaucoup plus léger que la version GPU)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-dotenv \
    langchain \
    langchain-community \
    langchain-core \
    langchain-chroma \
    langchain-google-genai \
    langchain-huggingface \
    chromadb \
    sentence-transformers \
    pydantic

# Pré-télécharger le modèle multilingue (supporte FR, EN, AR, ES, Darija)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

COPY app/ ./app/
COPY ingest_guide.py .

ENV CHROMA_PATH=./chroma_ecoscan
ENV COLLECTION_NAME=ecoscan_knowledge
ENV EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ENV GEMINI_MODEL=gemini-flash-lite-latest

# Construire le ChromaDB avec toutes les données du guide au moment du build
RUN python ingest_guide.py

EXPOSE 10000

COPY start.sh .
RUN chmod +x start.sh

CMD ["sh", "start.sh"]
