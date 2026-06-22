import logging
from langchain_google_genai import GoogleGenAIEmbeddings
from config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

_embedding_model = None

def get_embedding_model():
    """
    Returns the Google GenAI API-based embedding model.
    Instantiated lazily to avoid memory overhead on startup.
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info("Initializing Google GenAI Embeddings (models/text-embedding-004)...")
        print("[EMBEDDINGS] Initializing Google GenAI Embeddings (models/text-embedding-004)...")
        try:
            if not GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY is not defined in the environment variables.")
            
            _embedding_model = GoogleGenAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=GEMINI_API_KEY
            )
            print("[EMBEDDINGS] Google GenAI Embeddings initialized successfully.")
            logger.info("Google GenAI Embeddings successfully initialized.")
        except Exception as e:
            print(f"[EMBEDDINGS] [ERROR] Failed to initialize Google GenAI Embeddings: {e}")
            logger.critical(f"Error initializing embeddings: {e}", exc_info=True)
            raise e
    return _embedding_model