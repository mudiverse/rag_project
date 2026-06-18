import logging
from langchain_chroma import Chroma
from embeddings.embedding_model import get_embedding_model
from config import CHROMA_PATH

logger = logging.getLogger(__name__)

def get_vectorstore():
    print("[VECTORSTORE] Connecting to Chroma DB...")
    logger.info(f"Connecting to Chroma DB at path: {CHROMA_PATH}")
    try:
        embedding_model = get_embedding_model()
        store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_model
        )
        print("[VECTORSTORE] Connected to Chroma DB successfully.")
        logger.info("Chroma DB successfully initialized.")
        return store
    except Exception as e:
        print(f"[VECTORSTORE] [ERROR] Failed to initialize Chroma DB: {e}")
        logger.error(f"Error initializing Chroma DB: {e}", exc_info=True)
        raise e

def add_docs(chunks):
    print(f"[VECTORSTORE] Adding {len(chunks)} chunks to database...")
    logger.info(f"Adding {len(chunks)} documents to vector store.")
    try:
        db = get_vectorstore()
        db.add_documents(chunks)
        print("[VECTORSTORE] Chunks successfully saved and persisted to Chroma DB.")
        logger.info("Chunks successfully persisted.")
    except Exception as e:
        print(f"[VECTORSTORE] [ERROR] Failed to save chunks to database: {e}")
        logger.error(f"Error saving chunks: {e}", exc_info=True)
        raise e


