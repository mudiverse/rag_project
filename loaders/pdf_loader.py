import os
import logging
from langchain_community.document_loaders import PyMuPDFLoader

logger = logging.getLogger(__name__)

def load_pdf(pdf_path):
    """Loads a PDF file and returns a list of Document objects (one per page)."""
    print(f"[LOADER] Loading PDF from: {pdf_path}...")
    logger.info(f"Loading PDF file: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        error_msg = f"PDF file not found at: {pdf_path}"
        print(f"[LOADER] [ERROR] {error_msg}")
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    try:
        pdf_loader = PyMuPDFLoader(pdf_path)
        docs = pdf_loader.load()
        print(f"[LOADER] Successfully loaded {len(docs)} pages.")
        logger.info(f"Loaded {len(docs)} pages from {pdf_path}.")
        return docs
    except Exception as e:
        print(f"[LOADER] [ERROR] Failed to read PDF {pdf_path}: {e}")
        logger.error(f"Error reading PDF {pdf_path}: {e}", exc_info=True)
        raise e

