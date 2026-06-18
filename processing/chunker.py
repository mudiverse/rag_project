import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from embeddings.embedding_model import get_embedding_model

logger = logging.getLogger(__name__)

def recurcive_chunker(documents):
    print("[CHUNKER] Running Recursive Character Text Splitter...")
    logger.info(f"Running recursive chunker on {len(documents)} pages...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = splitter.split_documents(documents)
    print(f"[CHUNKER] Recursive chunker generated {len(chunks)} chunks.")
    return chunks

def semantic_chunker(documents):
    print("[CHUNKER] Initializing Semantic Chunker...")
    logger.info("Initializing SemanticChunker with embedding model...")
    try:
        embeddings = get_embedding_model()
        sem_splitter = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile"
        )
        print("[CHUNKER] Splitting documents semantically...")
        logger.info(f"Running semantic chunker on {len(documents)} pages...")
        chunks = sem_splitter.split_documents(documents)
        print(f"[CHUNKER] Semantic chunker completed. Generated {len(chunks)} chunks.")
        logger.info(f"Semantic chunking complete. Generated {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        print(f"[CHUNKER] [ERROR] Semantic chunker failed: {e}. Falling back to Recursive Chunker.")
        logger.error(f"Semantic chunker error: {e}. Falling back to recursive chunker.", exc_info=True)
        return recurcive_chunker(documents)

