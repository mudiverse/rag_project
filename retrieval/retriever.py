import logging
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from vectorstore.chroma_store import get_vectorstore

logger = logging.getLogger(__name__)

# Cache for global hybrid retriever to avoid rebuilding BM25 on every query
_cached_hybrid_retriever = None
_cached_doc_count = -1

def get_retriever(filter_dict=None):
    """
    Creates and returns a Hybrid Retriever (Vector Search + BM25 Lexical Search).
    Supports dynamic metadata filtering.
    """
    global _cached_hybrid_retriever, _cached_doc_count
    
    print(f"\n[RETRIEVER] Building retriever (Filters: {filter_dict})...")
    logger.info(f"Building retriever with filters: {filter_dict}")
    
    try:
        db = get_vectorstore()
        
        # Fetch all documents to construct the BM25 index in memory
        stored_data = db.get()
        doc_ids = stored_data.get("ids", [])
        current_count = len(doc_ids)
        
        logger.info(f"Vector store contains {current_count} documents.")
        
        # Configure vector search kwargs
        vector_search_kwargs = {"k": 5}
        if filter_dict:
            vector_search_kwargs["filter"] = filter_dict
            
        vector_retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs=vector_search_kwargs
        )
        
        if current_count == 0:
            print("[RETRIEVER] [WARNING] Vector store is empty. Returning basic vector retriever.")
            logger.warning("Vector store is empty. Cannot build hybrid search; returning basic retriever.")
            return vector_retriever
            
        # Reconstruct Document objects from Chroma database data
        documents = []
        for text, meta in zip(stored_data.get("documents", []), stored_data.get("metadatas", [])):
            documents.append(Document(page_content=text, metadata=meta))
            
        # If dynamic filters are active, filter BM25 documents as well
        if filter_dict:
            filtered_docs = []
            for doc in documents:
                matches_filter = True
                for key, val in filter_dict.items():
                    if doc.metadata.get(key) != val:
                        matches_filter = False
                        break
                if matches_filter:
                    filtered_docs.append(doc)
            documents = filtered_docs
            print(f"[RETRIEVER] Filtered documents for BM25 index. Match count: {len(documents)} / {current_count}")
            logger.info(f"Filtered docs for BM25: {len(documents)} active chunks.")
            
            # Since filters are active, we must build a dynamic EnsembleRetriever and not cache it globally
            if len(documents) > 0:
                bm25_k = min(5, len(documents))
                bm25_retriever = BM25Retriever.from_documents(documents)
                bm25_retriever.k = bm25_k
                
                print(f"[RETRIEVER] Constructing dynamic Hybrid Retriever (Vector + BM25)...")
                hybrid_retriever = EnsembleRetriever(
                    retrievers=[bm25_retriever, vector_retriever],
                    weights=[0.4, 0.6]  # 40% lexical priority, 60% semantic priority
                )
                return hybrid_retriever
            else:
                print("[RETRIEVER] [WARNING] No documents matched the filter criteria for BM25. Returning vector retriever.")
                return vector_retriever
                
        # If no filters are active, check cache
        if _cached_hybrid_retriever is not None and current_count == _cached_doc_count:
            print("[RETRIEVER] Using cached Hybrid Retriever.")
            logger.info("Using cached Hybrid Retriever.")
            return _cached_hybrid_retriever
            
        # Rebuild and cache hybrid retriever
        print(f"[RETRIEVER] Rebuilding Hybrid Retriever (Vector + BM25) for {current_count} documents...")
        bm25_retriever = BM25Retriever.from_documents(documents)
        bm25_retriever.k = min(5, len(documents))
        
        hybrid_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6]
        )
        
        _cached_hybrid_retriever = hybrid_retriever
        _cached_doc_count = current_count
        print("[RETRIEVER] Hybrid Retriever initialized and cached successfully.")
        logger.info("Hybrid Retriever successfully initialized and cached.")
        return hybrid_retriever
        
    except Exception as e:
        print(f"[RETRIEVER] [ERROR] Failed to build hybrid retriever: {e}. Falling back to basic vector retriever.")
        logger.error(f"Failed to build hybrid retriever: {e}", exc_info=True)
        try:
            db = get_vectorstore()
            return db.as_retriever(search_kwargs={"k": 5})
        except Exception as err:
            logger.critical(f"Critical retriever fallback error: {err}")
            raise err