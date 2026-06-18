import logging
from chains.rag_chain import build_rag_chain

logger = logging.getLogger(__name__)

print("[QUERY] Initializing Query Pipeline Chain...")
chain = build_rag_chain()

def ask_question(question):
    """
    Submits a query to the dynamic RAG pipeline.
    The input 'question' can be:
      - A simple string: "What is a hospital?"
      - A dictionary with filters: {"question": "What is a hospital?", "filter": {"pdf_id": "policy1"}}
    """
    print(f"\n[QUERY] Submitting query to RAG chain...")
    logger.info(f"Query pipeline received query: {question}")
    try:
        response = chain.stream(question)
        return response
    except Exception as e:
        print(f"[QUERY] [ERROR] Failed to execute query chain: {e}")
        logger.error(f"Error executing RAG chain: {e}", exc_info=True)
        # Yield a clean error fallback
        def error_generator():
            yield f"[QUERY PIPELINE ERROR: {e}]"
        return error_generator()