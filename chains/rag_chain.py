import logging
from config import GEMINI_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI
from retrieval.retriever import get_retriever
from prompts.rag_prompts import RAG_PROMPT
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

# Initialize Google Gemini LLM with API validation
try:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not defined in the environment variables.")
    print("[CHAIN] Initializing Gemini LLM (gemini-2.5-flash)...")
    logger.info("Initializing Google Gemini API connection.")
    llm_model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.0,  # Factual temperature for RAG
        google_api_key=GEMINI_API_KEY
    )
except Exception as e:
    print(f"[CHAIN] [ERROR] Failed to initialize Gemini LLM: {e}")
    logger.critical(f"Critical error initializing LLM: {e}", exc_info=True)
    # Fallback to prevent app crashes, yields error
    class FallbackLLM:
        def stream(self, prompt, **kwargs):
            yield f"[LLM ERROR: Gemini API key or connection failed. Details: {e}]"
        def invoke(self, prompt, **kwargs):
            from langchain_core.outputs import ChatResult, ChatGeneration, AIMessage
            return AIMessage(content=f"[LLM ERROR: Gemini API key or connection failed. Details: {e}]")
    llm_model = FallbackLLM()

def retrieve_documents_dynamic(input_data):
    """Retrieves document chunks using the hybrid search, dynamically passing filters."""
    if isinstance(input_data, dict):
        question = input_data.get("question")
        filter_dict = input_data.get("filter", None)
    else:
        question = input_data
        filter_dict = None

    print(f"[CHAIN] Query: '{question}' | Filters: {filter_dict}")
    logger.info(f"Retrieving context for query: {question} with filters: {filter_dict}")
    
    try:
        retriever = get_retriever(filter_dict=filter_dict)
        docs = retriever.invoke(question)
        print(f"[CHAIN] Retrieved {len(docs)} context chunks.")
        
        # Log and print metadata details of retrieved docs
        for idx, doc in enumerate(docs):
            p = doc.metadata.get("page_number", "Unknown")
            c = doc.metadata.get("clause", "General")
            pdf = doc.metadata.get("pdf_id", "Unknown")
            print(f"  - Chunk {idx+1}: Document={pdf} | Page={p} | Clause='{c}' | Length={len(doc.page_content)} chars")
            
        return docs
    except Exception as e:
        print(f"[CHAIN] [ERROR] Context retrieval failed: {e}")
        logger.error(f"Retrieval error in chain: {e}", exc_info=True)
        return []

def format_docs_with_metadata(docs):
    """Formats retrieved document pages and clauses to construct context block."""
    if not docs:
        return "No relevant context found."
        
    formatted_chunks = []
    for doc in docs:
        page = doc.metadata.get("page_number", "Unknown")
        clause = doc.metadata.get("clause", "General")
        pdf_id = doc.metadata.get("pdf_id", "Unknown")
        formatted_chunks.append(
            f"--- Context Block (Source: {pdf_id}.pdf, Page: {page}, Clause: {clause}) ---\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(formatted_chunks)

def get_question(input_data):
    """Extracts query string from dictionary or text inputs."""
    if isinstance(input_data, dict):
        return input_data.get("question")
    return input_data

def build_rag_chain():
    """Assembles the full dynamic LCEL RAG chain."""
    print("[CHAIN] Constructing LCEL pipeline components...")
    
    # 1. Parallelize retrieving context and matching question
    pipeline_inputs = RunnableParallel({
        "context": RunnableLambda(retrieve_documents_dynamic) | RunnableLambda(format_docs_with_metadata),
        "question": RunnableLambda(get_question)
    })
    
    # 2. Assemble RAG pipeline
    chain = pipeline_inputs | RAG_PROMPT | llm_model | StrOutputParser()
    print("[CHAIN] LCEL RAG pipeline assembled successfully.")
    return chain

        
       

