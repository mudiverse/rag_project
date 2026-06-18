# Upgrades and Improvements Report: LangChain RAG Pipeline

This document details the upgrades implemented in the RAG pipeline. The pipeline now leverages semantic chunking, hybrid keyword + semantic search, rich metadata schemas, dynamic filtering, structured logging, and robust API error handling.

---

## 1. Semantic Chunker Implementation
We completed the `semantic_chunker` function in `processing/chunker.py`. It uses LangChain's `SemanticChunker` coupled with the local HuggingFace embeddings (`all-MiniLM-L6-v2`) to split documents based on semantic shifts between sentences rather than static character counts. It also includes a fallback to recursive chunking if semantic splitting fails.

**File:** [chunker.py](file:///c:/Users/mudit/Desktop/RAG_proj/rag_app/processing/chunker.py)
```python
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
```

---

## 2. Rich Metadata Schema Enrichment
During document ingestion, chunks are enriched with a metadata schema designed for structured document lookup (such as policies). The schema records:
- `pdf_id`: Name of the document (e.g. `policy1`).
- `page_number`: 1-indexed page where the chunk is located.
- `clause`: The section or clause title, extracted using regex and line-analysis patterns or propagated from previous chunks.
- `chunk_index`: The position index of the chunk in the document.

**File:** [ingest_pipeline.py](file:///c:/Users/mudit/Desktop/RAG_proj/rag_app/ingestion/ingest_pipeline.py)
```python
def extract_clause_from_text(text):
    """Utility to extract clause/section titles from text content."""
    section_patterns = [
        r'(?i)\b(?:section|clause|article)\s+\d+(?:\.\d+)*\b',
        r'(?i)\b(?:section|clause|article)\s+[A-Z\d]+\b',
        r'(?m)^\s*(\d+\.\d+)\s+([A-Z][A-Za-z\s]{3,40})',
    ]
    for pattern in section_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0).strip()
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    for line in lines[:3]:
        if 4 <= len(line) <= 60 and line.isupper() and not line.endswith(('.', '?', '!')):
            return line
    return "General"
```

---

## 3. Hybrid Search Retriever & Cache
We replaced the simple vector search with a Hybrid Search Retriever in `retrieval/retriever.py` that merges results from two algorithms using reciprocal rank fusion (`EnsembleRetriever`):
1. **Lexical (BM25 Search)**: For keyword matching (specific section names, policy codes).
2. **Semantic (Chroma Vector Search)**: For conceptual/synonym matching.

To optimize performance, the BM25 index is cached globally and only rebuilt when the count of documents in the vector store changes.

**File:** [retriever.py](file:///c:/Users/mudit/Desktop/RAG_proj/rag_app/retrieval/retriever.py)
```python
def get_retriever(filter_dict=None):
    """Creates a Hybrid Retriever supporting dynamic metadata filtering."""
    # Reconstructs Document objects from Chroma DB and sets up BM25 + Vector Ensemble
    # If dynamic filters are active, BM25 indexes are dynamically rebuilt using filtered docs.
    # If no filters are active, the hybrid retriever is cached.
    ...
```

---

## 4. Dynamic Filtering in RAG Chain
The RAG chain in `chains/rag_chain.py` was refactored using **LangChain Expression Language (LCEL)** to dynamically apply metadata filters. It supports two input formats:
- Simple query string: `"What is a hospital?"`
- Query dictionary: `{"question": "What is a hospital?", "filter": {"pdf_id": "policy1"}}`

It uses `RunnableLambda` to fetch the retriever dynamically with filters and `RunnableParallel` to format context and questions.

**File:** [rag_chain.py](file:///c:/Users/mudit/Desktop/RAG_proj/rag_app/chains/rag_chain.py)
```python
def build_rag_chain():
    """Assembles the full dynamic LCEL RAG chain."""
    pipeline_inputs = RunnableParallel({
        "context": RunnableLambda(retrieve_documents_dynamic) | RunnableLambda(format_docs_with_metadata),
        "question": RunnableLambda(get_question)
    })
    chain = pipeline_inputs | RAG_PROMPT | llm_model | StrOutputParser()
    return chain
```

---

## 5. API Error Handling
We integrated error handling throughout the application:
- **LLM API validation**: If the `GEMINI_API_KEY` is missing or fails, it falls back to a descriptive error generator rather than crashing the script.
- **Vector database exception safety**: DB connections, retrieval, and insertions are wrapped in `try-except` blocks.
- **Loader safety**: Checks if the target PDF exists and is readable, reporting descriptive errors.

---

## 6. Structured Logging and Prints
We configured logging to both stdout and a persistent log file (`rag_pipeline.log`) alongside highlighted console markers indicating current execution steps:
- `[LOADER]` - PDF loading status.
- `[CHUNKER]` - Text splitting status.
- `[VECTORSTORE]` - DB indexing actions.
- `[RETRIEVER]` - Index construction, active filters, and query parameters.
- `[CHAIN]` - Executed query, retrieved context details, and document metadata.
- `[QUERY]` - Chain streaming and output lifecycle.
- `[INGEST]` - Multi-stage document indexing pipeline.
