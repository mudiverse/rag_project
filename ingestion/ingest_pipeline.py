import os
import re
import logging
from loaders.pdf_loader import load_pdf
from processing.chunker import semantic_chunker
from vectorstore.chroma_store import add_docs

logger = logging.getLogger(__name__)

def extract_clause_from_text(text):
    """Utility to extract clause/section titles from text content."""
    # Look for Section X.Y, Clause X.Y, Article X, etc.
    section_patterns = [
        r'(?i)\b(?:section|clause|article)\s+\d+(?:\.\d+)*\b',
        r'(?i)\b(?:section|clause|article)\s+[A-Z\d]+\b',
        r'(?m)^\s*(\d+\.\d+)\s+([A-Z][A-Za-z\s]{3,40})',  # E.g. "3.1 Waiting Periods" at start of line
    ]
    
    for pattern in section_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0).strip()
            
    # Check if first few lines look like headers (short and uppercase)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    for line in lines[:3]:
        if 4 <= len(line) <= 60 and line.isupper() and not line.endswith(('.', '?', '!')):
            return line
            
    return "General"

def ingest_document(pdf_path):
    print(f"\n[INGEST] ===============================================")
    print(f"[INGEST] Starting ingestion pipeline for: {pdf_path}")
    print(f"[INGEST] ===============================================")
    logger.info(f"Ingesting PDF document: {pdf_path}")
    
    try:
        # Extract pdf_id from filename (e.g., policy1.pdf -> policy1)
        pdf_id = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"[INGEST] Extracted PDF ID: {pdf_id}")
        
        # 1. Load the doc
        print("[INGEST] Loading pages using PDF loader...")
        docs = load_pdf(pdf_path)
        print(f"[INGEST] Successfully loaded {len(docs)} pages.")
        
        # 2. Chunk the data semantically
        print("[INGEST] Chunking document content...")
        chunks = semantic_chunker(docs)
        
        # 3. Enrich chunk metadata schema
        print("[INGEST] Enriching chunk metadata schema...")
        enriched_chunks = []
        last_clause = "General"
        
        for idx, chunk in enumerate(chunks):
            # Page number in LangChain is 0-indexed by default, let's create a 1-indexed page_number
            page_0_indexed = chunk.metadata.get("page", 0)
            page_number = page_0_indexed + 1
            
            # Extract section/clause heading from content
            clause = extract_clause_from_text(chunk.page_content)
            if clause == "General" and last_clause != "General":
                # Propagate last heading if current chunk has none
                clause = last_clause
            else:
                last_clause = clause
                
            # Update chunk metadata dictionary
            chunk.metadata["pdf_id"] = pdf_id
            chunk.metadata["page_number"] = page_number
            chunk.metadata["clause"] = clause
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["source"] = pdf_path  # Keep standard source path
            
            enriched_chunks.append(chunk)
            
        print(f"[INGEST] Metadata schema enriched for {len(enriched_chunks)} chunks.")
        logger.info(f"Enriched {len(enriched_chunks)} chunks with schema (pdf_id={pdf_id}).")
        
        # 4. Add to Vector Store
        print("[INGEST] Embedding chunks and saving to vector store...")
        add_docs(enriched_chunks)
        print(f"[INGEST] Ingestion pipeline successfully completed for {pdf_path}!")
        logger.info(f"Successfully completed ingestion pipeline for {pdf_path}.")
        
    except Exception as e:
        print(f"[INGEST] [ERROR] Ingestion pipeline failed: {e}")
        logger.error(f"Ingestion pipeline failed for {pdf_path}: {e}", exc_info=True)
        raise e