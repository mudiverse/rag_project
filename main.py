import sys
import time
import os
import logging
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from ingestion.ingest_pipeline import ingest_document
from query.query_pipeline import ask_question
from vectorstore.chroma_store import get_vectorstore
from retrieval.retriever import get_retriever

# Configure standard logging to both console and a log file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("rag_pipeline.log", mode="w", encoding="utf-8")
    ]
)

logger = logging.getLogger("MAIN")

# Initialize FastAPI App
app = FastAPI(title="FastAPI RAG Backend")

# Define Pydantic Models for requests/responses
class IngestRequest(BaseModel):
    documentId: str
    pdfUrl: str

class QueryRequest(BaseModel):
    documentId: str
    question: str

class SourceItem(BaseModel):
    page: int
    snippet: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]

@app.post("/ingest")
async def ingest(request: IngestRequest):
    logger.info(f"Ingest request received for documentId: {request.documentId}")
    try:
        # 1. Download PDF file from Cloudinary URL
        os.makedirs("data/policies", exist_ok=True)
        pdf_path = f"data/policies/{request.documentId}.pdf"
        
        logger.info(f"Downloading PDF from URL: {request.pdfUrl}")
        async with httpx.AsyncClient() as client:
            response = await client.get(request.pdfUrl)
            if response.status_code != 200:
                logger.error(f"Failed to download PDF. Status code: {response.status_code}")
                raise HTTPException(status_code=400, detail="Failed to download PDF from the provided URL")
            
            with open(pdf_path, "wb") as f:
                f.write(response.content)
        
        # 2. Run ingestion pipeline
        logger.info(f"Triggering ingestion pipeline for file: {pdf_path}")
        ingest_document(pdf_path)
        
        return {"status": "INDEXED"}
    except Exception as e:
        logger.error(f"Ingestion failed for documentId {request.documentId}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/rag/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    logger.info(f"Query request received for documentId: {request.documentId}")
    try:
        # 1. Retrieve the source documents using the filter matching this documentId
        logger.info(f"Retrieving source documents for documentId: {request.documentId}")
        retriever = get_retriever(filter_dict={"pdf_id": request.documentId})
        retrieved_docs = retriever.invoke(request.question)
        
        sources = []
        for doc in retrieved_docs:
            sources.append(SourceItem(
                page=doc.metadata.get("page_number", 1),
                snippet=doc.page_content
            ))
        
        # 2. Get answer from the dynamic RAG pipeline using the existing chain
        logger.info(f"Invoking ask_question with question and pdf_id filter")
        response_stream = ask_question({
            "question": request.question,
            "filter": {"pdf_id": request.documentId}
        })
        
        # Concatenate response generator chunks into a single string
        answer_chunks = []
        for chunk in response_stream:
            answer_chunks.append(chunk)
        answer = "".join(answer_chunks)
        
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        logger.error(f"Query execution failed for documentId {request.documentId}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.delete("/rag/document/{documentId}")
async def delete_rag_document(documentId: str):
    logger.info(f"Delete request received for documentId: {documentId}")
    try:
        db = get_vectorstore()
        
        # Fetch chunk IDs matching pdf_id
        stored_data = db.get(where={"pdf_id": documentId})
        doc_ids = stored_data.get("ids", [])
        
        if doc_ids:
            logger.info(f"Deleting {len(doc_ids)} chunks from Chroma DB for pdf_id: {documentId}")
            db.delete(ids=doc_ids)
        else:
            logger.info(f"No chunks found in Chroma DB for pdf_id: {documentId}")
            
        # Delete local PDF file if it exists
        pdf_path = f"data/policies/{documentId}.pdf"
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            logger.info(f"Deleted local PDF file: {pdf_path}")
            
        return {"status": "DELETED"}
    except Exception as e:
        logger.error(f"Delete operation failed for documentId {documentId}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


def run_test_suite():
    print("\n" + "="*70)
    print("            RAG PIPELINE UPGRADE DEMONSTRATION RUN")
    print("="*70)
    
    policy1_path = "data/policies/policy1.pdf"
    policy2_path = "data/policies/policy2.pdf"
    
    # 1. Connect to Chroma and check existing chunks
    try:
        db = get_vectorstore()
        stored = db.get()
        num_docs = len(stored.get("ids", []))
        print(f"\n[MAIN] Connected to Chroma DB. Current database contains {num_docs} chunks.")
    except Exception as e:
        print(f"[MAIN] [ERROR] Failed to query database: {e}")
        logger.error(f"Failed to query database: {e}", exc_info=True)
        return

    # 2. Automatically ingest documents if store is empty
    if num_docs == 0:
        print("\n[MAIN] Database is empty! Running ingestion for policies...")
        try:
            ingest_document(policy1_path)
            ingest_document(policy2_path)
            stored = db.get()
            num_docs = len(stored.get("ids", []))
            print(f"[MAIN] Ingestion complete. Database now has {num_docs} chunks.")
        except Exception as e:
            print(f"[MAIN] [ERROR] Ingestion failed: {e}")
            return
    else:
        print("\n[MAIN] Using existing documents indexed in database.")
        print("[MAIN] Hint: To re-ingest clean datasets, delete the folder 'data/chroma_db'.")

    # 3. Test queries
    test_query = "Can we Migrate the policy to other insurance?"

    # --- TEST 2: Retrieval WITH Metadata Filter (Restricted to policy1) ---
    print("\n" + "="*60)
    print("TEST 2: Query WITH Filter (pdf_id = 'policy1')")
    print("="*60)
    filtered_query_p1 = {
        "question": test_query,
        "filter": {"pdf_id": "policy1"}
    }
    print(f"Query: '{test_query}' | Filter: pdf_id='policy1'\n")
    
    response = ask_question(filtered_query_p1)
    print("\nResponse:")
    for chunk in response:
        sys.stdout.write(chunk)
        sys.stdout.flush()
        time.sleep(0.01)
    print("\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_test_suite()
    else:
        print("[API] Starting FastAPI application on http://localhost:8000")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
