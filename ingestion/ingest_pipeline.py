"""WIll call all relavan process in INdexing or
Ingestion"""

from loaders.pdf_loader import load_pdf
from processing.chunker import recurcive_chunker
from vectorstore.chroma_store import add_docs


def ingest_document(pdf_path):
    #load the doc
    print("Loading the Docs..\n")
    docs = load_pdf(pdf_path)
    #chunk the data
    print("Chunking the Docs..\n")
    chunks = recurcive_chunker(docs)

    #convert to embedding and add to ChromaDB
    print("Embedding into the vectors../n")
    add_docs(chunks)