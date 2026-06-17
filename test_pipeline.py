from loaders.pdf_loader import load_pdf
from processing.chunker import recurcive_chunker

from embeddings.embedding_model import (
    get_embedding_model
)

from vectorstore.chroma_store import (
    get_vectorstore
)

PDF_PATH = "data/policies/policy1.pdf"

# -----------------------
# 1. LOAD
# -----------------------

docs = load_pdf(PDF_PATH)

print("\n" + "="*50)
print("LOADED DOCUMENTS")
print("="*50)

print(f"Total Pages Loaded: {len(docs)}")

print("\nFirst Page Content:\n")
print(docs[0].page_content[:1000])

print("\nMetadata:")
print(docs[0].metadata)


# -----------------------
# 2. CHUNK
# -----------------------

chunks = recurcive_chunker(docs)

print("\n" + "="*50)
print("CHUNKS")
print("="*50)

print(f"Total Chunks: {len(chunks)}")

for i, chunk in enumerate(chunks[:3]):

    print(f"\nChunk {i+1}")
    print("-"*30)

    print(chunk.page_content[:500])

    print("\nMetadata:")
    print(chunk.metadata)


# -----------------------
# 3. EMBEDDING
# -----------------------

embedding_model = get_embedding_model()

sample_embedding = embedding_model.embed_query(
    chunks[0].page_content
)

print("\n" + "="*50)
print("EMBEDDING")
print("="*50)

print(f"Embedding Length: {len(sample_embedding)}")

print("\nFirst 10 Values:")

print(sample_embedding[:10])


# -----------------------
# 4. STORE
# -----------------------

db = get_vectorstore()

db.add_documents(chunks)

print("\n" + "="*50)
print("STORED IN CHROMA")
print("="*50)

stored = db.get()

print(f"Total Docs In Store: {len(stored['ids'])}")

print("\nSample Metadata:")

print(stored["metadatas"][0])

print("\nSample Stored Text:")

print(stored["documents"][0][:500])


# -----------------------
# 5. RETRIEVE TEST
# -----------------------

print("\n" + "="*50)
print("RETRIEVAL TEST")
print("="*50)

results = db.similarity_search(
    "waiting period",
    k=3
)

for i, doc in enumerate(results):

    print(f"\nResult {i+1}")
    print("-"*30)

    print(doc.page_content[:500])

    print("\nMetadata:")
    print(doc.metadata)