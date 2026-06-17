"""Will store the Data in Local vectorStore for now"""

from langchain_chroma import Chroma
from embeddings.embedding_model import get_embedding_model

#get the dezired embedding model

def get_vectorstore():
    #create A Store for first Time
    print("Creating a vec store \n")
    store = Chroma(
        persist_directory="data/chroma_db",
        embedding_function=get_embedding_model()
    )

    return store

def add_docs(chunks):
    db = get_vectorstore()
    db.add_documents(chunks) #add the list of docs
    

