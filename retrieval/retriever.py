from vectorstore.chroma_store import get_vectorstore

#function creates a retriver Object
def get_retriever():
    db = get_vectorstore()
    #this object can be used for CHAIns [convert db into a retirver comonent]
    return db.as_retriever(
        search_kwargs={"k":5}
    )