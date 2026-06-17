"""full pipeline, which uses in Querying"""

from chains.rag_chain import build_rag_chain

chain = build_rag_chain()

def ask_question(question): #take an input query
    response = chain.invoke(question)
    return response