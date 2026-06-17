"""will have the main RAG chain will call other"""

#bringing gemini api
from config import GEMINI_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI

llm_model = ChatGoogleGenerativeAI(
    model ="gemini-3.5-flash",
    temperature=1.0,
    google_api_key=GEMINI_API_KEY
)


#main rag chain
from retrieval.retriever import get_retriever
from prompts.rag_prompts import RAG_PROMPT
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
def build_rag_chain():

    retriever = get_retriever()
    

       
   
    return  {
        "context":retriever,
        "question":RunnablePassthrough()
        }|RAG_PROMPT | llm_model |StrOutputParser()
        
       

