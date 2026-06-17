from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
def get_embedding_model():
    
    #Change to replace with other embedding
    model = HuggingFaceEmbeddings(
        model_name ="all-MiniLM-L6-v2",
        encode_kwargs={'normalize_embeddings':True}
    )
    return  model #ret obj of model