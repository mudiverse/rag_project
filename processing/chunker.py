"""This WIll help to split+chunk the docs properly"""

#TODO: update to SemanticTextlitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings #semantic
# from langchain.text_splitters
#  
def recurcive_chunker(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    #takes a list of texts ,return chunks+meta
    return splitter.split_documents(documents)

def semantic_chunker(documents):

    # sem_splitter = SemanticChunker(

    # )
    return None
