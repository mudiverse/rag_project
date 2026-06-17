#Loading the PDF for the RAG application
from langchain_community.document_loaders import PyMuPDFLoader
#right now app supports PDF so PDF later make it muli data

def load_pdf(pdf_path):
    pdf_loader = PyMuPDFLoader(pdf_path)

    #load the data as a Documents 
    return pdf_loader.load() #return the list of Docs created
