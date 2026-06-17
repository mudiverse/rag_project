import os
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

CHROMA_PATH = "data/chroma_db"

TOP_K = 5 # can change