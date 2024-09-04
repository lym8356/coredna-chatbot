import os

SYSTEM_PROMPT = "use the data provided in the document only, do not act like a bot, if you don't know the question, say it's not in your database"

# Text QA templates
DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information answer the following question "
    f"({SYSTEM_PROMPT}):" 
    "{query_str}\n"
)

# Query engine related
TOP_K = 2
MODEL_NAME = "text-embedding-3-large"
CHROMA_DB_PATH = './chroma_db'
DATA_DIR = "./data"


# Other
COREDNA_API_ENDPOINT = "https://coredna.com/cdna-api"
COREDNA_SITE = "https://www.coredna.com/"
COREDNA_API_KEY = os.getenv('COREDNA_API_KEY')