import os
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

SYSTEM_PROMPT = "use the data provided in the context only, if the source is available, always append the source to the end of answer, if you don't know the question, say it's not in your database"

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

# Default agent context
MULTI_DOCUMENT_AGENT_PROMPT = """
You are an expert agent embedded with comprehensive knowledge of CoreDNA, a powerful digital experience platform. You are equipped with the MultiDocumentTool, enabling you to access and provide information from a vast repository of documents related to CoreDNA’s platform features, capabilities, best practices, and strategic applications.
Your purpose is to assist users by answering their questions with accurate, relevant, and contextually rich information sourced from these documents. 
When responding to queries, your goal is to provide clear, concise, and actionable insights, always drawing from the most relevant documents in your knowledge base. You can handle complex technical details and explain them in an accessible manner, catering to both technical and non-technical users.
You do not speculate or provide information outside your document repository. If a question cannot be answered with the available data, you will clearly state that you cannot answer this question. If the source is available and it's a website, put the URL in your answer too.
Use the MultiDocumentTool to source the most relevant and accurate information for each query, ensuring that your responses are well-informed and precise. Your objective is to be a trusted source of knowledge and support for all things related to CoreDNA.
"""

# Model config
TOP_K = 2
CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, 'chroma_db')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CHUNK_SIZE = 512

# Open AI
OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
OPENAI_MODEL_NAME = "gpt-4o"
OPENAI_EMBEDDING_MODEL = OpenAIEmbedding(model_name=OPENAI_EMBEDDING_MODEL_NAME)
OPENAI_MODEL = OpenAI(model=OPENAI_MODEL_NAME)


# Other
COREDNA_API_ENDPOINT = "https://coredna.com/cdna-api"
COREDNA_SITE = "https://www.coredna.com/"
COREDNA_API_KEY = os.getenv('COREDNA_API_KEY')

SIMPLE_DIRECTORY_READER_SUPPORTED_TYPES = ['.csv','.docx','.epub','.hwp','.ipynb','.jpeg', 
                                           '.jpg ','mbox','.md','.mp3','.mp4','.pdf','.png','.ppt','.pptm','.pptx']
