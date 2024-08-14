import os
import dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter

# Load environment variables
# dotenv.load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Constants
MODEL_NAME = "text-embedding-3-large"
CHROMA_DB_PATH = './chroma_db'
DATA_DIR = "./data"
TOP_K = 2
SYSTEM_PROMPT = "if you don't know the answer, just say it's not in the context, use the data provided in the document only"

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

# Global variable to store the index
index = None

def initialize_index():
    """Initialize the vector index from the ChromaDB or create a new one if necessary."""
    global index
    
    # Initialize ChromaDB client
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embed_model = OpenAIEmbedding(model_name=MODEL_NAME)
    chroma_collection = db.get_or_create_collection("FAQ_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection, embed_model=embed_model)

    # Update Settings for VectorStoreIndex
    Settings.embed_model = embed_model
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 20

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Load data if the collection is empty
    if chroma_collection.count() == 0:
        load_and_initialize_data(storage_context)
    else:
        index = VectorStoreIndex.from_vector_store(vector_store)

def load_and_initialize_data(storage_context):
    """Load documents from the directory and initialize the vector store and index."""
    global index
    splitter = SentenceSplitter()
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context, 
        embed_model=Settings.embed_model, 
        transformations=[splitter]
    )

# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route('/query', methods=['POST'])
@cross_origin()
def query_index():
    """Endpoint to query the vector index and return the result."""
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "Query not provided"}), 400
    
    # Perform the query on the index
    try:

        TEXT_QA_TEMPLATE = PromptTemplate(DEFAULT_TEXT_QA_PROMPT_TMPL)

        query_engine = index.as_query_engine(
            similarity_top_k=TOP_K,
            text_qa_template=TEXT_QA_TEMPLATE
        )
        
        response = query_engine.query(query)
        return jsonify({"response": str(response)})
    except Exception as e:
        return jsonify({"error": f"Failed to process query: {str(e)}"}), 500

if __name__ == '__main__':
    initialize_index()
    app.run(host='0.0.0.0', port=80)
