from constants import DEFAULT_TEXT_QA_PROMPT_TMPL, CHROMA_DB_PATH, MODEL_NAME, DATA_DIR, COREDNA_SITE, COREDNA_API_ENDPOINT, COREDNA_API_KEY,TOP_K

import chromadb, urllib3, os, json
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.web import SitemapReader

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

    # load webpage data from crawler
    document_pages = load_webpages_to_documents()

    # load csv data from local folders
    document_csv = SimpleDirectoryReader(DATA_DIR).load_data()

    # combine all documents
    documents = document_pages + document_csv
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=Settings.embed_model, 
        transformations=[splitter]
    )

# Load webpage content into global documents
def load_webpages_to_documents():

    # pages_to_scan = get_list_of_pages()

    # urls = [f"https:{page['url']}" for page in pages_to_scan]

    # documents = SimpleWebPageReader(html_to_text=True).load_data(urls = urls)

    # for doc, url in zip(documents, urls):
    #     doc.metadata = {'url': url}

    loader = SitemapReader()

    documents = loader.load_data(
        sitemap_url=f"{COREDNA_SITE}sitemap.xml",
        filter={COREDNA_SITE}
    )

    return documents

def get_list_of_pages():

    # Make API calls to get list of pages to crawl
    response = urllib3.request(
        'GET', 
        f"{COREDNA_API_ENDPOINT}/pages/pages?properties={{url}}",
        headers = {
            'Authorization': COREDNA_API_KEY
        }
    )
    content = json.loads(response.data.decode('utf-8'))

    return content['data']



# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
@cross_origin()
def index():
    return jsonify({"status": "OK"})


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
            text_qa_template=TEXT_QA_TEMPLATE,
            include_metadata=True
        )
        
        response = query_engine.query(query)

        answer = str(response)
        source_links = [node.metadata.get('Source', '') for node in response.source_nodes if node.metadata.get('Source')]

        if source_links:
            answer += f"\n\nIf you would like to read more, here is the link: {source_links[0]}"

        return jsonify({"response": answer})
    
    except Exception as e:
        return jsonify({"error": f"Failed to process query: {str(e)}"}), 500

if __name__ == '__main__':
    initialize_index()
    app.run(host='0.0.0.0', port=80)
