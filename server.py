import chromadb
import os
import dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.llms import openai
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import (
    SentenceSplitter
)


index = None
model = "text-embedding-3-large"
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def initialize_index():
    global index
    # create ChromaDB client and access existing vector store
    db = chromadb.PersistentClient(path='./chroma_db')
    chroma_collection = db.get_or_create_collection("FAQ_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)


    # Initialize index
    embed_model = OpenAIEmbedding(model_name=model)
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    Settings.chunk_overlap = 64
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # check if the collection has data
    if chroma_collection.count() == 0:
        load_data(vector_store, storage_context)
    else:
        index = VectorStoreIndex.from_vector_store(vector_store)



def load_data(vector_store, storage_context):
    global index
    splitter = SentenceSplitter()
    # Load data from the directory and update the vector store
    documents = SimpleDirectoryReader("./data").load_data()
    embed_model = OpenAIEmbedding(model_name = model)
    vector_store = ChromaVectorStore(documents=documents, embed_model=embed_model)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model, transformations=[splitter])

    # Save the index back to the storage
    vector_store.add(documents)

app = Flask(__name__)
CORS(app)

@app.route('/query', methods=['POST'])
@cross_origin()
def query_index():
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "Query not provided"}), 400
    
    # Perform the query on the index
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return jsonify({"response": str(response)})

if __name__ == '__main__':
    initialize_index()
    app.run(host='0.0.0.0', port=80)