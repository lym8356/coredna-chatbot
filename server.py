import os, traceback
from typing import Dict
from dotenv import load_dotenv
from multiprocessing import Lock
from multiprocessing.managers import BaseManager

# from agent import MultiDocumentReActAgent, ActionAnalyzerAgent
from agent.ActionAgent import ActionAgent
from agent.MultiDocumentReActAgent import MultiDocumentReActAgent
from agent.RouterAgent import RouterAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from constants import ROUTER_AGENT_PROMPT
from storage.Index import Index
from utils import load_data_from_sitemap
import logging
# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

# Global dictionary to store Index and ReActAgent instances for different collections
index_handlers: Dict[str, Index] = {}

def initialize_index(collection_name: str):

    """Initialize the vector index for a given collection name."""

    if (collection_name in index_handlers):
        return index_handlers[collection_name]
    
    # Create an instance of the Index class with the specified collection name
    index_handler = Index(collection_name=collection_name)
    # Load or create the index, this should return an instance of VectorIndex
    index_handler.load_or_create_index()

    # Store the index handler in the global dictionary
    index_handlers[collection_name] = index_handler

    print(f"Index for collection '{collection_name}' has been initialized.")
    return index_handler
    
# Set up logging
logging.basicConfig(filename='app.log', level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s')
logger = logging.getLogger(__name__)


# Initialize Flask app
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
@cross_origin()
def health_check():
    return jsonify({"status": "OK"})


# Endpoint to initialize index with a collection name
@app.route('/initialize', methods=['POST'])
@cross_origin()
def initialize():
    data = request.get_json()
    collection_name = data.get('collection_name')
    global agent
    if not collection_name:
        return jsonify({"error": "Collection name is required"}), 400
    
    try:
        index_handler = initialize_index(collection_name)
        rag_agent = MultiDocumentReActAgent(index_handler=index_handler).create_openai_agent()
        action_agent = ActionAgent(index_handler=index_handler).create_openai_agent()

        agents = {
            "rag_agent": rag_agent,
            "action_agent": action_agent
        }

        rag_tool = QueryEngineTool(
            query_engine=rag_agent,
            metadata=ToolMetadata(
                name="rag_agent_tool",
                description="This tool uses the rag_agent to answer queries"
            ),
        )

        action_tool = QueryEngineTool(
            query_engine=action_agent,
            metadata=ToolMetadata(
                name="action_agent_tool",
                description="This tool uses the action_agent to check if a specific HTML exists in a given URL"
            ),
        )

        # agent = OpenAIAgent.from_tools(
        #     tool_retriever=obj_index.as_retriever(similarity_top_k=1),
        #     system_prompt=ROUTER_AGENT_PROMPT,
        #     verbose=True
        # )

        agent = RouterAgent([rag_tool, action_tool]).create_agent()

        return jsonify({"status": f"Index for collection '{collection_name}' has been initialized."}), 200
    
    except Exception as e:
        return jsonify({"error": f"Failed to initialize index: {str(e)}"}), 500


# endpoint to accpet a file 


# endpoint to handle sitemap URL
@app.route('/sitemap', methods=['PUT'])
@cross_origin()
def handle_sitemap():

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        sitemap_url = data.get('sitemap_url')
        domain = data.get('domain')
        collection_name = data.get('collection_name')

         # Validate input data
        if not sitemap_url or not isinstance(sitemap_url, str) or not sitemap_url.strip():
            return jsonify({"error": "Invalid or missing 'sitemap_url'"}), 400

        if not domain or not isinstance(domain, str) or not domain.strip():
            return jsonify({"error": "Invalid or missing 'domain'"}), 400
        
        if not collection_name or not isinstance(collection_name, str) or not collection_name.strip():
            return jsonify({"error": "Invalid or missing 'collection_name'"}), 400

        documents = load_data_from_sitemap(sitemap_url, domain)

        # Add documents to the index
        index_handler = index_handlers[collection_name]

        for doc in documents:
            index_handler.index.insert(doc)
        index_handler.index.storage_context.persist(persist_dir=index_handler.get_collection_path())
        index_handlers[collection_name] = index_handler

        # TODO: Recreate agent

        # Return a success message along with the number of documents processed
        return jsonify({"status": "OK", "documents_processed": len(documents)}), 200

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except ConnectionError as e:
        return jsonify({"error": str(e)}), 502

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500



@app.route('/query', methods=['POST'])
@cross_origin()
def query():

    query = request.json.get('query')
    if not query:
        logger.error("No query provided")
        return jsonify({"error": "Query not provided"}), 400
    
    global agent
    if agent is None:
        logger.error("Agent not initialized")
        return jsonify({"error": "Agent is not initialized. Please initialize the index and agent first."}), 400
    
    try:
        answer = agent.chat(query)
        logger.info(f"Query processed successfully: {query}")
        return jsonify({"response": str(answer)})
    
    except Exception as e:
        logger.exception("Failed to process query")
        return jsonify({"error": f"Failed to process query: {str(e)}"}), 500


# enable this for development mode
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=80)
