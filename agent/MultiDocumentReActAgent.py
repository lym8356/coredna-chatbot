from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.core.chat_engine.types import AgentChatResponse

import chromadb
from utils import load_data_from_sitemap, load_data_from_files
from llama_index.vector_stores.chroma import ChromaVectorStore
from constants import CHROMA_DB_PATH, DEFAULT_AGENT_CONTEXT

# There should be 1 
class MultiDocumentReActAgent:

    def __init__(self) -> None:

        Settings.chunk_size = 512
        Settings.embed_model = OllamaEmbedding(model_name='snowflake-arctic-embed:33m')
        Settings.llm = Ollama(model='mistral:latest')

        # database connection

         # Initialize ChromaDB client
        db = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        # TODO: allow users to upload files and create new collection base on file name

        faq_collection = db.get_or_create_collection("faq_collection")
        sitemap_collection = db.get_or_create_collection("sitemap_collection")
        faq_vector_store = ChromaVectorStore(chroma_collection=faq_collection)
        sitemap_vector_store = ChromaVectorStore(chroma_collection=sitemap_collection)

        self.faq_storage_context = StorageContext.from_defaults(vector_store=faq_vector_store)
        self.sitemap_storage_context = StorageContext.from_defaults(vector_store=sitemap_vector_store)
        self.index_loaded = False
        self.index = None
        self.query_engine_tools = []


    def load_from_existing_context(self):

        try:
            # TODO: load in a loop base on collection
            self._index = load_index_from_storage(storage_context=self.faq_storage_context)
            self.ge_index = load_index_from_storage(storage_context=self.sitemap_storage_context)
            self.index_loaded = True
        except Exception as e:
            self.index_loaded = False

        if not self.index_loaded:
            # load data
            # TODO: the folder name should be dynamic, this is passed in as a parameter when the class is created
            file_docs = load_data_from_files(directory='coredna')

            # build index
            self.ge_index = VectorStoreIndex.from_documents(documents=ge_docs, storage_context=self.ge_storage_context)
            
            
        
        def create_query_engine_and_tools(self):

            ge_engine = self.ge_index.as_query_engine(similarity_top_k=3)

            self.query_engine_tools = [
                QueryEngineTool(
                    query_engine=ge_engine,
                    metadata=ToolMetadata(
                        name="multi_document_engine",
                        description=(
                            "Provides detailed financial information about GE for the year 2023. "
                            "Input a specific plain text financial query for the tool"
                        ),
                    ),
                )
            ]

        def create_agent(self):

            agent = ReActAgent.from_tools(
                self.query_engine_tools,
                llm=Settings.llm,
                verbose=True,
                context=DEFAULT_AGENT_CONTEXT
            )
            return agent