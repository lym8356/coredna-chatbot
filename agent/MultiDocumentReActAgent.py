from llama_index.core import Settings
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent

from constants import MULTI_DOCUMENT_AGENT_PROMPT, CHUNK_SIZE, OPENAI_EMBEDDING_MODEL, OPENAI_MODEL
from storage.Index import Index

"""
This agent is ONLY responsible for fetching content from the vector database
Content includes static content like web pages, PDFs, CSVs, videos etc

All the tools built for this agent are from different data sources
Optional: Create 1 tool for each category of data source
Example: 1 tool for reading FAQ, 1 tool for web pages

All content needs to be summarized
There should be 1 data node which takes multiple data sources, all data sources should combine into 1 data node

Use SummaryIndex provided by LLamaIndex to convert this conbined data node to index object
Create a query engine base on the index
Create a tool base on the query engine
"""

class MultiDocumentReActAgent:
    def __init__(self, index_handler) -> None:

        """
        Initialize the MultiDocumentReActAgent.

        Args:
            index_handler (Index): An instance of the Index class, containing the loaded index to use for the agent.
        """

        Settings.chunk_size = CHUNK_SIZE
        Settings.embed_model = OPENAI_EMBEDDING_MODEL
        Settings.llm = OPENAI_MODEL

        # TODO: external vector database connection

        # Initialize Index class to handle index operations
        self.index_handler: Index = index_handler

        self.index = self.index_handler.get_index()  # This should be an instance of VectorStoreIndex
        self.index_loaded = self.index is not None 

        self.query_engine_tools = []

        self.create_query_engine_and_tools()


    def get_index_handler(self):
        return self.index_handler
    
    def get_engine_tools(self):
        return self.query_engine_tools
        
    def create_query_engine_and_tools(self):

        """
        Create a query engine and associated tools for the agent.
        """

        if not self.index_loaded:
            raise ValueError("Index is not loaded. Ensure the provided index_handler has a loaded index.")

        qa_engine = self.index.as_query_engine()

        self.query_engine_tools = [
            QueryEngineTool(
                query_engine=qa_engine,
                metadata=ToolMetadata(
                        name="multi_document_engine_tool",
                        description=(
                            "Provides detailed information about CoreDNA"
                        ),
                    ),
                )
            ]
        print("tools created")

    def create_agent(self):

        """
        Create a ReAct agent using the query engine tools.

        Returns:
            ReActAgent: The created agent.
        """

        if not self.query_engine_tools:
            raise ValueError("Query engine tools are not created. Call 'create_query_engine_and_tools' first.")

        agent = ReActAgent.from_tools(
            self.query_engine_tools,
            llm=Settings.llm,
            verbose=True,
            context=MULTI_DOCUMENT_AGENT_PROMPT
        )
        print("agent created")
        return agent