from storage.Index import Index
from tools.http_request_tool import http_tool
from tools.coredna.sharpspring_tool import sharpspring_tool
from tools.webpage_scanner_tool import tag_exists_tool, fetch_field_tool
from llama_index.core.agent import ReActAgent
from llama_index.core import Settings
from constants import CHUNK_SIZE, OPENAI_EMBEDDING_MODEL, OPENAI_MODEL, ACTION_ANALYZER_AGENT_PROMPT
from llama_index.core.prompts import PromptTemplate



class ActionAnalyzerAgent:
    def __init__(self, index_handler) -> None:
        self.index_handler: Index = index_handler
        self.function_tools = [
            http_tool(),
            tag_exists_tool(),
            fetch_field_tool(),
            sharpspring_tool()
        ]

        Settings.chunk_size = CHUNK_SIZE
        Settings.embed_model = OPENAI_EMBEDDING_MODEL
        Settings.llm = OPENAI_MODEL

    def get_index_handler(self):
        return self.index_handler
    
    
    def create_agent(self):
        agent = ReActAgent.from_tools(
            tools=self.function_tools,
            llm=Settings.llm,
            verbose=True,
            context=ACTION_ANALYZER_AGENT_PROMPT,
            max_iterations=20
        )
        # react_system_prompt = PromptTemplate(ACTION_ANALYZER_AGENT_PROMPT)
        # agent.update_prompts({"agent_worker:system_prompt": react_system_prompt})
        # agent.reset()
        print("action agent created")
        return agent
        