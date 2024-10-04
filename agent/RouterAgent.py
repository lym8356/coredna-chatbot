from llama_index.agent.openai_legacy import FnRetrieverOpenAIAgent
from constants import ROUTER_AGENT_PROMPT
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping
from llama_index.core import VectorStoreIndex


class RouterAgent:
    def __init__(self, agent_tools: list) -> None:
        self.tools = agent_tools
        # self.agent = self.create_agent()
        
    def create_agent(self):

        # selector = LLMSingleSelector.from_defaults()

        # Create a router query engine
        # router_query_engine = RouterQueryEngine(
        #     selector=selector,
        #     query_engine_tools=self.tools,
        #     verbose=True,
        # )

        tool_mapping = SimpleToolNodeMapping.from_objects(self.tools)

        obj_index = ObjectIndex.from_objects(self.tools, tool_mapping=tool_mapping, index_cls=VectorStoreIndex)

        retriever = obj_index.as_retriever(similarity_top_k=2)

        agent = FnRetrieverOpenAIAgent.from_retriever(
            retriever,
            system_prompt=ROUTER_AGENT_PROMPT,
            verbose=True
        )

        print("router agent created")

        return agent
    
    
