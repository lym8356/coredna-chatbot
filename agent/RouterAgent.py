from llama_index.agent.openai_legacy import FnRetrieverOpenAIAgent
from constants import ROUTER_AGENT_PROMPT

class RouterAgent:
    def __init__(self) -> None:
        self.agent = self.create_agent()

    def create_agent(self):

        agent = FnRetrieverOpenAIAgent.from_retriever(
            system_prompt=ROUTER_AGENT_PROMPT,
            verbose=True
        )
        print("router agent created")

        return agent
    
