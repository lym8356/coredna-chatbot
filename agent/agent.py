from llama_index.core.agent import ReActAgent

agent = None

def get_agent():
    return agent


def build_agent(tools):
    global agent
    agent = ReActAgent.from_tools(tools, verbose=True)