from llama_index.core.agent import ReActAgent, FunctionCallingAgentWorker, AgentRunner
from llama_index.agent.openai import OpenAIAgent, OpenAIAssistantAgent
from llama_index.llms.openai import OpenAI
from typing import Dict, List, Tuple, Any, cast, Optional
from llama_index.core import (
    VectorStoreIndex,
    SummaryIndex,
    ServiceContext,
    Document,
)
from llama_index.core.agent.react.formatter import ReActChatFormatter
from llama_index.core.agent.react.prompts import REACT_CHAT_SYSTEM_HEADER


agent = None


def get_agent():
    return agent


def build_agent(
        system_prompt: str,
        # rag_params: RAGParams,
        docs: List[Document],
        vector_index: Optional[VectorStoreIndex] = None,
        additional_tools: Optional[List] = None,
):
    """Construct agent from docs / parameters / indices."""


# Currently testing this
def build_react_agent(tools: dict):

    llm = OpenAI(model="gpt-4o")

    agent = ReActAgent.from_tools(
        tools, 
        llm=llm, 
        verbose=True,
        max_iterations=10,
        react_chat_formatter=ReActChatFormatter(
            system_header="If it's a question you can find answer in the knowledge base tool, and the source is available, always append the source at the end of answer. Always use the provided tools to find your answer, if you cannot find the answer, just say you don't know" + "\n" + REACT_CHAT_SYSTEM_HEADER
        ),
        context="If it's a question you can find answer in the knowledge base tool, and the source is available, always append the source at the end of answer. Always use the provided tools to find your answer, if you cannot find the answer, just say you don't know"
    )

    return agent


def build_function_calling_agent(tools: dict):
    llm = OpenAI(model="gpt-4o")
    agent_worker = FunctionCallingAgentWorker.from_tools(
        tools=tools,
        llm=llm,
        verbose=True,
        allow_parallel_tool_calls=False
    )

    agent = AgentRunner(agent_worker)
    return agent


# Do not use, doesn't work
def build_openai_agent(tools: dict):
    agent = OpenAIAssistantAgent.from_new(
        name="QA bot",
        instructions="You are a bot designed to answer questions or perform actions base on user's queries, if it is a question, only use the tool provided to search the entire question and always search the metadata and append the source",
        openai_tools=[],
        tools=tools,
        verbose=True,
        run_retrieve_sleep_time=1.0
    )
    return agent