from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from rag import config
from rag.tools import ALL_TOOLS

SYSTEM_PROMPT = (
    "You are an advanced AI assistant with two tools: "
    "search_internal_knowledge for company policy, internal documents, and private notes, "
    "and search_global_internet for breaking news, current events, and public facts. "
    "Use the internal knowledge tool for questions about this organization. "
    "Use web search for questions the local documents do not cover. "
    "Answer from the tool results and cite the source when one is provided."
)


def create_my_agent():
    llm = ChatOpenAI(
        model=config.CHAT_MODEL,
        api_key=config.OPENROUTER_API_KEY,
        base_url=config.OPENROUTER_BASE_URL,
        temperature=0,
    ).bind_tools(ALL_TOOLS)

    def call_model(state: MessagesState):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> str:
        last_message = state["messages"][-1]
        if not getattr(last_message, "tool_calls", None):
            return END
        return "tools"

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(ALL_TOOLS))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    return workflow.compile()
