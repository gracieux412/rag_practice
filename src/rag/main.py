from langchain_core.messages import HumanMessage

from rag import config
from rag.agent import create_my_agent
from rag.ingestion import ensure_knowledge_base


def message_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def main() -> None:
    config.configure_stdio()
    print("Custom Agent Engine Active.")
    print("Ask anything. The agent routes to ChromaDB or the internet.")
    print("Type 'exit' or 'quit' to close the session.")
    print("Put .txt files in the data folder, then restart, to replace the sample index.")
    print("-" * 60)

    ensure_knowledge_base()
    agent_executor = create_my_agent()
    session_messages = []

    while True:
        try:
            user_input = ""
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Shutting down agent execution loop. Goodbye!")
                break

            session_messages.append(HumanMessage(content=user_input))
            response_state = agent_executor.invoke(
                {"messages": session_messages},
                config={"recursion_limit": 12},
            )
            final_message = response_state["messages"][-1]
            print(f"\nAgent: {message_text(final_message)}")
            session_messages = list(response_state["messages"])

        except (KeyboardInterrupt, EOFError):
            print("\nShutting down agent loop. Goodbye!")
            break
        except Exception as exc:
            print(f"\nAn internal graph error occurred: {type(exc).__name__}: {exc}")
            if session_messages and isinstance(session_messages[-1], HumanMessage):
                if session_messages[-1].content == user_input:
                    session_messages.pop()


if __name__ == "__main__":
    main()
