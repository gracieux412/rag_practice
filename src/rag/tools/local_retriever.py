from langchain_core.tools import tool

from rag.database import get_vector_store


@tool
def search_internal_knowledge(query: str) -> str:
    """Searches proprietary internal documents, company history, and private notes.
    Use this first if the user asks about local policy, private details, or company files."""
    retriever = get_vector_store().as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(query)
    if not docs:
        return "No specific internal data matches found."
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source')}]: {doc.page_content}" for doc in docs
    )
