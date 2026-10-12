from ddgs import DDGS
from langchain_core.tools import tool


@tool
def search_global_internet(query: str) -> str:
    """Searches the live open internet for breaking news, current events, and global info."""
    try:
        results = DDGS().text(
            query,
            region="us-en",
            safesearch="moderate",
            max_results=5,
        )
    except Exception as exc:
        return f"Web search failed: {exc}"

    if not results:
        return "No web results found."

    lines = []
    for row in results:
        title = row.get("title", "")
        body = row.get("body", "")
        href = row.get("href", "")
        lines.append(f"{title}\n{body}\n{href}".strip())
    return "\n\n".join(lines)
