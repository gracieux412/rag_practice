from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from rag import config

_vector_store: Chroma | None = None


def get_vector_store(collection_name: str = "agent_knowledge") -> Chroma:
    """Connect to the local Chroma collection, creating it on first use."""
    global _vector_store
    if _vector_store is None:
        embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url=config.OPENROUTER_BASE_URL,
            # OpenRouter accepts raw text. Token-id batches are an OpenAI-only path.
            check_embedding_ctx_length=False,
        )
        _vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(config.CHROMA_DB_DIR),
        )
    return _vector_store
