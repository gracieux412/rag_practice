from rag.tools.internet_search import search_global_internet
from rag.tools.local_retriever import search_internal_knowledge

ALL_TOOLS = [search_internal_knowledge, search_global_internet]
