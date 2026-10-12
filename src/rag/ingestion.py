import hashlib

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import config
from rag.database import get_vector_store

SAMPLE_DOCUMENTS = [
    Document(
        page_content=(
            "Our remote work policy permits employees to work from anywhere "
            "globally for up to 90 days per calendar year."
        ),
        metadata={"source": "internal_hr_manual.txt"},
    ),
    Document(
        page_content=(
            "The primary office tech stack relies on Python 3.12, LangChain, "
            "and ChromaDB for AI backend applications."
        ),
        metadata={"source": "engineering_manifesto.txt"},
    ),
]


def load_source_documents() -> tuple[list[Document], str]:
    """Load .txt files from data/, or the built-in samples when the folder is empty."""
    txt_files = sorted(config.DATA_DIR.glob("*.txt"))
    if not txt_files:
        return list(SAMPLE_DOCUMENTS), "sample"

    documents = []
    for file_path in txt_files:
        documents.append(
            Document(
                page_content=file_path.read_text(encoding="utf-8"),
                metadata={"source": file_path.name},
            )
        )
    return documents, "files"


def _fingerprint(documents: list[Document]) -> str:
    payload = "\n".join(
        f"{doc.metadata.get('source', '')}\n{doc.page_content}" for doc in documents
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _chunk_ids(chunks: list[Document]) -> list[str]:
    counts: dict[str, int] = {}
    ids: list[str] = []
    for chunk in chunks:
        source = str(chunk.metadata.get("source", "unknown"))
        index = counts.get(source, 0)
        counts[source] = index + 1
        ids.append(f"{source}:{index}")
    return ids


def ensure_knowledge_base() -> None:
    """Index local documents when the source text has changed."""
    config.configure_stdio()
    documents, kind = load_source_documents()
    fingerprint = _fingerprint(documents)
    stored = ""
    if config.FINGERPRINT_PATH.exists():
        stored = config.FINGERPRINT_PATH.read_text(encoding="utf-8").strip()
    if stored == fingerprint:
        return

    if kind == "sample":
        print(
            f"No source .txt files found in {config.DATA_DIR}. "
            "Indexing built-in sample documents."
        )
    else:
        for document in documents:
            print(f"Reading {document.metadata['source']}...")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    if not chunks:
        print("No text to index.")
        return

    db = get_vector_store()
    if db._collection.count() > 0:
        db.reset_collection()

    print(f"Embedding {len(chunks)} text chunks into the vector store...")
    db.add_documents(chunks, ids=_chunk_ids(chunks))
    config.FINGERPRINT_PATH.write_text(fingerprint, encoding="utf-8")
    print("Ingestion complete. Local documentation is indexed.")


def run() -> None:
    print("Starting ChromaDB data ingestion...")
    if config.FINGERPRINT_PATH.exists():
        config.FINGERPRINT_PATH.unlink()
    ensure_knowledge_base()


if __name__ == "__main__":
    run()
