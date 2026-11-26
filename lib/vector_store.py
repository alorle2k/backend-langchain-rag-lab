
"""Vector store and retrieval helpers."""

from lib.config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
)


def build_embeddings():
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=EMBEDDING_MODEL)


def build_vector_store():
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=build_embeddings(),
    )


def retrieve_context(question, *, vector_store=None, top_k=DEFAULT_TOP_K):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be blank.")

    store = vector_store if vector_store is not None else build_vector_store()

    return store.similarity_search_with_score(
        question.strip(),
        k=top_k,
    )
