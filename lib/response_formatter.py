
"""Format API responses and source metadata."""

CHAIN_EXPRESSION = "prompt | chat_model | StrOutputParser"

COMPONENTS = [
    "ChatPromptTemplate",
    "ChatModel",
    "StrOutputParser",
]

FALLBACK_ANSWER = (
    "I do not have enough approved runbook context "
    "to answer this question."
)


def format_sources(scored_documents):
    """Extract source metadata from retrieved documents."""

    sources = []

    for document, distance in scored_documents:
        metadata = document.metadata or {}

        sources.append({
            "source_id": metadata.get("source_id", "unknown"),
            "title": metadata.get("title", "Untitled"),
            "category": metadata.get("category", "Uncategorized"),
            "section": metadata.get("section", "Unspecified"),
            "chunk_id": metadata.get("chunk_id", "unknown"),
            "distance": round(float(distance), 4),
        })

    return sources


def format_langchain_debug(scored_documents, context, top_k, fallback):
    """Build LangChain workflow debug information."""

    return {
        "chain_expression": CHAIN_EXPRESSION,
        "components": COMPONENTS,
        "retrieved_count": len(scored_documents),
        "retrieved_chunk_ids": [
            (document.metadata or {}).get("chunk_id", "unknown")
            for document, _ in scored_documents
        ],
        "top_k": top_k,
        "context_characters": len(context),
        "fallback": fallback,
        "score_type": "distance",
    }


def format_success_response(answer, sources, debug):
    """Format a successful API response."""

    return {
        "answer": answer.strip(),
        "sources": sources,
        "langchain": debug,
    }


def format_fallback_response(debug):
    """Format a safe response when no usable context exists."""

    return {
        "answer": FALLBACK_ANSWER,
        "sources": [],
        "langchain": debug,
    }


def format_error_response(error, message):
    """Format an API error response."""

    return {
        "error": error,
        "message": message,
    }
