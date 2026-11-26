
"""LangChain-supported retrieval augmented generation service."""

from lib.vector_store import retrieve_context
from lib.prompt_templates import build_rag_prompt
from lib.response_formatter import (
    format_sources,
    format_langchain_debug,
    format_success_response,
    format_fallback_response,
)


DEFAULT_TOP_K = 3


class LangChainServiceError(Exception):
    """Raised when the LangChain service fails."""


def has_usable_context(scored_documents):
    return any(
        isinstance(getattr(document, "page_content", None), str)
        and document.page_content.strip()
        for document, _ in scored_documents
    )


def format_context(scored_documents):
    blocks = []

    for document, distance in scored_documents:
        metadata = document.metadata or {}

        block = (
            f"Source ID: {metadata.get('source_id', 'unknown')}\n"
            f"Title: {metadata.get('title', 'Untitled')}\n"
            f"Category: {metadata.get('category', 'unknown')}\n"
            f"Section: {metadata.get('section', 'unknown')}\n"
            f"Chunk ID: {metadata.get('chunk_id', 'unknown')}\n"
            f"Distance: {distance}\n"
            f"Text:\n{document.page_content}"
        )

        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def answer_question(
    question,
    *,
    vector_store=None,
    chain=None,
    top_k=DEFAULT_TOP_K,
):
    try:
        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError("Question cannot be blank.")

        scored_documents = retrieve_context(
            cleaned_question,
            vector_store=vector_store,
            top_k=top_k,
        )

        if not has_usable_context(scored_documents):
            debug = format_langchain_debug(
                [],
                "",
                top_k,
                True,
            )
            return format_fallback_response(debug)

        context = format_context(scored_documents)

        if chain is None:
            from langchain_core.output_parsers import StrOutputParser

            from lib.service import build_chat_model

            chain = build_rag_prompt() | build_chat_model() | StrOutputParser()

        result = chain.invoke({
            "context": context,
            "question": cleaned_question,
        })

        if not isinstance(result, str) or not result.strip():
            raise LangChainServiceError("The model returned an empty response.")

        sources = format_sources(scored_documents)

        debug = format_langchain_debug(
            scored_documents,
            context,
            top_k,
            False,
        )

        return format_success_response(
            result.strip(),
            sources,
            debug,
        )

    except LangChainServiceError:
        raise

    except Exception as exc:
        raise LangChainServiceError(str(exc)) from exc
