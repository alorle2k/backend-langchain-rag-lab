
"""LangChain-supported retrieval augmented generation service."""

from langchain_ollama import ChatOllama

from lib.config import CHAT_MODEL, DEFAULT_TOP_K
from lib.vector_store import retrieve_context
from lib.response_formatter import (
    format_sources,
    format_langchain_debug,
    format_success_response,
    format_fallback_response,
)


class LangChainServiceError(Exception):
    """Raised when the LangChain service fails."""


def build_chat_model():
    """Create the Ollama chat model."""

    return ChatOllama(
        model=CHAT_MODEL,
        temperature=0,
    )


def build_chain():
    """Build the complete LangChain runnable sequence."""

    from langchain_core.output_parsers import StrOutputParser
    from lib import prompt_templates

    prompt = prompt_templates.build_rag_prompt()
    chat_model = build_chat_model()
    output_parser = StrOutputParser()

    return prompt | chat_model | output_parser


def has_usable_context(scored_documents):
    """Check whether retrieved documents contain usable text."""

    return any(
        isinstance(getattr(document, "page_content", None), str)
        and document.page_content.strip()
        for document, _ in scored_documents
    )


def format_context(scored_documents):
    """Format retrieved documents and their metadata."""

    blocks = []

    for index, (document, distance) in enumerate(
        scored_documents,
        start=1,
    ):
        metadata = document.metadata or {}

        block = (
            f"[Context {index}]\n"
            f"Source ID: {metadata.get('source_id', 'unknown')}\n"
            f"Title: {metadata.get('title', 'Untitled')}\n"
            f"Category: {metadata.get('category', 'unknown')}\n"
            f"Section: {metadata.get('section', 'unknown')}\n"
            f"Chunk ID: {metadata.get('chunk_id', 'unknown')}\n"
            f"Distance: {float(distance):.4f}\n"
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
    """Retrieve context and generate a grounded answer."""

    try:
        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError("Question cannot be blank.")

        # Step 1: Retrieve approved runbook context.
        scored_documents = retrieve_context(
            cleaned_question,
            vector_store=vector_store,
            top_k=top_k,
        )

        # Step 2: Handle missing context.
        if not has_usable_context(scored_documents):
            debug = format_langchain_debug(
                [],
                "",
                top_k,
                True,
            )

            return format_fallback_response(debug)

        # Step 3: Format retrieved context.
        context = format_context(scored_documents)

        # Step 4: Build the LangChain workflow.
        if chain is None:
            chain = build_chain()

        # Step 5: Execute the chain.
        result = chain.invoke({
            "context": context,
            "question": cleaned_question,
        })

        # Step 6: Validate model output.
        if not isinstance(result, str) or not result.strip():
            raise LangChainServiceError(
                "The model returned an empty answer."
            )

        # Step 7: Extract source metadata.
        sources = format_sources(scored_documents)

        # Step 8: Build debug information.
        debug = format_langchain_debug(
            scored_documents,
            context,
            top_k,
            False,
        )

        # Step 9: Return final response.
        return format_success_response(
            result.strip(),
            sources,
            debug,
        )

    except LangChainServiceError:
        raise

    except Exception as exc:
        raise LangChainServiceError(str(exc)) from exc
