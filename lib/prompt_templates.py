
"""Prompt template helpers for the LangChain RAG workflow."""

from langchain_core.prompts import ChatPromptTemplate


SYSTEM_PROMPT = """
You are an internal reliability assistant.

Your job is to help engineers troubleshoot incidents
using approved company reliability runbooks.

IMPORTANT RULES:

1. Answer using ONLY the approved retrieved context.
2. Do not use general knowledge to invent company procedures.
3. Do not invent policies, metrics, or incident steps.
4. If the retrieved context does not contain enough
   information, clearly say that you do not have enough
   approved runbook context to answer.
5. Be concise, specific, and useful during incidents.
6. Identify the relevant source information when possible.

The approved runbook context is provided below.
"""


def build_rag_prompt():
    """Build the LangChain chat prompt template."""

    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Approved retrieved context:\n"
                "{context}\n\n"
                "Question:\n"
                "{question}",
            ),
        ]
    )
