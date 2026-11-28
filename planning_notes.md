
# LangChain RAG Lab - Planning Notes

## 1. System Goal

The goal is to refactor the existing Flask RAG API
using LangChain while preserving its existing behavior.

The system must answer engineering questions using
approved internal reliability runbooks.

## 2. Manual RAG Workflow

1. Receive the user's question.
2. Validate the question.
3. Retrieve relevant runbook chunks from Chroma.
4. Format retrieved documents and metadata.
5. Insert the context and question into a prompt.
6. Send the prompt to the Ollama chat model.
7. Parse the model output.
8. Return the answer and source metadata.

## 3. LangChain Components

- ChatPromptTemplate: Creates the grounded prompt.
- OllamaEmbeddings: Creates document embeddings.
- Chroma: Stores and retrieves runbook documents.
- ChatOllama: Generates the answer.
- StrOutputParser: Parses the model response.

## 4. Grounding and Safety

The model must answer using only approved retrieved
runbook context.

If no usable context is available, the API returns
a safe fallback response without calling the model.

The system must not invent company procedures.

## 5. Source Attribution

Each retrieved document preserves its metadata:

- Source ID
- Title
- Category
- Section
- Chunk ID
- Retrieval distance

## 6. API Contract

The POST /api/ask endpoint accepts a JSON question.

A successful response includes:

- answer
- sources
- langchain debug metadata

Invalid requests return HTTP 400.
LangChain service failures return HTTP 502.

## 7. Verification

The application is tested using Pytest.

The tests verify request validation, retrieval,
prompt construction, LangChain workflow structure,
fallback behavior, source attribution, and API responses.
