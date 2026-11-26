
"""Flask API for the LangChain RAG lab."""

from flask import Flask, jsonify, request

from lib.langchain_rag_service import (
    LangChainServiceError,
    answer_question,
)
from lib.response_formatter import format_error_response
from lib.validation import validate_question_payload


def create_app():
    """Create and configure the Flask application."""

    app = Flask(__name__)

    @app.post("/api/ask")
    def ask():
        """Accept a question and return a source-backed LangChain RAG response."""

        # Get JSON request body
        payload = request.get_json(silent=True)

        # Validate request
        try:
            question = validate_question_payload(payload)
        except (ValueError, TypeError) as error:
            return jsonify(
                format_error_response(
                    "invalid_request",
                    str(error),
                )
            ), 400

        # Generate answer
        try:
            response = answer_question(question)

            return jsonify(response), 200

        except LangChainServiceError as error:
            return jsonify(
                format_error_response(
                    "langchain_service_error",
                    str(error),
                )
            ), 502

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
