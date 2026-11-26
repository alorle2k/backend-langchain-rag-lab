
"""Flask API for the LangChain RAG lab."""

from flask import Flask, jsonify, request

from lib.langchain_rag_service import LangChainServiceError, answer_question
from lib.response_formatter import format_error_response
from lib.validation import validate_question_payload


def create_app():
    app = Flask(__name__)

    @app.post("/api/ask")
    def ask():
        payload = request.get_json(silent=True)

        question, error = validate_question_payload(payload)

        if error:
            return jsonify(error), 400

        try:
            response = answer_question(question)
            return jsonify(response), 200

        except LangChainServiceError as exc:
            return jsonify(
                format_error_response(
                    "service_error",
                    str(exc),
                )
            ), 502

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)
