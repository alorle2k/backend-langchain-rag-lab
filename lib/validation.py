
"""Validate incoming question payloads."""

MAX_QUESTION_LENGTH = 500
MIN_QUESTION_LENGTH = 3


def validate_question_payload(payload):
    if not isinstance(payload, dict):
        return None, {
            "error": "invalid_request",
            "message": "Request body must be a JSON object.",
        }

    if "question" not in payload:
        return None, {
            "error": "missing_question",
            "message": "A question field is required.",
        }

    question = payload["question"]

    if not isinstance(question, str):
        return None, {
            "error": "invalid_question",
            "message": "Question must be a string.",
        }

    question = question.strip()

    if not question:
        return None, {
            "error": "empty_question",
            "message": "Question cannot be empty.",
        }

    if len(question) < MIN_QUESTION_LENGTH:
        return None, {
            "error": "short_question",
            "message": "Question must contain at least 3 characters.",
        }

    if len(question) > MAX_QUESTION_LENGTH:
        return None, {
            "error": "long_question",
            "message": "Question cannot exceed 500 characters.",
        }

    return question, None
