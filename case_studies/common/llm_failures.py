"""Failure taxonomy for LLM propose/action calls.

Hosted models (gpt-4o-mini, etc.) rarely fail the propose/action call itself.
Local models served through Ollama fail far more often, and in qualitatively
different ways: the server isn't running, a CPU-bound model times out, or the
model returns prose instead of the requested JSON. Lumping all of that into a
single "exception" bucket (the previous behavior) makes it impossible to tell
those apart when comparing models in an ablation run.

This module gives every propose/action call site a common vocabulary for
"why did this fail" and a place to attach the raw model output that caused
it, so it ends up in the trace even when parsing/validation never succeeds.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from pydantic import ValidationError

# Stable set of failure categories written to metrics CSVs and traces.
CONNECTION_ERROR = "connection_error"
TIMEOUT = "timeout"
JSON_PARSE_ERROR = "json_parse_error"
SCHEMA_VALIDATION_ERROR = "schema_validation_error"
EMPTY_RESPONSE = "empty_response"
UNKNOWN_ERROR = "unknown_error"


class LLMActionError(RuntimeError):
    """Raised when a propose/action LLM call fails.

    Carries a `failure_type` from the constants above plus whatever raw model
    output was available, so callers can log it to the trace before the
    exception propagates and the run is marked as failed.
    """

    def __init__(
        self,
        failure_type: str,
        message: str,
        *,
        raw_response: Any = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.failure_type = failure_type
        self.raw_response = raw_response
        self.cause = cause


def classify_exception(exc: BaseException) -> str:
    """Best-effort mapping from an arbitrary exception to a failure category.

    Used both to build `LLMActionError.failure_type` at the call site and as
    a fallback further up the stack for exceptions that were never wrapped
    (e.g. a connection error that bypassed the propose/action helper).
    """
    if isinstance(exc, ValidationError):
        return SCHEMA_VALIDATION_ERROR
    if isinstance(exc, json.JSONDecodeError):
        return JSON_PARSE_ERROR

    name = type(exc).__name__.lower()
    msg = str(exc).lower()

    if "timeout" in name or "timed out" in msg or "timeout" in msg:
        return TIMEOUT

    connection_markers = (
        "connect",
        "connection refused",
        "failed to establish a new connection",
        "max retries exceeded",
        "name or service not known",
        "no route to host",
    )
    if "connect" in name or any(marker in msg for marker in connection_markers):
        return CONNECTION_ERROR

    if "empty" in msg and "response" in msg:
        return EMPTY_RESPONSE

    return UNKNOWN_ERROR


def failure_type_of(exc: BaseException) -> str:
    """Return the failure_type for any exception, wrapped or not."""
    failure_type = getattr(exc, "failure_type", None)
    if isinstance(failure_type, str) and failure_type:
        return failure_type
    return classify_exception(exc)
