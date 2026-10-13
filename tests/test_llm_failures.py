"""Tests for case_studies/common/llm_failures.py."""

import json

import pytest
from pydantic import BaseModel, ValidationError

from common import llm_failures as lf


class _Schema(BaseModel):
    x: int


def _validation_error() -> ValidationError:
    with pytest.raises(ValidationError) as info:
        _Schema(x="not an int")
    return info.value


class ReadTimeout(Exception):
    pass


class ConnectError(Exception):
    pass


def test_constants_are_distinct():
    values = [
        lf.CONNECTION_ERROR,
        lf.TIMEOUT,
        lf.JSON_PARSE_ERROR,
        lf.SCHEMA_VALIDATION_ERROR,
        lf.EMPTY_RESPONSE,
        lf.UNKNOWN_ERROR,
    ]
    assert len(set(values)) == len(values)


def test_llm_action_error_attributes():
    cause = ValueError("x")
    err = lf.LLMActionError(lf.TIMEOUT, "slow", raw_response="raw", cause=cause)
    assert isinstance(err, RuntimeError)
    assert str(err) == "slow"
    assert (err.failure_type, err.raw_response, err.cause) == (lf.TIMEOUT, "raw", cause)
    bare = lf.LLMActionError(lf.UNKNOWN_ERROR, "m")
    assert bare.raw_response is None and bare.cause is None


def test_classify_validation_error():
    assert lf.classify_exception(_validation_error()) == lf.SCHEMA_VALIDATION_ERROR


def test_classify_json_decode_error():
    with pytest.raises(json.JSONDecodeError) as info:
        json.loads("prose, not json")
    assert lf.classify_exception(info.value) == lf.JSON_PARSE_ERROR


@pytest.mark.parametrize(
    "exc, expected",
    [
        (TimeoutError("x"), lf.TIMEOUT),
        (ReadTimeout("x"), lf.TIMEOUT),
        (RuntimeError("Request timed out"), lf.TIMEOUT),
        (RuntimeError("read TIMEOUT=30"), lf.TIMEOUT),
        (ConnectError("x"), lf.CONNECTION_ERROR),
        (ConnectionRefusedError("x"), lf.CONNECTION_ERROR),
        (OSError("Connection refused"), lf.CONNECTION_ERROR),
        (OSError("Failed to establish a new connection"), lf.CONNECTION_ERROR),
        (OSError("Max retries exceeded with url"), lf.CONNECTION_ERROR),
        (OSError("Name or service not known"), lf.CONNECTION_ERROR),
        (OSError("No route to host"), lf.CONNECTION_ERROR),
        (ValueError("Empty response from model"), lf.EMPTY_RESPONSE),
        (ValueError("empty"), lf.UNKNOWN_ERROR),
        (ValueError("bad response"), lf.UNKNOWN_ERROR),
        (KeyError("x"), lf.UNKNOWN_ERROR),
    ],
)
def test_classify_by_name_and_message(exc, expected):
    assert lf.classify_exception(exc) == expected


def test_timeout_wins_over_connection():
    assert lf.classify_exception(OSError("connect timed out")) == lf.TIMEOUT


def test_failure_type_of_prefers_attached_type():
    err = lf.LLMActionError(lf.EMPTY_RESPONSE, "connection refused")
    assert lf.failure_type_of(err) == lf.EMPTY_RESPONSE


@pytest.mark.parametrize("attached", ["", None, 3])
def test_failure_type_of_falls_back_to_classification(attached):
    exc = RuntimeError("timed out")
    exc.failure_type = attached
    assert lf.failure_type_of(exc) == lf.TIMEOUT


def test_failure_type_of_plain_exception():
    assert lf.failure_type_of(_validation_error()) == lf.SCHEMA_VALIDATION_ERROR
