"""Shared fixtures for the offline test suite.

The case-study loops import LangGraph, LangChain and python-dotenv at module
level. CI installs none of them, so this file registers minimal stand-ins for
any that are missing. Tests drive the deterministic parts of the loops (state
handling, safety checks, rollouts, metrics) and never call a real LLM.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_STUDIES = REPO_ROOT / "case_studies"
for path in (
    CASE_STUDIES,
    CASE_STUDIES / "cstr_case",
    CASE_STUDIES / "mixer_case",
):
    if str(path) not in sys.path:
        sys.path.append(str(path))


def _missing(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except ImportError:
        return True
    return False


def _install_stub(name: str, **attrs) -> None:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(sys.modules[parent], child, module)


class _StubStateGraph:
    def __init__(self, state_schema):
        self.state_schema = state_schema
        self.nodes = {}
        self.edges = []
        self.conditional_edges = []

    def add_node(self, name, func):
        self.nodes[name] = func

    def add_edge(self, start, end):
        self.edges.append((start, end))

    def add_conditional_edges(self, source, path, path_map=None):
        self.conditional_edges.append((source, path, path_map))

    def compile(self, **_kwargs):
        return self


class _StubChatModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@contextlib.contextmanager
def _stub_openai_callback():
    yield types.SimpleNamespace(
        total_tokens=0, prompt_tokens=0, completion_tokens=0, total_cost=0.0
    )


def _register_optional_stubs() -> None:
    if _missing("langgraph.graph"):
        _install_stub("langgraph")
        _install_stub(
            "langgraph.graph",
            StateGraph=_StubStateGraph,
            START="__start__",
            END="__end__",
        )
    if _missing("langchain.callbacks"):
        if _missing("langchain"):
            _install_stub("langchain")
        _install_stub("langchain.callbacks", get_openai_callback=_stub_openai_callback)
    if _missing("langchain_openai"):
        _install_stub("langchain_openai", ChatOpenAI=_StubChatModel)
    if _missing("langchain_ollama"):
        _install_stub("langchain_ollama", ChatOllama=_StubChatModel)
    if _missing("dotenv"):
        _install_stub("dotenv", load_dotenv=lambda *a, **k: False)


_register_optional_stubs()


class _DupStdout:
    """Stand-in for sys.stdout while a loop module is imported.

    Both loops run ``sys.stdout = open(sys.stdout.fileno(), ...)`` at import.
    Handing them a duplicate descriptor keeps pytest's capture intact when the
    replacement file object is later garbage-collected and closed.
    """

    @staticmethod
    def fileno():
        return os.dup(1)


def import_case_module(name: str):
    """Import a case-study loop module without hijacking pytest's stdout."""
    if name in sys.modules:
        return sys.modules[name]
    real_stdout = sys.stdout
    sys.stdout = _DupStdout()
    try:
        return importlib.import_module(name)
    finally:
        sys.stdout = real_stdout


def load_module_from_path(alias: str, path: Path):
    """Load a file under a unique module name (both cases ship the same names)."""
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def cstr_case():
    return import_case_module("cstr_case")


@pytest.fixture(scope="session")
def mixer_case():
    return import_case_module("mixer_case")


class FakeStructuredLLM:
    """Replays queued structured outputs in the ``include_raw=True`` shape.

    Queue a pydantic object for a parsed reply, or an exception to raise it.
    """

    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def invoke(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, dict):
            return item
        raw = types.SimpleNamespace(
            content=item.model_dump_json(), usage_metadata=None, response_metadata={}
        )
        return {"raw": raw, "parsed": item, "parsing_error": None}


class FakeChatModel:
    """Minimal chat model whose structured output comes from a queue."""

    def __init__(self, responses):
        self.structured = FakeStructuredLLM(responses)

    def with_structured_output(self, *_args, **_kwargs):
        return self.structured

    def invoke(self, prompt, *args, **kwargs):
        return self.structured.invoke(prompt, *args, **kwargs)


@pytest.fixture
def fake_llm_factory():
    return FakeChatModel
