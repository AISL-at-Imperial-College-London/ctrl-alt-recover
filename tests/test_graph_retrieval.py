"""Tests for the two graph_retrieval_code.py modules (CSTR and mixer)."""

import runpy
import types

import pytest
import requests

from conftest import CASE_STUDIES, load_module_from_path


@pytest.fixture(
    scope="module",
    params=[("cstr", "mixerModuleImperial>"), ("mixer", "mixerModuleImperial_new>")],
    ids=["cstr", "mixer"],
)
def gr(request):
    case, graph = request.param
    module = load_module_from_path(
        f"graph_retrieval_{case}",
        CASE_STUDIES / f"{case}_case" / "graph_retrieval_code.py",
    )
    module.expected_graph = graph
    return module


def test_query_constants(gr):
    assert gr.ENDPOINT.startswith("http://localhost:7200/repositories/")
    for query in (gr.QUERY_PLANNING, gr.QUERY_ACTION, gr.QUERY_CSTR):
        assert query == query.strip()
        assert "CONSTRUCT {" in query and "WHERE {" in query
        assert query.count("{") == query.count("}")
    assert gr.QUERY_PLANNING.startswith("PREFIX :")
    assert "rdfs:comment ?trComment" in gr.QUERY_PLANNING
    assert "CPSMod:isChangedByActuator" in gr.QUERY_ACTION
    assert "FROM <http://www.hsu-ifa.de/graphs/CSTR_Simulation>" in gr.QUERY_CSTR
    assert f"GRAPH <http://www.hsu-ifa.de/graphs/{gr.expected_graph}" in (
        gr.QUERY_PLANNING
    )
    assert gr.QUERY_ACTION.count(gr.expected_graph) == 1


def test_run_construct_success(gr, monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return types.SimpleNamespace(status_code=200, text="@prefix : <x> .")

    monkeypatch.setattr(gr.requests, "post", fake_post)
    assert gr.run_construct("CONSTRUCT {}") == "@prefix : <x> ."
    assert calls == [
        (
            gr.ENDPOINT,
            {
                "data": {"query": "CONSTRUCT {}"},
                "headers": {"Accept": "text/turtle"},
                "timeout": 30,
            },
        )
    ]


def test_run_construct_http_error_exits_1(gr, monkeypatch, capsys):
    monkeypatch.setattr(
        gr.requests,
        "post",
        lambda *a, **k: types.SimpleNamespace(status_code=500, text="boom"),
    )
    with pytest.raises(SystemExit) as info:
        gr.run_construct("q")
    assert info.value.code == 1
    err = capsys.readouterr().err
    assert "HTTP 500" in err and "boom" in err


def test_run_construct_request_exception_exits_2(gr, monkeypatch, capsys):
    def fail(*a, **k):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(gr.requests, "post", fail)
    with pytest.raises(SystemExit) as info:
        gr.run_construct("q")
    assert info.value.code == 2
    assert "Could not reach the endpoint: refused" in capsys.readouterr().err


@pytest.mark.parametrize("case", ["cstr", "mixer"])
def test_main_prints_cstr_query_result(case, monkeypatch, capsys):
    sent = []

    def fake_post(url, data, **kwargs):
        sent.append(data["query"])
        return types.SimpleNamespace(status_code=200, text="TTL-OUT")

    monkeypatch.setattr(requests, "post", fake_post)
    path = CASE_STUDIES / f"{case}_case" / "graph_retrieval_code.py"
    namespace = runpy.run_path(str(path), run_name="__main__")
    assert sent == [namespace["QUERY_CSTR"]]
    assert capsys.readouterr().out == "TTL-OUT\n"
