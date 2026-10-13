"""Tests for case_studies/common/trace_logging.py."""

import csv
import hashlib
import json
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest

from common import trace_logging as tl
from common.trace_logging import TraceRecorder, _git_commit, _jsonable, _safe_name


@pytest.fixture
def no_git(monkeypatch):
    monkeypatch.setattr(tl, "_git_commit", lambda _root: "deadbeef")


def make_recorder(tmp_path, **kwargs):
    defaults = dict(
        base_dir=tmp_path,
        case_study="cstr",
        model="gpt-4o-mini",
        fault="fouling",
        run=3,
        experiment_id="exp1",
    )
    defaults.update(kwargs)
    return TraceRecorder(**defaults)


def read_events(rec):
    return [json.loads(line) for line in rec.events_path.read_text().splitlines()]


def test_utc_now_is_z_suffixed():
    assert tl._utc_now().endswith("Z")
    assert "+00:00" not in tl._utc_now()


@pytest.mark.parametrize(
    "value, expected",
    [
        ("plain", "plain"),
        ("a b/c:d", "a_b_c_d"),
        ("..hidden..", "hidden"),
        ("", "unknown"),
        ("///", "unknown"),
        ("llama3.1:8b", "llama3.1_8b"),
        (42, "42"),
    ],
)
def test_safe_name(value, expected):
    assert _safe_name(value) == expected


def test_jsonable_scalars():
    assert _jsonable(None) is None
    assert _jsonable(True) is True
    assert _jsonable("s") == "s"
    assert _jsonable(np.int64(7)) == 7 and type(_jsonable(np.int64(7))) is int
    assert type(_jsonable(np.float32(1.5))) is float
    assert _jsonable(float("nan")) == "nan"
    assert _jsonable(float("inf")) == "inf"
    assert _jsonable(np.float64("-inf")) == "-inf"
    assert _jsonable(Path("a/b")) == str(Path("a/b"))


def test_jsonable_containers_and_arrays():
    value = {1: (np.int32(1), {2.5, 2.5}), "arr": np.array([[1.0, np.nan]])}
    assert _jsonable(value) == {"1": [1, [2.5]], "arr": [[1.0, "nan"]]}


def test_jsonable_model_dump_dict_and_fallbacks():
    class Pyd2:
        def model_dump(self):
            return {"x": np.float64(1.0)}

    class Pyd1:
        def dict(self):
            return {"y": 2}

    class BrokenDump:
        def model_dump(self):
            raise RuntimeError

        def dict(self):
            raise RuntimeError

        def tolist(self):
            raise RuntimeError

        def __str__(self):
            return "broken"

    assert _jsonable(Pyd2()) == {"x": 1.0}
    assert _jsonable(Pyd1()) == {"y": 2}
    assert _jsonable(BrokenDump()) == "broken"
    assert _jsonable(object.__new__(object)).startswith("<object")


def test_git_commit_success(monkeypatch, tmp_path):
    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"], calls["cwd"] = cmd, kwargs["cwd"]
        return types.SimpleNamespace(stdout="abc123\n")

    monkeypatch.setattr(tl.subprocess, "run", fake_run)
    assert _git_commit(tmp_path) == "abc123"
    assert calls == {"cmd": ["git", "rev-parse", "HEAD"], "cwd": tmp_path}


def test_git_commit_empty_output_is_none(monkeypatch, tmp_path):
    monkeypatch.setattr(
        tl.subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout="  \n")
    )
    assert _git_commit(tmp_path) is None


def test_git_commit_failure_is_none(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(tl.subprocess, "run", boom)
    assert _git_commit(tmp_path) is None


def test_init_layout_metadata_and_first_event(tmp_path, no_git):
    rec = make_recorder(tmp_path, model="llama3:8b", metadata={"seed": np.int64(5)})
    expected_dir = tmp_path.resolve() / "cstr" / "exp1" / "llama3_8b" / "fouling"
    assert rec.run_dir == expected_dir / "run_003"
    for child in ("queries", "subgraphs", "prompts", "responses", "trajectories"):
        assert (rec.run_dir / child).is_dir()

    meta = json.loads((rec.run_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["trace_version"] == "1.0"
    assert meta["repository_commit"] == "deadbeef"
    assert meta["model"] == "llama3:8b" and meta["run"] == 3 and meta["seed"] == 5

    (event,) = read_events(rec)
    assert event["sequence"] == 1 and event["event"] == "run_started"
    assert event["metadata"]["experiment_id"] == "exp1"


def test_default_experiment_id_is_timestamp(tmp_path, no_git):
    rec = make_recorder(tmp_path, experiment_id=None)
    assert len(rec.experiment_id) == len("20260101_120000")
    assert rec.experiment_id[8] == "_"


def test_record_event_sequences_and_sanitises(tmp_path, no_git):
    rec = make_recorder(tmp_path)
    rec.record_event("custom", value=np.float64("nan"), arr=np.arange(3))
    events = read_events(rec)
    assert [e["sequence"] for e in events] == [1, 2]
    assert events[1]["value"] == "nan" and events[1]["arr"] == [0, 1, 2]


def test_record_sparql_writes_artifacts_with_hashes(tmp_path, no_git):
    rec = make_recorder(tmp_path)
    rec.record_sparql(
        name="planning query", query="SELECT *", subgraph=None, endpoint=None, loaded=1
    )
    query_file = rec.run_dir / "queries" / "planning_query.sparql"
    graph_file = rec.run_dir / "subgraphs" / "planning_query.ttl"
    assert query_file.read_text() == "SELECT *"
    assert graph_file.read_bytes() == b""

    event = read_events(rec)[-1]
    assert event["event"] == "sparql_retrieval" and event["loaded"] is True
    assert event["query_artifact"] == {
        "path": "queries/planning_query.sparql",
        "bytes": 8,
        "sha256": hashlib.sha256(b"SELECT *").hexdigest(),
    }
    assert event["subgraph_artifact"]["bytes"] == 0


def test_record_llm_call_writes_prompt_and_response(tmp_path, no_git):
    rec = make_recorder(tmp_path)
    for i in range(2):
        rec.record_llm_call(
            agent="action agent",
            iteration=i,
            messages=[{"role": "user", "content": "hi"}],
            raw_response={"text": "ok"},
            parsed_response=None,
            rationale="because",
            usage=None,
            latency_seconds=0.25,
            failure_type="timeout" if i else None,
        )
    stem = "call_0002_action_agent_iteration_0001"
    prompt = json.loads((rec.run_dir / "prompts" / f"{stem}.json").read_text())
    response = json.loads((rec.run_dir / "responses" / f"{stem}.json").read_text())
    assert prompt == {
        "agent": "action agent",
        "iteration": 1,
        "messages": [{"role": "user", "content": "hi"}],
    }
    assert response["usage"] == {} and response["failure_type"] == "timeout"
    assert response["agent_provided_rationale"] == "because"
    event = read_events(rec)[-1]
    assert event["event"] == "llm_call"
    assert event["response_artifact"]["path"] == f"responses/{stem}.json"


def test_write_trajectory_csv_and_sidecar(tmp_path, no_git):
    rec = make_recorder(tmp_path)
    rows = [
        {"t": 0, "T": np.float64(310.0), "tags": ["a"]},
        {"t": 1, "T": float("inf"), "extra": {"k": 1}},
    ]
    artifact = rec.write_trajectory(
        name="plant/main", rows=rows, accepted=1, condition="c1", metadata={"m": 2}
    )
    path = rec.run_dir / "trajectories" / "plant_main.csv"
    assert artifact["path"] == "trajectories/plant_main.csv"
    assert artifact["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == ["t", "T", "tags", "extra"]
        out = list(reader)
    assert out[0] == {"t": "0", "T": "310.0", "tags": '["a"]', "extra": ""}
    assert out[1]["T"] == "inf" and out[1]["extra"] == '{"k": 1}'

    sidecar = json.loads(
        (rec.run_dir / "trajectories" / "plant_main.metadata.json").read_text()
    )
    assert sidecar == {
        "name": "plant/main",
        "accepted": True,
        "condition": "c1",
        "rows": 2,
        "m": 2,
    }
    assert read_events(rec)[-1]["event"] == "trajectory_written"


def test_write_trajectory_empty(tmp_path, no_git):
    rec = make_recorder(tmp_path)
    assert (
        rec.write_trajectory(name="x", rows=[], accepted=False, condition="c") is None
    )
    event = read_events(rec)[-1]
    assert event["event"] == "trajectory_empty" and event["name"] == "x"
    assert not list((rec.run_dir / "trajectories").iterdir())


def test_live_rows_and_finalize_manifest(tmp_path, no_git):
    rec = make_recorder(tmp_path)
    p1 = rec.append_live_row(stream_name="cstr plant", row={"t": 1, "v": np.nan})
    p2 = rec.append_live_row(stream_name="cstr plant", row={"t": 2, "v": 1.0})
    assert p1 == p2 == rec.run_dir / "live" / "cstr_plant.jsonl"
    # Visible before finalize (line buffered + flushed).
    lines = [json.loads(x) for x in p1.read_text().splitlines()]
    assert [r["t"] for r in lines] == [1, 2] and lines[0]["v"] == "nan"
    assert "written_at_utc" in lines[0]
    started = [e for e in read_events(rec) if e["event"] == "live_stream_started"]
    assert len(started) == 1 and started[0]["path"] == "live/cstr_plant.jsonl"

    rec.finalize(result={"score": np.float32(0.5)}, status="failed")
    assert all(s.closed for s in rec._live_streams.values())

    final = json.loads((rec.run_dir / "final_result.json").read_text())
    assert final["status"] == "failed" and final["result"] == {"score": 0.5}
    assert read_events(rec)[-1]["event"] == "run_finished"

    manifest = json.loads((rec.run_dir / "manifest.json").read_text())
    assert manifest["run_directory"] == "cstr/exp1/gpt-4o-mini/fouling/run_003"
    paths = [a["path"] for a in manifest["artifacts"]]
    assert manifest["artifact_count"] == len(paths)
    assert paths == [
        "metadata.json",
        "final_result.json",
        "live/cstr_plant.jsonl",
        "events.jsonl",
    ]
    events_entry = manifest["artifacts"][-1]
    assert (
        events_entry["sha256"]
        == hashlib.sha256(rec.events_path.read_bytes()).hexdigest()
    )


def test_finalize_skips_deleted_live_file(tmp_path, no_git):
    rec = make_recorder(tmp_path)
    path = rec.append_live_row(stream_name="s", row={"a": 1})
    rec._live_streams[path].close()
    path.unlink()
    rec.finalize(result=None)
    manifest = json.loads((rec.run_dir / "manifest.json").read_text())
    assert "live/s.jsonl" not in [a["path"] for a in manifest["artifacts"]]
