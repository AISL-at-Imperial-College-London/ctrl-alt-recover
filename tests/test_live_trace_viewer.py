"""Tests for the read-only CSTR live dashboard (Agg backend, no blocking calls)."""

import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import live_trace_viewer as ltv


@pytest.fixture(autouse=True)
def isolated_matplotlib():
    # CstrDashboard calls plt.style.use(), which mutates global rcParams.
    with matplotlib.rc_context():
        yield
    plt.close("all")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def make_run(root, name, *, live="cstr_plant.jsonl", mtime_ns=None):
    run = root / name
    run.mkdir(parents=True)
    meta = run / "metadata.json"
    meta.write_text(json.dumps({"experiment_id": name, "model": "m", "fault": "f"}))
    if live:
        write_jsonl(run / "live" / live, [])
    if mtime_ns is not None:
        os.utime(meta, ns=(mtime_ns, mtime_ns))
    return run


def dashboard(tmp_path, run_dir=None, window=10, refresh_ms=10):
    return ltv.CstrDashboard(
        trace_root=tmp_path, run_dir=run_dir, window=window, refresh_ms=refresh_ms
    )


# --- JsonlTail ----------------------------------------------------------------


def test_tail_missing_file(tmp_path):
    assert ltv.JsonlTail(tmp_path / "nope.jsonl").read_new() == []


def test_tail_incremental_partial_and_invalid_lines(tmp_path):
    path = tmp_path / "s.jsonl"
    tail = ltv.JsonlTail(path)
    path.write_bytes(b'{"a": 1}\nnot json\n[1, 2]\n\xff\xfe\n{"b": 2')
    assert tail.read_new() == [{"a": 1}]
    held = tail.offset
    assert tail.read_new() == [] and tail.offset == held  # partial line waits
    with path.open("ab") as fh:
        fh.write(b'}\n{"c": 3}\n')
    assert tail.read_new() == [{"b": 2}, {"c": 3}]
    assert tail.read_new() == []


def test_tail_restarts_after_truncation(tmp_path):
    path = tmp_path / "s.jsonl"
    write_jsonl(path, [{"a": 1}, {"a": 2}])
    tail = ltv.JsonlTail(path)
    tail.read_new()
    path.write_text('{"z": 9}\n')
    assert tail.read_new() == [{"z": 9}]


# --- small helpers -------------------------------------------------------------


def test_number_and_series():
    row = {"t": "nan", "time": "3.5", "bad": "x", "inf": float("inf")}
    assert ltv._number(row, "t", "time") == 3.5
    assert np.isnan(ltv._number(row, "bad", "inf", "missing"))
    series = ltv._series([{"t": 1}, {"time": 2}, {}], "t", "time")
    assert series.dtype == float
    np.testing.assert_array_equal(series, [1.0, 2.0, np.nan])


def test_latest_run(tmp_path):
    assert ltv._latest_run(tmp_path / "missing") is None
    make_run(tmp_path, "no_live", live=None, mtime_ns=3 * 10**18)
    assert ltv._latest_run(tmp_path) is None
    make_run(tmp_path, "old", mtime_ns=10**18)
    newer = make_run(
        tmp_path, "newer", live="cstr_validation.jsonl", mtime_ns=2 * 10**18
    )
    assert ltv._latest_run(tmp_path) == newer


@pytest.mark.parametrize(
    "content, expected",
    [('{"model": "m"}', {"model": "m"}), ("[1]", {}), ("{bad", {}), (None, {})],
)
def test_read_metadata(tmp_path, content, expected):
    if content is not None:
        (tmp_path / "metadata.json").write_text(content, encoding="utf-8")
    assert ltv._read_metadata(tmp_path) == expected


@pytest.mark.parametrize(
    "event, text",
    [
        (
            {"event": "sparql_retrieval", "name": "q", "loaded": True},
            "KG read: q (loaded=True)",
        ),
        (
            {"event": "llm_call", "agent": "action", "iteration": 2},
            "LLM read/write: action call, iteration 2",
        ),
        ({"event": "llm_call"}, "LLM read/write: ? call, iteration ?"),
        (
            {"event": "validation_decision", "accepted": True, "summary": "ok"},
            "Validation ACCEPTED: ok",
        ),
        (
            {"event": "validation_decision", "fail_reason": "unsafe"},
            "Validation REJECTED: unsafe",
        ),
        (
            {"event": "trajectory_written", "name": "plant", "rows": 5},
            "Trajectory written: plant (5 rows)",
        ),
        ({"event": "live_stream_started", "stream": "s"}, "Live stream opened: s"),
        ({"event": "run_finished", "status": "completed"}, "Run finished: completed"),
        ({"event": "run_started"}, "run started"),
        ({}, "event"),
    ],
)
def test_event_text(event, text):
    assert ltv._event_text(event) == text


# --- dashboard -----------------------------------------------------------------


def test_dashboard_init_clamps_and_axes(tmp_path):
    dash = dashboard(tmp_path, window=1, refresh_ms=1)
    assert (dash.window, dash.refresh_ms) == (50, 100)
    assert dash.fixed_run_dir is None
    assert len(dash.fig.axes) == 6
    assert dash.fig.canvas.manager.get_window_title() == "CSTR live trace"


def test_update_without_any_run_shows_waiting(tmp_path):
    dash = dashboard(tmp_path)
    dash.update(0)
    assert dash.run_dir is None
    assert "Waiting for live plant data" in dash.temp_ax.texts[0].get_text()
    assert dash.validation_ax.texts[0].get_text() == "No validation rollout yet"
    activity = dash.activity_ax.texts[0].get_text()
    assert "Waiting for a CSTR trace" in activity and str(tmp_path) in activity
    assert dash.fig._suptitle is None


def test_discovery_is_throttled(tmp_path, monkeypatch):
    clock = iter([100.0, 101.0, 103.0])
    monkeypatch.setattr(ltv.time, "monotonic", lambda: next(clock))
    dash = dashboard(tmp_path)
    dash._select_run()  # t=100: nothing yet, next check at 102
    run = make_run(tmp_path, "r1")
    dash._select_run()  # t=101: throttled
    assert dash.run_dir is None
    dash._select_run()  # t=103: discovers
    assert dash.run_dir == run and dash.metadata["experiment_id"] == "r1"


def test_switching_runs_resets_buffers(tmp_path, monkeypatch):
    monkeypatch.setattr(ltv.time, "monotonic", iter([0.0, 10.0]).__next__)
    first = make_run(tmp_path, "first", mtime_ns=10**18)
    write_jsonl(first / "live" / "cstr_plant.jsonl", [{"t": 1}])
    dash = dashboard(tmp_path)
    dash._read()
    assert dash.plant_rows == [{"t": 1}]
    second = make_run(tmp_path, "second", mtime_ns=2 * 10**18)
    dash._read()
    assert dash.run_dir == second and dash.plant_rows == []
    dash.fixed_run_dir = second
    dash._select_run()  # same run: no reset
    assert dash.run_dir == second


@pytest.fixture
def populated_run(tmp_path):
    run = tmp_path / "exp" / "run_001"
    run.mkdir(parents=True)
    (run / "metadata.json").write_text(
        json.dumps({"experiment_id": "E1", "model": "gpt", "fault": "foul", "run": 1})
    )
    plant = [
        {
            "t": float(i),
            "T_meas": 310.0 + i,
            "T_true": 310.0,
            "T_sp": 310.0,
            "L_true": 9.0,
            "L_meas": 9.1,
            "L_sp": 10.0,
            "Fin": 0.03,
            "Fout": 0.03,
            "Fin_sp": 0.033,
            "u_valve": 0.5,
            "u_pump": 0.4,
            "u_cool": 0.3,
            "phase": "NORMAL",
            "control_zone": "UNSAFE" if i == 3 else "SAFE",
            "fault_tags": "FOUL",
        }
        for i in range(5)
    ]
    write_jsonl(run / "live" / "cstr_plant.jsonl", plant)
    validation = [
        {"action_call": 1, "t": 0.0, "T_meas": 300.0, "proposed_T_sp": 305.0},
        {"action_call": 2, "t": 0.0, "T_meas": 320.0, "proposed_T_sp": 305.0},
        {
            "action_call": 2,
            "t": 1.0,
            "T_meas": 330.0,
            "proposed_T_sp": 305.0,
            "control_zone": "UNSAFE",
        },
    ]
    write_jsonl(run / "live" / "cstr_validation.jsonl", validation)
    write_jsonl(
        run / "events.jsonl",
        [{"event": "run_started"}] + [{"event": "llm_call", "agent": "a"}] * 12,
    )
    return run


def test_update_with_fixed_run_draws_everything(tmp_path, populated_run):
    dash = dashboard(tmp_path, run_dir=populated_run)
    dash.update(0)
    assert dash.last_read == {"plant": 5, "validation": 3, "events": 13}

    temp_lines = {line.get_label(): line for line in dash.temp_ax.get_lines()}
    assert set(temp_lines) == {"T measured", "T true", "T setpoint"}
    np.testing.assert_array_equal(
        temp_lines["T measured"].get_ydata(), 310 + np.arange(5)
    )
    unsafe = dash.temp_ax.collections[0].get_offsets()
    np.testing.assert_array_equal(unsafe, [[3.0, 313.0]])
    assert dash.temp_ax.get_title() == "Plant temperature"
    assert dash.temp_ax.get_legend() is not None
    assert dash.actuator_ax.get_ylim() == (-0.05, 1.05)
    assert [l.get_label() for l in dash.flow_ax.get_lines()] == [
        "Fin",
        "Fout",
        "Fin setpoint",
    ]

    # Only the latest action call is plotted.
    assert dash.validation_ax.get_title().endswith("(action call 2)")
    t_meas, _proposed = dash.validation_ax.get_lines()
    np.testing.assert_array_equal(t_meas.get_ydata(), [320.0, 330.0])
    np.testing.assert_array_equal(
        dash.validation_ax.collections[0].get_offsets(), [[1.0, 330.0]]
    )

    activity = dash.activity_ax.texts[0].get_text()
    assert "Status: RUNNING" in activity
    assert "Experiment: E1" in activity and "Time: 4.0 s" in activity
    assert "Zone: SAFE" in activity and "Fault tag: FOUL" in activity
    assert activity.count("• ") == 9  # last nine events only
    assert dash.fig._suptitle.get_text() == "CSTR live trace — E1 / gpt / foul / run 1"

    # Second refresh reads nothing new; completion is picked up.
    (populated_run / "final_result.json").write_text("{}")
    dash.update(1)
    assert dash.last_read == {"plant": 0, "validation": 0, "events": 0}
    assert "Status: COMPLETE" in dash.activity_ax.texts[0].get_text()


def test_validation_without_action_call(tmp_path, populated_run):
    path = populated_run / "live" / "cstr_validation.jsonl"
    path.write_text(json.dumps({"t": 0.0, "T_meas": 300.0}) + "\n")
    dash = dashboard(tmp_path, run_dir=populated_run)
    dash.update(0)
    assert "no action-call identifier" in dash.validation_ax.texts[0].get_text()


def test_activity_without_plant_rows(tmp_path):
    run = make_run(tmp_path, "r", live="cstr_validation.jsonl")
    dash = dashboard(tmp_path, run_dir=run)
    dash.update(0)
    activity = dash.activity_ax.texts[0].get_text()
    assert "Time: nan s" in activity and "Phase: ?" in activity
    assert "Fault tag: ?" in activity


def test_activity_uses_legacy_fault_tag_key(tmp_path):
    run = make_run(tmp_path, "r")
    write_jsonl(run / "live" / "cstr_plant.jsonl", [{"time": 2, "fault_tag": "LEAK"}])
    dash = dashboard(tmp_path, run_dir=run)
    dash.update(0)
    assert "Fault tag: LEAK" in dash.activity_ax.texts[0].get_text()
    assert len(dash.temp_ax.collections) == 0  # no UNSAFE rows


def test_read_trims_buffers(tmp_path, populated_run):
    dash = dashboard(tmp_path, run_dir=populated_run, window=50)
    dash._select_run()
    dash.plant_rows = [{"t": i} for i in range(10_000)]
    dash.events = [{"event": "x"}] * 200
    dash._read()
    assert len(dash.plant_rows) == 10_000
    assert dash.plant_rows[-1]["t"] == 4.0 and dash.plant_rows[0] == {"t": 5}
    assert len(dash.events) == 100


def test_plant_window_limits_points(tmp_path, populated_run):
    dash = dashboard(tmp_path, run_dir=populated_run, window=50)
    dash._read()
    dash.plant_rows = [{"t": i, "T_meas": i} for i in range(80)]
    dash._draw_plant()
    xdata = dash.temp_ax.get_lines()[0].get_xdata()
    assert len(xdata) == 50 and xdata[0] == 30


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="live_trace_viewer.py:319 calls int(nan) for validation rows without "
    "action_call, crashing the dashboard instead of skipping them",
)
def test_validation_rows_missing_action_call_are_skipped(tmp_path, populated_run):
    write_jsonl(populated_run / "live" / "cstr_validation.jsonl", [{"t": 2.0}])
    dash = dashboard(tmp_path, run_dir=populated_run)
    dash.update(0)
    assert dash.validation_ax.get_title().endswith("(action call 2)")


def test_show_starts_animation_without_blocking(tmp_path, monkeypatch):
    created, shown = [], []

    def fake_animation(fig, func, **kwargs):
        created.append((fig, func, kwargs))
        return "anim"

    monkeypatch.setattr(ltv.animation, "FuncAnimation", fake_animation)
    monkeypatch.setattr(ltv.plt, "show", lambda: shown.append(True))
    dash = dashboard(tmp_path, refresh_ms=250)
    dash.show()
    assert dash.animation == "anim" and shown == [True]
    ((fig, func, kwargs),) = created
    assert fig is dash.fig and func == dash.update
    assert kwargs == {"interval": 250, "cache_frame_data": False}


def test_main_parses_arguments(tmp_path, monkeypatch):
    captured = {}

    def fake_show(self):
        captured.update(
            root=self.trace_root,
            run=self.fixed_run_dir,
            window=self.window,
            refresh=self.refresh_ms,
        )

    monkeypatch.setattr(ltv.CstrDashboard, "show", fake_show)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "live_trace_viewer.py",
            "--trace-root",
            str(tmp_path),
            "--run-dir",
            str(tmp_path / "r"),
            "--window",
            "300",
            "--refresh-ms",
            "750",
        ],
    )
    ltv.main()
    assert captured == {
        "root": tmp_path.resolve(),
        "run": (tmp_path / "r").resolve(),
        "window": 300,
        "refresh": 750,
    }


def test_main_defaults(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        ltv.CstrDashboard, "show", lambda self: captured.update(vars(self))
    )
    monkeypatch.setattr(sys, "argv", ["live_trace_viewer.py"])
    ltv.main()
    assert captured["trace_root"] == ltv.DEFAULT_TRACE_ROOT.resolve()
    assert captured["fixed_run_dir"] is None
    assert (captured["window"], captured["refresh_ms"]) == (1200, 500)
    assert ltv.DEFAULT_TRACE_ROOT == ltv.REPO_ROOT / "traces" / "cstr"


def test_missing_style_is_tolerated(tmp_path, monkeypatch):
    def no_style(_name):
        raise OSError("style not found")

    monkeypatch.setattr(ltv.plt.style, "use", no_style)
    assert len(dashboard(tmp_path).fig.axes) == 6
