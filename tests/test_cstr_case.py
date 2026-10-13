"""Offline tests for the CSTR monitoring/action loop (cstr_case.py).

No LLM, GraphDB or network: the LLM factory, SPARQL loader and LangGraph are
replaced with in-file fakes, and every file the loop writes goes to tmp_path.
"""

import argparse
import contextlib
import copy
import json
import sys
import types

import numpy as np
import pytest
from pydantic import ValidationError

from cstr_digital_twin import CSTRSimulation, FaultConfig
from common.llm_failures import LLMActionError

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def cc(cstr_case, monkeypatch):
    # Every test starts with no recorder and no KG, restored afterwards.
    monkeypatch.setattr(cstr_case, "_TRACE_RECORDER", None)
    monkeypatch.setattr(cstr_case, "_KG_LOADED", False)
    monkeypatch.setattr(cstr_case, "_KG_ACTION_TTL", "")
    return cstr_case


class SpyRecorder:
    """Records every TraceRecorder method call as (name, kwargs)."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, kwargs))

        return record

    def named(self, name):
        return [kw for n, kw in self.calls if n == name]


class RunnableStateGraph:
    """StateGraph stand-in that can also run the wired graph."""

    START = "__start__"
    END = "__end__"

    def __init__(self, schema):
        self.schema = schema
        self.nodes = {}
        self.edges = {}
        self.cond = {}

    def add_node(self, name, func):
        self.nodes[name] = func

    def add_edge(self, start, end):
        self.edges[start] = end

    def add_conditional_edges(self, source, route, targets):
        self.cond[source] = (route, list(targets))

    def compile(self, **_kwargs):
        return self

    def invoke(self, state, config=None):
        state = dict(state)
        node = self.edges[self.START]
        for _ in range(config["recursion_limit"]):
            if node == self.END:
                return state
            state = self.nodes[node](state)
            if node in self.cond:
                route, targets = self.cond[node]
                nxt = route(state)
                assert nxt in targets
                node = nxt
            else:
                node = self.edges[node]
        raise RecursionError("graph did not terminate")


@pytest.fixture
def runnable_graph(cc, monkeypatch):
    monkeypatch.setattr(cc, "StateGraph", RunnableStateGraph)
    monkeypatch.setattr(cc, "START", RunnableStateGraph.START)
    monkeypatch.setattr(cc, "END", RunnableStateGraph.END)
    return cc.build_graph()


def sim_row(t, *, T=310.0, L=10.0, u=0.5, phase="NORMAL", **extra):
    row = {
        "t": t,
        "phase": phase,
        "T_meas": T,
        "L_meas": L,
        "L_true": L,
        "Fin": 0.033,
        "Fout": 0.033,
        "u_valve": u,
        "u_pump": u,
        "u_cool": u,
        "F_over": 0.0,
    }
    row.update(extra)
    return row


class ScriptedPlant:
    """Twin replacement replaying fixed simulate_step outputs."""

    def __init__(self, rows, dt=1.0):
        self.rows = list(rows)
        self.dt = dt
        self.i = 0
        self.kwargs = []

    def simulate_step(self, **kwargs):
        self.kwargs.append(kwargs)
        row = self.rows[min(self.i, len(self.rows) - 1)]
        self.i += 1
        return dict(row)


@pytest.fixture(scope="module")
def warm_plant():
    """Plant settled in NORMAL phase at t=450 (own RNG, so deterministic)."""
    plant = CSTRSimulation(seed=7, isolated_rng=True)
    for _ in range(450):
        out = plant.simulate_step()
    assert out["phase"] == "NORMAL"
    return plant, out


class StructuredFailsModel:
    """Chat model without structured output; plain invoke returns `reply`."""

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def with_structured_output(self, *_a, **_k):
        raise NotImplementedError("no structured output")

    def invoke(self, prompt, *_a, **_k):
        self.prompts.append(prompt)
        if isinstance(self.reply, BaseException):
            raise self.reply
        return types.SimpleNamespace(content=self.reply)


def propose_state(**overrides):
    state = {
        "sim_last": sim_row(1200.0, T=311.5, L=10.1, u=0.6),
        "T_sp": 310.0,
        "L_sp": 10.0,
        "Fin_sp": 2.0 / 60.0,
        "reprompt_count": 0,
        "llm_model": "gpt-4o-mini",
        "eval_feedback": "",
        "symptoms": {"T_err": 1.5},
    }
    state.update(overrides)
    return state


def plan(cc, T=309.5, L=10.0, Fin=0.03, why="PO8 reduces load"):
    return cc.SetpointPlan(T_sp=T, L_sp=L, Fin_sp=Fin, reasoning=why)


@contextlib.contextmanager
def fake_callback():
    yield types.SimpleNamespace(prompt_tokens=11, completion_tokens=5, total_tokens=16)


# ---------------------------------------------------------------------------
# fault scenarios, row mapping, fallback, symptoms
# ---------------------------------------------------------------------------

FAULT_FLAGS = {
    "fouling": ("fouling_on", "fouling_t_start", True),
    "pump_degrade": ("pump_degrade_on", "pump_degrade_t", True),
    "cool_stuck_open": ("cool_stuck_open", "cool_stuck_open_t", True),
    "cool_stuck_closed": ("cool_stuck_closed", "cool_stuck_closed_t", True),
    "outlet_block": ("outlet_block_on", "outlet_block_t", False),
    "inlet_stuck_open": ("inlet_stuck_open", "inlet_stuck_open_t", False),
    "temp_bias": ("temp_bias_on", "temp_bias_t", False),
    "level_bias": ("level_bias_on", "level_bias_t", False),
    "leak": ("leak_on", "leak_t", False),
}
ALL_FLAGS = [flag for flag, _, _ in FAULT_FLAGS.values()]


@pytest.mark.parametrize("name", sorted(FAULT_FLAGS))
def test_fault_cfg_enables_exactly_one_fault(cc, name):
    flag, t_attr, jittered = FAULT_FLAGS[name]
    cfg = cc.fault_cfg_from_name(f"  {name.upper()} ")
    assert cfg.enable
    assert [f for f in ALL_FLAGS if getattr(cfg, f)] == [flag]
    t = getattr(cfg, t_attr)
    if jittered:
        assert 1800.0 <= t <= 2200.0
    else:
        assert t == 2000.0


def test_fault_cfg_normal_and_unknown(cc):
    assert cc.fault_cfg_from_name("Normal").enable is False
    with pytest.raises(ValueError, match="Unknown fault 'bogus'"):
        cc.fault_cfg_from_name("bogus")


def test_row_from_sim_maps_phase_and_errors(cc):
    state = {"T_sp": 310.0, "L_sp": 10.0}
    out = sim_row(12.0, T=312.5, L=9.5, phase="shutdown", Fin=0.05, Fout=0.03)
    row = cc.row_from_sim(out, state)
    assert row["phase"] == 3
    assert row["time"] == 12.0
    assert row["T_meas_error"] == pytest.approx(2.5)
    assert row["L_meas_error"] == pytest.approx(-0.5)
    assert row["mass_balance"] == pytest.approx(0.02)
    assert row["F_leak"] == 0.0

    empty = cc.row_from_sim({"phase": "weird"}, state)
    assert empty["phase"] == 2
    assert np.isnan(empty["T_meas"]) and np.isnan(empty["mass_balance"])
    assert cc.row_from_sim({"phase": "STARTUP"}, state)["phase"] == 1


def test_fallback_policy(cc):
    assert cc.fallback_policy({}) == {
        "T_sp": 310.0,
        "L_sp": 10.0,
        "Fin_sp": pytest.approx(1.0 / 60.0),
    }
    far = cc.fallback_policy({"T_sp": 340.0, "L_sp": 8.0, "Fin_sp": 0.01})
    assert far == {"T_sp": 340.0, "L_sp": 8.0, "Fin_sp": pytest.approx(0.5 / 60.0)}
    near = cc.fallback_policy({"T_sp": 315.0, "Fin_sp": 0.1})
    assert near["T_sp"] == 310.0 and near["Fin_sp"] == pytest.approx(0.05)


def test_compute_symptoms_flags(cc):
    hot = {"T_meas": 313.0, "L_meas": 9.0, "Fin": 0.1, "Fout": 0.05}
    hot.update(u_cool=0.01, u_valve=0.96, u_pump=0.97)
    sym = cc.compute_symptoms(hot, {"T_sp": 310.0, "L_sp": 10.0})
    assert sym["T_err"] == pytest.approx(3.0)
    assert sym["L_err"] == pytest.approx(-1.0)
    assert sym["cool_low"] and sym["cooling_ineffective"]
    assert sym["valve_high"] and sym["pump_high"] and sym["mass_imbalance"]
    assert not sym["cool_high"] and not sym["cooling_overactive"]

    cold = {"T_meas": 308.0, "u_cool": 0.99}
    sym = cc.compute_symptoms(cold, {})
    assert sym["cool_high"] and sym["cooling_overactive"]
    assert sym["L_err"] is None and not sym["mass_imbalance"]

    blank = cc.compute_symptoms({}, {})
    assert blank["T_err"] is None
    assert not any(v for k, v in blank.items() if k not in ("T_err", "L_err"))


# ---------------------------------------------------------------------------
# plant cloning
# ---------------------------------------------------------------------------


def test_clone_plant_obj_is_independent(cc):
    plant = CSTRSimulation(seed=1, isolated_rng=True)
    plant.simulate_step()
    twin = cc.clone_plant_obj(plant)
    assert twin is not plant and twin.t == plant.t
    twin.simulate_step()
    assert plant.t == 1.0 and twin.t == 2.0


def test_clone_plant_obj_rebuilds_when_deepcopy_fails(cc, monkeypatch):
    plant = CSTRSimulation(seed=1, isolated_rng=True, T_sp=305.0)
    plant.simulate_step()
    real = copy.deepcopy

    def picky(obj, *a):
        if isinstance(obj, CSTRSimulation):
            raise TypeError("cannot pickle")
        return real(obj, *a)

    monkeypatch.setattr(cc.copy, "deepcopy", picky)
    twin = cc.clone_plant_obj(plant)
    assert isinstance(twin, CSTRSimulation)
    assert twin.T_sp == 305.0 and twin.t == 1.0
    assert twin.x is not plant.x  # __dict__ was deep-copied


def test_clone_plant_obj_shallow_fallback(cc, monkeypatch):
    plant = CSTRSimulation(seed=1, isolated_rng=True)

    def broken(*_a):
        raise TypeError("nope")

    monkeypatch.setattr(cc.copy, "deepcopy", broken)
    twin = cc.clone_plant_obj(plant)
    assert twin is not plant
    assert twin.x is plant.x  # shallow dict copy shares the arrays


# ---------------------------------------------------------------------------
# routers, check_end, reprompting, action_apply
# ---------------------------------------------------------------------------


def test_route_after_monitoring_and_check_end(cc):
    assert cc.route_after_monitoring({"should_act": True}) == "action_propose"
    assert cc.route_after_monitoring({}) == "check_end"
    assert cc.route_after_check_end({"abort_reason": "max_steps"}) == cc.END
    assert cc.route_after_check_end({"abort_reason": ""}) == "plant"


def test_route_after_evaluation(cc):
    assert cc.route_after_evaluation({"eval_pass": True}) == "action_apply"
    state = {"eval_pass": False, "reprompt_count": 1, "reprompt_max": 2}
    assert cc.route_after_evaluation(state) == "reprompting"

    state = {"eval_pass": False, "reprompt_count": 2, "reprompt_max": 2}
    state.update(T_sp=310.0, L_sp=9.0, Fin_sp=0.04, eval_metrics={"x": 1})
    assert cc.route_after_evaluation(state) == "action_apply"
    assert state["proposed_sp"] == cc.fallback_policy(state)
    assert state["eval_pass"] is True
    assert state["eval_summary"].startswith("PASS(fallback)")
    assert state["eval_metrics"] == {}


def test_check_end(cc):
    state = {"itr": 5, "max_steps": 5, "sim_last": {"t": 1.0}}
    assert cc.check_end(state)["abort_reason"] == "max_steps"

    state = {"itr": 1, "max_steps": 5, "sim_last": {"t": 700.0}, "T_end": 700.0}
    assert cc.check_end(state)["abort_reason"] == "t_end_reached"

    state = {"itr": 1, "max_steps": 5, "sim_last": {"t": 699.0}, "T_end": 700.0}
    assert "abort_reason" not in cc.check_end(state)
    state = {"itr": 1, "max_steps": 5, "sim_last": {"t": float("nan")}}
    assert "abort_reason" not in cc.check_end(state)


@pytest.mark.parametrize(
    "reason, hint",
    [
        ("no_recovery", "Did not reach safe zone"),
        ("slow_recovery", "time_to_safe=42"),
        ("unsafe_fraction", "Too much time spent unsafe"),
        ("invalid_actuators", "Actuator invalid"),
        ("other", "hint="),
    ],
)
def test_reprompting_feedback(cc, reason, hint):
    state = {
        "reprompt_count": 1,
        "eval_fail_reason": reason,
        "eval_summary": "FAIL: x",
        "eval_metrics": {"time_to_safe": 42},
    }
    cc.reprompting(state)
    assert state["reprompt_count"] == 2
    assert state["eval_feedback"].startswith(f"FAIL: x; fail_reason={reason}")
    assert hint in state["eval_feedback"]
    if reason == "other":
        assert state["eval_feedback"].endswith("hint=")


def test_action_apply_uses_proposal_or_fallback(cc):
    state = {
        "proposed_sp": {"T_sp": 309.0, "L_sp": 9.5, "Fin_sp": 0.02},
        "should_act": True,
        "sim_last": {"t": 1234.0},
    }
    cc.action_apply(state)
    assert (state["T_sp"], state["L_sp"], state["Fin_sp"]) == (309.0, 9.5, 0.02)
    assert state["sp_change_done"] and not state["should_act"]
    assert state["t_sp_change"] == 1234.0

    state = {"proposed_sp": {}, "T_sp": 310.0, "L_sp": 10.0, "Fin_sp": 0.04}
    cc.action_apply(state)
    assert state["Fin_sp"] == pytest.approx(0.02)
    assert np.isnan(state["t_sp_change"])


# ---------------------------------------------------------------------------
# safety checkers
# ---------------------------------------------------------------------------


def test_safe_state_checker_boundaries(cc):
    sc = cc.SafeStateChecker(safe_window=2, recovery_window=2)
    r = sc.update(T_meas=313.0, L_true=11.0, T_sp=313.0, L_sp=11.0)
    assert r == {
        "safe_now": True,
        "recovered_now": True,
        "safe_persist": True,
        "recovered_persist": True,
    }
    r = sc.update(T_meas=313.01, L_true=10.0, T_sp=313.0, L_sp=10.0)
    assert not r["safe_now"] and not r["recovered_now"]
    assert not r["safe_persist"]  # 1/2 < safe_ratio=1.0
    assert not r["recovered_persist"]  # 1/2 < 0.9


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(T_meas=310.0, L_true=1.99),
        dict(T_meas=310.0, L_true=11.01),
        dict(T_meas=310.0, L_true=10.0, F_over=1e-6),
        dict(T_meas=float("nan"), L_true=10.0),
        dict(T_meas=310.0, L_true=float("inf")),
    ],
)
def test_safe_state_checker_unsafe_cases(cc, kwargs):
    r = cc.SafeStateChecker().update(T_sp=310.0, L_sp=10.0, **kwargs)
    assert not r["safe_now"] and not r["recovered_now"]


def test_safe_state_checker_recovery_bands(cc):
    sc = cc.SafeStateChecker()
    assert sc.update(T_meas=311.0, L_true=2.3, T_sp=310.0, L_sp=2.0)["recovered_now"]
    r = sc.update(T_meas=311.1, L_true=10.0, T_sp=310.0, L_sp=10.0)
    assert r["safe_now"] and not r["recovered_now"]
    r = sc.update(
        T_meas=310.0, L_true=10.0, T_sp=float("nan"), L_sp=10.0, F_over=np.nan
    )
    assert r["safe_now"] and not r["recovered_now"]


def test_control_safety_config_defaults(cc):
    cfg = cc.ControlSafetyConfig()
    assert (cfg.T_target, cfg.sat_threshold, cfg.hard_sat) == (310.0, 0.95, 0.99)
    assert (cfg.trend_eps, cfg.T_band_normal, cfg.T_band_when_sat) == (0.002, 1.0, 0.5)


def acts(u_valve=0.5, u_pump=0.5, u_cool=0.5, T=310.0):
    return {"T_meas": T, "u_valve": u_valve, "u_pump": u_pump, "u_cool": u_cool}


def test_control_safety_safe_and_tracking(cc):
    cs = cc.ControlSafetyChecker()
    r = cs.update(out=acts(T=312.0), T_sp=311.0)
    assert r == {
        "control_zone": "SAFE",
        "control_reasons": "",
        "actuator_valid": True,
        "actuator_saturated": False,
        "actuator_trending_to_sat": False,
        "T_target": 310.0,
    }
    # Not saturated: an off-setpoint temperature does not change the zone.
    assert cs.update(out=acts(T=320.0), T_sp=310.0)["control_zone"] == "SAFE"
    assert cs.update(out={}, T_sp=310.0)["control_zone"] == "UNSAFE"


def test_control_safety_invalid_actuators(cc):
    r = cc.ControlSafetyChecker().update(
        out=acts(u_valve=-0.1, u_pump=1.01, u_cool=np.nan), T_sp=310.0
    )
    assert r["control_zone"] == "UNSAFE" and not r["actuator_valid"]
    assert "invalid_actuators=['u_valve', 'u_pump', 'u_cool']" in r["control_reasons"]
    # u_pump=1.01 is finite, so it also counts as (hard) saturated.
    assert "hard_sat=['u_pump']" in r["control_reasons"]


def test_control_safety_saturation_zones(cc):
    cs = cc.ControlSafetyChecker()
    r = cs.update(out=acts(u_cool=0.95, T=310.5), T_sp=305.0)
    assert r["control_zone"] == "WARNING" and r["actuator_saturated"]
    assert r["control_reasons"] == "saturated=['u_cool']; at_limit_but_holding_T"

    r = cc.ControlSafetyChecker().update(out=acts(u_cool=0.99, T=310.6), T_sp=310.6)
    assert r["control_zone"] == "UNSAFE"
    assert "hard_sat=['u_cool']" in r["control_reasons"]
    assert "T_not_close_to_310 err=0.600K" in r["control_reasons"]

    r = cc.ControlSafetyChecker().update(out=acts(u_cool=0.96, T=np.nan), T_sp=310.0)
    assert r["control_zone"] == "UNSAFE" and "err=infK" in r["control_reasons"]


def test_control_safety_trending_to_saturation(cc):
    cs = cc.ControlSafetyChecker()
    assert not cs.update(out=acts(u_cool=0.95), T_sp=310.0)["actuator_trending_to_sat"]
    r = cs.update(out=acts(u_cool=0.9521), T_sp=310.0)
    assert r["actuator_trending_to_sat"]
    assert "trending_to_sat=['u_cool']" in r["control_reasons"]
    # Rising by exactly trend_eps or less is not "trending".
    r = cs.update(out=acts(u_cool=0.9531), T_sp=310.0)
    assert not r["actuator_trending_to_sat"]


# ---------------------------------------------------------------------------
# KG context + LLM factory
# ---------------------------------------------------------------------------


def test_kg_context_not_loaded(cc):
    assert cc.get_action_kg_context() == ""
    assert cc.is_kg_loaded() is False


def test_load_kg_context_success(cc, monkeypatch):
    ttl = "@prefix ex: <x#> .\nex:a ex:b ex:c .\n\nex:d ex:e ex:f ."
    queries = []
    monkeypatch.setattr(cc, "SPARQL_AVAILABLE", True)
    monkeypatch.setattr(cc, "run_construct", lambda q: queries.append(q) or ttl)
    assert cc.load_kg_context() is True
    assert queries == [cc.QUERY_CSTR]
    assert cc.is_kg_loaded() and cc._KG_ACTION_TTL == ttl
    ctx = cc.get_action_kg_context()
    assert "## KNOWLEDGE GRAPH DATA" in ctx and f"```turtle\n{ttl}\n```" in ctx


def test_load_kg_context_failures(cc, monkeypatch):
    monkeypatch.setattr(cc, "SPARQL_AVAILABLE", False)
    monkeypatch.setattr(cc, "run_construct", pytest.fail)
    assert cc.load_kg_context() is False

    monkeypatch.setattr(cc, "SPARQL_AVAILABLE", True)
    monkeypatch.setattr(cc, "run_construct", lambda q: "")
    assert cc.load_kg_context() is False and not cc.is_kg_loaded()

    def boom(_q):
        raise RuntimeError("graphdb down")

    monkeypatch.setattr(cc, "run_construct", boom)
    assert cc.load_kg_context() is False and not cc.is_kg_loaded()


def test_record_cached_kg_trace(cc, monkeypatch):
    monkeypatch.setattr(cc, "_KG_ACTION_TTL", "ex:a ex:b ex:c .")
    monkeypatch.setattr(cc, "_KG_LOADED", True)
    spy = SpyRecorder()
    cc._record_cached_kg_trace(spy)
    (call,) = spy.named("record_sparql")
    assert call["name"] == "action_context"
    assert call["subgraph"] == "ex:a ex:b ex:c ." and call["loaded"] is True
    assert call["endpoint"] == cc.ENDPOINT
    assert call["query"] == cc.QUERY_CSTR


def test_make_llm_picks_backend(cc, monkeypatch):
    made = []

    def factory(kind):
        return lambda **kw: made.append((kind, kw)) or kind

    monkeypatch.setattr(cc, "ChatOpenAI", factory("openai"))
    monkeypatch.setattr(cc, "ChatOllama", factory("ollama"))
    assert cc._make_llm("gpt-4o-mini", temperature=0.2, timeout=5.0) == "openai"
    assert made[-1][1] == {"model": "gpt-4o-mini", "temperature": 0.2, "timeout": 5.0}
    assert cc._make_llm() == "ollama"
    assert made[-1][1] == {
        "model": "llama3:8b",
        "temperature": 0.0,
        "timeout": 120.0,
        "base_url": "http://localhost:11434",
    }


def test_setpoint_plan_requires_all_fields(cc):
    p = cc.SetpointPlan(T_sp="309.5", Fin_sp=0.03, L_sp=10, reasoning="PO8")
    assert p.T_sp == 309.5 and isinstance(p.L_sp, float)
    with pytest.raises(ValidationError):
        cc.SetpointPlan(T_sp=309.5, Fin_sp=0.03, reasoning="no level")


# ---------------------------------------------------------------------------
# initializing / plant / live state
# ---------------------------------------------------------------------------


def test_initializing_defaults(cc):
    state = cc.initializing({})
    assert state["itr"] == 0 and state["T_end"] == 7000.0
    assert (state["T_sp"], state["L_sp"]) == (310.0, 10.0)
    assert state["Fin_sp"] == state["Fin_sp_normal"] == pytest.approx(2.0 / 60.0)
    assert state["mode"] == "RUN" and state["reprompt_max"] == 5
    assert state["fault_name"] == "normal" and state["fault_cfg"].enable is False
    assert state["max_time_to_safe"] is None
    assert state["control_zone"] == "SAFE" and state["sim_log"] == []
    assert state["_plant"].rng is None  # shared global stream by default
    assert all(state[k] == 0 for k in ("action_calls", "action_total_tokens_sum"))


def test_initializing_overrides(cc):
    state = cc.initializing(
        {
            "fault_name": "leak",
            "T_sp": 308,
            "L_sp": 9,
            "Fin_sp": 0.05,
            "T_end": 500,
            "reprompt_max": "2",
            "mode": "STOP",
            "isolated_rng": True,
        }
    )
    assert state["fault_cfg"].leak_on and state["_plant"].fault_cfg.leak_on
    assert state["_plant"].T_sp == 308.0 and state["_plant"].L_sp == 9.0
    assert state["Fin_sp_startup"] == 0.05 and state["T_end"] == 500.0
    assert state["reprompt_max"] == 2 and state["mode"] == "STOP"
    assert state["_plant"].rng is not None


def test_plant_step_logs_record(cc):
    state = cc.initializing({"Fin_sp": 0.04, "isolated_rng": True})
    state["control_zone"] = "WARNING"
    cc.plant(state)
    cc.plant(state)
    assert state["itr"] == 2
    assert state["sim_last"]["t"] == 2.0
    rec = state["sim_log"][-1]
    assert rec["Fin_sp"] == 0.04 and rec["fault_name"] == "normal"
    assert rec["control_zone"] == "WARNING" and rec["sp_change_done"] is False
    assert state["_plant"].Fin_sp_normal == 0.04


def test_append_live_plant_state(cc, monkeypatch):
    state = {"sim_log": [{"t": 1.0}], "itr": 3, "fault_flag": 1}
    cc._append_live_plant_state(state)  # no recorder -> no-op
    assert state["sim_log"][-1] == {"t": 1.0}

    spy = SpyRecorder()
    monkeypatch.setattr(cc, "_TRACE_RECORDER", spy)
    cc._append_live_plant_state({"sim_log": []})
    assert spy.calls == []
    cc._append_live_plant_state(state)
    (call,) = spy.named("append_live_row")
    assert call["stream_name"] == "cstr_plant"
    assert call["row"]["iteration"] == 3 and call["row"]["fault_flag"] is True
    assert call["row"]["actuator_valid"] is True and call["row"]["t"] == 1.0


# ---------------------------------------------------------------------------
# monitoring
# ---------------------------------------------------------------------------


def monitor(cc, state, out):
    state["sim_last"] = out
    state["sim_log"].append(dict(out))
    state["itr"] += 1
    return cc.monitoring(state)


def test_monitoring_outside_normal_phase(cc, monkeypatch):
    spy = SpyRecorder()
    monkeypatch.setattr(cc, "_TRACE_RECORDER", spy)
    state = cc.initializing({})
    state.update(itr=199, fault_flag=True, should_act=True)
    monitor(cc, state, sim_row(5.0, T=330.0, L=3.0, phase="STARTUP"))
    assert state["fault_flag_prev"] is True and state["fault_flag"] is False
    assert state["should_act"] is False and state["n_violations"] == 0
    assert state["safe_now"] is False and state["symptoms"]["T_err"] == 20.0
    (call,) = spy.named("append_live_row")
    assert call["row"]["iteration"] == 200 and call["row"]["safe_now"] is False


def test_monitoring_level_error_triggers_once_per_change(cc):
    state = cc.initializing({})
    monitor(cc, state, sim_row(500.0, T=310.0, L=10.8))
    assert state["should_act"] is True
    assert state["violated_params"] == "level_error_large(|L_meas-L_sp|>0.7)"
    state["sp_change_done"] = True
    monitor(cc, state, sim_row(501.0, T=310.0, L=10.8))
    assert state["should_act"] is False and state["n_violations"] == 1


def test_monitoring_recovery_flag(cc):
    state = cc.initializing({})
    monitor(cc, state, sim_row(500.0))
    assert state["safe_now"] and state["recovered_now"]
    assert state["safe_recovered_flag"] is True
    assert state["should_act"] is False and state["violated_params"] == ""


def test_monitoring_cooling_limited_rising_edge(cc):
    state = cc.initializing({})
    acts_seen = []
    for k in range(17):
        monitor(cc, state, sim_row(500.0 + k, T=313.0, u=0.2))
        acts_seen.append(state["should_act"])
    # 15-sample window must fill before the cooling-limited event confirms.
    assert acts_seen == [False] * 14 + [True, False, False]
    assert state["fault_flag"] is True
    assert state["violated_params"].startswith("cool_limited(u_cool<=0.3")


def test_monitoring_entered_unsafe_is_edge_triggered(cc):
    state = cc.initializing({})
    monitor(cc, state, sim_row(500.0, u_valve=1.5))
    assert state["control_zone"] == "UNSAFE" and state["should_act"] is True
    assert "entered_UNSAFE(control_zone)" in state["violated_params"]
    assert state["actuator_valid"] is False
    monitor(cc, state, sim_row(501.0, u_valve=1.5))
    assert state["should_act"] is False and state["violated_params"] == ""


def test_monitoring_temperature_detector(cc):
    state = cc.initializing({})
    for k in range(60):  # settle gate: 60 samples within 1 K of T_sp
        monitor(cc, state, sim_row(500.0 + k))
    assert not state["fault_flag"]
    fired = []
    for k in range(8):
        monitor(cc, state, sim_row(560.0 + k, T=313.0))
        fired.append(state["should_act"])
        assert state["violated_params"] == "T_meas_error>2.0"
    # persistence_ratio=0.6 over a 10-sample window -> 6th hot sample.
    assert fired == [False] * 5 + [True, False, False]
    assert state["anomaly_ratio"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# action_propose
# ---------------------------------------------------------------------------


def test_action_propose_structured_success(cc, monkeypatch, fake_llm_factory):
    llm = fake_llm_factory([plan(cc)])
    made = []
    monkeypatch.setattr(cc, "_make_llm", lambda **kw: made.append(kw) or llm)
    monkeypatch.setattr(cc, "get_openai_callback", fake_callback)
    monkeypatch.setattr(cc, "_KG_LOADED", True)
    monkeypatch.setattr(cc, "_KG_ACTION_TTL", "ex:PO8 ex:changes ex:Fin .")
    spy = SpyRecorder()
    monkeypatch.setattr(cc, "_TRACE_RECORDER", spy)

    state = propose_state(
        eval_feedback="FAIL: earlier",
        llm_timeout=7,
        eval_pass=True,
        eval_metrics={1: 2},
    )
    cc.action_propose(state)

    assert made == [{"model": "gpt-4o-mini", "temperature": 0.0, "timeout": 7.0}]
    assert state["proposed_sp"] == {"T_sp": 309.5, "L_sp": 10.0, "Fin_sp": 0.03}
    assert state["eval_pass"] is False and state["eval_metrics"] == {}
    assert state["action_calls"] == 1
    assert state["action_total_tokens_last"] == state["action_total_tokens_sum"] == 16
    assert state["action_prompt_tokens_sum"] == 11
    assert state["action_completion_tokens_sum"] == 5
    assert state["action_latency_avg_s"] == state["action_latency_last_s"] >= 0.0

    system, user = llm.structured.prompts[0]
    assert system[0] == "system" and "ex:PO8 ex:changes ex:Fin ." in system[1]
    assert "FAIL: earlier" in user[1] and "T_meas: 311.5" in user[1]
    assert "Previous proposal: None" in user[1]

    (call,) = spy.named("record_llm_call")
    assert call["agent"] == "action" and call["failure_type"] is None
    assert call["parsed_response"].T_sp == 309.5
    assert call["rationale"] == "PO8 reduces load"
    assert call["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 5,
        "total_tokens": 16,
    }
    assert json.loads(call["raw_response"].content)["Fin_sp"] == 0.03


def test_action_propose_nudges_repeated_proposal(cc, monkeypatch, fake_llm_factory):
    llm = fake_llm_factory([plan(cc), plan(cc, Fin=0.02)])
    monkeypatch.setattr(cc, "_make_llm", lambda **kw: llm)
    monkeypatch.setattr(cc, "get_openai_callback", fake_callback)
    previous = {"T_sp": 309.5, "L_sp": 10.0, "Fin_sp": 0.03}
    state = propose_state(reprompt_count=1, proposed_sp=dict(previous))
    state.update(action_calls=4, action_total_tokens_sum=100)
    cc.action_propose(state)
    assert state["proposed_sp"]["Fin_sp"] == pytest.approx(0.027)
    assert state["action_calls"] == 5 and state["action_total_tokens_sum"] == 116
    assert f"Previous proposal: {previous}" in llm.structured.prompts[0][1][1]

    state["proposed_sp"] = dict(previous)
    cc.action_propose(state)
    assert state["proposed_sp"]["Fin_sp"] == 0.02  # different -> untouched


def test_action_propose_local_model_skips_token_callback(
    cc, monkeypatch, fake_llm_factory
):
    llm = fake_llm_factory([plan(cc)])
    monkeypatch.setattr(cc, "_make_llm", lambda **kw: llm)
    monkeypatch.setattr(cc, "get_openai_callback", pytest.fail)
    state = propose_state(llm_model="llama3:8b")
    cc.action_propose(state)
    assert state["proposed_sp"]["T_sp"] == 309.5
    assert state["action_total_tokens_sum"] == 0


def test_action_propose_plain_json_fallback(cc, monkeypatch):
    reply = json.dumps({"T_sp": 309, "L_sp": 9.9, "Fin_sp": 0.025, "reasoning": "x"})
    model = StructuredFailsModel(reply)
    monkeypatch.setattr(cc, "_make_llm", lambda **kw: model)
    spy = SpyRecorder()
    monkeypatch.setattr(cc, "_TRACE_RECORDER", spy)
    state = propose_state(llm_model="qwen3:8b")
    cc.action_propose(state)
    assert state["proposed_sp"] == {"T_sp": 309.0, "L_sp": 9.9, "Fin_sp": 0.025}
    assert len(model.prompts) == 1
    (call,) = spy.named("record_llm_call")
    assert call["failure_type"] is None


@pytest.mark.parametrize(
    "reply, failure_type, detail",
    [
        ("Sure! Lower the flow.", "json_parse_error", "json_parse_error="),
        ('{"T_sp": 309}', "schema_validation_error", "json_parse_error="),
        (ConnectionError("Connection refused"), "connection_error", "plain_invoke"),
        (TimeoutError("request timed out"), "timeout", "plain_invoke"),
    ],
)
def test_action_propose_classifies_failures(
    cc, monkeypatch, reply, failure_type, detail
):
    monkeypatch.setattr(cc, "_make_llm", lambda **kw: StructuredFailsModel(reply))
    spy = SpyRecorder()
    monkeypatch.setattr(cc, "_TRACE_RECORDER", spy)
    state = propose_state(llm_model="llama3:8b")
    with pytest.raises(LLMActionError) as info:
        cc.action_propose(state)
    err = info.value
    assert err.failure_type == failure_type
    assert "model=llama3:8b" in str(err) and detail in str(err)
    assert "structured_output_error=no structured output" in str(err)
    if isinstance(reply, str):
        assert err.raw_response == reply
    else:
        assert err.raw_response == {
            "structured_output_error": "no structured output",
            "plain_invoke_error": str(reply),
        }
    assert state["action_calls"] == 1  # counted even though it failed
    (call,) = spy.named("record_llm_call")
    assert call["failure_type"] == failure_type and call["parsed_response"] is None
    assert "proposed_sp" not in state


def test_action_propose_unparsed_structured_reply(cc, monkeypatch, fake_llm_factory):
    raw = types.SimpleNamespace(content="prose instead of JSON")
    llm = fake_llm_factory([{"raw": raw, "parsed": None, "parsing_error": "bad"}])
    monkeypatch.setattr(cc, "_make_llm", lambda **kw: llm)
    with pytest.raises(LLMActionError) as info:
        cc.action_propose(propose_state(llm_model="llama3:8b"))
    assert info.value.failure_type == "schema_validation_error"
    assert info.value.raw_response is raw


# ---------------------------------------------------------------------------
# digital-twin rollout validation + evaluation node
# ---------------------------------------------------------------------------

PROPOSAL = {"T_sp": 310.0, "L_sp": 10.0, "Fin_sp": 0.03}


def rollout(cc, rows, *, t0=0.0, T_end=200.0, **kwargs):
    plant = ScriptedPlant(rows)
    state = {"T_end": T_end, "sim_last": {"t": t0}, "action_calls": 3}
    result = cc.rollout_validate_setpoints(
        base_plant=plant, state=state, proposed=PROPOSAL, **kwargs
    )
    assert plant.i == 0  # the base plant itself is never stepped
    return result


def safe_rows(n, t0=0.0, **kw):
    return [sim_row(t0 + k + 1, **kw) for k in range(n)]


def test_rollout_passes_after_grace_and_persistence(cc, monkeypatch):
    spy = SpyRecorder()
    monkeypatch.setattr(cc, "_TRACE_RECORDER", spy)
    ok, summary, reason, metrics, traj = rollout(cc, safe_rows(200))
    assert ok and reason == ""
    assert summary == "PASS: time_to_safe=119.0s, unsafe_fraction=0.000"
    assert metrics["time_to_safe"] == 119.0
    assert metrics["total_time_simulated"] == 200.0 and metrics["dt"] == 1.0
    assert metrics["worst_zone"] == "SAFE" and metrics["peak_u_cool"] == 0.5
    assert len(traj) == 200 and traj[0]["proposed_Fin_sp"] == 0.03
    assert traj[0]["action_call"] == 3 and traj[0]["control_zone"] == "SAFE"
    assert len(spy.named("append_live_row")) == 200
    assert spy.named("append_live_row")[0]["stream_name"] == "cstr_validation"


def test_rollout_uses_proposed_setpoints_and_mode(cc):
    plant = ScriptedPlant(safe_rows(5))
    state = {"T_end": 3.0, "sim_last": {"t": float("nan")}, "mode": "STOP"}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cc, "clone_plant_obj", lambda p: p)
        *_, metrics, _ = cc.rollout_validate_setpoints(
            base_plant=plant, state=state, proposed=PROPOSAL
        )
    assert metrics["t0"] == 0.0 and len(plant.kwargs) == 3
    assert plant.kwargs[0] == {
        "T_sp": 310.0,
        "L_sp": 10.0,
        "Fin_sp_normal": 0.03,
        "Fin_sp_startup": 0.03,
        "mode": "STOP",
    }


def test_rollout_invalid_actuators(cc):
    rows = safe_rows(200)
    rows[10] = sim_row(11, u_valve=1.2)
    ok, summary, reason, metrics, _ = rollout(cc, rows)
    assert not ok and summary == "FAIL: invalid actuators"
    assert reason == "invalid_actuators" and metrics["any_invalid_actuators"]
    assert metrics["worst_zone"] == "UNSAFE"


def test_rollout_no_recovery(cc):
    ok, summary, reason, metrics, _ = rollout(cc, safe_rows(200, T=320.0))
    assert not ok and reason == "no_recovery"
    assert summary == "FAIL: did not reach SAFE (persist 60.0s)"
    assert metrics["time_to_safe"] is None and metrics["peak_T_meas"] == 320.0
    assert metrics["unsafe_time"] == 141.0  # steps after the 60 s grace window


def test_rollout_slow_recovery(cc):
    ok, summary, reason, *_ = rollout(cc, safe_rows(200), deadline_s=100.0)
    assert not ok and reason == "slow_recovery"
    assert summary == "FAIL: recovery too slow (time_to_safe=119.0s)"


def test_rollout_unsafe_fraction(cc):
    rows = safe_rows(200)
    for k in range(60, 101):
        rows[k] = sim_row(k + 1, T=320.0)
    ok, summary, reason, metrics, _ = rollout(cc, rows, unsafe_fraction_max=0.1)
    assert not ok and reason == "unsafe_fraction"
    assert metrics["unsafe_fraction"] == pytest.approx(41 / 200)
    assert metrics["time_to_safe"] == 161.0


def test_rollout_phase_gate_and_warning_zone(cc):
    rows = safe_rows(200, phase="Phase.STARTUP", u_cool=0.96)
    ok, _, reason, metrics, _ = rollout(cc, rows)
    assert not ok and reason == "no_recovery"  # nothing judged outside NORMAL
    assert metrics["worst_zone"] == "WARNING"
    ok, *_ = rollout(cc, rows, enforce_only_normal_phase=False)
    assert ok
    for phase in ("3", "SHUTDOWN", "1"):
        assert not rollout(cc, safe_rows(200, phase=phase))[0]
    assert rollout(cc, safe_rows(200, phase="2"))[0]


def test_rollout_skips_non_finite_time_and_empty_horizon(cc):
    rows = safe_rows(200)
    rows[0] = sim_row(float("nan"), T=999.0)
    *_, metrics, traj = rollout(cc, rows)
    assert metrics["total_time_simulated"] == 199.0 and len(traj) == 199
    assert metrics["peak_T_meas"] == 310.0

    ok, _, reason, metrics, traj = rollout(cc, rows, t0=300.0, T_end=200.0)
    assert not ok and reason == "no_recovery" and traj == []
    assert metrics["unsafe_fraction"] == 1.0 and metrics["peak_T_meas"] is None


@pytest.mark.xfail(
    strict=True,
    reason="rollout reads s.get('recovered') but SafeStateChecker returns "
    "'recovered_now', so trajectory rows always log recovered_now=False",
)
def test_rollout_trajectory_reports_recovered_now(cc):
    *_, traj = rollout(cc, safe_rows(200))
    assert traj[-1]["recovered_now"] is True


def test_rollout_real_twin_passes_and_leaves_plant_untouched(cc, warm_plant):
    plant, out = warm_plant
    base = copy.deepcopy(plant)
    state = {"T_end": out["t"] + 130.0, "sim_last": out}
    ok, summary, _, metrics, traj = cc.rollout_validate_setpoints(
        base_plant=base, state=state, proposed=PROPOSAL
    )
    assert ok, summary
    assert metrics["t0"] == 450.0 and len(traj) == 130
    assert base.t == 450.0 and np.array_equal(base.x, plant.x)


def test_evaluation_missing_proposal(cc):
    state = {"_plant": None, "proposed_sp": {}}
    cc.evaluation(state)
    assert state["eval_pass"] is False
    assert state["eval_fail_reason"] == "missing_proposal"
    assert state["eval_summary"] == "FAIL: no proposed_sp found"


def test_evaluation_accepts_and_writes_trajectory(cc, monkeypatch, warm_plant):
    plant, out = warm_plant
    spy = SpyRecorder()
    monkeypatch.setattr(cc, "_TRACE_RECORDER", spy)
    state = {
        "_plant": copy.deepcopy(plant),
        "proposed_sp": dict(PROPOSAL),
        "T_end": out["t"] + 125.0,
        "sim_last": out,
        "fault_name": "normal",
        "action_calls": 2,
    }
    cc.evaluation(state)
    assert state["eval_pass"] is True and state["eval_fail_reason"] == ""
    assert state["eval_summary"].startswith("PASS")
    (event,) = spy.named("record_event")
    assert event["accepted"] is True and event["condition"] == "nominal"
    assert event["proposed_setpoints"] == PROPOSAL
    (traj,) = spy.named("write_trajectory")
    assert traj["name"] == "accepted_nominal_validation_action_0002"
    assert len(traj["rows"]) == 125
    assert traj["metadata"]["source"] == "digital_twin_validation"


def test_evaluation_rejects_without_trajectory(cc, monkeypatch):
    spy = SpyRecorder()
    monkeypatch.setattr(cc, "_TRACE_RECORDER", spy)
    state = {
        "_plant": ScriptedPlant(safe_rows(200, T=320.0)),
        "proposed_sp": dict(PROPOSAL),
        "T_end": 200.0,
        "sim_last": {"t": 0.0},
        "fault_name": "fouling",
    }
    cc.evaluation(state)
    assert state["eval_pass"] is False and state["eval_fail_reason"] == "no_recovery"
    assert state["eval_metrics"]["peak_T_meas"] == 320.0
    (event,) = spy.named("record_event")
    assert event["condition"] == "fault" and event["accepted"] is False
    assert spy.named("write_trajectory") == []


# ---------------------------------------------------------------------------
# graph wiring + end-to-end runs
# ---------------------------------------------------------------------------


def test_build_graph_wiring(runnable_graph):
    g = runnable_graph
    assert set(g.nodes) == {
        "initializing",
        "plant",
        "monitoring",
        "check_end",
        "action_propose",
        "evaluation",
        "reprompting",
        "action_apply",
    }
    assert g.edges == {
        g.START: "initializing",
        "initializing": "plant",
        "plant": "monitoring",
        "action_propose": "evaluation",
        "reprompting": "action_propose",
        "action_apply": "plant",
    }
    assert g.cond["monitoring"][1] == ["action_propose", "check_end"]
    assert g.cond["check_end"][1] == ["plant", g.END]
    assert g.cond["evaluation"][1] == ["action_apply", "reprompting"]


def make_args(**overrides):
    args = dict(
        mode="3sigma",
        max_steps=2000,
        T_sp=310.0,
        L_sp=10.0,
        Fin_sp=2.0 / 60.0,
        T_end=60.0,
        safe_hold=60.0,
        max_time_to_safe=-1.0,
        llm_model="gpt-4o-mini",
        reprompt_max=1,
        isolated_rng=True,
        trace_dir=None,
    )
    args.update(overrides)
    return argparse.Namespace(**args)


def read_events(run_dir):
    lines = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def only_run_dir(tmp_path):
    (path,) = [p for p in tmp_path.rglob("run_000") if p.is_dir()]
    return path


def test_run_single_experiment_without_action(cc, runnable_graph, tmp_path):
    args = make_args(trace_dir=str(tmp_path), T_end=40.0)
    metrics, out = cc.run_single_experiment_cstr(
        runnable_graph, fault="normal", run=0, args=args, experiment_id="exp"
    )
    assert out["abort_reason"] == "t_end_reached" and out["itr"] == 40
    assert metrics["reached_end"] == 1 and metrics["steps"] == 40
    assert metrics["acted"] == 0 and metrics["success"] == 0
    assert metrics["action_calls"] == 0 and metrics["llm_model"] == "gpt-4o-mini"
    assert metrics["safe_seconds"] == 40  # zone logged before monitoring runs
    assert metrics["_analysis"]["final_setpoints"]["T_sp"] == 310.0
    assert cc._TRACE_RECORDER is None

    run_dir = only_run_dir(tmp_path)
    assert run_dir.relative_to(tmp_path).parts[:4] == (
        "cstr",
        "exp",
        "gpt-4o-mini",
        "normal",
    )
    events = [e["event"] for e in read_events(run_dir)]
    assert events[:2] == ["run_started", "sparql_retrieval"]
    assert "plant_trajectory_decision" in events and events[-1] == "run_finished"
    final = json.loads((run_dir / "final_result.json").read_text(encoding="utf-8"))
    assert final["status"] == "completed" and "_analysis" not in final["result"]
    live = (run_dir / "live" / "cstr_plant.jsonl").read_text(encoding="utf-8")
    assert len(live.splitlines()) == 40


def test_run_single_experiment_with_reprompt_and_fallback(
    cc, runnable_graph, monkeypatch, fake_llm_factory
):
    # Fin_sp=0.1 drives the cooling valve into saturation; the loop must act.
    bad = plan(cc, L=12.0, Fin=0.1, why="overfill")
    llm = fake_llm_factory([bad, bad])
    monkeypatch.setattr(cc, "_make_llm", lambda **kw: llm)
    monkeypatch.setattr(cc, "get_openai_callback", fake_callback)
    args = make_args(Fin_sp=0.1, T_end=600.0, reprompt_max=1)
    metrics, out = cc.run_single_experiment_cstr(
        runnable_graph, fault="normal", run=0, args=args, verbose=False
    )
    assert len(llm.structured.prompts) == 2
    assert metrics["action_calls"] == 2 and metrics["reprompts"] == 1
    assert metrics["action_total_tokens_sum"] == 32
    # The second identical proposal was nudged, still failed, so the
    # deterministic fallback was applied.
    assert out["eval_summary"].startswith("PASS(fallback)")
    assert metrics["acted"] == 1 and metrics["success"] == 0
    assert metrics["final_Fin_sp"] == pytest.approx(0.05)
    assert metrics["final_L_sp"] == 10.0
    assert metrics["time_to_first_action"] == out["t_sp_change"]
    assert metrics["reached_end"] == 1


def test_run_single_experiment_error_finalizes_trace(cc, tmp_path, monkeypatch):
    class Failing:
        def invoke(self, state, config):
            raise LLMActionError("timeout", "model too slow")

    monkeypatch.setattr(cc, "_KG_ACTION_TTL", "ex:a ex:b ex:c .")
    args = make_args(trace_dir=str(tmp_path))
    with pytest.raises(LLMActionError):
        cc.run_single_experiment_cstr(Failing(), fault="fouling", run=0, args=args)
    assert cc._TRACE_RECORDER is None
    run_dir = only_run_dir(tmp_path)
    final = json.loads((run_dir / "final_result.json").read_text(encoding="utf-8"))
    assert final["status"] == "error"
    assert final["result"] == {
        "error_type": "LLMActionError",
        "error": "model too slow",
        "failure_type": "timeout",
    }
    ttl = (run_dir / "subgraphs" / "action_context.ttl").read_text(encoding="utf-8")
    assert ttl == "ex:a ex:b ex:c ."


def test_run_single_experiment_accepts_successful_fault_run(cc, tmp_path):
    out_state = {
        "abort_reason": "t_end_reached",
        "safe_now": True,
        "control_zone": "SAFE",
        "sp_change_done": True,
        "eval_pass": True,
        "reprompt_count": 0,
        "sim_log": [{"t": 1.0, "control_zone": "SAFE"}],
    }
    graph = types.SimpleNamespace(invoke=lambda state, config: dict(out_state))
    args = make_args(trace_dir=str(tmp_path), max_time_to_safe=30.0)
    metrics, _ = cc.run_single_experiment_cstr(graph, fault="leak", run=0, args=args)
    assert metrics["success"] == 1
    run_dir = only_run_dir(tmp_path)
    decision = [
        e for e in read_events(run_dir) if e["event"] == "plant_trajectory_decision"
    ]
    assert decision[0]["accepted"] is True and decision[0]["condition"] == "fault"
    assert (run_dir / "trajectories" / "accepted_fault_plant.csv").exists()


def test_extract_cstr_metrics(cc):
    out = {
        "sim_last": {"t": "bad", "phase": "NORMAL"},
        "abort_reason": "max_steps",
        "control_zone": "WARNING",
        "safe_now": True,
        "reprompt_count": 3,
        "sp_change_done": True,
        "eval_pass": True,
        "t_sp_change": "not-a-number",
        "fault_flag": True,
        "anomaly_ratio": 0.4,
        "itr": 3,
        "eval_metrics": {"time_to_safe": 12.0, "peak_u_cool": 0.9},
        "sim_log": [
            {"control_zone": "UNSAFE"},
            {"control_zone": "WARNING"},
            {"control_zone": "SAFE"},
            {},
        ],
        "action_calls": 2,
        "action_latency_sum_s": 1.5,
    }
    init = {"reprompt_max": 3, "llm_model": "m", "detector_mode": "strict"}
    m = cc._extract_cstr_metrics(out, fault="leak", run=4, wall_time=2, init_state=init)
    assert m["success"] == 0  # reprompt limit hit
    assert m["reached_end"] == 0 and m["abort_reason"] == "max_steps"
    assert m["sim_time_s"] is None and m["time_to_first_action"] is None
    assert (m["unsafe_seconds"], m["warning_seconds"], m["safe_seconds"]) == (1, 1, 1)
    assert m["final_fault_flag"] == 1 and m["final_safe_now"] == 1
    assert m["rollout_time_to_safe"] == 12.0 and m["rollout_peak_u_cool"] == 0.9
    assert m["rollout_unsafe_fraction"] is None
    assert m["detector_mode"] == "strict" and m["wall_time_s"] == 2.0
    assert m["action_latency_sum_s"] == 1.5 and m["action_total_tokens_sum"] == 0

    init["reprompt_max"] = 4
    out.update(t_sp_change=812.0, sim_last={"t": 900.0})
    m = cc._extract_cstr_metrics(out, fault="leak", run=4, wall_time=2, init_state=init)
    assert m["success"] == 1 and m["sim_time_s"] == 900.0
    assert m["time_to_first_action"] == 812.0
    assert (
        cc._extract_cstr_metrics({}, fault="x", run=0, wall_time=0, init_state={})[
            "steps"
        ]
        == 0
    )


# ---------------------------------------------------------------------------
# ablation study + main
# ---------------------------------------------------------------------------


class CannedGraph:
    def __init__(self, fail_on=()):
        self.fail_on = set(fail_on)
        self.calls = []

    def invoke(self, state, config):
        self.calls.append(state)
        if (state["fault_name"], len(self.calls)) in self.fail_on:
            raise LLMActionError("connection_error", "ollama not running")
        return {
            "abort_reason": "t_end_reached",
            "safe_now": True,
            "control_zone": "SAFE",
            "sp_change_done": True,
            "eval_pass": True,
            "reprompt_count": 0,
            "itr": 10,
            "sim_last": {"t": 10.0},
            "sim_log": [{"t": 1.0, "control_zone": "SAFE"}],
            "fault_cfg": FaultConfig(enable=False),
        }


def test_run_ablation_study(cc, tmp_path, monkeypatch):
    plots = []
    monkeypatch.setattr(cc, "plot_results", lambda *a, **k: plots.append(k["title"]))
    graph = CannedGraph(fail_on={("fouling", 4)})
    args = make_args(llm_model="llama3:8b", mode="strict")
    df = cc.run_ablation_study_cstr(
        graph,
        faults=["normal", "fouling"],
        n_runs=2,
        args=args,
        output_dir=str(tmp_path / "res"),
        save_plots=True,
    )
    assert list(df["fault"]) == ["normal", "normal", "fouling", "fouling"]
    assert list(df["success"]) == [1, 1, 1, 0]
    assert list(df["reprompts"]) == [0, 0, 0, 0]
    assert df["failure_type"].iloc[3] == "connection_error"
    assert df["abort_reason"].iloc[3] == "exception"
    assert plots == [
        "CSTR (normal) run=0",
        "CSTR (normal) run=1",
        "CSTR (fouling) run=0",
    ]

    files = sorted(p.name for p in (tmp_path / "res").iterdir())
    assert len(files) == 3
    assert all(f.startswith("cstr_ablation_llama38b_strict_") for f in files)
    summary_csv = next(p for p in (tmp_path / "res").iterdir() if "summary" in p.name)
    summary = summary_csv.read_text(encoding="utf-8").splitlines()
    assert summary[0].startswith("fault,n,success_mean,success_std")
    assert summary[2].startswith("fouling,2,0.5,")
    blob_path = next(p for p in (tmp_path / "res").iterdir() if p.suffix == ".json")
    blob = json.loads(blob_path.read_text(encoding="utf-8"))
    assert blob["meta"]["runs_per_fault"] == 2 and len(blob["runs"]) == 4
    assert blob["runs"][3]["failure_type"] == "connection_error"
    assert blob["runs"][0]["analysis"]["final_control_zone"] == "SAFE"


def test_main_single_run(cc, tmp_path, monkeypatch):
    graph = CannedGraph()
    loads, plots = [], []
    monkeypatch.setattr(cc, "build_graph", lambda: graph)
    monkeypatch.setattr(cc, "SPARQL_AVAILABLE", True)
    monkeypatch.setattr(cc, "load_kg_context", lambda: loads.append(1))
    monkeypatch.setattr(cc, "plot_results", lambda log, **k: plots.append(k))
    out_dir = tmp_path / "out"
    argv = ["cstr_case.py", "--fault", "normal", "--output", str(out_dir)]
    argv += ["--trace-dir", "", "--plot", "--llm-model", "gpt-4o", "--T-end", "50"]
    argv += ["--isolated-rng", "--max-time-to-safe", "120"]
    monkeypatch.setattr(sys, "argv", argv)
    cc.main()

    assert loads == [1] and len(graph.calls) == 1
    init = graph.calls[0]
    assert init["T_end"] == 50.0 and init["isolated_rng"] is True
    assert init["max_time_to_safe"] == 120.0 and init["llm_model"] == "gpt-4o"
    assert plots[0]["title"] == "CSTR Digital Twin (LLM + Eval + Apply)"
    names = sorted(p.name for p in out_dir.iterdir())
    assert len(names) == 2
    assert all(n.startswith("cstr_single_normal_gpt4o_3sigma_") for n in names)
    blob_path = next(p for p in out_dir.iterdir() if p.suffix == ".json")
    blob = json.loads(blob_path.read_text(encoding="utf-8"))
    assert blob["meta"]["args"]["fault"] == "normal"
    assert blob["metrics"]["success"] == 1 and "_analysis" not in blob["metrics"]


def test_main_all_faults_runs_ablation(cc, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cc, "build_graph", lambda: "graph")
    monkeypatch.setattr(cc, "SPARQL_AVAILABLE", False)
    monkeypatch.setattr(cc, "load_kg_context", pytest.fail)
    monkeypatch.setattr(
        cc, "run_ablation_study_cstr", lambda g, **kw: calls.append((g, kw))
    )
    argv = ["cstr_case.py", "--fault", "all", "--runs", "3"]
    argv += ["--output", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    cc.main()
    ((graph, kw),) = calls
    assert graph == "graph" and kw["n_runs"] == 3 and kw["save_plots"] is False
    assert kw["faults"] == ["fouling", "pump_degrade", "cool_stuck_closed"]
    assert kw["output_dir"] == str(tmp_path) and kw["args"].trace_dir.endswith("traces")


def test_main_rejects_unknown_fault(cc, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cstr_case.py", "--fault", "leak"])
    with pytest.raises(SystemExit):
        cc.main()


# ---------------------------------------------------------------------------
# results -> plot vectors, plotting
# ---------------------------------------------------------------------------


def test_extract_vectors(cc):
    log = [
        {"t": 1.0, "phase": "STARTUP", "x": np.arange(6.0), "t_sp_change": None},
        {"t": 2.0, "phase": "normal", "CA": 9.0, "t_sp_change": "oops"},
        {"t": 3.0, "phase": "SHUTDOWN", "fault_tags": "FOUL", "t_sp_change": np.nan},
        {"t": 4.0, "fault_tag": "PUMP", "t_sp_change": 3.5, "T_sp": 309.0},
        {"t": 5.0, "t_sp_change": 4.5},
    ]
    vec = cc.extract_vectors(log)
    assert vec["t_hist"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert vec["phase_hist"] == [1, 2, 3, 2, 2]
    assert vec["CA_hist"][:2] == [1.0, 9.0] and vec["CD_hist"][0] == 4.0
    assert np.isnan(vec["CB_hist"][1])
    assert vec["fault_tag_hist"] == ["OK", "OK", "FOUL", "PUMP", "OK"]
    assert vec["t_sp_change"] == 3.5
    assert vec["T_sp_hist"][3] == 309.0 and vec["leak_hist"] == [0.0] * 5
    assert cc.extract_vectors([])["t_sp_change"] is None


@pytest.fixture
def shown(cc, monkeypatch):
    """Capture figures that would be shown, then close them."""
    figs = []

    def show():
        fig = cc.plt.gcf()
        figs.append((fig._suptitle.get_text(), len(fig.axes), fig))
        cc.plt.close(fig)

    monkeypatch.setattr(cc.plt, "show", show)
    return figs


def test_plot_results(cc, shown, warm_plant):
    plant, _ = warm_plant
    twin = copy.deepcopy(plant)
    log = []
    for k in range(5):
        rec = dict(twin.simulate_step())
        rec.update(T_sp=310.0, L_sp=10.0, Fin_sp=0.03, t_sp_change=452.0)
        log.append(rec)
    cfg = FaultConfig(fouling_on=True, pump_degrade_on=True)
    cc.plot_results(log, fault_cfg=cfg, title="unit plot")
    cc.plot_results([], fault_cfg=FaultConfig(enable=False), title="empty")
    assert [(title, n) for title, n, _ in shown] == [("unit plot", 9), ("empty", 9)]
    temp_ax = shown[0][2].axes[1]
    labels = [line.get_label() for line in temp_ax.get_lines()]
    assert {"T (true)", "T_meas", "T_sp(t)"} <= set(labels)
    texts = {t.get_text() for t in temp_ax.texts}
    assert texts == {"SP change @ fault flag", "Fouling start", "Pump degrade start"}


def test_plot_digital_twin_constant_setpoints(cc, shown):
    t = [0.0, 1.0, 2.0]
    series = {
        name: [0.1, 0.2, 0.3]
        for name in (
            "L_hist V_hist Fin_hist Fout_hist Fin_sp_hist CA_hist CB_hist CC_hist "
            "CD_hist u_valve_hist u_pump_hist u_cool_hist T_hist Tmeas_hist Fc_hist "
            "UAeff_hist leak_hist over_hist"
        ).split()
    }
    cc.plot_cstr_digital_twin(
        t_hist=t,
        phase_hist=[1, 2, 3],
        t_sp_change=float("nan"),
        show_fault_markers=False,
        fouling_t_start=1.0,
        **series,
    )
    ((title, n_axes, fig),) = shown
    assert title == "CSTR Digital Twin with Fault Injection" and n_axes == 9
    level_ax = fig.axes[0]
    assert "L_sp" in [line.get_label() for line in level_ax.get_lines()]
    assert len(level_ax.texts) == 0
