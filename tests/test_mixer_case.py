"""Offline tests for the pure helpers in case_studies/mixer_case/mixer_case.py.

Graph nodes, routing and experiment drivers live in test_mixer_case_graph.py.
"""

import pydantic
import pytest


@pytest.fixture
def mc(mixer_case, monkeypatch):
    # Every test starts from "no KG loaded" and restores the module afterwards.
    monkeypatch.setattr(mixer_case, "_KG_PLANNING_TTL", "")
    monkeypatch.setattr(mixer_case, "_KG_ACTION_TTL", "")
    monkeypatch.setattr(mixer_case, "_KG_LOADED", False)
    monkeypatch.setattr(mixer_case, "_TRACE_RECORDER", None)
    return mixer_case


@pytest.fixture
def state(mc):
    return mc.initializing({"condition": "normal"})


class RecorderSpy:
    def __init__(self):
        self.calls = []

    def record_event(self, event, **payload):
        self.calls.append(("event", event, payload))

    def record_sparql(self, **payload):
        self.calls.append(("sparql", payload["name"], payload))


# --- trajectory rows / KG context ---------------------------------------------


def test_trajectory_rows_empty_state(mc):
    assert mc._mixer_trajectory_rows({}) == []


def test_trajectory_rows_pad_ragged_columns(mc):
    rows = mc._mixer_trajectory_rows(
        {
            "condition": "leak",
            "log_time": [0.0, 1.5],
            "log_curr_state": ["s0", "s1"],
            "log_level_B201": [0.022],
            "log_pump_power": [0.0, 1.0],
        }
    )
    assert len(rows) == 2
    assert rows[0]["iteration"] == 1 and rows[1]["iteration"] == 2
    assert rows[1]["fault_condition"] == "leak"
    assert rows[1]["time"] == 1.5 and rows[1]["state"] == "s1"
    assert rows[0]["level_B201"] == 0.022 and rows[1]["level_B201"] is None
    assert rows[0]["valve_out_B204"] is None
    assert rows[1]["pump_power"] == 1.0
    assert (
        mc._mixer_trajectory_rows({"log_time": [1]})[0]["fault_condition"] == "normal"
    )


def test_record_cached_kg_trace_without_sparql(mc, monkeypatch):
    monkeypatch.setattr(mc, "SPARQL_AVAILABLE", False)
    spy = RecorderSpy()
    mc._record_cached_kg_trace(spy)
    assert spy.calls == [("event", "sparql_unavailable", {})]


def test_record_cached_kg_trace_records_both_queries(mc, monkeypatch):
    monkeypatch.setattr(mc, "SPARQL_AVAILABLE", True)
    monkeypatch.setattr(mc, "_KG_PLANNING_TTL", ":a :b :c .")
    spy = RecorderSpy()
    mc._record_cached_kg_trace(spy)
    assert [c[1] for c in spy.calls] == ["planning", "action"]
    planning, action = spy.calls[0][2], spy.calls[1][2]
    assert planning["loaded"] is True and planning["subgraph"] == ":a :b :c ."
    assert planning["query"] == mc.QUERY_PLANNING
    assert action["loaded"] is False and action["endpoint"] == mc.ENDPOINT


def test_load_kg_context_without_sparql(mc, monkeypatch):
    monkeypatch.setattr(mc, "SPARQL_AVAILABLE", False)
    monkeypatch.setattr(mc, "run_construct", lambda q: pytest.fail("no query"))
    assert mc.load_kg_context() is False
    assert mc.is_kg_loaded() is False


def test_load_kg_context_success(mc, monkeypatch, capsys):
    monkeypatch.setattr(mc, "SPARQL_AVAILABLE", True)
    ttl = {
        mc.QUERY_PLANNING: "@prefix : <x#> .\n:s1 a :State .\n\n:s2 a :State .\n",
        mc.QUERY_ACTION: "@prefix : <x#> .\n:a1 a :Action .\n",
    }
    monkeypatch.setattr(mc, "run_construct", ttl.__getitem__)
    assert mc.load_kg_context() is True
    assert mc.is_kg_loaded() is True
    out = capsys.readouterr().out
    assert "Planning context: ~2 statements" in out
    assert "Action context: ~1 statements" in out
    assert ":s2 a :State ." in mc.get_planning_kg_context()
    assert ":a1 a :Action ." in mc.get_action_kg_context()


def test_load_kg_context_empty_response(mc, monkeypatch):
    monkeypatch.setattr(mc, "SPARQL_AVAILABLE", True)
    monkeypatch.setattr(mc, "run_construct", lambda q: "")
    assert mc.load_kg_context() is False
    assert mc.is_kg_loaded() is False


def test_load_kg_context_endpoint_error(mc, monkeypatch, capsys):
    def boom(_query):
        raise ConnectionError("GraphDB down")

    monkeypatch.setattr(mc, "SPARQL_AVAILABLE", True)
    monkeypatch.setattr(mc, "run_construct", boom)
    assert mc.load_kg_context() is False
    assert "Could not load KG context: GraphDB down" in capsys.readouterr().out


def test_kg_context_getters_empty_when_not_loaded(mc, monkeypatch):
    assert mc.get_planning_kg_context() == ""
    assert mc.get_action_kg_context() == ""
    monkeypatch.setattr(mc, "_KG_LOADED", True)  # loaded flag but empty TTL
    assert mc.get_planning_kg_context() == ""
    assert mc.get_action_kg_context() == ""


# --- configuration / prompts / LLM factory -------------------------------------


@pytest.mark.parametrize(
    "model, temp",
    [
        ("gpt-5", 1.0),
        ("GPT-5-mini", 1.0),
        ("my-gpt5", 1.0),
        ("gpt-4o", 0.0),
        ("qwen3:8b", 0.0),
    ],
)
def test_model_temperature(mc, model, temp):
    assert mc.get_model_temperature(model) == temp


@pytest.mark.parametrize("level", ["normal", "minimal"])
def test_prompts_for_level_without_kg(mc, level):
    prompts = mc.get_prompts_for_level(level)
    assert set(prompts) == {
        "planning_system",
        "planning_few_shot",
        "action_system",
        "action_few_shot",
    }
    assert prompts["planning_system"].startswith(mc.ONTOLOGY_CONTEXT)
    assert mc.VALID_STATES_STR in prompts["planning_system"]
    assert "KNOWLEDGE GRAPH DATA" not in prompts["planning_system"]
    if level == "minimal":
        assert prompts["planning_few_shot"] == prompts["action_few_shot"] == ""
        assert prompts["action_system"] == mc.ONTOLOGY_CONTEXT
    else:
        assert "## EXAMPLES" in prompts["planning_few_shot"]
        assert "## ACTION TASK" in prompts["action_system"]


def test_prompts_for_level_append_kg(mc, monkeypatch):
    monkeypatch.setattr(mc, "_KG_LOADED", True)
    monkeypatch.setattr(mc, "_KG_PLANNING_TTL", ":plan_ttl")
    monkeypatch.setattr(mc, "_KG_ACTION_TTL", ":action_ttl")
    prompts = mc.get_prompts_for_level("minimal")
    assert prompts["planning_system"].rstrip().endswith("```")
    assert ":plan_ttl" in prompts["planning_system"]
    assert ":action_ttl" in prompts["action_system"]
    assert ":plan_ttl" not in prompts["action_system"]


@pytest.mark.parametrize("level", ["full", "reduced", ""])
def test_prompts_for_unknown_level(mc, level):
    # NB: "full" is the module/CLI default for CURRENT_PROMPT_LEVEL but is rejected here.
    with pytest.raises(ValueError, match="Unknown prompt level"):
        mc.get_prompts_for_level(level)


class _Recorder:
    def __init__(self, kind):
        self.kind = kind

    def __call__(self, **kwargs):
        return (self.kind, kwargs)


def test_make_llm_openai_defaults(mc, monkeypatch):
    monkeypatch.setattr(mc, "ChatOpenAI", _Recorder("openai"))
    monkeypatch.setattr(mc, "ChatOllama", _Recorder("ollama"))
    monkeypatch.setattr(mc, "CURRENT_MODEL", "gpt-5-mini")
    monkeypatch.setattr(mc, "CURRENT_LLM_TIMEOUT", 7.5)
    assert mc._make_llm() == (
        "openai",
        {"model": "gpt-5-mini", "temperature": 1.0, "timeout": 7.5},
    )


def test_make_llm_ollama_with_explicit_args(mc, monkeypatch, capsys):
    monkeypatch.setattr(mc, "ChatOpenAI", _Recorder("openai"))
    monkeypatch.setattr(mc, "ChatOllama", _Recorder("ollama"))
    assert mc._make_llm("llama3:8b", temperature=0.3, timeout=900) == (
        "ollama",
        {"model": "llama3:8b", "temperature": 0.3, "timeout": 900},
    )
    assert "Non-GPT model 'llama3:8b'" in capsys.readouterr().out


def test_pydantic_models_validate(mc):
    plan = mc.ActionPlan(
        target_state="s", actions=[{"actuator": "P101", "value": "1"}], reasoning="r"
    )
    assert plan.actions[0].value == 1.0
    with pytest.raises(pydantic.ValidationError):
        mc.PathPlan(current_state="a", next_state="b")
    with pytest.raises(pydantic.ValidationError):
        mc.ActionItem(actuator="P101", value="open")


# --- actuators -----------------------------------------------------------------


@pytest.mark.parametrize(
    "name, key",
    [
        ("valve_in0", "valve_in_B201"),
        ("  VALVE_PUMP_B204_ON ", "valve_in_B204"),
        ("mixer0_valve_out_opening_actuator", "valve_out_B204"),
        ("p101", "pump_power"),
        ("bypass_pump_P102", "pump_power_alt"),
        ("valve_out_B203", "valve_out_B203"),
        ("pump_P103", None),
    ],
)
def test_normalize_actuator(mc, name, key):
    assert mc.normalize_actuator(name) == key


def test_apply_action_plan_is_exclusive(mc, capsys):
    state = {"valve_in_B201": 1, "valve_out_B204": 1, "pump_power_alt": 1.0}
    plan = mc.ActionPlan(
        target_state="state_emptying_tank_B202",
        actions=[
            mc.ActionItem(actuator="valve_pump_tank_B202", value=1),
            mc.ActionItem(actuator="valve_in_B204", value=1.0),
            mc.ActionItem(actuator="P101", value=0.8),
            mc.ActionItem(actuator="flux_capacitor", value=1),
        ],
        reasoning="",
    )
    out = mc.apply_action_plan(state, plan)
    assert out is state
    assert out["valve_out_B202"] == 1 and isinstance(out["valve_out_B202"], int)
    assert out["valve_in_B204"] == 1 and isinstance(out["valve_in_B204"], int)
    assert out["pump_power"] == 0.8
    assert out["valve_in_B201"] == out["valve_out_B204"] == 0
    assert out["pump_power_alt"] == 0.0
    assert "Unknown actuator: flux_capacitor" in capsys.readouterr().out


@pytest.mark.parametrize(
    "fault, expected",
    [
        ("pump_failure", True),
        ("pump_degradation", True),
        ("Clogging_Fault", True),
        ("leak", True),
        ("P101_stuck", True),
        ("sensor_fault", False),
        ("normal", False),
    ],
)
def test_is_pump_fault(mc, fault, expected):
    assert mc.is_pump_fault(fault) is expected


def test_validate_actuators(mc, state):
    assert mc._validate_actuators(state) == 0
    assert mc._validate_actuators({}) == 0
    bad = dict(state, valve_in_B201=2, valve_out_B204=-1, pump_power=1.5)
    assert mc._validate_actuators(bad) == 3
    assert mc._validate_actuators(dict(state, pump_power_alt="fast")) == 1
    assert mc._validate_actuators(dict(state, pump_power=None)) == 1


@pytest.mark.parametrize(
    "target, overrides, correct, n_errors",
    [
        ("state_filling_tank_B201", {"valve_in_B201": 1}, 10, 0),
        ("STATE_BYPASS_EMPTYING_TANK_B202", {"valve_out_B202": 1, "valve_in_B204": 1, "pump_power_alt": 0.4}, 10, 0),
        ("state_emptying_tank_B201", {"valve_out_B201": 1, "valve_in_B204": 1, "pump_power_alt": 1.0}, 8, 2),
        ("state_emptying_tank_B204", {}, 9, 1),
    ],
)  # fmt: skip
def test_check_actuators(mc, state, target, overrides, correct, n_errors):
    result = mc.check_actuators(dict(state, **overrides), target)
    assert result["total"] == 10
    assert result["correct"] == correct
    assert len(result["errors"]) == n_errors
    assert result["accuracy"] == correct / 10


def test_check_actuators_reports_mismatch_detail(mc, state):
    result = mc.check_actuators(dict(state, valve_in_B203=1), "state_filling_tank_B202")
    assert "valve_in_B202: expected=1, actual=0" in result["errors"]
    assert "valve_in_B203: expected=0, actual=1" in result["errors"]


def test_check_actuators_unknown_state(mc, state):
    assert mc.check_actuators(state, "state_initialStep") == {
        "correct": 0,
        "total": 0,
        "errors": [],
        "accuracy": 1.0,
    }


# --- state machine helpers -----------------------------------------------------


def test_tank_status_labels(mc, state):
    state.update(level_B201=0.04, level_B202=0.02, level_B203=0.03, level_B204=0.055)
    lines = mc.get_tank_status(state).splitlines()
    assert lines[0] == "Tank Levels:"
    assert lines[1].startswith("- B201: 0.0400") and lines[1].endswith("FULL")
    assert lines[2].endswith("EMPTY")
    assert lines[3].endswith("partial")
    assert lines[4].endswith("FULL")


FULL = {"level_B201": 0.034, "level_B202": 0.034, "level_B203": 0.034}
EMPTY = {"level_B201": 0.02, "level_B202": 0.02, "level_B203": 0.02, "level_B204": 0.02}
MID = {
    "level_B201": 0.028,
    "level_B202": 0.028,
    "level_B203": 0.028,
    "level_B204": 0.03,
}

TRANSITION_CASES = [
    ("state_filling_tank_B201", FULL, True),
    ("state_filling_tank_B202", MID, False),
    ("state_filling_tank_B203", FULL, True),
    ("state_emptying_tank_B201", EMPTY, True),
    ("state_bypass_emptying_tank_b202", EMPTY, True),
    ("state_emptying_tank_B203", MID, False),
    ("state_emptying_tank_B204", EMPTY, True),
    ("state_emptying_tank_B204", MID, False),
    ("state_unknown", EMPTY, False),
]


@pytest.mark.parametrize("curr, levels, expected", TRANSITION_CASES)
def test_transition_condition_helpers_agree(mc, state, curr, levels, expected):
    state.update(levels, curr_state=curr)
    assert mc.check_transition_condition(state) is expected
    assert mc.check_transition_condition_with_levels(state) is expected
    assert mc.check_transition_conditions(state) is expected


def test_initial_state_always_exits(mc, state):
    assert state["curr_state"] == "state_initialStep"
    assert mc.check_transition_condition_with_levels(state) is True
    assert mc.check_transition_conditions(state) is True
    # The older helper has no initial-state rule.
    assert mc.check_transition_condition(state) is False


@pytest.mark.parametrize(
    "curr, fault, levels, expected",
    [
        ("state_initialStep", "leak", MID, "state_filling_tank_B201"),
        ("state_filling_tank_B201", "normal", FULL, "state_filling_tank_B202"),
        ("state_filling_tank_B201", "normal", MID, "state_filling_tank_B201"),
        ("state_filling_tank_B202", "leak", FULL, "state_filling_tank_B203"),
        ("state_filling_tank_B202", "leak", MID, "state_filling_tank_B202"),
        ("state_filling_tank_B203", "normal", FULL, "state_emptying_tank_B201"),
        ("state_filling_tank_B203", "sensor_fault", FULL, "state_emptying_tank_B201"),
        ("state_filling_tank_B203", "pump_failure", FULL, "state_bypass_emptying_tank_b201"),
        ("state_filling_tank_B203", "pump_failure", MID, "state_filling_tank_B203"),
        ("state_emptying_tank_B201", "normal", EMPTY, "state_emptying_tank_B202"),
        ("state_emptying_tank_B201", "normal", MID, "state_emptying_tank_B201"),
        ("state_emptying_tank_B202", "normal", EMPTY, "state_emptying_tank_B203"),
        ("state_emptying_tank_B202", "normal", MID, "state_emptying_tank_B202"),
        ("state_emptying_tank_B203", "normal", EMPTY, "state_emptying_tank_B204"),
        ("state_emptying_tank_B203", "normal", MID, "state_emptying_tank_B203"),
        ("state_emptying_tank_B204", "normal", EMPTY, "state_emptying_tank_B204"),
        ("state_bypass_emptying_tank_b201", "leak", EMPTY, "state_bypass_emptying_tank_b202"),
        ("state_bypass_emptying_tank_b201", "leak", MID, "state_bypass_emptying_tank_b201"),
        ("state_bypass_emptying_tank_b202", "leak", EMPTY, "state_bypass_emptying_tank_b203"),
        ("state_bypass_emptying_tank_b202", "leak", MID, "state_bypass_emptying_tank_b202"),
        ("state_bypass_emptying_tank_b203", "leak", EMPTY, "state_emptying_tank_B204"),
        ("state_bypass_emptying_tank_b203", "leak", MID, "state_bypass_emptying_tank_b203"),
        ("state_mystery", "normal", EMPTY, "state_mystery"),
    ],
)  # fmt: skip
def test_expected_next_state(mc, state, curr, fault, levels, expected):
    state.update(levels)
    assert mc.get_expected_next_state(curr, fault, state) == expected


def planning_state(state, curr, chose, fault="normal", decision=None, **levels):
    state.update(levels, curr_state=curr, condition=fault)
    state["path"] = [curr, chose] if chose else []
    if decision is None:
        state.pop("decision_levels", None)
    else:
        state["decision_levels"] = {**MID, **decision}
    return state


def test_planning_decision_correct_from_decision_levels(mc, state, capsys):
    # Live levels say "not full", but the LLM decided on full levels.
    planning_state(
        state, "state_filling_tank_B201", "state_filling_tank_B202",
        decision={"level_B201": 0.034}, level_B201=0.025,
    )  # fmt: skip
    check = mc.check_planning_decision(state)
    assert check == {
        "exit_condition_met": True,
        "expected_next": "state_filling_tank_B202",
        "llm_chose": "state_filling_tank_B202",
        "planning_correct": True,
        "error_type": None,
    }
    assert "Using decision_levels" in capsys.readouterr().out
    assert state["level_B201"] == 0.025  # original state untouched


def test_planning_decision_with_live_levels(mc, state):
    planning_state(
        state, "state_filling_tank_B201", "state_filling_tank_B202",
        decision={"level_B201": 0.034}, level_B201=0.025,
    )  # fmt: skip
    check = mc.check_planning_decision(state, use_decision_levels=False)
    assert check["planning_correct"] is False
    assert check["error_type"] == "PREMATURE_TRANSITION"


@pytest.mark.parametrize(
    "curr, chose, fault, levels, error_type",
    [
        ("state_filling_tank_B201", None, "normal", FULL, "MISSED_TRANSITION"),
        ("state_filling_tank_B202", "state_filling_tank_B203", "normal", MID, "PREMATURE_TRANSITION"),
        ("state_filling_tank_B203", "state_emptying_tank_B201", "leak", FULL, "WRONG_PATH_SHOULD_BYPASS"),
        ("state_filling_tank_B203", "state_bypass_emptying_tank_b201", "sensor_fault", FULL, "WRONG_PATH_UNNECESSARY_BYPASS"),
        ("state_filling_tank_B203", "state_emptying_tank_B204", "normal", FULL, "WRONG_STATE"),
    ],
)  # fmt: skip
def test_planning_decision_error_types(
    mc, state, curr, chose, fault, levels, error_type
):
    planning_state(state, curr, chose, fault=fault, **levels)
    check = mc.check_planning_decision(state)
    assert check["planning_correct"] is False
    assert check["error_type"] == error_type
    assert check["llm_chose"] == (chose or curr)


def test_physical_outcome(mc, state):
    assert mc.check_physical_outcome(state, {}) == {
        "valid": True,
        "violations": [],
        "violation_count": 0,
    }
    state.update(level_B202=0.0431, level_B204=-0.001, level_B203=0.042)
    result = mc.check_physical_outcome(state, {})
    assert result["valid"] is False
    assert result["violations"] == [
        "OVERFLOW: B202 level=0.0431 > max=0.033",
        "NEGATIVE_LEVEL: B204 level=-0.0010",
    ]


def test_validate_sequence_exact_normal(mc):
    seq = ["state_initialStep"] + [
        s for s in mc.EXPECTED_SEQUENCE_NORMAL for _ in (0, 1)
    ]
    result = mc.validate_sequence(seq, "sensor_fault")
    assert result["sequence_correct"] is True
    assert result["filling_correct"] and result["emptying_correct"]
    assert result["sequence_similarity"] == 1.0
    assert result["actual_count"] == 7
    assert result["missing_states"] == result["extra_states"] == []
    assert result["skipped_states"] == 0


def test_validate_sequence_wrong_path(mc):
    result = mc.validate_sequence(mc.EXPECTED_SEQUENCE_NORMAL, "pump_failure")
    assert result["expected"] == mc.EXPECTED_SEQUENCE_BYPASS
    assert result["sequence_correct"] is False
    assert result["filling_correct"] is True and result["emptying_correct"] is False
    assert result["sequence_similarity"] == pytest.approx(4 / 7)
    assert result["skipped_states"] == 3
    assert sorted(result["extra_states"]) == [
        f"state_emptying_tank_b20{i}" for i in (1, 2, 3)
    ]


def test_validate_sequence_short_and_unknown_fault(mc):
    result = mc.validate_sequence(["state_filling_tank_B201"], "made_up_fault")
    assert result["expected"] == mc.EXPECTED_SEQUENCE_NORMAL
    assert result["actual"] == ["state_filling_tank_b201"]
    assert result["filling_correct"] is False
    assert result["sequence_similarity"] == pytest.approx(1 / 7)
    assert mc.validate_sequence([], "normal")["actual_count"] == 0


def test_fault_display_name(mc):
    assert mc.get_fault_display_name("leak") == "leakage_main_pump_line_P101"
    assert mc.get_fault_display_name("normal") == "normal"
    assert mc.get_fault_display_name("cosmic_ray") == "cosmic_ray"


def test_constants_are_consistent(mc):
    assert set(mc.FAULT_TYPES) == set(mc.EXPECTED_PATHS) == set(mc.FAULT_DISPLAY_NAMES)
    assert set(mc.EXPECTED_ACTUATORS) == set(mc.VALID_STATES)
    assert set(mc.EXPECTED_SEQUENCE_BYPASS) <= set(mc.VALID_STATES)
    assert set(mc.ACTUATOR_TO_STATE_KEY.values()) == set(
        mc.EXPECTED_ACTUATORS["state_filling_tank_B201"]
    )


def test_print_metrics(mc, state, capsys):
    state.update(
        m_start_wall=mc.time.perf_counter() - 2.5,
        m_success=True, itr=8, m_llm_planning_calls=4, m_llm_planning_latency_s=2.0,
        m_actuator_correct=9, m_actuator_total=10, m_planning_correct=3,
        m_planning_total=4, m_planning_input_tokens=100, m_planning_output_tokens=20,
        m_action_input_tokens=50, m_action_output_tokens=5,
    )  # fmt: skip
    mc._print_metrics(state)
    out = capsys.readouterr().out
    assert state["m_total_wall"] == pytest.approx(2.5, abs=0.5)
    assert "success: 1" in out
    assert "actuator_accuracy: 9/10 (90.0%)" in out
    assert "planning_accuracy: 3/4 (75.0%)" in out
    assert "planning_latency_avg_s: 0.500" in out
    assert "action_latency_avg_s: 0.000" in out
    assert "Total: 175 tokens" in out


def test_print_metrics_defaults(mc, capsys):
    mc._print_metrics({"m_start_wall": 0.0})
    out = capsys.readouterr().out
    assert "success: 0" in out
    assert "actuator_accuracy: 0/0 (100.0%)" in out
