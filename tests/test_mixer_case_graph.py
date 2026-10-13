"""Offline tests for the mixer LangGraph nodes, routing and experiment drivers.

LLM calls go through a queued fake model, the KG is disabled and the stub
StateGraph below is interpreted by a tiny runner, so nothing touches the network.
"""

import copy
import json
import os
import random
import runpy
import sys
import types

import pytest

from common.llm_failures import LLMActionError


@pytest.fixture
def mc(mixer_case, monkeypatch):
    monkeypatch.setattr(mixer_case, "_KG_PLANNING_TTL", "")
    monkeypatch.setattr(mixer_case, "_KG_ACTION_TTL", "")
    monkeypatch.setattr(mixer_case, "_KG_LOADED", False)
    monkeypatch.setattr(mixer_case, "_TRACE_RECORDER", None)
    monkeypatch.setattr(mixer_case, "CURRENT_MODEL", "gpt-4o-mini")
    monkeypatch.setattr(mixer_case, "CURRENT_PROMPT_LEVEL", "normal")
    random.seed(0)
    return mixer_case


@pytest.fixture
def llm(mc, monkeypatch, fake_llm_factory):
    """Queue structured replies; every _make_llm() call shares one queue."""
    model = fake_llm_factory([])
    monkeypatch.setattr(mc, "_make_llm", lambda *a, **k: model)
    return model.structured


def with_usage(parsed, input_tokens, output_tokens):
    raw = types.SimpleNamespace(
        content=parsed.model_dump_json(),
        usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens},
    )
    return {"raw": raw, "parsed": parsed, "parsing_error": None}


class RecorderSpy:
    def __init__(self):
        self.llm_calls = []
        self.events = []

    def record_llm_call(self, **kwargs):
        self.llm_calls.append(kwargs)

    def record_event(self, event, **payload):
        self.events.append((event, payload))


def plan(mc, next_state, current="x"):
    return mc.PathPlan(current_state=current, next_state=next_state, reasoning="why")


def act(mc, target, **actuators):
    actions = [mc.ActionItem(actuator=k, value=v) for k, v in actuators.items()]
    return mc.ActionPlan(target_state=target, actions=actions, reasoning="because")


LLM_NAMES = {"pump_power": "pump_P101", "pump_power_alt": "pump_P102"}


def correct_actions(mc, target):
    cfg = mc.EXPECTED_ACTUATORS[target]
    return act(mc, target, **{LLM_NAMES.get(k, k): v for k, v in cfg.items() if v})


def fresh_state(mc, fault="normal", curr=None, **levels):
    state = mc.initializing({"condition": fault})
    if curr:
        state["curr_state"] = curr
    state.update(levels)
    return mc.monitoring(state)  # itr=1, decision_levels, current_iteration_data


def with_twin(mc, state, target, actuators=None, **twin_levels):
    """Pretend planning chose `target` and action/simulation produced a twin."""
    state["path"] = [state["curr_state"], target]
    twin = {k: v for k, v in state.items() if k != "digital_twin_states"}
    twin.update(actuators if actuators is not None else mc.EXPECTED_ACTUATORS[target])
    twin.update(twin_levels, curr_state=target, sim_time=1.5)
    state["digital_twin_states"] = twin
    state["simulation_error"] = None
    return state


# --- initializing / logging / monitoring ---------------------------------------


def test_initializing_defaults(mc):
    state = mc.initializing({"condition": "leak"})
    assert state["condition"] == "leak"
    assert state["curr_state"] == "state_initialStep"
    assert state["level_B204"] == 0.022 and state["max_height_B204"] == 0.055
    assert state["pump_main"] == state["pump_alt"] == 150
    assert state["decision_levels"] == {f"level_B20{i}": 0.022 for i in range(1, 5)}
    assert state["m_success_itr"] == -1 and state["simulation_error"] is None
    assert mc._validate_actuators(state) == 0
    assert mc.initializing({})["condition"] == "normal"


def test_logging_current_state_appends_one_row(mc):
    state = mc.initializing({})
    mc.logging_current_state(state)
    state["level_B201"] = 0.03
    mc.logging_current_state(state)
    assert state["log_level_B201"] == [0.022, 0.03]
    assert state["log_curr_state"] == ["state_initialStep"] * 2
    assert len(state["log_pump_power_alt"]) == 2
    assert state["log_time"] == []  # monitoring owns the time log


def test_monitoring_advances_and_snapshots(mc, capsys):
    state = mc.initializing({})
    state.update(sim_time=2.0, level_B202=0.03, valve_in_B201=3)
    mc.monitoring(state)
    state["sim_time"] = 1.5
    mc.monitoring(state)
    assert state["itr"] == 2 and state["m_step_count"] == 2
    assert state["log_time"] == [2.0, 3.5]
    assert state["decision_levels"]["level_B202"] == 0.03
    assert state["m_invalid_actuator_count"] == 2  # one bad valve, counted twice
    data = state["current_iteration_data"]
    assert data["iteration"] == 2 and data["planning_curr_state"] == "state_initialStep"
    assert data["actuators_total"] == 10 and data["reprompt_time_s"] == 0.0
    assert state["m_success"] is False
    assert "[MONITORING] Iteration 2" in capsys.readouterr().out


def test_monitoring_marks_success_once(mc):
    state = mc.initializing({})
    state["curr_state"] = "state_emptying_tank_B204"
    state["sim_time"] = 4.0
    mc.monitoring(state)
    mc.monitoring(state)
    assert state["m_success"] is True
    assert state["m_success_itr"] == 1
    assert state["m_success_sim_time"] == 4.0


# --- planning node ---------------------------------------------------------------


def test_planning_accepts_valid_state_and_counts_tokens(mc, llm):
    state = fresh_state(mc)
    llm.responses.append(with_usage(plan(mc, "STATE_FILLING_TANK_B201"), 120, 30))
    mc.planning(state)
    assert state["path"] == ["state_initialStep", "state_filling_tank_B201"]
    assert state["m_planning_input_tokens"] == 120
    assert state["m_planning_output_tokens"] == 30
    assert state["m_llm_planning_calls"] == 1
    assert state["log_paths"][-1]["valid"] is True
    assert state["log_paths"][-1]["llm_choice"] == "STATE_FILLING_TANK_B201"
    data = state["current_iteration_data"]
    assert data["planning_chosen_state"] == "state_filling_tank_B201"
    assert data["planning_expected_state"] == "state_filling_tank_B201"
    assert data["planning_correct"] is True
    assert data["planning_input_tokens"] == 120
    (system, system_msg), (user, user_msg) = llm.prompts[0]
    assert (system, user) == ("system", "user")
    assert "## PLANNING TASK" in system_msg and "## EXAMPLES" in system_msg
    assert "Current State: state_initialStep" in user_msg
    assert "PREVIOUS ATTEMPT" not in user_msg


def test_planning_minimal_prompt_has_no_examples(mc, llm, monkeypatch):
    monkeypatch.setattr(mc, "CURRENT_PROMPT_LEVEL", "minimal")
    llm.responses.append(plan(mc, "state_filling_tank_B201"))
    mc.planning(fresh_state(mc, fault="leak"))
    system_msg, user_msg = llm.prompts[0][0][1], llm.prompts[0][1][1]
    assert "## EXAMPLES" not in system_msg
    assert "Fault Condition: leakage_main_pump_line_P101" in user_msg


def test_planning_feeds_back_and_clears_previous_error(mc, llm):
    state = fresh_state(mc)
    state["simulation_error"] = "MISSED_TRANSITION: go on"
    llm.responses.append(plan(mc, "state_filling_tank_B201"))
    mc.planning(state)
    user_msg = llm.prompts[0][1][1]
    assert "PREVIOUS ATTEMPT RESULTED IN ERROR:\nMISSED_TRANSITION: go on" in user_msg
    assert "- B201: 0.0220 (max: 0.033, min: 0.022)" in user_msg
    assert state["simulation_error"] is None


def test_planning_rejects_hallucinated_state(mc, llm):
    state = fresh_state(mc, curr="state_filling_tank_B202")
    llm.responses += [plan(mc, "state_mixing_tank_B999"), plan(mc, "done")]
    mc.planning(state)
    assert state["path"] == ["state_filling_tank_B202"] * 2
    assert state["simulation_error"].startswith(
        "INVALID STATE CHOSEN: 'state_mixing_tank_B999'"
    )
    assert state["log_paths"][-1]["valid"] is False
    assert (
        state["current_iteration_data"]["planning_correct"] is True
    )  # stayed, not full
    mc.planning(state)
    assert state["m_invalid_states"] == 2
    assert mc.reprompt_target(state) == "planning"


def test_planning_without_iteration_data(mc, llm):
    state = mc.initializing({})
    llm.responses.append(plan(mc, "state_filling_tank_B201"))
    mc.planning(state)
    assert state["current_iteration_data"] == {}
    assert len(state["log_paths"]) == 1


@pytest.mark.parametrize(
    "exc, failure_type",
    [
        (TimeoutError("Request timed out"), "timeout"),
        (ConnectionError("Connection refused"), "connection_error"),
        (json.JSONDecodeError("bad", "{", 0), "json_parse_error"),
        (RuntimeError("boom"), "unknown_error"),
    ],
)
def test_planning_invoke_failure_is_classified(mc, llm, monkeypatch, exc, failure_type):
    spy = RecorderSpy()
    monkeypatch.setattr(mc, "_TRACE_RECORDER", spy)
    state = fresh_state(mc)
    llm.responses.append(exc)
    with pytest.raises(LLMActionError) as err:
        mc.planning(state)
    assert err.value.failure_type == failure_type
    assert "LLM planning call failed" in str(err.value)
    assert "model=gpt-4o-mini" in str(err.value)
    assert err.value.raw_response is None
    assert state["m_llm_planning_calls"] == 1
    assert state["path"] == []
    (call,) = spy.llm_calls
    assert call["agent"] == "planning" and call["failure_type"] == failure_type
    assert call["parsed_response"] is None and call["iteration"] == 1


def test_planning_unparsed_reply_is_schema_error(mc, llm):
    raw = types.SimpleNamespace(content="Sure! Next is B202.", usage_metadata=None)
    llm.responses.append({"raw": raw, "parsed": None, "parsing_error": "bad"})
    with pytest.raises(LLMActionError) as err:
        mc.planning(fresh_state(mc))
    assert err.value.failure_type == "schema_validation_error"
    assert "raw_content='Sure! Next is B202.'" in str(err.value)
    assert err.value.raw_response is raw


def test_planning_records_trace(mc, llm, monkeypatch):
    spy = RecorderSpy()
    monkeypatch.setattr(mc, "_TRACE_RECORDER", spy)
    llm.responses.append(with_usage(plan(mc, "state_filling_tank_B201"), 5, 2))
    mc.planning(fresh_state(mc))
    (call,) = spy.llm_calls
    assert call["rationale"] == "why"
    assert call["usage"] == {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
    assert [m["role"] for m in call["messages"]] == ["system", "user"]
    assert call["failure_type"] is None


@pytest.mark.xfail(
    strict=True,
    reason="mixer_case.py:206/3917 default prompt level 'full' is not in PROMPT_LEVELS",
)
def test_default_prompt_level_is_supported(mixer_case):
    mixer_case.get_prompts_for_level(mixer_case.CURRENT_PROMPT_LEVEL)


# --- action node -----------------------------------------------------------------


def test_action_applies_plan_to_twin_only(mc, llm, capsys):
    state = fresh_state(mc, curr="state_filling_tank_B203")
    state["path"] = ["state_filling_tank_B203", "state_emptying_tank_B201"]
    state["current_iteration_data"][
        "planning_chosen_state"
    ] = "state_emptying_tank_B201"
    llm.responses.append(
        with_usage(
            act(mc, "state_emptying_tank_B201", valve_pump_B201=1, valve_in_B204=1,
                pump_P101=1.0, valve_x=1),
            40, 9,
        )
    )  # fmt: skip
    mc.action(state)
    twin = state["digital_twin_states"]
    assert twin["curr_state"] == "state_emptying_tank_B201"
    assert state["curr_state"] == "state_filling_tank_B203"
    assert (twin["valve_out_B201"], twin["valve_in_B204"], twin["pump_power"]) == (
        1,
        1,
        1.0,
    )
    assert state["valve_out_B201"] == 0 and state["pump_power"] == 0
    assert state["m_action_input_tokens"] == 40 and state["m_action_output_tokens"] == 9
    assert state["m_llm_action_calls"] == 1
    logged = state["log_actions_raw"][-1]
    assert logged["target"] == "state_emptying_tank_B201"
    assert {"actuator": "valve_x", "value": 1.0} in logged["actions"]
    data = state["current_iteration_data"]
    assert data["action_warnings"] == "valve_x"
    assert data["action_raw_output"].startswith(
        "valve_pump_B201=1.0, valve_in_B204=1.0"
    )
    assert data["ist_valve_out_B201"] == 1 and data["ist_pump_power"] == 1.0
    assert data["soll_valve_out_B201"] == 1 and data["soll_pump_power"] == 1.0
    assert data["soll_pump_power_alt"] == 0.0
    assert mc.check_actuators(twin, "state_emptying_tank_B201")["accuracy"] == 1.0
    assert "Unknown actuator: valve_x" in capsys.readouterr().out
    prompt = llm.prompts[0][1][1]
    assert prompt.startswith("TARGET STATE: state_emptying_tank_B201")


def test_action_soll_falls_back_to_target_for_unknown_choice(mc, llm):
    state = fresh_state(mc)
    state["path"] = ["state_initialStep", "state_filling_tank_B201"]
    state["current_iteration_data"]["planning_chosen_state"] = "state_bogus"
    llm.responses.append(correct_actions(mc, "state_filling_tank_B201"))
    mc.action(state)
    data = state["current_iteration_data"]
    assert data["ist_valve_in_B201"] == 1
    assert data["soll_valve_in_B201"] == 0  # no expected config for "state_bogus"


def test_action_without_path_targets_current_state(mc, llm):
    state = mc.initializing({})
    state["curr_state"] = "state_filling_tank_B202"
    llm.responses.append(correct_actions(mc, "state_filling_tank_B202"))
    mc.action(state)
    assert state["digital_twin_states"]["curr_state"] == "state_filling_tank_B202"
    assert state["digital_twin_states"]["valve_in_B202"] == 1
    assert state["current_iteration_data"] == {}


def test_action_feeds_back_and_clears_error(mc, llm, monkeypatch, capsys):
    monkeypatch.setattr(mc, "_KG_LOADED", True)
    monkeypatch.setattr(mc, "_KG_ACTION_TTL", ":s :doAction :a .")
    state = fresh_state(mc)
    state["path"] = ["state_initialStep", "state_filling_tank_B201"]
    state["simulation_error"] = "ACTUATOR_ERROR: wrong valves"
    llm.responses.append(correct_actions(mc, "state_filling_tank_B201"))
    mc.action(state)
    system_msg, user_msg = llm.prompts[0][0][1], llm.prompts[0][1][1]
    assert ":s :doAction :a ." in system_msg
    assert (
        "PREVIOUS ACTUATOR CONFIGURATION RESULTED IN ERROR:\nACTUATOR_ERROR" in user_msg
    )
    assert state["simulation_error"] is None
    assert "[ACTION KG]" in capsys.readouterr().out


def test_action_failure_is_classified(mc, llm, monkeypatch):
    spy = RecorderSpy()
    monkeypatch.setattr(mc, "_TRACE_RECORDER", spy)
    state = fresh_state(mc)
    state["path"] = ["state_initialStep", "state_filling_tank_B201"]
    llm.responses.append(ConnectionError("Max retries exceeded"))
    with pytest.raises(LLMActionError) as err:
        mc.action(state)
    assert err.value.failure_type == "connection_error"
    assert "LLM action call failed" in str(err.value)
    assert state["log_actions_raw"] == []
    assert spy.llm_calls[0]["agent"] == "action"


def test_action_unparsed_reply_is_schema_error(mc, llm):
    state = fresh_state(mc)
    llm.responses.append({"raw": "plain string reply", "parsed": None})
    with pytest.raises(LLMActionError) as err:
        mc.action(state)
    assert err.value.failure_type == "schema_validation_error"
    assert "raw_content='plain string reply'" in str(err.value)


def test_action_records_trace(mc, llm, monkeypatch):
    spy = RecorderSpy()
    monkeypatch.setattr(mc, "_TRACE_RECORDER", spy)
    state = fresh_state(mc)
    state["path"] = ["state_initialStep", "state_filling_tank_B201"]
    llm.responses.append(correct_actions(mc, "state_filling_tank_B201"))
    mc.action(state)
    (call,) = spy.llm_calls
    assert call["rationale"] == "because" and call["failure_type"] is None
    assert call["usage"]["total_tokens"] == 0


# --- simulation / plant nodes ----------------------------------------------------

FULL = 0.034
EMPTY_TARGET = {
    "state_filling_tank_B201": {},
    "state_filling_tank_B202": {},
    "state_filling_tank_B203": {},
    "state_emptying_tank_B201": {"level_B201": FULL},
    "state_bypass_emptying_tank_b201": {"level_B201": FULL},
    "state_emptying_tank_B202": {"level_B202": FULL},
    "state_bypass_emptying_tank_b202": {"level_B202": FULL},
    "state_emptying_tank_B203": {"level_B203": FULL},
    "state_bypass_emptying_tank_b203": {"level_B203": FULL},
    "state_emptying_tank_B204": {"level_B204": 0.05},
}


def expected_change(target, levels):
    tank = target[-4:].upper()
    if "filling" in target:
        assert levels[f"level_{tank}"] >= 0.033
    else:
        assert levels[f"level_{tank}"] <= 0.022
        if tank != "B204":
            assert levels["level_B204"] == pytest.approx(
                0.022 + FULL - levels[f"level_{tank}"]
            )


@pytest.mark.parametrize("target", list(EMPTY_TARGET))
def test_simulation_node_runs_twin_for_target(mc, target):
    state = fresh_state(mc)
    with_twin(mc, state, target, **EMPTY_TARGET[target])
    state["simulation_error"] = "stale"
    before = {
        k: state[k] for k in ("level_B201", "level_B202", "level_B203", "level_B204")
    }
    mc.simulation(state)
    twin = state["digital_twin_states"]
    assert state["simulation_error"] is None
    assert twin["sim_time"] > 0
    expected_change(target, twin)
    assert {k: state[k] for k in before} == before  # plant state untouched
    assert state["current_iteration_data"]["simulation_time_s"] >= 0


def test_simulation_node_unknown_state_is_noop(mc):
    state = fresh_state(mc)
    with_twin(mc, state, "state_initialStep", actuators={})
    mc.simulation(state)
    assert state["digital_twin_states"]["sim_time"] == 0
    assert state["simulation_error"] is None


def test_simulation_node_captures_twin_error(mc):
    state = fresh_state(mc)
    with_twin(mc, state, "state_emptying_tank_B201", actuators={"valve_out_B201": 1},
              level_B201=FULL)  # fmt: skip
    mc.simulation(state)
    assert "cannot empty the tankB201" in state["simulation_error"]
    assert state["should_reprompt"] is True
    assert mc.reprompt_target(state) == "action"


@pytest.mark.parametrize("target", list(EMPTY_TARGET))
def test_plant_node_runs_physics_on_state(mc, target):
    state = mc.initializing({})
    state.update(
        mc.EXPECTED_ACTUATORS[target], curr_state=target, **EMPTY_TARGET[target]
    )
    mc.plant(state)
    assert state["sim_time"] > 0
    expected_change(target, state)
    assert state["curr_state"] == target


def test_plant_node_initial_state_only_resets_time(mc):
    state = mc.initializing({})
    state["sim_time"] = 9.0
    mc.plant(state)
    assert state["sim_time"] == 0
    assert state["level_B201"] == 0.022


def test_plant_node_propagates_twin_errors(mc):
    state = mc.initializing({})
    state["curr_state"] = "state_filling_tank_B201"
    with pytest.raises(ValueError, match="Valve in B201 is closed"):
        mc.plant(state)


# --- evaluation node -------------------------------------------------------------


def test_evaluation_accepts_correct_step(mc, monkeypatch):
    spy = RecorderSpy()
    monkeypatch.setattr(mc, "_TRACE_RECORDER", spy)
    state = fresh_state(mc)
    state["current_iteration_data"]["planning_chosen_state"] = "state_filling_tank_B201"
    with_twin(mc, state, "state_filling_tank_B201", level_B201=0.0335)
    mc.evaluation(state)
    assert state["should_reprompt"] is False
    assert state["curr_state"] == "state_filling_tank_B201"
    assert state["level_B201"] == 0.0335 and state["valve_in_B201"] == 1
    assert state["sim_time"] == 1.5
    assert (state["m_planning_correct"], state["m_planning_total"]) == (1, 1)
    assert (state["m_actuator_correct"], state["m_actuator_total"]) == (10, 10)
    assert state["log_planning_checks"][-1]["exit_condition_met"] is True
    assert state["log_actuator_checks"][-1]["planning_was_correct"] is True
    detail = state["log_iteration_details"][-1]
    assert detail["was_reprompted"] is False and detail["actuators_accuracy"] == 1.0
    assert detail["iteration_total_time_s"] >= detail["evaluation_time_s"] >= 0
    event, payload = spy.events[-1]
    assert event == "validation_decision"
    assert payload["accepted"] is True and payload["condition"] == "nominal"


def test_evaluation_stays_in_state_when_llm_stays_correctly(mc):
    state = fresh_state(mc, curr="state_filling_tank_B202", level_B202=0.025)
    with_twin(mc, state, "state_filling_tank_B202", level_B202=0.034)
    mc.evaluation(state)
    assert state["should_reprompt"] is False
    assert state["curr_state"] == "state_filling_tank_B202"
    assert state["level_B202"] == 0.034


@pytest.mark.parametrize(
    "curr, target, fault, decision, error_type, phrase",
    [
        ("state_filling_tank_B202", "state_filling_tank_B203", "normal", {"level_B202": 0.025},
         "PREMATURE_TRANSITION", "PREMATURE_TRANSITION: You chose to transition to 'state_filling_tank_B203'"),
        ("state_filling_tank_B202", "state_filling_tank_B202", "normal", {"level_B202": FULL},
         "MISSED_TRANSITION", "MISSED_TRANSITION: You chose to stay in 'state_filling_tank_B202'"),
        ("state_filling_tank_B203", "state_emptying_tank_B201", "pump_failure", {"level_B203": FULL},
         "WRONG_PATH_SHOULD_BYPASS", "WRONG_STATE: Incorrect state transition chosen."),
    ],
)  # fmt: skip
def test_evaluation_rolls_back_wrong_plan(
    mc, curr, target, fault, decision, error_type, phrase
):
    state = fresh_state(mc, fault=fault, curr=curr, **decision)
    state["current_iteration_data"]["planning_chosen_state"] = target
    with_twin(mc, state, target, level_B201=0.03, level_B202=0.03, level_B203=0.03)
    state["level_B202"] = 0.0299  # live level drifted; rollback restores decision level
    mc.evaluation(state)
    assert state["should_reprompt"] is True
    assert state["curr_state"] == curr
    assert phrase in state["simulation_error"]
    assert state["level_B202"] == state["decision_levels"]["level_B202"]
    assert state["log_planning_checks"][-1]["error_type"] == error_type
    assert state["m_missed_transitions"] == (error_type == "MISSED_TRANSITION")
    assert state["m_planning_correct"] == 0
    assert state["m_actuator_total"] == 10  # actuators are checked even on rollback
    detail = state["log_iteration_details"][-1]
    assert detail["was_reprompted"] is True
    assert detail["reprompt_reason"] == state["simulation_error"][:200]
    assert mc.reprompt_target(state) == "planning"


def test_evaluation_flags_physical_violation(mc):
    state = fresh_state(mc)
    with_twin(mc, state, "state_filling_tank_B201", level_B201=0.05)
    mc.evaluation(state)
    assert state["should_reprompt"] is True
    assert state["m_physical_violations"] == 1
    assert state["simulation_error"].startswith(
        "PHYSICAL VIOLATION after correct planning"
    )
    assert state["curr_state"] == "state_initialStep"
    assert mc.reprompt_target(state) == "planning"


def test_evaluation_flags_actuator_error_after_correct_plan(mc):
    state = fresh_state(mc)
    wrong = dict(mc.EXPECTED_ACTUATORS["state_filling_tank_B201"], valve_in_B202=1)
    with_twin(mc, state, "state_filling_tank_B201", actuators=wrong, level_B201=0.0335)
    mc.evaluation(state)
    assert state["should_reprompt"] is True
    assert state["simulation_error"].startswith("ACTUATOR_ERROR")
    assert state["m_actuator_correct"] == 9
    assert state["curr_state"] == "state_initialStep"
    assert mc.reprompt_target(state) == "action"


def test_evaluation_simulation_error_blames_actuators(mc, monkeypatch):
    spy = RecorderSpy()
    monkeypatch.setattr(mc, "_TRACE_RECORDER", spy)
    state = fresh_state(mc, fault="leak")
    with_twin(mc, state, "state_bypass_emptying_tank_b201", actuators={})
    state["simulation_error"] = (
        "['valve_in_B204'] is closed, cannot empty the tankB201."
    )
    mc.evaluation(state)
    assert state["should_reprompt"] is True
    assert state["simulation_error"].startswith("ACTUATOR_ERROR")
    assert "'state_bypass_emptying_tank_b201'" in state["simulation_error"]
    assert state["m_actuator_total"] == 10 and state["m_actuator_correct"] < 10
    assert state["m_planning_total"] == 0  # planning not judged on a failed sim
    event, payload = spy.events[-1]
    assert payload["accepted"] is False and payload["condition"] == "fault"


def test_evaluation_simulation_error_with_correct_actuators_keeps_message(mc):
    state = fresh_state(mc)
    with_twin(mc, state, "state_emptying_tank_B204")
    state["simulation_error"] = "something else went wrong"
    mc.evaluation(state)
    assert state["simulation_error"] == "something else went wrong"
    assert state["should_reprompt"] is True


def test_evaluation_without_decision_levels_uses_twin_levels(mc, capsys):
    state = fresh_state(mc, curr="state_filling_tank_B201")
    del state["decision_levels"]
    state["current_iteration_data"] = {}
    with_twin(mc, state, "state_filling_tank_B202", level_B201=FULL)
    mc.evaluation(state)
    assert "decision_levels NOT SET" in capsys.readouterr().out
    assert state["log_planning_checks"][-1]["decision_levels"]["level_B201"] == FULL
    assert state["log_iteration_details"] == []


def test_evaluation_initializes_missing_metrics(mc):
    state = fresh_state(mc)
    with_twin(mc, state, "state_filling_tank_B201", level_B201=0.0335)
    for key in ("m_planning_correct", "m_actuator_total", "log_planning_checks"):
        del state[key]
    mc.evaluation(state)
    assert state["m_planning_correct"] == 1
    assert state["m_actuator_total"] == 10
    assert len(state["log_planning_checks"]) == 1


# --- reprompting and routing -----------------------------------------------------


def test_reprompting_counts_and_logs(mc):
    state = fresh_state(mc)
    state["simulation_error"] = "boom"
    mc.reprompting(state)
    mc.reprompting(state)
    assert state["m_reprompt_count"] == 2
    assert (
        state["log_reprompts"]
        == [{"itr": 1, "curr_state": "state_initialStep", "error": "boom"}] * 2
    )
    assert state["current_iteration_data"]["reprompt_time_s"] >= 0
    mc.reprompting(mc.initializing({}))  # no iteration data -> no timing update


def test_should_reprompt_routes(mc):
    state = fresh_state(mc)
    state["should_reprompt"] = False
    assert mc.should_reprompt(state) == "plant"
    state["should_reprompt"] = True
    state["log_reprompts"] = [{"itr": 1}] * 4 + [{"itr": 0}] * 3
    assert mc.should_reprompt(state) == "reprompting"


def test_should_reprompt_aborts_at_limit(mc, capsys):
    state = fresh_state(mc)
    state["should_reprompt"] = True
    state["log_reprompts"] = [{"itr": 1}] * 5
    assert mc.should_reprompt(state) == mc.END
    assert state["m_abort_reason"] == "reprompt_limit"
    assert state["should_reprompt"] is False and state["m_success"] is False
    detail = state["log_iteration_details"][-1]
    assert detail["aborted"] is True
    assert detail["reprompt_reason"] == "REPROMPT_LIMIT_REACHED: 5 attempts"
    assert "REPROMPT LIMIT" in capsys.readouterr().out


def test_should_reprompt_limit_without_iteration_data(mc):
    state = mc.initializing({})
    state.update(should_reprompt=True, log_reprompts=[{"itr": 0}] * 6)
    assert mc.should_reprompt(state) == mc.END
    assert state["log_iteration_details"] == []


@pytest.mark.parametrize(
    "error, route",
    [
        ("ACTUATOR_ERROR: wrong", "action"),
        ("actuator error in pump", "action"),
        ("['pump_main'] is closed, cannot empty the tankB202.", "action"),
        ("Valve in B203 is closed, cannot fill the tankB203.", "action"),
        ("MISSED_TRANSITION: stay", "planning"),
        ("planning_error", "planning"),
        ("Tank B201 already FULL", "planning"),
        ("INVALID STATE CHOSEN: 'x'", "planning"),
        ("Overflow in B204", "planning"),
        ("exit condition not met", "planning"),
        ("something odd", "action"),
        ("", "action"),
        (None, "action"),
    ],
)
def test_reprompt_target(mc, error, route):
    assert mc.reprompt_target({"simulation_error": error}) == route
    if error is None:
        assert mc.reprompt_target({}) == route


def test_should_act_ends_on_success(mc, capsys):
    state = fresh_state(mc, curr="state_emptying_tank_B204", level_B204=0.021)
    assert mc.should_act(state) == mc.END
    assert state["m_abort_reason"] == "success"
    assert "Process complete" in capsys.readouterr().out


def test_should_act_ends_on_iteration_cap(mc):
    state = fresh_state(mc, curr="state_filling_tank_B201")
    state["itr"] = 31
    assert mc.should_act(state) == mc.END
    assert state["m_abort_reason"] == "max_iterations"


def test_should_act_plans_or_runs_plant(mc):
    state = fresh_state(mc, curr="state_emptying_tank_B204", level_B204=0.03)
    assert mc.should_act(state) == "planning"  # should_act flag set by monitoring
    state["should_act"] = False
    assert mc.should_act(state) == "plant"
    state["level_B204"] = 0.0
    state["curr_state"] = "state_bypass_emptying_tank_b203"
    assert mc.should_act(state) == "planning"  # exit condition met


# --- graph builder and end-to-end runs -------------------------------------------


class StubStateGraph:
    """Records the topology; compile() returns an interpreter for it."""

    def __init__(self, schema):
        self.schema = schema
        self.nodes, self.edges, self.conditional = {}, [], {}

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def add_edge(self, start, end):
        self.edges.append((start, end))

    def add_conditional_edges(self, source, fn, targets):
        self.conditional[source] = (fn, list(targets))

    def compile(self, **_kwargs):
        return GraphRunner(self)


class GraphRunner:
    def __init__(self, builder):
        self.builder = builder

    def invoke(self, input, config=None):
        mc = sys.modules["mixer_case"]
        state = dict(input)
        edges = dict(self.builder.edges)
        node = edges[mc.START]
        for _ in range(config["recursion_limit"]):
            state = self.builder.nodes[node](state)
            if node in self.builder.conditional:
                fn, targets = self.builder.conditional[node]
                node = fn(state)
                assert node in targets
            else:
                node = edges[node]
            if node == mc.END:
                return state
        raise RuntimeError("recursion limit")


def test_build_graph_topology(mc, monkeypatch):
    monkeypatch.setattr(mc, "StateGraph", StubStateGraph)
    builder = mc.build_graph().builder
    assert builder.schema is mc.GraphState
    assert builder.nodes == {
        name: getattr(mc, name)
        for name in ("initializing", "monitoring", "planning", "action",
                     "simulation", "evaluation", "reprompting", "plant")
    }  # fmt: skip
    assert builder.edges == [
        (mc.START, "initializing"),
        ("initializing", "monitoring"),
        ("planning", "action"),
        ("action", "simulation"),
        ("simulation", "evaluation"),
        ("plant", "monitoring"),
    ]
    assert builder.conditional == {
        "evaluation": (mc.should_reprompt, ["reprompting", "plant", mc.END]),
        "reprompting": (mc.reprompt_target, ["planning", "action"]),
        "monitoring": (mc.should_act, ["planning", mc.END]),
    }


def scripted_run(mc, fault):
    """Planning/action replies for a perfect run, plus one wrong first plan."""
    path = (
        mc.EXPECTED_SEQUENCE_BYPASS
        if mc.is_pump_fault(fault)
        else mc.EXPECTED_SEQUENCE_NORMAL
    )
    replies = [
        plan(mc, "state_filling_tank_B202"),
        correct_actions(mc, "state_filling_tank_B202"),
    ]
    for target in path:
        replies += [
            with_usage(plan(mc, target), 10, 2),
            with_usage(correct_actions(mc, target), 7, 1),
        ]
    return replies


@pytest.mark.parametrize("fault", ["normal", "leak"])
def test_run_single_experiment_end_to_end(
    mc, llm, monkeypatch, tmp_path, fault, capsys
):
    monkeypatch.setattr(mc, "StateGraph", StubStateGraph)
    monkeypatch.setattr(mc, "SPARQL_AVAILABLE", False)
    llm.responses += scripted_run(mc, fault)
    metrics = mc.run_single_experiment(
        mc.build_graph(), fault, run=2, trace_dir=str(tmp_path), experiment_id="exp"
    )
    assert llm.responses == []
    assert mc._TRACE_RECORDER is None
    assert metrics["success"] == 1 and metrics["reached_end"] == 1
    assert metrics["abort_reason"] == "success"
    assert metrics["final_state"] == "state_emptying_tank_B204"
    assert metrics["sequence_correct"] == 1 and metrics["correct_path"] == 1
    assert metrics["reprompts"] == 1
    assert (metrics["planning_calls"], metrics["action_calls"]) == (8, 8)
    assert (metrics["planning_correct"], metrics["planning_total"]) == (7, 8)
    assert metrics["planning_accuracy"] == pytest.approx(7 / 8)
    assert metrics["actuator_accuracy"] == 1.0  # the detour's actuators fit its target
    assert metrics["total_tokens"] == 7 * (12 + 8)
    assert metrics["transitions_attempted"] == 7
    assert metrics["first_try_rate_planning"] == pytest.approx(6 / 7)
    assert metrics["attempts_per_transition"] == pytest.approx(8 / 7)
    assert metrics["process_completion"] == 1.0
    assert metrics["physical_violations"] == 0
    assert "SUCCESS: ✅ YES" in capsys.readouterr().out

    run_dir = tmp_path / "mixer" / "exp" / "gpt-4o-mini" / fault / "run_002"
    final = json.loads((run_dir / "final_result.json").read_text(encoding="utf-8"))
    assert final["status"] == "completed" and final["result"]["success"] == 1
    condition = "nominal" if fault == "normal" else "fault"
    trajectory = run_dir / "trajectories" / f"accepted_{condition}_plant.csv"
    header, *rows = trajectory.read_text(encoding="utf-8").splitlines()
    assert header.startswith("iteration,fault_condition,time,state,")
    assert len(rows) == 8
    assert len(list((run_dir / "prompts").iterdir())) == 16
    events = (run_dir / "events.jsonl").read_text(encoding="utf-8")
    assert '"sparql_unavailable"' in events


def test_run_single_experiment_failure_is_traced(mc, tmp_path):
    class Exploding:
        def invoke(self, input, config):
            raise LLMActionError("timeout", "planning timed out")

    with pytest.raises(LLMActionError):
        mc.run_single_experiment(
            Exploding(), "leak", verbose=False, trace_dir=str(tmp_path)
        )
    assert mc._TRACE_RECORDER is None
    (final,) = tmp_path.rglob("final_result.json")
    data = json.loads(final.read_text(encoding="utf-8"))
    assert data["status"] == "error"
    assert data["result"]["failure_type"] == "timeout"
    assert data["result"]["error"] == "planning timed out"


class CannedGraph:
    def __init__(self, result):
        self.result = result

    def invoke(self, input, config):
        assert config == {"recursion_limit": 500}
        return {"condition": input["condition"], **copy.deepcopy(self.result)}


@pytest.mark.parametrize(
    "result, abort_reason",
    [
        ({"curr_state": "state_filling_tank_B202", "itr": 31}, "max_iterations"),
        ({"curr_state": "state_filling_tank_B202", "itr": 4, "m_reprompt_count": 5}, "reprompt_limit"),
        ({"curr_state": "state_emptying_tank_B204", "level_B204": 0.03}, "unknown"),
        ({}, "unknown"),
    ],
)  # fmt: skip
def test_run_single_experiment_abort_reasons(mc, result, abort_reason):
    metrics = mc.run_single_experiment(CannedGraph(result), "normal", verbose=False)
    assert metrics["abort_reason"] == abort_reason
    assert metrics["success"] == 0


def test_run_single_experiment_iteration_metrics(mc, capsys):
    details = [
        {"iteration": 1, "planning_correct": True, "actuators_accuracy": 1.0},
        {"iteration": 1, "planning_correct": False, "actuators_accuracy": 0.0},
        {"iteration": 2, "planning_correct": False, "actuators_accuracy": 0.5},
        {"itr": 3, "planning_correct": True, "actuators_accuracy": 1.0},
    ]
    result = {
        "curr_state": "state_bypass_emptying_tank_b202",
        "log_curr_state": ["state_initialStep", "state_filling_tank_B201", "state_bogus"],
        "log_iteration_details": details,
        "m_llm_planning_calls": 6, "m_llm_action_calls": 3,
        "m_llm_planning_latency_s": 3.0, "m_llm_action_latency_s": 0.6,
        "m_actuator_correct": 18, "m_actuator_total": 20,
    }  # fmt: skip
    metrics = mc.run_single_experiment(
        CannedGraph(result), "pump_failure", verbose=True
    )
    assert metrics["transitions_attempted"] == 3
    assert metrics["first_try_rate_planning"] == pytest.approx(2 / 3)
    assert metrics["first_try_rate_action"] == pytest.approx(2 / 3)
    assert metrics["attempts_per_transition"] == 2.0
    assert metrics["process_completion"] == pytest.approx(3 / 7)
    assert metrics["planning_latency_avg"] == 0.5
    assert metrics["action_latency_avg"] == pytest.approx(0.2)
    assert metrics["actuator_accuracy"] == 0.9 and metrics["planning_accuracy"] == 1.0
    assert metrics["correct_path"] == 0  # pump fault but no bypass state visited
    assert metrics["expected_path"] == "bypass"
    assert metrics["filling_correct"] == 0 and metrics[
        "sequence_similarity"
    ] == pytest.approx(1 / 7)
    assert metrics["iteration_details"] == details
    out = capsys.readouterr().out
    assert "Extra states: ['state_bogus']" in out
    assert "Missing states:" in out and "SUCCESS: ❌ NO" in out


SUCCESS_RESULT = {
    "curr_state": "state_emptying_tank_B204",
    "itr": 8,
    "level_B204": 0.02,
    "m_success": True,
    "log_curr_state": ["state_initialStep"],  # replaced per fault below
    "log_iteration_details": [
        {"iteration": 1, "planning_correct": True, "action_raw_output": "a=1"}
    ],
}


class FaultAwareGraph:
    """Succeeds for faults with a bypass path, raises for everything else."""

    def __init__(self, mc):
        self.mc = mc

    def invoke(self, input, config):
        fault = input["condition"]
        if fault == "sensor_fault":
            raise LLMActionError("json_parse_error", "prose instead of JSON")
        result = dict(SUCCESS_RESULT, condition=fault)
        result["log_curr_state"] = list(self.mc.EXPECTED_SEQUENCE_BYPASS)
        return result


def test_run_ablation_study_writes_csvs(mc, tmp_path, monkeypatch):
    monkeypatch.setattr(mc, "CURRENT_MODEL", "gpt-4.1-mini")
    out = tmp_path / "results"
    df = mc.run_ablation_study(
        FaultAwareGraph(mc), ["leak", "sensor_fault"], n_runs=2, output_dir=str(out)
    )
    assert len(df) == 4
    assert list(df["run"]) == [0, 1, 0, 1]
    leak, sensor = df[df.fault == "leak"], df[df.fault == "sensor_fault"]
    assert set(leak["success"]) == {1} and set(leak["abort_reason"]) == {"success"}
    assert set(sensor["abort_reason"]) == {"exception"}
    assert set(sensor["failure_type"]) == {"json_parse_error"}

    files = sorted(p.name for p in out.iterdir())
    assert len(files) == 3
    runs_csv, iterations_csv, summary_csv = (
        next(out.glob(f"ablation_4.1mini_normal_*_{kind}.csv"))
        for kind in ("runs", "iterations", "summary")
    )
    import pandas as pd

    runs = pd.read_csv(runs_csv)
    assert list(runs.columns[:5]) == [
        "model",
        "prompt_level",
        "fault",
        "run",
        "success",
    ]
    assert "failure_type" not in runs.columns
    iterations = pd.read_csv(iterations_csv)
    assert len(iterations) == 2  # only the two successful runs carry details
    assert set(iterations["action_raw_output"]) == {"a=1"}
    summary = pd.read_csv(summary_csv).set_index("fault")
    assert summary.loc["leak", "success_mean"] == 1.0
    assert summary.loc["sensor_fault", "success_sum"] == 0
    assert summary.loc["leak", "aborted_success"] == 2


def test_run_ablation_study_single_run_traces(mc, tmp_path):
    df = mc.run_ablation_study(
        FaultAwareGraph(mc), ["pump_failure"], n_runs=1,
        output_dir=str(tmp_path / "out"), trace_dir=str(tmp_path / "traces"),
    )  # fmt: skip
    assert df.loc[0, "success"] == 1
    assert len(list((tmp_path / "traces").rglob("final_result.json"))) == 1
    assert len(list((tmp_path / "traces").rglob("accepted_fault_plant.csv"))) == 1


# --- CLI ---------------------------------------------------------------------------


class _DupStdout:
    @staticmethod
    def fileno():
        return os.dup(1)


class NoInvokeGraph(StubStateGraph):
    def compile(self, **_kwargs):
        return object()  # graph.invoke -> AttributeError -> recorded as exception


@pytest.mark.parametrize(
    "argv, faults, n_rows",
    [
        ([], ["pump_failure"], 1),
        (["-f", "leak", "-n", "2", "-p", "minimal", "-m", "gpt-4.1-nano"], ["leak"], 2),
        (["-f", "all", "-n", "1", "--llm-timeout", "5"], None, 6),
    ],
)
def test_main_cli(mixer_case, monkeypatch, tmp_path, argv, faults, n_rows):
    import pandas as pd

    monkeypatch.setattr(sys.modules["langgraph.graph"], "StateGraph", NoInvokeGraph)
    retrieval = sys.modules.get("graph_retrieval_code")
    if retrieval is not None:
        ttl = ":s :p :o ." if "all" in argv else ""  # KG loads in one variant
        monkeypatch.setattr(retrieval, "run_construct", lambda q: ttl)
    out, traces = tmp_path / "out", tmp_path / "traces"
    monkeypatch.setattr(
        sys, "argv",
        ["mixer_case.py", *argv, "-o", str(out), "--trace-dir", str(traces)],
    )  # fmt: skip
    monkeypatch.setattr(sys, "stdout", _DupStdout())
    try:
        ns = runpy.run_path(mixer_case.__file__, run_name="__main__")
    finally:
        if not isinstance(sys.stdout, _DupStdout):
            sys.stdout.close()
    df = ns["df"]
    assert len(df) == n_rows
    assert set(df["fault"]) == set(faults or mixer_case.FAULT_TYPES)
    assert set(df["abort_reason"]) == {"exception"}
    assert ns["CURRENT_LLM_TIMEOUT"] == (5.0 if "--llm-timeout" in argv else 120.0)
    (runs_csv,) = out.glob("*_runs.csv")
    assert len(pd.read_csv(runs_csv)) == n_rows
    assert len(list(traces.rglob("final_result.json"))) == n_rows
