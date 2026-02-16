from __future__ import annotations

import os
import json
from datetime import datetime
import pandas as pd

import random
import argparse
import copy
import time
from dataclasses import dataclass
from collections import deque
from typing import TypedDict, Dict, Any, List, Optional, Literal, Tuple

import numpy as np
import matplotlib.pyplot as plt

from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from langchain.callbacks import get_openai_callback


from cstr_digital_twin import CSTRSimulation, FaultConfig
from cstr_anomaly_threshold_test import OnlineTempThresholdDetector, TempThresholdConfig

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


from textwrap import dedent
import json
import sys
from dotenv import load_dotenv

sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
load_dotenv()
try:
    from graph_retrieval_code import run_construct, QUERY_CSTR

    SPARQL_AVAILABLE = True
except ImportError:
    print("[WARNING] graph_retrieval.py not found - running without KG context")
    SPARQL_AVAILABLE = False

_KG_PLANNING_TTL: str = ""
_KG_ACTION_TTL: str = ""
_KG_LOADED: bool = False


def load_kg_context() -> bool:
    """Load KG context from GraphDB (called once at startup)."""
    global _KG_PLANNING_TTL, _KG_ACTION_TTL, _KG_LOADED

    if not SPARQL_AVAILABLE:
        return False

    print("[KG] Loading Knowledge Graph context from GraphDB...")

    try:
        _KG_ACTION_TTL = run_construct(QUERY_CSTR)

        if _KG_ACTION_TTL:
            _KG_LOADED = True
            # Count approximate triples
            action_lines = len(
                [
                    l
                    for l in _KG_ACTION_TTL.split("\n")
                    if l.strip() and not l.startswith("@")
                ]
            )
            print(f"[KG] ✅ Action context: ~{action_lines} statements")
            return True
        else:
            print("[KG] ⚠️ Empty response from GraphDB")
            return False
    except Exception as e:
        print(f"[KG] ⚠️ Could not load KG context: {e}")
        return False


def get_action_kg_context() -> str:
    """Get Action KG context as formatted prompt section."""
    if not _KG_LOADED or not _KG_ACTION_TTL:
        return ""

    return f"""

## KNOWLEDGE GRAPH DATA (State → Action → Actuator mappings)

The following Turtle data shows which `UML:Action` instances are linked to which states,
and which `VDI2206:Actuator` instances execute those actions via `CPSMod:isChangedByActuator`.

```turtle
{_KG_ACTION_TTL}
```
"""


def is_kg_loaded() -> bool:
    """Check if KG context was successfully loaded."""
    return _KG_LOADED


# =============================================================================
# LLM setup
# =============================================================================


def _make_llm(model: str = "llama3:8b", temperature: float = 0.0):
    """
    Uses OpenAI if model starts with 'gpt', else Ollama.
    """
    if model.startswith("gpt"):
        return ChatOpenAI(model=model, temperature=temperature, timeout=60)
    return ChatOllama(
        model=model,
        temperature=temperature,
        timeout=60,
        base_url="http://localhost:11434",
    )


class SetpointPlan(BaseModel):
    T_sp: float = Field(..., description="New temperature setpoint [K]")
    Fin_sp: float = Field(..., description="New inlet flow setpoint [L/s]")
    L_sp: float = Field(..., description="New level setpoint [L]")
    reasoning: str = Field(..., description="Short justification referencing rules")


# =============================================================================
# Plotting
# =============================================================================


def plot_cstr_digital_twin(
    *,
    t_hist,
    L_hist,
    V_hist,
    phase_hist,
    Fin_hist,
    Fout_hist,
    Fin_sp_hist,
    CA_hist,
    CB_hist,
    CC_hist,
    CD_hist,
    u_valve_hist,
    u_pump_hist,
    u_cool_hist,
    T_hist,
    Tmeas_hist,
    Fc_hist,
    UAeff_hist,
    leak_hist,
    over_hist,
    fault_tag_hist=None,
    L_sp_hist=None,
    T_sp_hist=None,
    T_startup: float = 1000.0,
    T_normal: float = 5000.0,
    T_shutdown: float = 200.0,
    L_sp: float = 10.0,
    T_sp: float = 310.0,
    t_sp_change: Optional[float] = None,
    label_sp_change: str = "Setpoint change",
    show_fault_markers: bool = True,
    fouling_t_start: Optional[float] = None,
    pump_degrade_t: Optional[float] = None,
    title: str = "CSTR Digital Twin with Fault Injection",
):
    T_total = T_startup + T_normal + T_shutdown
    phase_colors = {1: "lightblue", 2: "lightgreen", 3: "lightsalmon"}

    def add_phase_backgrounds(ax):
        ax.axvspan(0, T_startup, alpha=0.2, color=phase_colors[1], label="Startup")
        ax.axvspan(
            T_startup,
            T_startup + T_normal,
            alpha=0.2,
            color=phase_colors[2],
            label="Normal",
        )
        ax.axvspan(
            T_startup + T_normal,
            T_total,
            alpha=0.2,
            color=phase_colors[3],
            label="Shutdown",
        )

    def add_event_lines(ax):
        if t_sp_change is not None and np.isfinite(t_sp_change):
            ax.axvline(t_sp_change, linestyle="--", linewidth=2.0, alpha=0.9)
            ax.text(
                t_sp_change,
                0.98,
                label_sp_change,
                transform=ax.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="right",
                fontsize=9,
            )

        if show_fault_markers:
            if fouling_t_start is not None and np.isfinite(fouling_t_start):
                ax.axvline(fouling_t_start, linestyle=":", linewidth=2.0, alpha=0.9)
                ax.text(
                    fouling_t_start,
                    0.80,
                    "Fouling start",
                    transform=ax.get_xaxis_transform(),
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=9,
                )
            if pump_degrade_t is not None and np.isfinite(pump_degrade_t):
                ax.axvline(pump_degrade_t, linestyle=":", linewidth=2.0, alpha=0.9)
                ax.text(
                    pump_degrade_t,
                    0.62,
                    "Pump degrade start",
                    transform=ax.get_xaxis_transform(),
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=9,
                )

    fig, axes = plt.subplots(4, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Level
    ax = axes[0, 0]
    add_phase_backgrounds(ax)
    ax.plot(t_hist, L_hist, linewidth=2, label="Level [L]")
    if L_sp_hist is not None:
        ax.step(
            t_hist,
            L_sp_hist,
            where="post",
            linestyle="--",
            linewidth=1.5,
            label="L_sp(t)",
        )
    else:
        ax.axhline(L_sp, linestyle="--", linewidth=1.5, label="L_sp")
    add_event_lines(ax)
    ax.set_title("Reactor Level")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("L [L]")
    ax.grid(alpha=0.3)
    ax.legend()

    # Temperature
    ax = axes[0, 1]
    add_phase_backgrounds(ax)
    ax.plot(t_hist, T_hist, linewidth=2, label="T (true)")
    ax.plot(t_hist, Tmeas_hist, linewidth=1, alpha=0.6, label="T_meas")
    if T_sp_hist is not None:
        ax.step(
            t_hist,
            T_sp_hist,
            where="post",
            linestyle="--",
            linewidth=1.5,
            label="T_sp(t)",
        )
    else:
        ax.axhline(T_sp, linestyle="--", linewidth=1.5, label="T_sp")
    add_event_lines(ax)
    ax.set_title("Reactor Temperature")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("T [K]")
    ax.grid(alpha=0.3)
    ax.legend()

    # Flows
    ax = axes[1, 0]
    add_phase_backgrounds(ax)
    ax.plot(t_hist, Fin_hist, label="Fin", linewidth=2)
    ax.plot(t_hist, Fout_hist, label="Fout", linewidth=2)
    ax.step(
        t_hist,
        Fin_sp_hist,
        where="post",
        linestyle="--",
        label="Fin_sp(t)",
        linewidth=1.5,
    )
    add_event_lines(ax)
    ax.set_title("Process Flows")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Flow [L/s]")
    ax.grid(alpha=0.3)
    ax.legend()

    # Cooling Flow
    ax = axes[1, 1]
    add_phase_backgrounds(ax)
    ax.plot(t_hist, Fc_hist, label="Fc", linewidth=2)
    add_event_lines(ax)
    ax.set_title("Cooling Water Flow")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Fc [L/s]")
    ax.grid(alpha=0.3)
    ax.legend()

    # Concentrations
    ax = axes[2, 0]
    add_phase_backgrounds(ax)
    ax.plot(t_hist, CA_hist, label="CA", linewidth=2)
    ax.plot(t_hist, CD_hist, label="CD", linewidth=2)
    ax.plot(t_hist, CB_hist, label="CB", linewidth=2)
    ax.plot(t_hist, CC_hist, label="CC", linewidth=2)
    add_event_lines(ax)
    ax.set_title("Reactor Concentrations")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("mol/L")
    ax.grid(alpha=0.3)
    ax.legend()

    # UA_eff + leak/overflow
    ax = axes[2, 1]
    add_phase_backgrounds(ax)
    ax.plot(t_hist, UAeff_hist, label="UA_eff", linewidth=2)
    add_event_lines(ax)
    ax2 = ax.twinx()
    ax2.plot(t_hist, over_hist, label="Overflow outflow", linewidth=1.5, alpha=0.8)
    ax2.plot(t_hist, leak_hist, label="Leak outflow", linewidth=1.5, alpha=0.8)
    ax.set_title("UA_eff + Leak/Overflow")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("UA [J/s/K]")
    ax2.set_ylabel("Extra outflows [L/s]")
    ax.grid(alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")

    # Actuators
    ax = axes[3, 0]
    add_phase_backgrounds(ax)
    ax.plot(t_hist, u_valve_hist, label="inlet valve", linewidth=2)
    ax.plot(t_hist, u_pump_hist, label="outlet pump", linewidth=2)
    ax.plot(t_hist, u_cool_hist, label="cooling valve", linewidth=2)
    add_event_lines(ax)
    ax.set_title("Actuator Positions")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("0-1")
    ax.set_ylim([-0.05, 1.05])
    ax.grid(alpha=0.3)
    ax.legend()

    # Phase trace
    ax = axes[3, 1]
    ax.plot(t_hist, phase_hist, linewidth=3)
    if t_sp_change is not None and np.isfinite(t_sp_change):
        ax.axvline(t_sp_change, linestyle="--", linewidth=2.0, alpha=0.9)
    ax.set_title("Operation Phase")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Phase")
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["Startup", "Normal", "Shutdown"])
    ax.grid(alpha=0.3)
    ax.set_ylim([0.5, 3.5])

    plt.tight_layout()
    plt.show()


# =============================================================================
# Graph state
# =============================================================================


class GraphState(TypedDict, total=False):
    itr: int
    max_steps: int
    T_end: float

    # setpoints
    T_sp: float
    L_sp: float
    Fin_sp: float
    Fin_sp_normal: float
    Fin_sp_startup: float

    mode: Literal["RUN", "STOP"]

    # fault config
    fault_name: str
    fault_cfg: FaultConfig

    # monitoring outputs
    fault_flag: bool
    fault_flag_prev: bool
    anomaly_ratio: float
    violated_params: str
    n_violations: int

    # action trigger
    should_act: bool
    sp_change_done: bool
    t_sp_change: Optional[float]

    # safety zone / control margin
    control_zone: Literal["SAFE", "WARNING", "UNSAFE"]
    control_reasons: str
    actuator_valid: bool
    actuator_saturated: bool
    actuator_trending_to_sat: bool
    T_target: float

    # safe checker flags
    safe_now: bool
    recovered_now: bool
    safe_recovered_flag: bool

    # LLM propose/eval
    proposed_sp: Dict[str, float]
    eval_pass: bool
    eval_summary: str
    eval_fail_reason: str
    eval_metrics: Dict[str, Any]
    eval_feedback: str

    reprompt_count: int
    reprompt_max: int

    # acceptance knobs
    safe_hold_seconds: float
    max_time_to_safe: Optional[float]

    abort_reason: str

    # logs
    sim_last: Dict[str, Any]
    sim_log: List[Dict[str, Any]]

    # objects
    _plant: Any
    _detector: Any
    _safe_checker: Any
    _control_safety: Any

    action_calls: int
    action_latency_last_s: float
    action_latency_sum_s: float
    action_latency_avg_s: float
    action_prompt_tokens_last: int
    action_completion_tokens_last: int
    action_total_tokens_last: int
    action_prompt_tokens_sum: int
    action_completion_tokens_sum: int
    action_total_tokens_sum: int


# =============================================================================
# Safety + control checkers
# =============================================================================


@dataclass
class ControlSafetyConfig:
    T_target: float = 310.0
    sat_threshold: float = 0.95
    hard_sat: float = 0.99
    trend_eps: float = 0.002
    T_band_normal: float = 1.0
    T_band_when_sat: float = 0.5


class SafeStateChecker:
    """
    Hard safety + recovery persistence.
    """

    def __init__(
        self,
        *,
        T_safe_max: float = 313.0,
        L_min_safe: float = 2.0,
        L_max_safe: float = 11.0,
        T_band: float = 1.0,
        L_band: float = 0.3,
        safe_window: int = 30,
        safe_ratio: float = 1.0,
        recovery_window: int = 120,
        recovery_ratio: float = 0.9,
    ):
        self.T_safe_max = float(T_safe_max)
        self.L_min_safe = float(L_min_safe)
        self.L_max_safe = float(L_max_safe)
        self.T_band = float(T_band)
        self.L_band = float(L_band)
        self.safe_buf = deque(maxlen=int(safe_window))
        self.rec_buf = deque(maxlen=int(recovery_window))
        self.safe_ratio = float(safe_ratio)
        self.recovery_ratio = float(recovery_ratio)

    def update(self, *, T_meas, L_true, T_sp, L_sp, F_over=0.0):
        T_meas = float(T_meas) if np.isfinite(T_meas) else np.nan
        L_true = float(L_true) if np.isfinite(L_true) else np.nan
        T_sp = float(T_sp) if np.isfinite(T_sp) else np.nan
        L_sp = float(L_sp) if np.isfinite(L_sp) else np.nan
        F_over = float(F_over) if np.isfinite(F_over) else 0.0

        safe_now = (
            np.isfinite(T_meas)
            and np.isfinite(L_true)
            and (T_meas <= self.T_safe_max)
            and (self.L_min_safe <= L_true <= self.L_max_safe)
            and (F_over <= 1e-12)
        )
        recovered_now = (
            safe_now
            and np.isfinite(T_sp)
            and np.isfinite(L_sp)
            and (abs(T_meas - T_sp) <= self.T_band)
            and (abs(L_true - L_sp) <= self.L_band)
        )

        self.safe_buf.append(1 if safe_now else 0)
        self.rec_buf.append(1 if recovered_now else 0)

        safe_persist = (
            (sum(self.safe_buf) / len(self.safe_buf) >= self.safe_ratio)
            if self.safe_buf
            else False
        )
        recovered_persist = (
            (sum(self.rec_buf) / len(self.rec_buf) >= self.recovery_ratio)
            if self.rec_buf
            else False
        )

        return {
            "safe_now": safe_now,
            "recovered_now": recovered_now,
            "safe_persist": safe_persist,
            "recovered_persist": recovered_persist,
        }


class ControlSafetyChecker:
    """
    Validates actuator values, detects saturation / trending-to-saturation, and enforces:
      if saturated or trending -> T must be close to 310 (cfg.T_target within cfg.T_band_when_sat)
    """

    def __init__(self, cfg: ControlSafetyConfig = ControlSafetyConfig()):
        self.cfg = cfg
        self.prev = {"u_valve": None, "u_pump": None, "u_cool": None}

    def update(self, *, out: Dict[str, Any], T_sp: float) -> Dict[str, Any]:
        cfg = self.cfg

        T_meas = float(out.get("T_meas", np.nan))
        u_valve = float(out.get("u_valve", np.nan))
        u_pump = float(out.get("u_pump", np.nan))
        u_cool = float(out.get("u_cool", np.nan))

        u = {"u_valve": u_valve, "u_pump": u_pump, "u_cool": u_cool}

        actuator_valid = True
        invalid = []
        for k, v in u.items():
            if (not np.isfinite(v)) or (v < 0.0) or (v > 1.0):
                actuator_valid = False
                invalid.append(k)

        saturated = [
            k for k, v in u.items() if np.isfinite(v) and v >= cfg.sat_threshold
        ]
        hard_saturated = [
            k for k, v in u.items() if np.isfinite(v) and v >= cfg.hard_sat
        ]

        trending = []
        for k, v in u.items():
            pv = self.prev.get(k)
            if pv is not None and np.isfinite(v) and np.isfinite(pv):
                if (v >= cfg.sat_threshold) and ((v - pv) > cfg.trend_eps):
                    trending.append(k)
            self.prev[k] = v

        actuator_saturated = len(saturated) > 0
        actuator_trending_to_sat = len(trending) > 0

        T_err_target = (
            abs(T_meas - cfg.T_target) if np.isfinite(T_meas) else float("inf")
        )
        if actuator_saturated or actuator_trending_to_sat:
            T_ok = T_err_target <= cfg.T_band_when_sat
        else:
            T_ok = (
                (abs(T_meas - float(T_sp)) <= cfg.T_band_normal)
                if np.isfinite(T_meas)
                else False
            )

        reasons = []
        zone = "SAFE"

        if not actuator_valid:
            zone = "UNSAFE"
            reasons.append(f"invalid_actuators={invalid}")

        if actuator_saturated:
            reasons.append(f"saturated={saturated}")
        if actuator_trending_to_sat:
            reasons.append(f"trending_to_sat={trending}")
        if hard_saturated:
            reasons.append(f"hard_sat={hard_saturated}")

        if (actuator_saturated or actuator_trending_to_sat) and (not T_ok):
            zone = "UNSAFE"
            reasons.append(
                f"T_not_close_to_310 err={T_err_target:.3f}K (req <= {cfg.T_band_when_sat}K)"
            )

        if zone == "SAFE" and actuator_saturated:
            zone = "WARNING"
            reasons.append("at_limit_but_holding_T")

        return {
            "control_zone": zone,
            "control_reasons": "; ".join(reasons),
            "actuator_valid": actuator_valid,
            "actuator_saturated": actuator_saturated,
            "actuator_trending_to_sat": actuator_trending_to_sat,
            "T_target": cfg.T_target,
        }


# =============================================================================
# Fault scenarios
# =============================================================================


def fault_cfg_from_name(name: str) -> FaultConfig:
    name = name.lower().strip()
    if name == "normal":
        return FaultConfig(enable=False)

    cfg = FaultConfig(enable=True)

    cfg.fouling_on = False
    cfg.pump_degrade_on = False
    cfg.cool_stuck_open = False
    cfg.cool_stuck_closed = False
    cfg.outlet_block_on = False
    cfg.inlet_stuck_open = False
    cfg.temp_bias_on = False
    cfg.level_bias_on = False
    cfg.leak_on = False

    if name == "fouling":
        cfg.fouling_on = True
        cfg.fouling_t_start = 2000.0 + random.uniform(-200.0, 200.0)
    elif name == "pump_degrade":
        cfg.pump_degrade_on = True
        cfg.pump_degrade_t = 2000.0 + random.uniform(-200.0, 200.0)
    elif name == "cool_stuck_open":
        cfg.cool_stuck_open = True
        cfg.cool_stuck_open_t = 2000.0 + random.uniform(-200.0, 200.0)
    elif name == "cool_stuck_closed":
        cfg.cool_stuck_closed = True
        cfg.cool_stuck_closed_t = 2000.0 + random.uniform(-200.0, 200.0)
    elif name == "outlet_block":
        cfg.outlet_block_on = True
        cfg.outlet_block_t = 2000.0
    elif name == "inlet_stuck_open":
        cfg.inlet_stuck_open = True
        cfg.inlet_stuck_open_t = 2000.0
    elif name == "temp_bias":
        cfg.temp_bias_on = True
        cfg.temp_bias_t = 2000.0
    elif name == "level_bias":
        cfg.level_bias_on = True
        cfg.level_bias_t = 2000.0
    elif name == "leak":
        cfg.leak_on = True
        cfg.leak_t = 2000.0
    else:
        raise ValueError(
            f"Unknown fault '{name}'. Use: normal,fouling,pump_degrade,cool_stuck_open,cool_stuck_closed,"
            f"outlet_block,inlet_stuck_open,temp_bias,level_bias,leak"
        )

    return cfg


# =============================================================================
# Mapping sim output -> detector row
# =============================================================================


def row_from_sim(sim_out: Dict[str, Any], state: GraphState) -> Dict[str, Any]:
    phase_map = {"STARTUP": 1, "NORMAL": 2, "SHUTDOWN": 3}
    phase_num = phase_map.get(str(sim_out.get("phase", "NORMAL")).upper(), 2)

    T_meas = float(sim_out.get("T_meas", np.nan))
    L_meas = float(sim_out.get("L_meas", np.nan))
    Fin = float(sim_out.get("Fin", np.nan))
    Fout = float(sim_out.get("Fout", np.nan))

    return {
        "time": float(sim_out.get("t", np.nan)),
        "phase": phase_num,
        "T_meas": T_meas,
        "L_meas": L_meas,
        "Fin": Fin,
        "Fout": Fout,
        "Fc": float(sim_out.get("Fc", np.nan)),
        "u_valve": float(sim_out.get("u_valve", np.nan)),
        "u_pump": float(sim_out.get("u_pump", np.nan)),
        "u_cool": float(sim_out.get("u_cool", np.nan)),
        "UA_eff": float(sim_out.get("UA_eff", np.nan)),
        "mass_balance": Fin - Fout,
        "T_meas_error": T_meas - float(state["T_sp"]),
        "L_meas_error": L_meas - float(state["L_sp"]),
        "F_leak": float(sim_out.get("F_leak", 0.0)),
        "F_over": float(sim_out.get("F_over", 0.0)),
    }


# =============================================================================
# Fallback policy (used if LLM fails repeatedly)
# =============================================================================


def fallback_policy(state: GraphState) -> Dict[str, float]:
    """
    Conservative deterministic move:
    - Keep T_sp near 310 (per your saturation rule)
    - Reduce Fin_sp to reduce heat generation/load
    - Keep L_sp constant
    """
    T_sp = float(state.get("T_sp", 310.0))
    L_sp = float(state.get("L_sp", 10.0))
    Fin_sp = float(state.get("Fin_sp", 2.0 / 60.0))

    Fin_new = max(0.5 * Fin_sp, 0.5 / 60.0)
    return {
        "T_sp": 310.0 if abs(T_sp - 310.0) < 20 else T_sp,
        "L_sp": L_sp,
        "Fin_sp": Fin_new,
    }


# =============================================================================
# Nodes
# =============================================================================


def initializing(state: GraphState) -> GraphState:
    state["itr"] = 0
    state["abort_reason"] = ""
    state["should_act"] = False
    state["sp_change_done"] = False
    state["t_sp_change"] = None

    # termination
    state["T_end"] = float(state.get("T_end", 7000.0))

    # monitoring flags
    state["fault_flag"] = False
    state["fault_flag_prev"] = False
    state["anomaly_ratio"] = 0.0
    state["violated_params"] = ""
    state["n_violations"] = 0

    state["mode"] = state.get("mode", "RUN")

    # initial setpoints
    state["T_sp"] = float(state.get("T_sp", 310.0))
    state["L_sp"] = float(state.get("L_sp", 10.0))
    state["Fin_sp"] = float(state.get("Fin_sp", 2.0 / 60.0))
    state["Fin_sp_normal"] = state["Fin_sp"]
    state["Fin_sp_startup"] = state["Fin_sp"]

    # evaluation knobs
    state["safe_hold_seconds"] = float(state.get("safe_hold_seconds", 60.0))
    # Optional: enforce a max time-to-safe (seconds). Use None to disable.
    state["max_time_to_safe"] = state.get("max_time_to_safe", None)

    # reprompt config
    state["reprompt_count"] = 0
    state["reprompt_max"] = int(state.get("reprompt_max", 5))

    # LLM/EVAL state
    state["proposed_sp"] = {}
    state["eval_pass"] = False
    state["eval_summary"] = ""
    state["eval_fail_reason"] = ""
    state["eval_metrics"] = {}
    state["eval_feedback"] = ""

    # safety flags
    state["safe_now"] = True
    state["recovered_now"] = True
    state["safe_recovered_flag"] = False

    # fault config
    fault_name = state.get("fault_name", "normal")
    cfg = fault_cfg_from_name(fault_name)
    state["fault_name"] = fault_name
    state["fault_cfg"] = cfg

    # logs
    state["sim_last"] = {}
    state["sim_log"] = []

    # detector
    state["_detector"] = OnlineTempThresholdDetector(
        TempThresholdConfig(
            allowed_phases={2},  # NORMAL only
            delta_above_sp=2.0,  # 310 -> trigger >312
            persistence_window=10,
            persistence_ratio=0.6,
            require_full_window=True,
            enable_settle_gate=True,
            settle_band=1.0,
            settle_required_samples=60,
        )
    )

    state["symptoms"] = {}

    # plant
    state["_plant"] = CSTRSimulation(
        dt=1.0,
        T_sp=state["T_sp"],
        L_sp=state["L_sp"],
        Fin_sp_normal=state["Fin_sp_normal"],
        Fin_sp_startup=state["Fin_sp_startup"],
        fault_cfg=state["fault_cfg"],
    )

    # safety checkers
    state["_safe_checker"] = SafeStateChecker(
        T_safe_max=313.0,
        L_min_safe=2.0,
        L_max_safe=11.0,
        T_band=1.0,
        L_band=0.3,
        safe_window=30,
        safe_ratio=1.0,
        recovery_window=120,
        recovery_ratio=0.9,
    )
    state["_control_safety"] = ControlSafetyChecker(
        ControlSafetyConfig(
            T_target=310.0,
            sat_threshold=0.95,
            hard_sat=0.99,
            trend_eps=0.002,
            T_band_normal=1.0,
            T_band_when_sat=0.5,
        )
    )
    state["control_zone"] = "SAFE"
    state["control_reasons"] = ""
    state["actuator_valid"] = True
    state["actuator_saturated"] = False
    state["actuator_trending_to_sat"] = False
    state["T_target"] = 310.0

    # -------------------------------------------------------------------------
    # LLM metrics init (no logic change)
    # -------------------------------------------------------------------------
    state["action_calls"] = 0
    state["action_latency_last_s"] = 0.0
    state["action_latency_sum_s"] = 0.0
    state["action_latency_avg_s"] = 0.0

    state["action_prompt_tokens_last"] = 0
    state["action_completion_tokens_last"] = 0
    state["action_total_tokens_last"] = 0

    state["action_prompt_tokens_sum"] = 0
    state["action_completion_tokens_sum"] = 0
    state["action_total_tokens_sum"] = 0
    return state


def plant(state: GraphState) -> GraphState:
    plant_obj: CSTRSimulation = state["_plant"]

    state["Fin_sp_normal"] = float(state["Fin_sp"])
    state["Fin_sp_startup"] = float(state["Fin_sp"])

    out = plant_obj.simulate_step(
        T_sp=float(state["T_sp"]),
        L_sp=float(state["L_sp"]),
        Fin_sp_normal=float(state["Fin_sp_normal"]),
        Fin_sp_startup=float(state["Fin_sp_startup"]),
        mode=str(state.get("mode", "RUN")),
    )

    state["sim_last"] = out

    rec = dict(out)
    rec["T_sp"] = float(state["T_sp"])
    rec["L_sp"] = float(state["L_sp"])
    rec["Fin_sp"] = float(state["Fin_sp"])
    rec["fault_name"] = state["fault_name"]
    rec["sp_change_done"] = bool(state["sp_change_done"])
    rec["t_sp_change"] = state.get("t_sp_change", None)
    rec["control_zone"] = state.get("control_zone", "")
    state["sim_log"].append(rec)

    state["itr"] += 1
    return state


def monitoring(state: GraphState) -> GraphState:
    """
    NORMAL-phase-only monitoring.

    Keeps your existing temp-threshold detector (great for fouling),
    and adds a second detector for "cooling limited / valve closed-ish" defined as:
        u_cool <= 0.30 persistently (window) while T is above setpoint by margin.

    Triggers action on rising edges (one-shot events) + entered_unsafe + big level error.

    No fault-name branching. No extra graph nodes.
    """
    det: OnlineTempThresholdDetector = state["_detector"]

    out = state.get("sim_last", {}) or {}
    row = row_from_sim(out, state)

    # ---- phase gate: only monitor/trigger in NORMAL ----
    phase_num = int(row.get("phase", 2))
    in_normal = phase_num == 2

    # Always compute symptoms + safety + control zone (so state is consistent)
    sym = compute_symptoms(row, state)
    state["symptoms"] = sym

    # -------------------------
    # SAFE/RECOVERY CHECK (always)
    # -------------------------
    sc: SafeStateChecker = state["_safe_checker"]
    T_meas = float(out.get("T_meas", float("nan")))
    L_true = float(out.get("L_true", float("nan")))
    F_over = float(out.get("F_over", 0.0))

    s = sc.update(
        T_meas=T_meas,
        L_true=L_true,
        T_sp=float(state["T_sp"]),
        L_sp=float(state["L_sp"]),
        F_over=F_over,
    )
    state["safe_now"] = bool(s.get("safe_now", False))
    state["recovered_now"] = bool(s.get("recovered_now", False))
    if bool(s.get("recovered_persist", False)):
        state["safe_recovered_flag"] = True

    # -------------------------
    # CONTROL SAFETY ZONE (always) + bugfix entered_unsafe
    # -------------------------
    cs: ControlSafetyChecker = state["_control_safety"]
    prev_zone = str(state.get("control_zone", "SAFE"))  # old zone

    cs_out = cs.update(out=out, T_sp=float(state["T_sp"]))
    new_zone = str(cs_out.get("control_zone", "SAFE"))

    state["control_zone"] = new_zone
    state["control_reasons"] = str(cs_out.get("control_reasons", ""))
    state["actuator_valid"] = bool(cs_out.get("actuator_valid", True))
    state["actuator_saturated"] = bool(cs_out.get("actuator_saturated", False))
    state["actuator_trending_to_sat"] = bool(
        cs_out.get("actuator_trending_to_sat", False)
    )
    state["T_target"] = float(cs_out.get("T_target", 310.0))

    entered_unsafe = (new_zone == "UNSAFE") and (prev_zone != "UNSAFE")

    # If not NORMAL: disable fault/anomaly triggers and return (but keep state fields sane)
    if not in_normal:
        state["fault_flag_prev"] = bool(state.get("fault_flag", False))
        state["fault_flag"] = False
        state["anomaly_ratio"] = 0.0
        state["violated_params"] = ""
        state["n_violations"] = 0
        state["should_act"] = False

        # light debug occasionally
        if state["itr"] % 200 == 0:
            print(
                f"[MON] t={row.get('time'):.1f} phase={phase_num} (not NORMAL) "
                f"safe={state['safe_now']} recovered={state['recovered_now']} "
                f"zone={state['control_zone']} {state.get('control_reasons','')}"
            )
        return state

    # =============================================================================
    # NORMAL-PHASE DETECTION STARTS HERE
    # =============================================================================

    # -------------------------
    # 1) Fouling-friendly detector (temp-threshold) - unchanged
    # -------------------------
    det_out = det.update(row)
    temp_fault_flag = bool(det_out.get("fault_flag", False))
    temp_raw_anom = bool(det_out.get("is_anomaly_raw", False))
    thr = det_out.get("threshold", None)

    state["fault_flag_prev"] = bool(state.get("fault_flag", False))
    state["anomaly_ratio"] = float(det_out.get("anomaly_ratio", 0.0))

    # rising edge of the temp detector event
    # (note: we compute a combined fault_flag later; this temp rising edge remains useful)
    prev_temp_flag = bool(state.get("_temp_fault_flag_prev", False))
    temp_rising_edge = temp_fault_flag and (not prev_temp_flag)
    state["_temp_fault_flag_prev"] = temp_fault_flag

    # -------------------------
    # 2) Cooling-limited detector (u_cool capped at <= 0.30) with persistence + rising edge
    # -------------------------
    # Lazy-init buffers (so you don't have to edit initializing())
    if "_cool_limited_buf" not in state:
        state["_cool_limited_buf"] = deque(maxlen=15)  # 15s window
        state["_cool_limited_prev"] = False

    u_cool = float(out.get("u_cool", np.nan))
    T_sp = float(state.get("T_sp", 310.0))
    T_err = (T_meas - T_sp) if (np.isfinite(T_meas) and np.isfinite(T_sp)) else np.nan

    # raw condition: cooling stuck low AND temperature is actually high
    # (temperature condition avoids flagging normal operation where cooling isn't needed)
    cool_cap = 0.30
    T_high_margin = 2.0
    cool_limited_raw = (np.isfinite(u_cool) and (u_cool <= cool_cap)) and (
        np.isfinite(T_err) and (T_err > T_high_margin)
    )

    # persistence: require >=70% of last 15 seconds be true
    buf = state["_cool_limited_buf"]
    buf.append(1 if cool_limited_raw else 0)
    cool_limited_confirmed = (len(buf) == buf.maxlen) and (
        (sum(buf) / len(buf)) >= 0.70
    )

    cool_limited_prev = bool(state.get("_cool_limited_prev", False))
    cool_limited_rising = cool_limited_confirmed and (not cool_limited_prev)
    state["_cool_limited_prev"] = cool_limited_confirmed

    # -------------------------
    # 3) Level error hook (your existing idea)
    # -------------------------
    L_err = sym.get("L_err", None)
    big_level_error = (L_err is not None) and (abs(float(L_err)) > 0.7)

    # -------------------------
    # 4) Combine into violated_params + n_violations (NORMAL only)
    # -------------------------
    violations = []
    if temp_raw_anom:
        violations.append(f"T_meas_error>{thr}")
    if cool_limited_confirmed:
        violations.append(f"cool_limited(u_cool<={cool_cap} & T_err>{T_high_margin}K)")
    if big_level_error:
        violations.append("level_error_large(|L_meas-L_sp|>0.7)")
    if entered_unsafe:
        violations.append("entered_UNSAFE(control_zone)")

    state["violated_params"] = "; ".join(violations)
    state["n_violations"] = int(len(violations))

    # -------------------------
    # 5) fault_flag + rising edge (combined)
    # -------------------------
    combined_fault_flag = bool(temp_fault_flag or cool_limited_confirmed)
    state["fault_flag"] = combined_fault_flag
    rising_edge = combined_fault_flag and (not state["fault_flag_prev"])

    # -------------------------
    # 6) Trigger action ONLY once per event type + entered_unsafe + level error
    #    (sp_change_done prevents repeated actions)
    # -------------------------
    state["should_act"] = (not bool(state.get("sp_change_done", False))) and (
        # your original fouling-style trigger
        rising_edge
        # additional one-shot event for cooling limited
        or cool_limited_rising
        # bugfixed unsafe entry
        or entered_unsafe
        # your level hook
        or big_level_error
    )

    # -------------------------
    # 7) Debug print
    # -------------------------
    if state["itr"] % 50 == 0 or state["should_act"]:
        print(
            f"[MON] t={row.get('time'):.1f} phase={phase_num} "
            f"temp_raw={temp_raw_anom} ratio={state['anomaly_ratio']:.2f} "
            f"temp_flag={temp_fault_flag} cool_lim={cool_limited_confirmed} "
            f"fault_flag={state['fault_flag']} should_act={state['should_act']} "
            f"safe={state['safe_now']} recovered={state['recovered_now']} "
            f"zone={state['control_zone']} {state.get('control_reasons','')} "
            f"violations={state.get('violated_params','')}"
        )

    return state


# =============================================================================
# LLM propose -> evaluation -> reprompt -> apply
# =============================================================================


def action_propose(state: GraphState) -> GraphState:
    out = state.get("sim_last", {})
    row = row_from_sim(out, state)

    prev_prop = (
        state.get("proposed_sp", None) if state.get("reprompt_count", 0) > 0 else None
    )
    feedback = state.get("eval_feedback", "")

    action_kg = get_action_kg_context() or ""
    sym = state.get("symptoms", {}) or {}

    ACTION_SYSTEM_PROMPT_KG = (
        dedent(
            """
        You are the **Corrective Setpoint Action Agent** for a Continuous Stirred Tank Reactor (CSTR).

        You must propose ONLY these setpoints:
        - T_sp (temperature setpoint)
        - L_sp (level setpoint)
        - Fin_sp (inlet flow setpoint)

        You will be given a Knowledge Graph (KG) Turtle snippet. You MUST use it.

        # How to use the KG (MANDATORY)
        From the KG, infer these causal links (examples from the KG you will see):
        - PO7_TempPID: reverse-acting temperature control:
        "e = T_meas - T_sp (reverse-acting)" → if T_meas > T_sp then controller increases cooling command u_cool_cmd.
        - PO8_FlowPID: Fin_sp affects u_valve_cmd → changes Fin through PO1_InletFlow.
        - PO6_LevelPID: L_sp affects u_pump_cmd → changes Fout through PO2_OutletFlow.
        - PO5_StateDerivatives: dV/dt = Fin - Fout - F_leak - F_overflow and dT/dt depends on Fin and cooling.

        So:
        - If cooling is saturated (u_cool near 1.0) and temperature is high, you cannot "cool harder".
        You must reduce heat input / load, typically by reducing Fin_sp (via PO8 -> u_valve_cmd -> Fin).
        - If level is drifting out of safe bounds, adjust L_sp (via PO6 -> u_pump_cmd -> Fout) and/or reduce Fin_sp.

        # Output format (STRICT)
        Return JSON ONLY with:
        {
        "T_sp": <float>,
        "L_sp": <float>,
        "Fin_sp": <float>,
        "reasoning": "<brief kg-grounded reasoning>"
        }

        # Hard rules
        1) If evaluation failed previously, propose a DIFFERENT setpoint triple than previous proposal.
        2) Prefer SMALL changes unless the snapshot shows actuator saturation or unsafe zone persists.
        3) If u_cool/u_valve/u_pump is near saturation (>=0.95), avoid pushing it further.
        4) If u_cool is saturated, prefer reducing Fin_sp before changing T_sp.
        5) Do not change setpoints without stating which KG relationship justifies it (cite PO6/PO7/PO8 in reasoning).
        """
        ).strip()
        + "\n\n"
        + action_kg
    )

    user_prompt = f"""
        CURRENT SNAPSHOT:
        - time: {row.get("time")}
        - phase: {row.get("phase")}
        - T_meas: {row.get("T_meas")}
        - L_meas: {row.get("L_meas")}
        - u_valve: {row.get("u_valve")}
        - u_pump: {row.get("u_pump")}
        - u_cool: {row.get("u_cool")}
        - anomaly_ratio: {state.get("anomaly_ratio")}
        - violated_params: {state.get("violated_params")}
        - control_zone: {state.get("control_zone")}
        - control_reasons: {state.get("control_reasons")}
        - current setpoints: T_sp={state.get("T_sp")}, L_sp={state.get("L_sp")}, Fin_sp={state.get("Fin_sp")}
        - symptoms: {sym}
        EVALUATION FEEDBACK (if any):
        {feedback}
        
        IMPORTANT RULES:
        1) If evaluation failed previously, you MUST propose a DIFFERENT set of setpoints than the previous proposal.
        Previous proposal: {prev_prop}
        2) Goal: reach SAFE and HOLD SAFE for at least {state.get("safe_hold_seconds", 60.0)} seconds in the digital-twin rollout.
        3) If u_cool/u_valve/u_pump is near saturation (>=0.95) or trending upward, temperature must stay close to 310 K.
        4) Setpoints can only be change in increments of 0.05 for T_sp and L_sp, and 0.005 for Fin_sp only.
        Now propose setpoints. Return ONLY JSON (no markdown).
        """

    llm_model = str(state.get("llm_model", "gpt-4o-mini"))
    llm = _make_llm(model=llm_model, temperature=0.0)
    plan = None

    # ----- metrics locals -----
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    t_start = time.perf_counter()

    # structured output if available
    try:
        plan_llm = llm.with_structured_output(SetpointPlan, include_raw=True)

        # tokens only reliable for OpenAI gpt-* with callback
        if llm_model.startswith("gpt"):
            with get_openai_callback() as cb:
                raw = plan_llm.invoke(
                    [("system", ACTION_SYSTEM_PROMPT_KG), ("user", user_prompt)]
                )
                prompt_tokens = int(getattr(cb, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(cb, "completion_tokens", 0) or 0)
                total_tokens = int(getattr(cb, "total_tokens", 0) or 0)
        else:
            raw = plan_llm.invoke(
                [("system", ACTION_SYSTEM_PROMPT_KG), ("user", user_prompt)]
            )

    except Exception as e:
        if llm_model.startswith("gpt"):
            with get_openai_callback() as cb:
                raw = llm.invoke(
                    [("system", ACTION_SYSTEM_PROMPT_KG), ("user", user_prompt)]
                )
                prompt_tokens = int(getattr(cb, "prompt_tokens", 0) or 0)
                completion_tokens = int(getattr(cb, "completion_tokens", 0) or 0)
                total_tokens = int(getattr(cb, "total_tokens", 0) or 0)
        else:
            raw = llm.invoke(
                [("system", ACTION_SYSTEM_PROMPT_KG), ("user", user_prompt)]
            )

        try:
            obj = json.loads(getattr(raw, "content", raw))
            plan = SetpointPlan(**obj)
        except Exception:
            raise RuntimeError(f"LLM output parse failed. Original error: {e}")

    plan = raw["parsed"]
    latency_s = time.perf_counter() - t_start

    # ----- update state counters (no logic change) -----
    state["action_calls"] = int(state.get("action_calls", 0)) + 1

    state["action_latency_last_s"] = float(latency_s)
    state["action_latency_sum_s"] = float(
        state.get("action_latency_sum_s", 0.0)
    ) + float(latency_s)
    state["action_latency_avg_s"] = float(state["action_latency_sum_s"]) / max(
        1, int(state["action_calls"])
    )

    state["action_prompt_tokens_last"] = int(prompt_tokens)
    state["action_completion_tokens_last"] = int(completion_tokens)
    state["action_total_tokens_last"] = int(total_tokens)

    state["action_prompt_tokens_sum"] = int(
        state.get("action_prompt_tokens_sum", 0)
    ) + int(prompt_tokens)
    state["action_completion_tokens_sum"] = int(
        state.get("action_completion_tokens_sum", 0)
    ) + int(completion_tokens)
    state["action_total_tokens_sum"] = int(
        state.get("action_total_tokens_sum", 0)
    ) + int(total_tokens)

    proposed = {
        "T_sp": float(plan.T_sp),
        "L_sp": float(plan.L_sp),
        "Fin_sp": float(plan.Fin_sp),
    }

    # Force change if it repeats exactly after failure
    if state.get("reprompt_count", 0) > 0 and isinstance(prev_prop, dict):
        same = all(
            abs(proposed.get(k, 0.0) - float(prev_prop.get(k, 1e9))) < 1e-12
            for k in ["T_sp", "L_sp", "Fin_sp"]
        )
        if same:
            # Deterministic nudge: reduce Fin_sp by 10%
            proposed["Fin_sp"] = 0.9 * proposed["Fin_sp"]

    state["proposed_sp"] = proposed
    state["eval_pass"] = False
    state["eval_summary"] = ""
    state["eval_fail_reason"] = ""
    state["eval_metrics"] = {}

    print("\n[ACTION_PROPOSE] Proposed setpoints:")
    print(" ", state["proposed_sp"])
    print("  reasoning:", getattr(plan, "reasoning", ""))
    print(
        f"  action_latency={state['action_latency_last_s']:.3f}s "
        f"tokens(prompt={state['action_prompt_tokens_last']}, "
        f"completion={state['action_completion_tokens_last']}, "
        f"total={state['action_total_tokens_last']})"
    )

    return state


def clone_plant_obj(plant_obj: CSTRSimulation) -> CSTRSimulation:
    try:
        return copy.deepcopy(plant_obj)
    except Exception:
        new_obj = CSTRSimulation(
            dt=float(getattr(plant_obj, "dt", 1.0)),
            T_sp=float(getattr(plant_obj, "T_sp", 310.0)),
            L_sp=float(getattr(plant_obj, "L_sp", 10.0)),
            Fin_sp_normal=float(getattr(plant_obj, "Fin_sp_normal", 2.0 / 60.0)),
            Fin_sp_startup=float(getattr(plant_obj, "Fin_sp_startup", 2.0 / 60.0)),
            fault_cfg=getattr(plant_obj, "fault_cfg", FaultConfig(enable=False)),
        )
        try:
            new_obj.__dict__.update(copy.deepcopy(plant_obj.__dict__))
        except Exception:
            new_obj.__dict__.update(dict(plant_obj.__dict__))
        return new_obj


def compute_symptoms(row: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    T = float(row.get("T_meas", np.nan))
    L = float(row.get("L_meas", np.nan))
    Fin = float(row.get("Fin", np.nan))
    Fout = float(row.get("Fout", np.nan))

    u_cool = float(row.get("u_cool", np.nan))
    u_valve = float(row.get("u_valve", np.nan))
    u_pump = float(row.get("u_pump", np.nan))

    T_sp = float(state.get("T_sp", 310.0))
    L_sp = float(state.get("L_sp", 10.0))

    T_err = (T - T_sp) if np.isfinite(T) else np.nan
    L_err = (L - L_sp) if np.isfinite(L) else np.nan

    cool_high = np.isfinite(u_cool) and (u_cool >= 0.95)
    cool_low = np.isfinite(u_cool) and (u_cool <= 0.05)
    valve_high = np.isfinite(u_valve) and (u_valve >= 0.95)
    pump_high = np.isfinite(u_pump) and (u_pump >= 0.95)

    cooling_ineffective = (np.isfinite(T_err) and T_err > 2.0) and cool_low
    cooling_overactive = (np.isfinite(T_err) and T_err < -1.5) and cool_high

    mass_imbalance = np.isfinite(Fin) and np.isfinite(Fout) and ((Fin - Fout) > 0.02)

    return {
        "T_err": float(T_err) if np.isfinite(T_err) else None,
        "L_err": float(L_err) if np.isfinite(L_err) else None,
        "cool_high": bool(cool_high),
        "cool_low": bool(cool_low),
        "valve_high": bool(valve_high),
        "pump_high": bool(pump_high),
        "cooling_ineffective": bool(cooling_ineffective),
        "cooling_overactive": bool(cooling_overactive),
        "mass_imbalance": bool(mass_imbalance),
    }


def rollout_validate_setpoints(
    *,
    base_plant: CSTRSimulation,
    state: GraphState,
    proposed: Dict[str, float],
    # --- acceptance knobs ---
    grace_s: float = 60.0,  # allow initial transient violations
    safe_persist_s: float = 60.0,  # must be safe this long consecutively
    deadline_s: float = 600.0,  # must reach safe within this time
    unsafe_fraction_max: float = 0.20,  # optional cap
    enforce_only_normal_phase: bool = True,  # ignore shutdown for safety judgement
) -> Tuple[bool, str, str, Dict[str, Any]]:
    """
    Recovery-based rollout validation.

    PASS if:
      - actuators always valid
      - reaches SAFE (safe_now + control_zone != UNSAFE) within deadline
      - remains SAFE for safe_persist_s consecutively
      - unsafe_fraction <= unsafe_fraction_max  (optional)

    Returns: ok, summary, fail_reason, metrics
    """
    twin = clone_plant_obj(base_plant)

    safe_checker = SafeStateChecker(
        T_safe_max=313.0,
        L_min_safe=2.0,
        L_max_safe=11.0,
        T_band=1.0,
        L_band=0.3,
        safe_window=30,
        safe_ratio=1.0,
        recovery_window=120,
        recovery_ratio=0.9,
    )

    control_checker = ControlSafetyChecker(
        ControlSafetyConfig(
            T_target=310.0,
            sat_threshold=0.95,
            hard_sat=0.99,
            trend_eps=0.002,
            T_band_normal=1.0,
            T_band_when_sat=0.5,
        )
    )

    T_sp_new = float(proposed["T_sp"])
    L_sp_new = float(proposed["L_sp"])
    Fin_sp_new = float(proposed["Fin_sp"])

    # end time is the experiment end (you want 7000s)
    T_end = float(state.get("t_end", 8000.0))
    out0 = state.get("sim_last", {})
    t0 = float(out0.get("t", 0.0)) if np.isfinite(float(out0.get("t", 0.0))) else 0.0

    dt = float(getattr(twin, "dt", 1.0))
    n_steps = int(max(0, np.ceil((T_end - t0) / max(dt, 1e-9))))

    # tracking
    unsafe_time = 0.0
    total_time = 0.0
    time_to_safe = None
    consecutive_safe = 0
    consecutive_required = int(np.ceil(safe_persist_s / max(dt, 1e-9)))

    any_invalid = False
    worst_zone = "SAFE"
    first_fail_reason = ""

    peak_T_meas = -np.inf
    peak_u_cool = -np.inf
    peak_u_valve = -np.inf
    peak_u_pump = -np.inf

    mode = str(state.get("mode", "RUN"))

    T_end = 7000  # <-- correct key

    dt = float(getattr(twin, "dt", 1.0))
    n_steps = int(max(0, np.ceil((T_end - t0) / max(dt, 1e-9))))  # <-- no hardcoding
    for i in range(n_steps):
        out = twin.simulate_step(
            T_sp=T_sp_new,
            L_sp=L_sp_new,
            Fin_sp_normal=Fin_sp_new,
            Fin_sp_startup=Fin_sp_new,
            mode=mode,
        )

        t = float(out.get("t", np.nan))
        if not np.isfinite(t):
            continue

        # track peaks for reporting
        T_meas = float(out.get("T_meas", np.nan))
        peak_T_meas = max(peak_T_meas, T_meas if np.isfinite(T_meas) else -np.inf)
        peak_u_cool = max(peak_u_cool, float(out.get("u_cool", -np.inf)))
        peak_u_valve = max(peak_u_valve, float(out.get("u_valve", -np.inf)))
        peak_u_pump = max(peak_u_pump, float(out.get("u_pump", -np.inf)))

        # phase gate (optional but recommended)
        phase_raw = out.get("phase", "NORMAL")
        phase = str(phase_raw).strip().upper()
        phase_num = 2
        if phase in ("STARTUP", "PHASE.STARTUP", "1"):
            phase_num = 1
        elif phase in ("NORMAL", "PHASE.NORMAL", "2"):
            phase_num = 2
        elif phase in ("SHUTDOWN", "PHASE.SHUTDOWN", "3"):
            phase_num = 3

        total_time += dt

        # update safety + control
        L_true = float(out.get("L_true", np.nan))
        F_over = float(out.get("F_over", 0.0))

        s = safe_checker.update(
            T_meas=T_meas,
            L_true=L_true,
            T_sp=T_sp_new,
            L_sp=L_sp_new,
            F_over=F_over,
        )
        c = control_checker.update(out=out, T_sp=T_sp_new)

        zone = str(c.get("control_zone", "SAFE"))
        if zone == "UNSAFE":
            worst_zone = "UNSAFE"
        elif zone == "WARNING" and worst_zone == "SAFE":
            worst_zone = "WARNING"

        if not bool(c.get("actuator_valid", True)):
            any_invalid = True
            if not first_fail_reason:
                first_fail_reason = "invalid_actuators"

        # Determine whether we *count* this step towards unsafe/safe judging
        # If you only care about recovery during NORMAL:
        if enforce_only_normal_phase and phase_num != 2:
            continue

        # Define "SAFE ZONE" for recovery
        safe_zone_now = bool(s.get("safe_now", False)) and (zone != "UNSAFE")

        # Grace period: don't penalize instability before grace_s
        if (t - t0) < grace_s:
            continue

        if not safe_zone_now:
            unsafe_time += dt
            consecutive_safe = 0
        else:
            consecutive_safe += 1
            # time_to_safe is first time we reach required persistence
            if time_to_safe is None and consecutive_safe >= consecutive_required:
                time_to_safe = t - t0

    unsafe_fraction = (unsafe_time / total_time) if total_time > 1e-9 else 1.0

    metrics = {
        "t0": t0,
        "T_end": T_end,
        "dt": dt,
        "total_time_simulated": total_time,
        "worst_zone": worst_zone,
        "unsafe_time": unsafe_time,
        "unsafe_fraction": unsafe_fraction,
        "time_to_safe": time_to_safe,
        "peak_T_meas": peak_T_meas if np.isfinite(peak_T_meas) else None,
        "peak_u_cool": peak_u_cool if np.isfinite(peak_u_cool) else None,
        "peak_u_valve": peak_u_valve if np.isfinite(peak_u_valve) else None,
        "peak_u_pump": peak_u_pump if np.isfinite(peak_u_pump) else None,
        "any_invalid_actuators": any_invalid,
    }

    # --- Decide acceptance ---
    if any_invalid:
        return (
            False,
            "FAIL: invalid actuators",
            first_fail_reason or "invalid_actuators",
            metrics,
        )

    if time_to_safe is None:
        return (
            False,
            f"FAIL: did not reach SAFE (persist {safe_persist_s}s)",
            "no_recovery",
            metrics,
        )

    if time_to_safe > deadline_s:
        return (
            False,
            f"FAIL: recovery too slow (time_to_safe={time_to_safe:.1f}s)",
            "slow_recovery",
            metrics,
        )

    if unsafe_fraction > unsafe_fraction_max:
        return (
            False,
            f"FAIL: unsafe_fraction too high ({unsafe_fraction:.3f})",
            "unsafe_fraction",
            metrics,
        )

    return (
        True,
        f"PASS: time_to_safe={time_to_safe:.1f}s, unsafe_fraction={unsafe_fraction:.3f}",
        "",
        metrics,
    )


def evaluation(state: GraphState) -> GraphState:
    print(
        "\n[EVAL] Validating proposed setpoints in digital twin rollout to END TIME..."
    )

    plant_obj: CSTRSimulation = state["_plant"]
    proposed = state.get("proposed_sp", None)

    if not proposed:
        state["eval_pass"] = False
        state["eval_summary"] = "FAIL: no proposed_sp found"
        state["eval_fail_reason"] = "missing_proposal"
        state["eval_metrics"] = {}
        return state

    ok, summary, fail_reason, metrics = rollout_validate_setpoints(
        base_plant=plant_obj,
        state=state,
        proposed=proposed,
        grace_s=60.0,
        safe_persist_s=60.0,
        deadline_s=600.0,
        unsafe_fraction_max=0.20,
        enforce_only_normal_phase=True,
    )

    state["eval_pass"] = bool(ok)
    state["eval_summary"] = summary
    state["eval_fail_reason"] = fail_reason
    state["eval_metrics"] = metrics

    print(f"[EVAL] {summary}")
    if not ok:
        print(f"[EVAL] fail_reason: {fail_reason}")
        print(f"[EVAL] metrics: {metrics}")

    return state


def reprompting(state: GraphState) -> GraphState:
    state["reprompt_count"] = int(state.get("reprompt_count", 0)) + 1
    reason = str(state.get("eval_fail_reason", "unknown"))
    metrics = state.get("eval_metrics", {}) or {}

    hint = ""
    if reason == "no_recovery":
        hint = "Did not reach safe zone. Reduce load (Fin_sp down) and/or reduce T_sp slightly."
    elif reason == "slow_recovery":
        hint = f"Recovery too slow (time_to_safe={metrics.get('time_to_safe')}). Try stronger change: lower Fin_sp more or lower T_sp more."
    elif reason == "unsafe_fraction":
        hint = "Too much time spent unsafe. Make a more conservative change or reduce oscillation."
    elif "invalid" in reason:
        hint = "Actuator invalid occurred. Choose milder setpoints."

    state["eval_feedback"] = (
        f"{state.get('eval_summary','')}; fail_reason={reason}; hint={hint}"
    )
    print(
        f"\n[REPROMPT] attempt {state['reprompt_count']}/{state.get('reprompt_max', 5)}"
    )
    return state


def action_apply(state: GraphState) -> GraphState:
    print("\n[ACTION_APPLY] Applying validated setpoints to plant...")

    proposed = state.get("proposed_sp", None)
    if not proposed:
        print("[ACTION_APPLY] Missing proposal; using fallback.")
        proposed = fallback_policy(state)

    state["L_sp"] = float(proposed["L_sp"])
    state["T_sp"] = float(proposed["T_sp"])
    state["Fin_sp"] = float(proposed["Fin_sp"])

    state["sp_change_done"] = True
    state["should_act"] = False

    out = state.get("sim_last", {})
    state["t_sp_change"] = float(out.get("t", np.nan))

    print(f"  at t={state['t_sp_change']}")
    print(f"  -> L_sp={state['L_sp']}  T_sp={state['T_sp']}  Fin_sp={state['Fin_sp']}")
    print(f"  eval_summary: {state.get('eval_summary')}")

    return state


def check_end(state: GraphState) -> GraphState:
    # hard limit: max_steps
    if state["itr"] >= state["max_steps"]:
        state["abort_reason"] = "max_steps"
        return state

    # ONLY terminate at end time
    t_now = float(state.get("sim_last", {}).get("t", 0.0))
    if np.isfinite(t_now) and (t_now >= float(state.get("T_end", 7000.0)) - 1e-9):
        state["abort_reason"] = "t_end_reached"
        return state

    return state


# =============================================================================
# Routers
# =============================================================================


def route_after_monitoring(state: GraphState):
    # if fault rising edge: propose action
    if state.get("should_act", False):
        return "action_propose"
    # otherwise check termination
    return "check_end"


def route_after_check_end(state: GraphState):
    return END if state.get("abort_reason") else "plant"


def route_after_evaluation(state: GraphState):
    if bool(state.get("eval_pass", False)):
        return "action_apply"

    if int(state.get("reprompt_count", 0)) >= int(state.get("reprompt_max", 5)):
        # Fallback instead of dying
        print("\n[EVAL] Reprompt limit hit -> using fallback_policy()")
        state["proposed_sp"] = fallback_policy(state)
        state["eval_pass"] = True
        state["eval_summary"] = "PASS(fallback): applying deterministic fallback policy"
        state["eval_fail_reason"] = ""
        state["eval_metrics"] = {}
        return "action_apply"

    return "reprompting"


# =============================================================================
# Graph builder
# =============================================================================


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("initializing", initializing)
    g.add_node("plant", plant)
    g.add_node("monitoring", monitoring)
    g.add_node("check_end", check_end)

    g.add_node("action_propose", action_propose)
    g.add_node("evaluation", evaluation)
    g.add_node("reprompting", reprompting)
    g.add_node("action_apply", action_apply)

    g.add_edge(START, "initializing")
    g.add_edge("initializing", "plant")
    g.add_edge("plant", "monitoring")

    g.add_conditional_edges(
        "monitoring", route_after_monitoring, ["action_propose", "check_end"]
    )
    g.add_conditional_edges("check_end", route_after_check_end, ["plant", END])

    g.add_edge("action_propose", "evaluation")
    g.add_conditional_edges(
        "evaluation", route_after_evaluation, ["action_apply", "reprompting"]
    )
    g.add_edge("reprompting", "action_propose")
    g.add_edge("action_apply", "plant")

    return g.compile()


# =============================================================================
# Results -> plot vectors
# =============================================================================


def extract_vectors(sim_log: List[Dict[str, Any]]):
    t_hist = [float(r.get("t", np.nan)) for r in sim_log]
    phase_map = {"STARTUP": 1, "NORMAL": 2, "SHUTDOWN": 3}
    phase_hist = [
        phase_map.get(str(r.get("phase", "NORMAL")).upper(), 2) for r in sim_log
    ]

    L_hist = [float(r.get("L_true", np.nan)) for r in sim_log]
    V_hist = [float(r.get("V", np.nan)) for r in sim_log]

    Fin_hist = [float(r.get("Fin", np.nan)) for r in sim_log]
    Fout_hist = [float(r.get("Fout", np.nan)) for r in sim_log]
    Fc_hist = [float(r.get("Fc", np.nan)) for r in sim_log]

    u_valve_hist = [float(r.get("u_valve", np.nan)) for r in sim_log]
    u_pump_hist = [float(r.get("u_pump", np.nan)) for r in sim_log]
    u_cool_hist = [float(r.get("u_cool", np.nan)) for r in sim_log]

    T_hist = [float(r.get("T_true", np.nan)) for r in sim_log]
    Tmeas_hist = [float(r.get("T_meas", np.nan)) for r in sim_log]

    UAeff_hist = [float(r.get("UA_eff", np.nan)) for r in sim_log]
    leak_hist = [float(r.get("F_leak", 0.0)) for r in sim_log]
    over_hist = [float(r.get("F_over", 0.0)) for r in sim_log]

    L_sp_hist = [float(r.get("L_sp", np.nan)) for r in sim_log]
    T_sp_hist = [float(r.get("T_sp", np.nan)) for r in sim_log]
    Fin_sp_hist = [float(r.get("Fin_sp", np.nan)) for r in sim_log]

    def _x_at(idx: int, fallback_key: str):
        out = []
        for r in sim_log:
            x = r.get("x", None)
            if isinstance(x, (list, tuple, np.ndarray)) and len(x) > idx:
                out.append(float(x[idx]))
            else:
                out.append(float(r.get(fallback_key, np.nan)))
        return out

    CA_hist = _x_at(1, "CA")
    CB_hist = _x_at(2, "CB")
    CC_hist = _x_at(3, "CC")
    CD_hist = _x_at(4, "CD")

    fault_tag_hist = [
        str(r.get("fault_tags", r.get("fault_tag", "OK"))) for r in sim_log
    ]

    t_sp_change = None
    for r in sim_log:
        v = r.get("t_sp_change", None)
        if v is None:
            continue
        try:
            fv = float(v)
        except Exception:
            continue
        if np.isfinite(fv):
            t_sp_change = fv
            break

    return {
        "t_hist": t_hist,
        "phase_hist": phase_hist,
        "L_hist": L_hist,
        "V_hist": V_hist,
        "Fin_hist": Fin_hist,
        "Fout_hist": Fout_hist,
        "Fc_hist": Fc_hist,
        "u_valve_hist": u_valve_hist,
        "u_pump_hist": u_pump_hist,
        "u_cool_hist": u_cool_hist,
        "T_hist": T_hist,
        "Tmeas_hist": Tmeas_hist,
        "UAeff_hist": UAeff_hist,
        "leak_hist": leak_hist,
        "over_hist": over_hist,
        "CA_hist": CA_hist,
        "CB_hist": CB_hist,
        "CC_hist": CC_hist,
        "CD_hist": CD_hist,
        "fault_tag_hist": fault_tag_hist,
        "L_sp_hist": L_sp_hist,
        "T_sp_hist": T_sp_hist,
        "Fin_sp_hist": Fin_sp_hist,
        "t_sp_change": t_sp_change,
    }


def plot_results(sim_log: List[Dict[str, Any]], fault_cfg: FaultConfig, title: str):
    vec = extract_vectors(sim_log)

    T_startup = 1000.0
    T_normal = 6000.0
    T_shutdown = 1000.0

    L_sp_final = float(sim_log[-1].get("L_sp", 10.0)) if sim_log else 10.0
    T_sp_final = float(sim_log[-1].get("T_sp", 310.0)) if sim_log else 310.0

    plot_cstr_digital_twin(
        t_hist=vec["t_hist"],
        L_sp_hist=vec["L_sp_hist"],
        T_sp_hist=vec["T_sp_hist"],
        L_hist=vec["L_hist"],
        V_hist=vec["V_hist"],
        phase_hist=vec["phase_hist"],
        Fin_hist=vec["Fin_hist"],
        Fout_hist=vec["Fout_hist"],
        Fin_sp_hist=vec["Fin_sp_hist"],
        CA_hist=vec["CA_hist"],
        CB_hist=vec["CB_hist"],
        CC_hist=vec["CC_hist"],
        CD_hist=vec["CD_hist"],
        u_valve_hist=vec["u_valve_hist"],
        u_pump_hist=vec["u_pump_hist"],
        u_cool_hist=vec["u_cool_hist"],
        T_hist=vec["T_hist"],
        Tmeas_hist=vec["Tmeas_hist"],
        Fc_hist=vec["Fc_hist"],
        UAeff_hist=vec["UAeff_hist"],
        leak_hist=vec["leak_hist"],
        over_hist=vec["over_hist"],
        fault_tag_hist=vec["fault_tag_hist"],
        T_startup=T_startup,
        T_normal=T_normal,
        T_shutdown=T_shutdown,
        L_sp=L_sp_final,
        T_sp=T_sp_final,
        t_sp_change=vec["t_sp_change"],
        label_sp_change="SP change @ fault flag",
        show_fault_markers=True,
        fouling_t_start=fault_cfg.fouling_t_start if fault_cfg.fouling_on else None,
        pump_degrade_t=fault_cfg.pump_degrade_t if fault_cfg.pump_degrade_on else None,
        title=title,
    )


def _extract_cstr_metrics(
    out: GraphState,
    *,
    fault: str,
    run: int,
    wall_time: float,
    init_state: Dict[str, Any],
) -> Dict[str, Any]:
    sim_last = out.get("sim_last", {}) or {}
    t_final = sim_last.get("t", None)
    phase_final = sim_last.get("phase", None)

    eval_metrics = out.get("eval_metrics", {}) or {}

    # basic “success” definition for this CSTR setting:
    # - reached T_end AND ended in SAFE zone (or at least not UNSAFE) AND safe_now true
    reached_end = out.get("abort_reason") == "t_end_reached"
    final_zone = out.get("control_zone", None)
    safe_now = bool(out.get("safe_now", False))

    reprompt_count = int(out.get("reprompt_count", 0))
    reprompt_max = int(init_state.get("reprompt_max", 5))  # use the configured value
    reprompt_hit = reprompt_count >= reprompt_max

    success = int(
        bool(out.get("sp_change_done", False))
        and bool(out.get("eval_pass", False))
        and (not reprompt_hit)
    )

    # reprompts used (your reprompt_count increments in reprompting())
    reprompts = int(out.get("reprompt_count", 0))

    # how many times did we decide to act?
    # (sp_change_done is only a single change in your design; still useful)
    acted = int(bool(out.get("sp_change_done", False)))

    # estimate “time_to_first_action”
    t_sp_change = out.get("t_sp_change", None)
    try:
        time_to_first_action = (
            float(t_sp_change)
            if t_sp_change is not None and np.isfinite(float(t_sp_change))
            else None
        )
    except Exception:
        time_to_first_action = None

    # get detector outcome at end
    fault_flag_end = int(bool(out.get("fault_flag", False)))
    anomaly_ratio_end = float(out.get("anomaly_ratio", 0.0))

    # optional: count how many seconds were UNSAFE in sim_log (rough but paper-friendly)
    sim_log = out.get("sim_log", []) or []
    unsafe_seconds = 0
    warning_seconds = 0
    safe_seconds = 0
    for r in sim_log:
        z = str(r.get("control_zone", ""))
        if z == "UNSAFE":
            unsafe_seconds += 1
        elif z == "WARNING":
            warning_seconds += 1
        elif z == "SAFE":
            safe_seconds += 1

    total_steps = int(out.get("itr", 0))
    sim_time = None
    try:
        sim_time = float(t_final) if t_final is not None else None
    except Exception:
        sim_time = None

    # final setpoints
    final_T_sp = out.get("T_sp", None)
    final_L_sp = out.get("L_sp", None)
    final_Fin_sp = out.get("Fin_sp", None)

    # evaluation pass/fail at last action
    eval_pass = int(bool(out.get("eval_pass", False)))

    # Paper-friendly: pull key rollout metrics if present
    time_to_safe = eval_metrics.get("time_to_safe", None)
    unsafe_fraction = eval_metrics.get("unsafe_fraction", None)
    peak_T_meas = eval_metrics.get("peak_T_meas", None)
    peak_u_cool = eval_metrics.get("peak_u_cool", None)
    peak_u_valve = eval_metrics.get("peak_u_valve", None)
    peak_u_pump = eval_metrics.get("peak_u_pump", None)

    return {
        # identifiers
        "fault": fault,
        "run": run,
        "llm_model": init_state.get("llm_model"),
        "detector_mode": init_state.get("detector_mode"),
        # outcomes
        "success": success,
        "reached_end": int(reached_end),
        "abort_reason": out.get("abort_reason", ""),
        "eval_pass": eval_pass,
        # timing
        "wall_time_s": float(wall_time),
        "sim_time_s": sim_time,
        "steps": total_steps,
        # state at end
        "final_t": t_final,
        "final_phase": phase_final,
        "final_control_zone": final_zone,
        "final_safe_now": int(safe_now),
        "final_fault_flag": fault_flag_end,
        "final_anomaly_ratio": anomaly_ratio_end,
        "final_control_reasons": out.get("control_reasons", ""),
        # action behavior
        "acted": acted,
        "t_sp_change": t_sp_change,
        "time_to_first_action": time_to_first_action,
        "reprompts": reprompts,
        # safety exposure (rough from logs)
        "unsafe_seconds": unsafe_seconds,
        "warning_seconds": warning_seconds,
        "safe_seconds": safe_seconds,
        # final setpoints
        "final_T_sp": final_T_sp,
        "final_L_sp": final_L_sp,
        "final_Fin_sp": final_Fin_sp,
        # rollout/eval metrics (if any)
        "rollout_time_to_safe": time_to_safe,
        "rollout_unsafe_fraction": unsafe_fraction,
        "rollout_peak_T_meas": peak_T_meas,
        "rollout_peak_u_cool": peak_u_cool,
        "rollout_peak_u_valve": peak_u_valve,
        "rollout_peak_u_pump": peak_u_pump,
        # LLM usage (action agent)
        "action_calls": int(out.get("action_calls", 0)),
        "action_latency_sum_s": float(out.get("action_latency_sum_s", 0.0)),
        "action_latency_avg_s": float(out.get("action_latency_avg_s", 0.0)),
        "action_prompt_tokens_sum": int(out.get("action_prompt_tokens_sum", 0)),
        "action_completion_tokens_sum": int(out.get("action_completion_tokens_sum", 0)),
        "action_total_tokens_sum": int(out.get("action_total_tokens_sum", 0)),
    }


def run_single_experiment_cstr(
    graph,
    *,
    fault: str,
    run: int,
    args: argparse.Namespace,
    recursion_limit: int = 200000,
    verbose: bool = True,
) -> Dict[str, Any]:
    init_state: GraphState = {
        "fault_name": fault,
        "detector_mode": args.mode,
        "max_steps": args.max_steps,
        "mode": "RUN",
        "T_sp": args.T_sp,
        "L_sp": args.L_sp,
        "Fin_sp": args.Fin_sp,
        "T_end": args.T_end,
        "safe_hold_seconds": args.safe_hold,
        "max_time_to_safe": (
            None if args.max_time_to_safe <= 0 else float(args.max_time_to_safe)
        ),
        "llm_model": args.llm_model,
        "reprompt_max": args.reprompt_max,
    }

    if verbose:
        print(f"\n--- Run {run+1} fault={fault} ---")

    t0 = time.perf_counter()
    out = graph.invoke(init_state, config={"recursion_limit": recursion_limit})
    wall = time.perf_counter() - t0

    metrics = _extract_cstr_metrics(
        out, fault=fault, run=run, wall_time=wall, init_state=init_state
    )

    # include eval summary text for debugging / later analysis (short field)
    metrics["eval_summary"] = str(out.get("eval_summary", ""))[:500]
    metrics["eval_fail_reason"] = str(out.get("eval_fail_reason", ""))[:200]

    # optional: keep a pointer-like “analysis blob” per run
    metrics["_analysis"] = {
        "final_setpoints": {
            "T_sp": out.get("T_sp"),
            "L_sp": out.get("L_sp"),
            "Fin_sp": out.get("Fin_sp"),
        },
        "last_eval_metrics": out.get("eval_metrics", {}),
        "last_eval_summary": out.get("eval_summary", ""),
        "last_eval_fail_reason": out.get("eval_fail_reason", ""),
        "final_control_zone": out.get("control_zone", ""),
        "final_control_reasons": out.get("control_reasons", ""),
        "final_fault_flag": bool(out.get("fault_flag", False)),
        "final_violated_params": out.get("violated_params", ""),
    }

    return metrics, out


def run_ablation_study_cstr(
    graph,
    *,
    faults: List[str],
    n_runs: int,
    args: argparse.Namespace,
    output_dir: str = "./results",
    save_plots: bool = False,  # plots are usually annoying in ablations
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_metrics: List[Dict[str, Any]] = []
    analysis_runs: List[Dict[str, Any]] = []

    print("\n" + "=" * 70)
    print("STARTING CSTR ABLATION STUDY")
    print("=" * 70)
    print(f"LLM model: {args.llm_model}")
    print(f"Detector mode: {args.mode}")
    print(f"Faults: {faults}")
    print(f"Runs per fault: {n_runs}")
    print(f"Total experiments: {len(faults) * n_runs}")
    print("=" * 70 + "\n")

    for fault in faults:
        print(f"\n{'='*45}")
        print(f"FAULT: {fault}")
        print(f"{'='*45}")

        for run in range(n_runs):
            try:
                metrics, out = run_single_experiment_cstr(
                    graph,
                    fault=fault,
                    run=run,
                    args=args,
                    verbose=True,
                )
                all_metrics.append(
                    {k: v for k, v in metrics.items() if k != "_analysis"}
                )
                analysis_runs.append(
                    {
                        "fault": fault,
                        "run": run,
                        "llm_model": args.llm_model,
                        "mode": args.mode,
                        "metrics": {
                            k: v for k, v in metrics.items() if k != "_analysis"
                        },
                        "analysis": metrics.get("_analysis", {}),
                    }
                )

                print(
                    f"  success={metrics['success']} "
                    f"reached_end={metrics['reached_end']} "
                    f"eval_pass={metrics['eval_pass']} "
                    f"zone={metrics['final_control_zone']} "
                    f"reprompts={metrics['reprompts']} "
                    f"unsafe_s={metrics['unsafe_seconds']} "
                    f"wall={metrics['wall_time_s']:.2f}s"
                )

                # Optional: plot only for single-run or if requested
                if save_plots:
                    fault_cfg = out.get("fault_cfg", FaultConfig(enable=False))
                    plot_results(
                        out.get("sim_log", []),
                        fault_cfg=fault_cfg,
                        title=f"CSTR ({fault}) run={run}",
                    )

            except Exception as e:
                print(f"  ERROR: {e}")
                all_metrics.append(
                    {
                        "fault": fault,
                        "run": run,
                        "llm_model": args.llm_model,
                        "detector_mode": args.mode,
                        "success": 0,
                        "reached_end": 0,
                        "abort_reason": "exception",
                        "eval_pass": 0,
                        "wall_time_s": 0.0,
                        "sim_time_s": None,
                        "steps": 0,
                        "final_control_zone": "",
                        "final_safe_now": 0,
                        "reprompts": 0,
                        "unsafe_seconds": None,
                        "warning_seconds": None,
                        "safe_seconds": None,
                        "error": str(e)[:500],
                    }
                )
                analysis_runs.append(
                    {
                        "fault": fault,
                        "run": run,
                        "llm_model": args.llm_model,
                        "mode": args.mode,
                        "error": str(e),
                    }
                )

    df = pd.DataFrame(all_metrics)

    # filenames
    model_short = (
        str(args.llm_model).replace("gpt-", "gpt").replace(":", "").replace("/", "_")
    )
    base = f"cstr_ablation_{model_short}_{args.mode}_{timestamp}"

    runs_csv = os.path.join(output_dir, f"{base}_runs.csv")
    df.to_csv(runs_csv, index=False)
    print(f"\n✓ Per-run results saved to: {runs_csv}")

    # summary per fault
    summary_rows = []
    for fault in faults:
        fdf = df[df["fault"] == fault]
        if len(fdf) == 0:
            continue
        summary_rows.append(
            {
                "fault": fault,
                "n": int(len(fdf)),
                "success_mean": (
                    float(fdf["success"].mean()) if "success" in fdf else None
                ),
                "success_std": (
                    float(fdf["success"].std())
                    if len(fdf) > 1 and "success" in fdf
                    else None
                ),
                "reached_end_mean": (
                    float(fdf["reached_end"].mean()) if "reached_end" in fdf else None
                ),
                "eval_pass_mean": (
                    float(fdf["eval_pass"].mean()) if "eval_pass" in fdf else None
                ),
                "reprompts_mean": (
                    float(fdf["reprompts"].mean()) if "reprompts" in fdf else None
                ),
                "unsafe_seconds_mean": (
                    float(fdf["unsafe_seconds"].mean())
                    if "unsafe_seconds" in fdf
                    else None
                ),
                "wall_time_mean": (
                    float(fdf["wall_time_s"].mean()) if "wall_time_s" in fdf else None
                ),
                "wall_time_std": (
                    float(fdf["wall_time_s"].std())
                    if len(fdf) > 1 and "wall_time_s" in fdf
                    else None
                ),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, f"{base}_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"✓ Summary results saved to: {summary_csv}")

    # “analysis artifact” JSON (nice for paper / debug)
    analysis_json = os.path.join(output_dir, f"{base}_analysis.json")
    analysis_blob = {
        "meta": {
            "timestamp": timestamp,
            "llm_model": args.llm_model,
            "detector_mode": args.mode,
            "runs_per_fault": n_runs,
            "faults": faults,
            "args": vars(args),
        },
        "per_fault_summary": summary_rows,
        "runs": analysis_runs,
    }
    with open(analysis_json, "w", encoding="utf-8") as f:
        json.dump(analysis_blob, f, indent=2)
    print(f"✓ Analysis JSON saved to: {analysis_json}")

    # console table
    print("\nPer-Fault Summary:")
    print("-" * 80)
    print(
        f"{'Fault':<18} {'Succ%':>7} {'End%':>7} {'Eval%':>7} {'Rep':>5} {'Unsafe_s':>9} {'Wall_s':>8}"
    )
    print("-" * 80)
    for _, r in summary_df.iterrows():
        succ = (
            100.0 * float(r["success_mean"])
            if r.get("success_mean") is not None
            else 0.0
        )
        endp = (
            100.0 * float(r["reached_end_mean"])
            if r.get("reached_end_mean") is not None
            else 0.0
        )
        evalp = (
            100.0 * float(r["eval_pass_mean"])
            if r.get("eval_pass_mean") is not None
            else 0.0
        )
        rep = float(r["reprompts_mean"]) if r.get("reprompts_mean") is not None else 0.0
        us = (
            float(r["unsafe_seconds_mean"])
            if r.get("unsafe_seconds_mean") is not None
            else 0.0
        )
        ws = float(r["wall_time_mean"]) if r.get("wall_time_mean") is not None else 0.0
        print(
            f"{str(r['fault']):<18} {succ:>6.1f}% {endp:>6.1f}% {evalp:>6.1f}% {rep:>5.1f} {us:>9.1f} {ws:>8.2f}"
        )
    print("-" * 80)

    return df


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fault",
        type=str,
        default="fouling",
        choices=[
            "all",
            "fouling",
            "pump_degrade",
            "cool_stuck_closed",
        ],
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="Runs per fault (for ablation)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Output directory for CSV/JSON analysis",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="3sigma",
        choices=["strict", "normal", "relaxed", "3sigma"],
    )
    parser.add_argument("--max-steps", type=int, default=20000)

    parser.add_argument("--T-sp", type=float, default=310.0)
    parser.add_argument("--L-sp", type=float, default=10.0)
    parser.add_argument("--Fin-sp", type=float, default=2.0 / 60.0)

    parser.add_argument("--T-end", type=float, default=8000.0)

    parser.add_argument("--safe-hold", type=float, default=60.0)
    parser.add_argument(
        "--max-time-to-safe", type=float, default=-1.0, help="<=0 means disabled"
    )

    parser.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--reprompt-max", type=int, default=5)

    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show plot (single run) / save plots (ablation)",
    )
    args = parser.parse_args()

    # KG loading stays as you had it
    if SPARQL_AVAILABLE:
        load_kg_context()

    graph = build_graph()

    print("\n" + "=" * 90)
    print("CSTR — monitoring-triggered LLM setpoint proposal + digital-twin evaluation")
    print("=" * 90)
    print(
        f"fault={args.fault}, runs={args.runs}, end_time={args.T_end}, safe_hold={args.safe_hold}s, llm={args.llm_model}"
    )
    print("=" * 90 + "\n")

    if args.fault == "all":
        faults = ["fouling", "pump_degrade", "cool_stuck_closed"]
        df = run_ablation_study_cstr(
            graph,
            faults=faults,
            n_runs=int(args.runs),
            args=args,
            output_dir=args.output,
            save_plots=bool(args.plot),
        )
        # no plotting by default in ablation; CSV/JSON are saved
        return

    # ---- single run (keeps your old behavior, but also writes analysis files) ----
    metrics, out = run_single_experiment_cstr(
        graph, fault=args.fault, run=0, args=args, verbose=True
    )

    # Save single-run analysis files too (nice for paper/debug)
    os.makedirs(args.output, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = (
        str(args.llm_model).replace("gpt-", "gpt").replace(":", "").replace("/", "_")
    )
    base = f"cstr_single_{args.fault}_{model_short}_{args.mode}_{timestamp}"

    runs_csv = os.path.join(args.output, f"{base}_run.csv")
    pd.DataFrame([{k: v for k, v in metrics.items() if k != "_analysis"}]).to_csv(
        runs_csv, index=False
    )

    analysis_json = os.path.join(args.output, f"{base}_analysis.json")
    with open(analysis_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {"timestamp": timestamp, "args": vars(args)},
                "metrics": {k: v for k, v in metrics.items() if k != "_analysis"},
                "analysis": metrics.get("_analysis", {}),
            },
            f,
            indent=2,
        )

    print("\n=== FINISHED (single run) ===")
    for k in [
        "success",
        "reached_end",
        "eval_pass",
        "abort_reason",
        "final_control_zone",
        "reprompts",
        "unsafe_seconds",
        "wall_time_s",
    ]:
        if k in metrics:
            print(f"{k}: {metrics[k]}")
    print(f"\n✓ Saved: {runs_csv}")
    print(f"✓ Saved: {analysis_json}")

    if args.plot:
        fault_cfg = out.get("fault_cfg", FaultConfig(enable=False))
        plot_results(
            out.get("sim_log", []),
            fault_cfg=fault_cfg,
            title="CSTR Digital Twin (LLM + Eval + Apply)",
        )


if __name__ == "__main__":
    main()
