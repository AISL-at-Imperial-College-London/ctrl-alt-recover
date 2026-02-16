from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Literal, List, Dict, Any, Optional
import math, os, pandas as pd, time
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph
from dotenv import load_dotenv
from textwrap import dedent
import json
import sys
import copy

sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
load_dotenv()

from mixer_module.simulation import Simulation

# =============================================================================
# SPARQL/KG Integration via graph_retrieval.py
# =============================================================================

try:
    from graph_retrieval_code import run_construct, QUERY_PLANNING, QUERY_ACTION

    SPARQL_AVAILABLE = True
except ImportError:
    print("[WARNING] graph_retrieval.py not found - running without KG context")
    SPARQL_AVAILABLE = False

# KG Context Cache (loaded once at startup)
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
        _KG_PLANNING_TTL = run_construct(QUERY_PLANNING)
        _KG_ACTION_TTL = run_construct(QUERY_ACTION)

        if _KG_PLANNING_TTL and _KG_ACTION_TTL:
            _KG_LOADED = True
            # Count approximate triples
            planning_lines = len(
                [
                    l
                    for l in _KG_PLANNING_TTL.split("\n")
                    if l.strip() and not l.startswith("@")
                ]
            )
            action_lines = len(
                [
                    l
                    for l in _KG_ACTION_TTL.split("\n")
                    if l.strip() and not l.startswith("@")
                ]
            )
            print(f"[KG] ✅ Planning context: ~{planning_lines} statements")
            print(f"[KG] ✅ Action context: ~{action_lines} statements")
            return True
        else:
            print("[KG] ⚠️ Empty response from GraphDB")
            return False
    except Exception as e:
        print(f"[KG] ⚠️ Could not load KG context: {e}")
        return False


def get_planning_kg_context() -> str:
    """Get Planning KG context as formatted prompt section."""
    if not _KG_LOADED or not _KG_PLANNING_TTL:
        return ""

    return f"""

## KNOWLEDGE GRAPH DATA (UML:StateMachine with Transitions)

The following Turtle data shows the state machine structure including `rdfs:comment` 
on transitions that contain **domain knowledge**.

**CRITICAL**: Read the `rdfs:comment` on transitions to understand fault conditions!

```turtle
{_KG_PLANNING_TTL}
```
"""


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
# GLOBAL CONFIGURATION (set via CLI or defaults)
# =============================================================================

CURRENT_MODEL = "gpt-4o"  # Default model
CURRENT_PROMPT_LEVEL = "full"  # Default prompt level

# Available models
AVAILABLE_MODELS = [
    "gpt-4o",
    "gpt_4.1",
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-5",
    "gpt-5-mini",
    "llama3:8b",
    "qwen3:8b",
    # "gpt-5-nano", (slow)
]


# Temperature per model family
def get_model_temperature(model_name: str) -> float:
    """
    Get appropriate temperature for model.
    - GPT-5 models: temperature=1 (recommended by OpenAI)
    - GPT-4 models: temperature=0 (deterministic)
    """
    model_lower = model_name.lower()

    if "gpt-5" in model_lower or "gpt5" in model_lower:
        return 1.0
    else:
        return 0.0


# Available prompt levels
PROMPT_LEVELS = ["normal", "minimal"]


# =============================================================================
# STATE DEFINITION
# =============================================================================


class GraphState(TypedDict):
    actions: List[str]
    digital_twin_states: Dict[str, Any]

    log_level_B201: list
    log_level_B202: list
    log_level_B203: list
    log_level_B204: list

    log_tank_B201_state: list
    log_tank_B202_state: list
    log_tank_B203_state: list
    log_tank_B204_state: list

    level_B201: float
    level_B202: float
    level_B203: float
    level_B204: float

    tank_B201_state: str
    tank_B202_state: str
    tank_B203_state: str
    tank_B204_state: str

    max_height_B201: float
    min_height_B201: float
    max_height_B202: float
    min_height_B202: float
    max_height_B203: float
    min_height_B203: float
    max_height_B204: float
    min_height_B204: float
    time: float
    sim_time: float

    valve_in_B201: int
    valve_in_B202: int
    valve_in_B203: int
    valve_in_B204: int
    valve_out_B201: int
    valve_out_B202: int
    valve_out_B203: int
    valve_out_B204: int

    final_state: str
    curr_state: str
    path: List[str]

    pump_main: float
    pump_alt: float

    pump_power: float
    pump_power_alt: float
    should_act: bool
    should_reprompt: bool

    timer: float
    log_time: list
    log_valve_in_B201: list
    log_valve_in_B202: list
    log_valve_in_B203: list
    log_valve_in_B204: list
    log_valve_out_B201: list
    log_valve_out_B202: list
    log_valve_out_B203: list
    log_valve_out_B204: list

    log_pump_main: list
    log_pump_alt: list

    log_pump_power: list
    log_pump_power_alt: list
    log_should_act: list
    log_should_reprompt: list

    itr: int
    condition: str

    # ---- Metrics ----
    m_start_wall: float
    m_total_wall: float
    m_llm_planning_calls: int
    m_llm_action_calls: int
    m_llm_planning_latency_s: float
    m_llm_action_latency_s: float
    m_reprompt_count: int
    m_invalid_actuator_count: int
    m_step_count: int
    m_success: bool
    m_success_itr: int
    m_success_sim_time: float
    m_abort_reason: (
        str  # Why experiment ended: 'success', 'reprompt_limit', 'max_iterations'
    )
    m_cycle_count: int
    m_last_state: str

    # ---- Token Usage Metrics ----
    m_planning_input_tokens: int
    m_planning_output_tokens: int
    m_action_input_tokens: int
    m_action_output_tokens: int

    # ---- Extra Logs ----
    log_curr_state: list
    log_paths: list
    log_actions_raw: list
    log_reprompts: list
    log_llm_applied_actions: list
    prev_state: str
    simulation_error: str

    # ---- Enhanced Evaluation Metrics ----
    m_planning_correct: int
    m_planning_total: int
    m_missed_transitions: int
    m_physical_violations: int
    m_actuator_correct: int
    m_actuator_total: int
    log_planning_checks: list
    log_actuator_checks: list
    log_physical_checks: list
    log_feasibility_checks: list
    prev_levels: dict

    # ---- Decision Levels (for rollback) ----
    decision_levels: Dict[str, float]  # Levels at decision time (BEFORE simulation)

    # ---- Detailed Iteration Logging ----
    log_iteration_details: list  # Detailed per-iteration data for CSV export
    current_iteration_data: dict  # Temporary storage for current iteration

    # ---- Detailed Iteration Logging ----
    log_iterations_detailed: list  # Detailed per-iteration data for CSV export
    iteration_timing: Dict[str, float]  # Timing for current iteration phases


# =============================================================================
# PYDANTIC MODELS FOR STRUCTURED LLM OUTPUT
# =============================================================================


class PathPlan(BaseModel):
    """Structured output for path planning"""

    current_state: str = Field(..., description="The current state we are in")
    next_state: str = Field(..., description="The next state to transition to")
    reasoning: str = Field(..., description="Brief explanation of why this transition")


class ActionItem(BaseModel):
    """Single actuator command"""

    actuator: str = Field(
        ...,
        description="Actuator name: valve_in_B201, valve_out_B201, pump_P101, bypass_pump_P102, etc.",
    )
    value: float = Field(..., description="Value: 0 or 1 for valves, 0.0-1.0 for pumps")


class ActionPlan(BaseModel):
    """Action plan with explicit actuator commands"""

    target_state: str = Field(
        ..., description="The state we are configuring actuators for"
    )
    actions: List[ActionItem] = Field(..., description="List of actuator commands")
    reasoning: str = Field(..., description="Brief explanation of why these actuators")


# =============================================================================
# ACTUATOR MAPPING
# =============================================================================

ACTUATOR_TO_STATE_KEY = {
    # Inlet valves
    "valve_in_B201": "valve_in_B201",
    "valve_in0": "valve_in_B201",
    "valve_in_B202": "valve_in_B202",
    "valve_in1": "valve_in_B202",
    "valve_in_B203": "valve_in_B203",
    "valve_in2": "valve_in_B203",
    "mixer0_valve_in0_opening_actuator": "valve_in_B201",
    "mixer0_valve_in1_opening_actuator": "valve_in_B202",
    "mixer0_valve_in2_opening_actuator": "valve_in_B203",
    "mixer0_valve_out_opening_actuator": "valve_out_B204",
    # Outlet/pump valves
    "valve_out_B201": "valve_out_B201",
    "valve_pump_tank_B201": "valve_out_B201",
    "valve_pump_B201": "valve_out_B201",
    "valve_pump_B201_on": "valve_out_B201",
    "mixer0_valve_pump_tank_b201_opening_actuator": "valve_out_B201",
    "valve_out_B202": "valve_out_B202",
    "valve_pump_tank_B202": "valve_out_B202",
    "valve_pump_B202": "valve_out_B202",
    "valve_pump_B202_on": "valve_out_B202",
    "mixer0_valve_pump_tank_b202_opening_actuator": "valve_out_B202",
    "valve_out_B203": "valve_out_B203",
    "valve_pump_tank_B203": "valve_out_B203",
    "valve_pump_B203": "valve_out_B203",
    "valve_pump_B203_on": "valve_out_B203",
    "mixer0_valve_pump_tank_b203_opening_actuator": "valve_out_B203",
    "valve_out_B204": "valve_out_B204",
    "valve_out": "valve_out_B204",
    # Inlet to collection tank
    "valve_in_B204": "valve_in_B204",
    "valve_pump_tank_B204": "valve_in_B204",
    "valve_pump_B204": "valve_in_B204",
    "valve_pump_B204_on": "valve_in_B204",
    "mixer0_valve_pump_tank_b204_opening_actuator": "valve_in_B204",
    # Pumps
    "pump_P101": "pump_power",
    "pump_main": "pump_power",
    "P101": "pump_power",
    "mixer0_pump_p101_n_in_actuator": "pump_power",
    "bypass_pump_P102": "pump_power_alt",
    "pump_P102": "pump_power_alt",
    "pump_alt": "pump_power_alt",
    "P102": "pump_power_alt",
    "pump_speed_P102": "pump_power_alt",
    "mixer0_pump_n_in_actuator": "pump_power_alt",
}


def normalize_actuator(name: str) -> Optional[str]:
    """Normalize actuator name to state key."""
    name_lower = name.lower().strip()
    for key, state_key in ACTUATOR_TO_STATE_KEY.items():
        if key.lower() == name_lower:
            return state_key
    return None


def apply_action_plan(state: dict, action_plan: ActionPlan) -> dict:
    """
    Apply action plan to state.
    EXCLUSIVE: First reset ALL actuators to 0, then apply only what's in the plan.
    """
    # Reset all actuators
    for key in [
        "valve_in_B201",
        "valve_in_B202",
        "valve_in_B203",
        "valve_in_B204",
        "valve_out_B201",
        "valve_out_B202",
        "valve_out_B203",
        "valve_out_B204",
    ]:
        state[key] = 0
    state["pump_power"] = 0.0
    state["pump_power_alt"] = 0.0

    # Apply actions from plan
    for action in action_plan.actions:
        state_key = normalize_actuator(action.actuator)
        if state_key:
            if "pump" in state_key:
                state[state_key] = float(action.value)
            else:
                state[state_key] = int(action.value)
            print(f"[ACTUATOR] {action.actuator} → {state_key} = {action.value}")
        else:
            print(f"[ACTUATOR WARNING] Unknown actuator: {action.actuator}")

    return state


# =============================================================================
# ONTOLOGY CONTEXT (T-Box description for all prompt levels)
# =============================================================================

ONTOLOGY_CONTEXT = """\
## KNOWLEDGE GRAPH STRUCTURE

You are working with a Knowledge Graph that describes an industrial system using 
standardized Ontology Design Patterns (ODPs). The data uses these namespaces:

```
PREFIX :        <http://example.org/mixer#>        # Instance namespace (A-Box)
PREFIX UML:     <http://www.hsu-ifa.de/ontologies/UMLStateMachine#>
PREFIX VDI2206: <http://www.w3id.org/hsu-aut/VDI2206#>
PREFIX VDI3682: <http://www.w3id.org/hsu-aut/VDI3682#>
PREFIX CPSMod:  <http://www.hsu-ifa.de/ontologies/CPSMod#>
```

### Behavior Model (UML:StateMachine)

The system behavior is modeled as a **UML:StateMachine** containing:

| Class | Description | Key Properties |
|-------|-------------|----------------|
| `UML:State` | A discrete system state | `rdfs:label`, `UML:doAction` |
| `UML:Transition` | Directed edge between states | `UML:sourceState`, `UML:targetState`, `rdfs:comment` |
| `UML:Event` | Trigger for transition | `rdfs:label` |
| `UML:Action` | Actuator commands for a state | `CPSMod:isChangedByActuator` |

**Transition Properties:**
- `UML:sourceState` → State where transition starts
- `UML:targetState` → State where transition ends  
- `UML:transitionEvent` → Event that triggers transition
- `UML:transitionGuard` → Condition string (e.g., "level >= 0.033 m")
- `rdfs:comment` → **CRITICAL: Contains domain knowledge about WHEN to take this path**

**State Naming Convention** (semantic encoding in instance names):
- `:state_filling_tank_X` → Fill-Operation for tank X (uses inlet valve only, NO pump)
- `:state_emptying_tank_X` → Empty-Operation (uses outlet valve + pump P101)
- `:state_bypass_emptying_tank_X` → Bypass Empty (uses pump P102 instead of P101 to empty tanks B201-B203)

### Structure Model (VDI2206)

Physical components follow VDI2206 hierarchy:

```
VDI2206:System (:MODVA1234)
    └── VDI2206:Module (:mixerModule1234)
            ├── VDI2206:Component (:tank_B201, :tank_B202, :tank_B203, :tank_B204)
            ├── VDI2206:Actuator (valves and pumps)
            └── VDI2206:Sensor (level sensors)
```

**Actuator Instances (VDI2206:Actuator):**
- Inlet valves of tanks b201 to b204
- Outlet valves of all tanks
- Main and bypass pumps 

### Process Model (VDI3682)

- `:mixing1234` a `VDI3682:ProcessOperator`
- `VDI3682:isAssignedTo` → `:mixerModule1234`
- `CPSMod:processRealizesBehavior` → `:MixerStateMachine`

### Tips

** TIP 1: Only use EXISTING UML:State instances**
You MUST only output states that exist as `UML:State` instances in the knowledge graph.
NEVER invent or hallucinate state names. If unsure, stay in current state.

** TIP 2: Read rdfs:comment on Transitions**
The `rdfs:comment` property on `UML:Transition` instances contains domain knowledge
about fault conditions. Always consult these comments
to determine the correct transition when multiple options exist.


"""


# =============================================================================
# VALID STATES (for validation against hallucination)
# =============================================================================

VALID_STATES = [
    # Filling states (no pump needed)
    "state_filling_tank_B201",
    "state_filling_tank_B202",
    "state_filling_tank_B203",
    # Normal emptying states (uses P101)
    "state_emptying_tank_B201",
    "state_emptying_tank_B202",
    "state_emptying_tank_B203",
    # Bypass emptying states (uses P102)
    "state_bypass_emptying_tank_b201",
    "state_bypass_emptying_tank_b202",
    "state_bypass_emptying_tank_b203",
    # Final drain
    "state_emptying_tank_B204",
]

VALID_STATES_STR = """
VALID UML:State INSTANCES (use ONLY these exact names):
  FILLING: state_filling_tank_B201, state_filling_tank_B202, state_filling_tank_B203
  NORMAL EMPTYING: state_emptying_tank_B201, state_emptying_tank_B202, state_emptying_tank_B203
  BYPASS EMPTYING: state_bypass_emptying_tank_b201, state_bypass_emptying_tank_b202, state_bypass_emptying_tank_b203
  FINAL: state_emptying_tank_B204
"""


# =============================================================================
# ABLATION STUDY CONSTANTS
# =============================================================================

# Available fault types
FAULT_TYPES = [
    "pump_failure",
    "pump_degradation",
    "clogging_fault",
    "sensor_fault",
    "leak",
    "normal",
]

# Expected paths for validation
EXPECTED_PATHS = {
    "pump_failure": "bypass",
    "pump_degradation": "bypass",
    "clogging_fault": "bypass",
    "sensor_fault": "normal",
    "leak": "bypass",
    "normal": "normal",
}

# Expected STATE SEQUENCES for validation (7 states total)
EXPECTED_SEQUENCE_NORMAL = [
    "state_filling_tank_B201",
    "state_filling_tank_B202",
    "state_filling_tank_B203",
    "state_emptying_tank_B201",
    "state_emptying_tank_B202",
    "state_emptying_tank_B203",
    "state_emptying_tank_B204",
]

EXPECTED_SEQUENCE_BYPASS = [
    "state_filling_tank_B201",
    "state_filling_tank_B202",
    "state_filling_tank_B203",
    "state_bypass_emptying_tank_b201",
    "state_bypass_emptying_tank_b202",
    "state_bypass_emptying_tank_b203",
    "state_emptying_tank_B204",
]

# Expected actuator configurations per state
EXPECTED_ACTUATORS = {
    "state_filling_tank_B201": {
        "valve_in_B201": 1,
        "valve_in_B202": 0,
        "valve_in_B203": 0,
        "valve_in_B204": 0,
        "valve_out_B201": 0,
        "valve_out_B202": 0,
        "valve_out_B203": 0,
        "valve_out_B204": 0,
        "pump_power": 0.0,
        "pump_power_alt": 0.0,
    },
    "state_filling_tank_B202": {
        "valve_in_B201": 0,
        "valve_in_B202": 1,
        "valve_in_B203": 0,
        "valve_in_B204": 0,
        "valve_out_B201": 0,
        "valve_out_B202": 0,
        "valve_out_B203": 0,
        "valve_out_B204": 0,
        "pump_power": 0.0,
        "pump_power_alt": 0.0,
    },
    "state_filling_tank_B203": {
        "valve_in_B201": 0,
        "valve_in_B202": 0,
        "valve_in_B203": 1,
        "valve_in_B204": 0,
        "valve_out_B201": 0,
        "valve_out_B202": 0,
        "valve_out_B203": 0,
        "valve_out_B204": 0,
        "pump_power": 0.0,
        "pump_power_alt": 0.0,
    },
    # Normal emptying (P101)
    "state_emptying_tank_B201": {
        "valve_in_B201": 0,
        "valve_in_B202": 0,
        "valve_in_B203": 0,
        "valve_in_B204": 1,
        "valve_out_B201": 1,
        "valve_out_B202": 0,
        "valve_out_B203": 0,
        "valve_out_B204": 0,
        "pump_power": 1.0,
        "pump_power_alt": 0.0,
    },
    "state_emptying_tank_B202": {
        "valve_in_B201": 0,
        "valve_in_B202": 0,
        "valve_in_B203": 0,
        "valve_in_B204": 1,
        "valve_out_B201": 0,
        "valve_out_B202": 1,
        "valve_out_B203": 0,
        "valve_out_B204": 0,
        "pump_power": 1.0,
        "pump_power_alt": 0.0,
    },
    "state_emptying_tank_B203": {
        "valve_in_B201": 0,
        "valve_in_B202": 0,
        "valve_in_B203": 0,
        "valve_in_B204": 1,
        "valve_out_B201": 0,
        "valve_out_B202": 0,
        "valve_out_B203": 1,
        "valve_out_B204": 0,
        "pump_power": 1.0,
        "pump_power_alt": 0.0,
    },
    # Bypass emptying (P102)
    "state_bypass_emptying_tank_b201": {
        "valve_in_B201": 0,
        "valve_in_B202": 0,
        "valve_in_B203": 0,
        "valve_in_B204": 1,
        "valve_out_B201": 1,
        "valve_out_B202": 0,
        "valve_out_B203": 0,
        "valve_out_B204": 0,
        "pump_power": 0.0,
        "pump_power_alt": 1.0,
    },
    "state_bypass_emptying_tank_b202": {
        "valve_in_B201": 0,
        "valve_in_B202": 0,
        "valve_in_B203": 0,
        "valve_in_B204": 1,
        "valve_out_B201": 0,
        "valve_out_B202": 1,
        "valve_out_B203": 0,
        "valve_out_B204": 0,
        "pump_power": 0.0,
        "pump_power_alt": 1.0,
    },
    "state_bypass_emptying_tank_b203": {
        "valve_in_B201": 0,
        "valve_in_B202": 0,
        "valve_in_B203": 0,
        "valve_in_B204": 1,
        "valve_out_B201": 0,
        "valve_out_B202": 0,
        "valve_out_B203": 1,
        "valve_out_B204": 0,
        "pump_power": 0.0,
        "pump_power_alt": 1.0,
    },
    # Final drain
    "state_emptying_tank_B204": {
        "valve_in_B201": 0,
        "valve_in_B202": 0,
        "valve_in_B203": 0,
        "valve_in_B204": 0,
        "valve_out_B201": 0,
        "valve_out_B202": 0,
        "valve_out_B203": 0,
        "valve_out_B204": 1,
        "pump_power": 0.0,
        "pump_power_alt": 0.0,
    },
}


# =============================================================================
# IMPROVED PROMPTS WITH FEW-SHOT EXAMPLES
# =============================================================================

# =============================================================================
# PROMPT LEVEL VARIANTS
# =============================================================================

# --- LEVEL: REDUCED (ONTOLOGY_CONTEXT + Valid States + weniger Few-Shots) ---

PLANNING_SYSTEM_PROMPT_NORMAL = (
    ONTOLOGY_CONTEXT
    + """

## PLANNING TASK

Role: You are a planning agent responsible for deciding which state the system should transition to next, in both normal operation and fault scenarios.

You have access to the complete state machine depicting states, transitions between those states, and actuators to be activated in each state. The state machine contains paths for normal operation as well as alternative paths for fault conditions.

Task: Determine the next UML:State based on current state, tank levels, fault condition, and the UML:StateMachine in the Knowledge Graph. Consult rdfs:comment on transitions for domain knowledge about path selection.

"""
    + VALID_STATES_STR
    + """

"""
)

PLANNING_FEW_SHOT_NORMAL = """\
## EXAMPLES

### Example 1: Filling continues regardless of fault
Current: state_filling_tank_B202, Fault: pump_failure, B202 is full
The UML:StateMachine in the Knowledge Graph shows that state_filling_tank_B202 is the UML:sourceState of a transition with UML:targetState state_filling_tank_B203.
→ Next: state_filling_tank_B203
Reasoning: Still in filling phase. Pump faults don't affect this state.

### Example 2: Transition to emptying 
Current: state_filling_tank_B203, Fault: sensor_fault, B203 is full
The UML:StateMachine shows that state_filling_tank_B203 has transitions to multiple target states. Use the domain knowledge about the system, described in rdfs:comment, to decide which transition to take.
→ Next: state_emptying_tank_B201
Reasoning: B203 full, sensor_fault doesn't affect pump operation.
"""

ACTION_SYSTEM_PROMPT_NORMAL = (
    ONTOLOGY_CONTEXT
    + """

## ACTION TASK

Role: You are an action agent responsible for finding which actuators need to be active in the target state provided by the planning agent.

Task: Based on the target state from the planning agent, query the Knowledge Graph to find which actuators are connected to that state via UML:doAction and CPSMod:isChangedByActuator. Output only actuators that are explicitly linked to that target state instance in the ttl file, no other connection!!!
** Actuator Mapping**
Only(!!!) use the actuator (VDI2206) which are connected to the TARGET STATE provided by the planning agent before via the chain UML:State -> UML:doAction -> UML:Action -> CPSMod:isChangedByActuator.!!! Even if you have potentially other ideas, stick with the version in the KG (ttl.file). Thise actuators are always 1 (on), not 0(closed)!

"""
)

ACTION_FEW_SHOT_NORMAL = """\
## EXAMPLES

### Example 1
Target: state_filling_tank_B202
The Knowledge Graph shows that state_filling_tank_B202 has a UML:doAction linked via CPSMod:isChangedByActuator to valve_in_B202. No other actuators are connected to this state.
Actions: [{"actuator": "valve_in_B202", "value": 1}]

"""


# --- LEVEL: MINIMAL (only ONTOLOGY_CONTEXT + Valid States, no examples) ---

PLANNING_SYSTEM_PROMPT_MINIMAL = (
    ONTOLOGY_CONTEXT
    + """

"""
    + VALID_STATES_STR
)

PLANNING_FEW_SHOT_MINIMAL = ""  # No examples

ACTION_SYSTEM_PROMPT_MINIMAL = ONTOLOGY_CONTEXT

ACTION_FEW_SHOT_MINIMAL = ""  # No examples


# =============================================================================
# PROMPT SELECTOR FUNCTION
# =============================================================================


def get_prompts_for_level(level: str) -> dict:
    """
    Get the appropriate prompts for the given difficulty level.
    """
    # Get KG context if available
    planning_kg = get_planning_kg_context() if is_kg_loaded() else ""
    action_kg = get_action_kg_context() if is_kg_loaded() else ""

    if level == "normal":
        return {
            "planning_system": PLANNING_SYSTEM_PROMPT_NORMAL + planning_kg,
            "planning_few_shot": PLANNING_FEW_SHOT_NORMAL,
            "action_system": ACTION_SYSTEM_PROMPT_NORMAL + action_kg,
            "action_few_shot": ACTION_FEW_SHOT_NORMAL,
        }
    elif level == "minimal":
        return {
            "planning_system": PLANNING_SYSTEM_PROMPT_MINIMAL + planning_kg,
            "planning_few_shot": PLANNING_FEW_SHOT_MINIMAL,
            "action_system": ACTION_SYSTEM_PROMPT_MINIMAL + action_kg,
            "action_few_shot": ACTION_FEW_SHOT_MINIMAL,
        }
    else:
        raise ValueError(f"Unknown prompt level: {level}. Choose from: {PROMPT_LEVELS}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _make_llm(model_name: str = None, temperature: float = None):
    """Create LLM instance. Uses global CURRENT_MODEL if model_name not specified."""
    global CURRENT_MODEL
    model = model_name if model_name else CURRENT_MODEL
    print(f"[LLM] Using model: {model}")
    # Auto-select temperature if not specified
    if temperature is None:
        temperature = get_model_temperature(model)
    if model.startswith("gpt"):
        return ChatOpenAI(model=model, temperature=temperature, timeout=60)
    else:
        # For non-GPT models, return a dummy or default LLM instance
        print(f"[LLM] Warning: Non-GPT model '{model}' may not be supported.")
        return ChatOllama(model=model, temperature=temperature, timeout=60)


def is_pump_fault(fault: str) -> bool:
    """Check if fault affects pump P101 path (requires bypass)."""
    pump_faults = [
        "pump_failure",
        "pump_degradation",
        "clogging_fault",
        "clogging",
        "leak",
        "p101",
    ]
    fault_lower = fault.lower()
    return any(pf in fault_lower for pf in pump_faults)


def get_tank_status(state: GraphState) -> str:
    """Get human-readable tank status."""
    return f"""Tank Levels:
- B201: {state['level_B201']:.4f} (max: {state['max_height_B201']}, min: {state['min_height_B201']}) - {'FULL' if state['level_B201'] >= state['max_height_B201'] else 'EMPTY' if state['level_B201'] <= state['min_height_B201'] else 'partial'}
- B202: {state['level_B202']:.4f} (max: {state['max_height_B202']}, min: {state['min_height_B202']}) - {'FULL' if state['level_B202'] >= state['max_height_B202'] else 'EMPTY' if state['level_B202'] <= state['min_height_B202'] else 'partial'}
- B203: {state['level_B203']:.4f} (max: {state['max_height_B203']}, min: {state['min_height_B203']}) - {'FULL' if state['level_B203'] >= state['max_height_B203'] else 'EMPTY' if state['level_B203'] <= state['min_height_B203'] else 'partial'}
- B204: {state['level_B204']:.4f} (max: {state['max_height_B204']}, min: {state['min_height_B204']}) - {'FULL' if state['level_B204'] >= state['max_height_B204'] else 'EMPTY' if state['level_B204'] <= state['min_height_B204'] else 'partial'}"""


def check_transition_condition(state: GraphState) -> bool:
    """Check if current state's exit condition is met."""
    curr = state["curr_state"].lower()

    # Filling states: transition when tank is full
    if "filling" in curr:
        if "b201" in curr:
            return state["level_B201"] >= state["max_height_B201"]
        elif "b202" in curr:
            return state["level_B202"] >= state["max_height_B202"]
        elif "b203" in curr:
            return state["level_B203"] >= state["max_height_B203"]

    # Emptying states (both normal and bypass): transition when tank is empty
    elif "emptying" in curr or "bypass" in curr:
        if "b201" in curr:
            return state["level_B201"] <= state["min_height_B201"]
        elif "b202" in curr:
            return state["level_B202"] <= state["min_height_B202"]
        elif "b203" in curr:
            return state["level_B203"] <= state["min_height_B203"]
        elif "b204" in curr:
            return state["level_B204"] <= state["min_height_B204"]

    return False


def _validate_actuators(state: dict) -> int:
    """Count invalid actuator values."""
    invalid = 0
    for k in [
        "valve_in_B201",
        "valve_in_B202",
        "valve_in_B203",
        "valve_in_B204",
        "valve_out_B201",
        "valve_out_B202",
        "valve_out_B203",
        "valve_out_B204",
    ]:
        if state.get(k, 0) not in (0, 1):
            invalid += 1
    for k in ["pump_power", "pump_power_alt"]:
        v = state.get(k, 0.0)
        try:
            if not (0.0 <= float(v) <= 1.0):
                invalid += 1
        except:
            invalid += 1
    return invalid


# =============================================================================
# ENHANCED EVALUATION FUNCTIONS
# =============================================================================


def get_expected_next_state(curr_state: str, fault: str, state: dict) -> str:
    """
    Calculate the EXPECTED next state based on:
    1. Current state
    2. Fault condition (determines bypass vs normal)
    3. Tank levels (determines if transition should happen)

    Returns the expected next state name.
    """
    curr_lower = curr_state.lower()
    use_bypass = is_pump_fault(fault)

    # STATE MACHINE LOGIC:

    # INITIAL STATE: Always transition to filling B201
    if "initial" in curr_lower or curr_lower == "state_initialstep":
        return "state_filling_tank_B201"

    # Filling phase: B201 → B202 → B203 (independent of faults)
    if "filling" in curr_lower:
        if "b201" in curr_lower:
            if state["level_B201"] >= state["max_height_B201"]:
                return "state_filling_tank_B202"
            else:
                return curr_state  # Stay - not full yet
        elif "b202" in curr_lower:
            if state["level_B202"] >= state["max_height_B202"]:
                return "state_filling_tank_B203"
            else:
                return curr_state  # Stay - not full yet
        elif "b203" in curr_lower:
            if state["level_B203"] >= state["max_height_B203"]:
                # Transition to emptying - path depends on fault!
                if use_bypass:
                    return "state_bypass_emptying_tank_b201"
                else:
                    return "state_emptying_tank_B201"
            else:
                return curr_state  # Stay - not full yet

    # Normal emptying phase: B201 → B202 → B203 → B204
    elif "emptying" in curr_lower and "bypass" not in curr_lower:
        if "b201" in curr_lower:
            if state["level_B201"] <= state["min_height_B201"]:
                return "state_emptying_tank_B202"
            else:
                return curr_state
        elif "b202" in curr_lower:
            if state["level_B202"] <= state["min_height_B202"]:
                return "state_emptying_tank_B203"
            else:
                return curr_state
        elif "b203" in curr_lower:
            if state["level_B203"] <= state["min_height_B203"]:
                return "state_emptying_tank_B204"
            else:
                return curr_state
        elif "b204" in curr_lower:
            return curr_state  # Final state - stay here

    # Bypass emptying phase: b201 → b202 → b203 → B204
    elif "bypass" in curr_lower:
        if "b201" in curr_lower:
            if state["level_B201"] <= state["min_height_B201"]:
                return "state_bypass_emptying_tank_b202"
            else:
                return curr_state
        elif "b202" in curr_lower:
            if state["level_B202"] <= state["min_height_B202"]:
                return "state_bypass_emptying_tank_b203"
            else:
                return curr_state
        elif "b203" in curr_lower:
            if state["level_B203"] <= state["min_height_B203"]:
                return "state_emptying_tank_B204"
            else:
                return curr_state

    # Default: stay in current state
    return curr_state


def check_planning_decision(
    state: GraphState, use_decision_levels: bool = True
) -> dict:
    """
    Check if the LLM made the correct planning decision.

    Args:
        state: Current state
        use_decision_levels: If True, check against levels at decision time (BEFORE sim).
                            If False, check against current levels (AFTER sim).

    Returns:
        dict with:
        - exit_condition_met: bool - should transition have happened?
        - expected_next: str - what should the next state be?
        - llm_chose: str - what did LLM choose?
        - planning_correct: bool - did LLM choose correctly?
        - error_type: str - type of error if any
    """
    curr = state["curr_state"]
    fault = state.get("condition", "normal")

    # Get what LLM chose as next state
    path = state.get("path", [])
    llm_chose = path[1] if len(path) > 1 else curr
    llm_chose_lower = llm_chose.lower()

    # Use decision_levels (what LLM saw) or current levels (after sim)?
    if use_decision_levels and "decision_levels" in state:
        # Create a temporary state dict with the levels at decision time
        check_state = dict(state)
        decision_levels = state["decision_levels"]
        check_state["level_B201"] = decision_levels["level_B201"]
        check_state["level_B202"] = decision_levels["level_B202"]
        check_state["level_B203"] = decision_levels["level_B203"]
        check_state["level_B204"] = decision_levels["level_B204"]

        print(f"[PLANNING CHECK] Using decision_levels (what LLM saw when deciding):")
        print(
            f"  B201={decision_levels['level_B201']:.4f}, B202={decision_levels['level_B202']:.4f}, B203={decision_levels['level_B203']:.4f}, B204={decision_levels['level_B204']:.4f}"
        )
    else:
        check_state = state

    # Calculate expected next state based on levels at decision time
    expected = get_expected_next_state(curr, fault, check_state)
    expected_lower = expected.lower()

    # Check if exit condition was met at decision time
    exit_met = check_transition_condition_with_levels(check_state)

    # Determine if planning was correct
    planning_correct = llm_chose_lower == expected_lower

    # Classify error type
    error_type = None
    if not planning_correct:
        if exit_met and llm_chose_lower == curr.lower():
            error_type = "MISSED_TRANSITION"
        elif not exit_met and llm_chose_lower != curr.lower():
            error_type = "PREMATURE_TRANSITION"
        elif "bypass" in expected_lower and "bypass" not in llm_chose_lower:
            error_type = "WRONG_PATH_SHOULD_BYPASS"
        elif "bypass" not in expected_lower and "bypass" in llm_chose_lower:
            error_type = "WRONG_PATH_UNNECESSARY_BYPASS"
        else:
            error_type = "WRONG_STATE"

    return {
        "exit_condition_met": exit_met,
        "expected_next": expected,
        "llm_chose": llm_chose,
        "planning_correct": planning_correct,
        "error_type": error_type,
    }


def check_transition_condition_with_levels(state: dict) -> bool:
    """
    Check if exit condition is met for current state.
    Uses whatever levels are in the state dict.
    """
    curr = state["curr_state"].lower()

    # INITIAL STATE: Exit condition is always met (always start process)
    if "initial" in curr or curr == "state_initialstep":
        return True

    # Filling states - exit when tank is full
    if "filling" in curr:
        if "b201" in curr:
            return state["level_B201"] >= state["max_height_B201"]
        elif "b202" in curr:
            return state["level_B202"] >= state["max_height_B202"]
        elif "b203" in curr:
            return state["level_B203"] >= state["max_height_B203"]

    # Emptying states - exit when tank is empty
    elif "emptying" in curr or "bypass" in curr:
        if "b201" in curr:
            return state["level_B201"] <= state["min_height_B201"]
        elif "b202" in curr:
            return state["level_B202"] <= state["min_height_B202"]
        elif "b203" in curr:
            return state["level_B203"] <= state["min_height_B203"]
        elif "b204" in curr:
            return state["level_B204"] <= state["min_height_B204"]

    return False


def check_actuators(state: dict, target_state: str) -> dict:
    """
    Check if actuators match expected configuration for target state.

    Returns:
        dict with 'correct', 'total', 'errors', 'accuracy'
    """
    target_lower = target_state.lower()

    # Find matching expected config (case-insensitive)
    expected = None
    for state_name, config in EXPECTED_ACTUATORS.items():
        if state_name.lower() == target_lower:
            expected = config
            break

    if not expected:
        return {"correct": 0, "total": 0, "errors": [], "accuracy": 1.0}

    correct = 0
    total = len(expected)
    errors = []

    for actuator, expected_val in expected.items():
        actual_val = state.get(actuator, 0)

        # For pumps, check if both are 0 or both are >0
        if "pump" in actuator:
            expected_on = expected_val > 0
            actual_on = actual_val > 0
            if expected_on == actual_on:
                correct += 1
            else:
                errors.append(
                    f"{actuator}: expected={expected_val}, actual={actual_val}"
                )
        else:
            # For valves, exact match
            if int(actual_val) == int(expected_val):
                correct += 1
            else:
                errors.append(
                    f"{actuator}: expected={expected_val}, actual={actual_val}"
                )

    accuracy = correct / total if total > 0 else 1.0

    return {
        "correct": correct,
        "total": total,
        "errors": errors,
        "accuracy": accuracy,
    }


# def validate_state_feasibility(state: GraphState, target_state: str) -> dict:
#     """
#     PRE-VALIDATION: Check if the target state is physically feasible.

#     This runs BEFORE simulation to catch invalid LLM decisions early.

#     Rules:
#     - Can't fill a tank that's already full
#     - Can't empty a tank that's already empty
#     - Must transition when exit condition is met

#     Returns:
#         dict with 'feasible', 'reason', 'should_reprompt'
#     """
#     target_lower = target_state.lower()

#     # Define thresholds
#     FULL_THRESHOLD = 0.001  # 1mm tolerance

#     # Check filling states - can't fill if already full
#     if 'filling' in target_lower:
#         if 'b201' in target_lower:
#             if state['level_B201'] >= state['max_height_B201'] - FULL_THRESHOLD:
#                 return {
#                     'feasible': False,
#                     'reason': f"Cannot fill B201: already full (level={state['level_B201']:.4f} >= max={state['max_height_B201']}). MUST transition to state_filling_tank_B202!",
#                     'should_reprompt': True,
#                     'expected_state': 'state_filling_tank_B202',
#                 }
#         elif 'b202' in target_lower:
#             if state['level_B202'] >= state['max_height_B202'] - FULL_THRESHOLD:
#                 return {
#                     'feasible': False,
#                     'reason': f"Cannot fill B202: already full (level={state['level_B202']:.4f} >= max={state['max_height_B202']}). MUST transition to state_filling_tank_B203!",
#                     'should_reprompt': True,
#                     'expected_state': 'state_filling_tank_B203',
#                 }
#         elif 'b203' in target_lower:
#             if state['level_B203'] >= state['max_height_B203'] - FULL_THRESHOLD:
#                 # B203 full = transition to emptying phase
#                 fault = state.get('condition', 'normal')
#                 if is_pump_fault(fault):
#                     expected = 'state_bypass_emptying_tank_b201'
#                 else:
#                     expected = 'state_emptying_tank_B201'
#                 return {
#                     'feasible': False,
#                     'reason': f"Cannot fill B203: already full (level={state['level_B203']:.4f} >= max={state['max_height_B203']}). MUST transition to {expected}!",
#                     'should_reprompt': True,
#                     'expected_state': expected,
#                 }

#     # Check emptying states - can't empty if already empty
#     if 'emptying' in target_lower or 'bypass' in target_lower:
#         if 'b201' in target_lower:
#             if state['level_B201'] <= state['min_height_B201'] + FULL_THRESHOLD:
#                 next_state = 'state_bypass_emptying_tank_b202' if 'bypass' in target_lower else 'state_emptying_tank_B202'
#                 return {
#                     'feasible': False,
#                     'reason': f"Cannot empty B201: already empty (level={state['level_B201']:.4f} <= min={state['min_height_B201']}). MUST transition to {next_state}!",
#                     'should_reprompt': True,
#                     'expected_state': next_state,
#                 }
#         elif 'b202' in target_lower:
#             if state['level_B202'] <= state['min_height_B202'] + FULL_THRESHOLD:
#                 next_state = 'state_bypass_emptying_tank_b203' if 'bypass' in target_lower else 'state_emptying_tank_B203'
#                 return {
#                     'feasible': False,
#                     'reason': f"Cannot empty B202: already empty (level={state['level_B202']:.4f} <= min={state['min_height_B202']}). MUST transition to {next_state}!",
#                     'should_reprompt': True,
#                     'expected_state': next_state,
#                 }
#         elif 'b203' in target_lower:
#             if state['level_B203'] <= state['min_height_B203'] + FULL_THRESHOLD:
#                 return {
#                     'feasible': False,
#                     'reason': f"Cannot empty B203: already empty (level={state['level_B203']:.4f} <= min={state['min_height_B203']}). MUST transition to state_emptying_tank_B204!",
#                     'should_reprompt': True,
#                     'expected_state': 'state_emptying_tank_B204',
#                 }
#         elif 'b204' in target_lower:
#             if state['level_B204'] <= state['min_height_B204'] + FULL_THRESHOLD:
#                 return {
#                     'feasible': False,
#                     'reason': f"B204 is empty - process complete!",
#                     'should_reprompt': False,  # Not an error, just done
#                     'expected_state': 'COMPLETE',
#                 }

#     # State is feasible
#     return {
#         'feasible': True,
#         'reason': None,
#         'should_reprompt': False,
#         'expected_state': target_state,
#     }


def check_physical_outcome(state: GraphState, prev_levels: dict) -> dict:
    """
    Check if the physical outcome is valid/sensible.

    Checks for:
    - Tank overflow (filling beyond max)
    - Negative levels
    """
    curr = state["curr_state"].lower()
    violations = []

    # Define limits with minimal tolerance (0.5mm = 0.0005m)
    # This catches overflow quickly while allowing for floating point imprecision
    OVERFLOW_TOLERANCE = 0.01  # 0.5mm tolerance - much smaller than before!

    # Check each tank
    tanks = [
        (
            "B201",
            state["level_B201"],
            state["max_height_B201"],
            state["min_height_B201"],
        ),
        (
            "B202",
            state["level_B202"],
            state["max_height_B202"],
            state["min_height_B202"],
        ),
        (
            "B203",
            state["level_B203"],
            state["max_height_B203"],
            state["min_height_B203"],
        ),
        (
            "B204",
            state["level_B204"],
            state["max_height_B204"],
            state["min_height_B204"],
        ),
    ]

    for tank_name, level, max_h, min_h in tanks:
        # Check overflow - trigger when level exceeds max (with tiny tolerance for float imprecision)
        if level > max_h + OVERFLOW_TOLERANCE:
            violations.append(f"OVERFLOW: {tank_name} level={level:.4f} > max={max_h}")

        # Check negative/impossible levels
        if level < 0:
            violations.append(f"NEGATIVE_LEVEL: {tank_name} level={level:.4f}")

    return {
        "valid": len(violations) == 0,
        "violations": violations,
        "violation_count": len(violations),
    }


def _print_metrics(state: GraphState):
    """Print final metrics summary."""
    state["m_total_wall"] = time.perf_counter() - state["m_start_wall"]

    p_calls = max(1, state.get("m_llm_planning_calls", 0))
    a_calls = max(1, state.get("m_llm_action_calls", 0))

    # Calculate actuator accuracy
    act_correct = state.get("m_actuator_correct", 0)
    act_total = state.get("m_actuator_total", 0)
    act_accuracy = (act_correct / act_total * 100) if act_total > 0 else 100.0

    # Calculate planning accuracy
    plan_correct = state.get("m_planning_correct", 0)
    plan_total = state.get("m_planning_total", 0)
    plan_accuracy = (plan_correct / plan_total * 100) if plan_total > 0 else 100.0

    # Token usage
    planning_tokens = state.get("m_planning_input_tokens", 0) + state.get(
        "m_planning_output_tokens", 0
    )
    action_tokens = state.get("m_action_input_tokens", 0) + state.get(
        "m_action_output_tokens", 0
    )
    total_tokens = planning_tokens + action_tokens

    print("\n===== METRICS SUMMARY =====")
    print(f"success: {1 if state.get('m_success') else 0}")
    print(f"final_state: {state.get('curr_state')}")
    print(f"iterations: {state.get('itr', 0)}")
    print(f"sim_time_total: {state.get('timer', 0.0):.3f}")
    print(f"reprompts: {state.get('m_reprompt_count', 0)}")
    print(f"actuator_accuracy: {act_correct}/{act_total} ({act_accuracy:.1f}%)")
    print(f"planning_calls: {state.get('m_llm_planning_calls', 0)}")
    print(f"action_calls: {state.get('m_llm_action_calls', 0)}")
    print(
        f"planning_latency_avg_s: {(state.get('m_llm_planning_latency_s', 0.0) / p_calls):.3f}"
    )
    print(
        f"action_latency_avg_s: {(state.get('m_llm_action_latency_s', 0.0) / a_calls):.3f}"
    )
    print(f"wall_time_total_s: {state.get('m_total_wall', 0.0):.3f}")

    print(f"\n--- TOKEN USAGE ---")
    print(
        f"Planning: {state.get('m_planning_input_tokens', 0)} in + {state.get('m_planning_output_tokens', 0)} out = {planning_tokens}"
    )
    print(
        f"Action: {state.get('m_action_input_tokens', 0)} in + {state.get('m_action_output_tokens', 0)} out = {action_tokens}"
    )
    print(f"Total: {total_tokens} tokens")

    # Enhanced evaluation metrics
    print("\n--- ENHANCED EVALUATION ---")
    print(f"planning_accuracy: {plan_correct}/{plan_total} ({plan_accuracy:.1f}%)")
    print(f"missed_transitions: {state.get('m_missed_transitions', 0)}")
    print(f"physical_violations: {state.get('m_physical_violations', 0)}")
    print("===========================\n")


# =============================================================================
# GRAPH NODE FUNCTIONS
# =============================================================================


def initializing(state: GraphState) -> GraphState:
    """Initialize all state variables."""
    state["digital_twin_states"] = {}
    state["level_B201"] = 0.022
    state["level_B202"] = 0.022
    state["level_B203"] = 0.022
    state["level_B204"] = 0.022

    state["tank_B201_state"] = "Empty"
    state["tank_B202_state"] = "Empty"
    state["tank_B203_state"] = "Empty"
    state["tank_B204_state"] = "Empty"

    state["max_height_B201"] = 0.033
    state["min_height_B201"] = 0.022
    state["max_height_B202"] = 0.033
    state["min_height_B202"] = 0.022
    state["max_height_B203"] = 0.033
    state["min_height_B203"] = 0.022
    state["max_height_B204"] = 0.055
    state["min_height_B204"] = 0.022
    state["time"] = 1e6
    state["sim_time"] = 0

    # All actuators off initially
    state["valve_in_B201"] = 0
    state["valve_in_B202"] = 0
    state["valve_in_B203"] = 0
    state["valve_in_B204"] = 0
    state["valve_out_B201"] = 0
    state["valve_out_B202"] = 0
    state["valve_out_B203"] = 0
    state["valve_out_B204"] = 0

    state["pump_main"] = 150
    state["pump_alt"] = 150
    state["pump_power"] = 0
    state["pump_power_alt"] = 0

    state["should_act"] = True
    state["should_reprompt"] = False
    state["timer"] = 0
    state["itr"] = 0

    # Start state - LLM will decide first action
    state["curr_state"] = "state_initialStep"
    state["path"] = []

    # Initialize decision_levels (levels LLM sees when deciding)
    state["decision_levels"] = {
        "level_B201": state["level_B201"],
        "level_B202": state["level_B202"],
        "level_B203": state["level_B203"],
        "level_B204": state["level_B204"],
    }

    # Don't override condition from inputs!
    state["condition"] = state.get("condition", "normal")
    state["simulation_error"] = None

    # Initialize logs
    state["log_time"] = []
    state["log_level_B201"] = []
    state["log_level_B202"] = []
    state["log_level_B203"] = []
    state["log_level_B204"] = []
    state["log_tank_B201_state"] = []
    state["log_tank_B202_state"] = []
    state["log_tank_B203_state"] = []
    state["log_tank_B204_state"] = []
    state["log_valve_in_B201"] = []
    state["log_valve_in_B202"] = []
    state["log_valve_in_B203"] = []
    state["log_valve_in_B204"] = []
    state["log_valve_out_B201"] = []
    state["log_valve_out_B202"] = []
    state["log_valve_out_B203"] = []
    state["log_valve_out_B204"] = []
    state["log_pump_main"] = []
    state["log_pump_alt"] = []
    state["log_pump_power"] = []
    state["log_pump_power_alt"] = []
    state["log_should_act"] = []
    state["log_should_reprompt"] = []
    state["log_curr_state"] = []
    state["log_paths"] = []
    state["log_actions_raw"] = []
    state["log_reprompts"] = []
    state["log_llm_applied_actions"] = []

    # Metrics
    state["m_start_wall"] = time.perf_counter()
    state["m_total_wall"] = 0.0
    state["m_llm_planning_calls"] = 0
    state["m_llm_action_calls"] = 0
    state["m_llm_planning_latency_s"] = 0.0
    state["m_llm_action_latency_s"] = 0.0
    state["m_reprompt_count"] = 0
    state["m_invalid_actuator_count"] = 0
    state["m_step_count"] = 0
    state["m_success"] = False
    state["m_success_itr"] = -1
    state["m_success_sim_time"] = -1.0
    state["m_abort_reason"] = ""  # Will be set when experiment ends
    state["m_cycle_count"] = 0
    state["m_last_state"] = state["curr_state"]

    # Token usage metrics
    state["m_planning_input_tokens"] = 0
    state["m_planning_output_tokens"] = 0
    state["m_action_input_tokens"] = 0
    state["m_action_output_tokens"] = 0

    # Enhanced evaluation metrics
    state["m_planning_correct"] = 0
    state["m_planning_total"] = 0
    state["m_missed_transitions"] = 0
    state["m_physical_violations"] = 0
    state["m_actuator_correct"] = 0
    state["m_actuator_total"] = 0
    state["log_planning_checks"] = []
    state["log_actuator_checks"] = []
    state["log_physical_checks"] = []
    state["log_feasibility_checks"] = []
    state["prev_levels"] = {}

    # Initialize detailed iteration logging
    state["log_iteration_details"] = []
    state["current_iteration_data"] = {}

    return state


def logging_current_state(state: GraphState) -> GraphState:
    """Log all current state variables."""
    state["log_level_B201"].append(state["level_B201"])
    state["log_level_B202"].append(state["level_B202"])
    state["log_level_B203"].append(state["level_B203"])
    state["log_level_B204"].append(state["level_B204"])
    state["log_tank_B201_state"].append(state["tank_B201_state"])
    state["log_tank_B202_state"].append(state["tank_B202_state"])
    state["log_tank_B203_state"].append(state["tank_B203_state"])
    state["log_tank_B204_state"].append(state["tank_B204_state"])
    state["log_valve_in_B201"].append(state["valve_in_B201"])
    state["log_valve_in_B202"].append(state["valve_in_B202"])
    state["log_valve_in_B203"].append(state["valve_in_B203"])
    state["log_valve_in_B204"].append(state["valve_in_B204"])
    state["log_valve_out_B201"].append(state["valve_out_B201"])
    state["log_valve_out_B202"].append(state["valve_out_B202"])
    state["log_valve_out_B203"].append(state["valve_out_B203"])
    state["log_valve_out_B204"].append(state["valve_out_B204"])
    state["log_pump_power"].append(state["pump_power"])
    state["log_pump_power_alt"].append(state["pump_power_alt"])
    state["log_curr_state"].append(state["curr_state"])
    return state


# FAULT MAPPING FOR FAULT LOCALISATION"
FAULT_DISPLAY_NAMES = {
    "pump_failure": "pump_P101_failure",
    "pump_degradation": "pump_P101_degradation",
    "clogging_fault": "clogging_main_pump_line_P101",
    "leak": "leakage_main_pump_line_P101",
    "sensor_fault": "sensor_fault",
    "normal": "normal",
}


def get_fault_display_name(fault: str) -> str:
    """Get descriptive fault name for LLM prompt."""
    return FAULT_DISPLAY_NAMES.get(fault, fault)


# PLANNING Prompt'
def planning(state: GraphState) -> GraphState:
    """
    IMPROVED PLANNING: LLM decides next state based on:
    1. Current state
    2. Tank levels (to know if transition condition is met)
    3. Fault condition (to decide normal vs bypass path)
    """
    global CURRENT_PROMPT_LEVEL, CURRENT_MODEL

    # Track timing start
    planning_start = time.perf_counter()

    curr_state = state["curr_state"]
    fault = state.get("condition", "normal")
    fault_display = get_fault_display_name(fault)  # Fault Localisation
    tank_status = get_tank_status(state)

    # Get prompts based on current level
    prompts = get_prompts_for_level(CURRENT_PROMPT_LEVEL)
    system_msg = prompts["planning_system"]
    if prompts["planning_few_shot"]:
        system_msg += "\n\n" + prompts["planning_few_shot"]

    # Add valid states list to help LLM (especially for reduced prompts)

    # Add error feedback if previous attempt failed
    # NOTE: Guide the LLM to re-analyze the KG context, especially UML:Transition!
    error_context = ""
    if state.get("simulation_error"):
        error_msg = state["simulation_error"]

        # # Build specific guidance based on error type
        # if 'MISSED_TRANSITION' in error_msg or 'exit condition was already met' in error_msg:
        #     # Show current tank levels to make it clear WHY transition is needed

        error_context = f"""
⚠️ PREVIOUS ATTEMPT RESULTED IN ERROR:
{error_msg}

CURRENT TANK LEVELS:
- B201: {state['level_B201']:.4f} (max: {state['max_height_B201']}, min: {state['min_height_B201']})
- B202: {state['level_B202']:.4f} (max: {state['max_height_B202']}, min: {state['min_height_B202']})
- B203: {state['level_B203']:.4f} (max: {state['max_height_B203']}, min: {state['min_height_B203']})
- B204: {state['level_B204']:.4f} (max: {state['max_height_B204']}, min: {state['min_height_B204']})
"""
        # '⛔ YOU MUST NOT CHOOSE '{curr_state}' AGAIN!

        # CHOOSE THE NEXT STATE IN THE SEQUENCE - DO NOT STAY!
        # """
        #         elif 'PREMATURE_TRANSITION' in error_msg or 'exit condition was NOT met' in error_msg:
        #             error_context = f"""
        # ⚠️ CRITICAL ERROR - PREMATURE TRANSITION:
        # {error_msg}

        # The exit condition for the current state has NOT been met yet.
        # You should STAY in '{curr_state}' until the tank reaches its threshold.

        # Re-check the tank levels and thresholds before deciding.
        # """
        #         elif 'WRONG_PATH' in error_msg or 'bypass' in error_msg.lower():
        #             error_context = f"""
        # ⚠️ CRITICAL ERROR - WRONG PATH SELECTED:
        # {error_msg}

        # Re-analyze the fault condition and UML:Transition relations:
        # Check rdfs:comment on transitions to understand which path to take.
        # """
        #         else:
        #             error_context = f"""
        # ⚠️ PREVIOUS ATTEMPT RESULTED IN ERROR:
        # {error_msg}'

        # Please re-evaluate by:
        # 1. Re-reading the UML:Transition relations for the current state
        # 2. Checking rdfs:comment annotations for transition conditions
        # 3. Considering the fault condition and its effect on the path
        # """

        # Clear the error after showing it
        state["simulation_error"] = None

    user_msg = f"""\
CURRENT SITUATION:
- Current State: {curr_state}
- Fault Condition: {fault_display}


{tank_status}
{error_context}
{VALID_STATES_STR}

TASK: Determine the next state to transition to.

Think step by step:
1. What phase am I in? (filling or emptying)
2. Is the current state's exit condition met? (check: is tank FULL for filling, EMPTY for emptying)
3. If exit condition IS met → transition to the NEXT state in the sequence
4. If exit condition is NOT met → stay in current state

Return your decision as JSON with: current_state, next_state, reasoning
"""

    print(f"\n[PLANNING] Current: {curr_state}, Fault: {fault}")

    # DEBUG: Show if KG context was loaded for Planning
    # if is_kg_loaded():
    #     kg_excerpt = _KG_PLANNING_TTL[:600] if _KG_PLANNING_TTL else "(empty)"
    #     # print(f"[PLANNING KG] ✅ KG context loaded ({len(_KG_PLANNING_TTL)} chars)")
    #     # print(f"[PLANNING KG] Excerpt (first 600 chars):\n{kg_excerpt}...")
    # else:
    #     print(f"[PLANNING KG] ⚠️ No KG context available - using static prompts only")

    # llm = _make_llm(temperature=1)  # Uses global CURRENT_MODEL
    # plan_llm = llm.with_structured_output(PathPlan)

    # t0 = time.perf_counter()
    # response = plan_llm.invoke([("system", system_msg), ("user", user_msg)])
    # dt = time.perf_counter() - t0

    # state["m_llm_planning_calls"] += 1
    # state["m_llm_planning_latency_s"] += dt

    llm = _make_llm()  # Uses global CURRENT_MODEL
    plan_llm = llm.with_structured_output(PathPlan, include_raw=True)
    t0 = time.perf_counter()
    raw_response = plan_llm.invoke([("system", system_msg), ("user", user_msg)])
    dt = time.perf_counter() - t0

    response = raw_response["parsed"]

    # Token-Usage
    token_usage = raw_response["raw"].usage_metadata or {}
    input_tokens = token_usage.get("input_tokens", 0)
    output_tokens = token_usage.get("output_tokens", 0)

    state["m_planning_input_tokens"] += input_tokens
    state["m_planning_output_tokens"] += output_tokens
    state["m_llm_planning_calls"] += 1
    state["m_llm_planning_latency_s"] += dt

    print(f"[PLANNING] Tokens: {input_tokens} in, {output_tokens} out")

    # Validate the chosen state
    next_state = response.next_state
    next_state_lower = next_state.lower()

    # Check if state is valid (case-insensitive match)
    valid_match = None
    for valid in VALID_STATES:
        if valid.lower() == next_state_lower:
            valid_match = valid
            break

    if valid_match:
        next_state = valid_match  # Use correct casing
        print(f"[PLANNING] LLM decided: {response.current_state} → {next_state}")
    else:
        # Invalid state! Log error and trigger reprompt
        print(f"[PLANNING] ⚠️ LLM chose INVALID state: {next_state}")
        print(f"[PLANNING] Valid states are: {VALID_STATES}")
        print(f"[PLANNING] Setting error for reprompt...")

        # Set error so reprompt will trigger
        state["simulation_error"] = (
            f"INVALID STATE CHOSEN: '{response.next_state}' is not a valid state. "
            f"Valid states are: {VALID_STATES}. "
            f"Please choose a valid state from this list."
        )

        # Keep current state but mark for reprompt
        next_state = curr_state

        # Track invalid state choices
        if "m_invalid_states" not in state:
            state["m_invalid_states"] = 0
        state["m_invalid_states"] += 1

    print(f"[PLANNING] Reasoning: {response.reasoning}")

    # Store the path
    state["path"] = [curr_state, next_state]

    # Log
    state["log_paths"].append(
        {
            "itr": state.get("itr", 0),
            "src": curr_state,
            "llm_choice": response.next_state,
            "next": next_state,
            "valid": valid_match is not None,
            "fault": fault,
            "reasoning": response.reasoning,
            "latency_s": dt,
        }
    )

    # Track timing end and store planning data for detailed logging
    planning_end = time.perf_counter()
    if "current_iteration_data" in state and state["current_iteration_data"]:
        state["current_iteration_data"]["planning_time_s"] = (
            planning_end - planning_start
        )
        state["current_iteration_data"]["planning_chosen_state"] = next_state
        state["current_iteration_data"]["planning_reasoning"] = (
            response.reasoning[:500] if response.reasoning else ""
        )

        # Calculate expected state for comparison
        expected = get_expected_next_state(curr_state, fault, state)
        state["current_iteration_data"]["planning_expected_state"] = expected
        state["current_iteration_data"]["planning_correct"] = (
            next_state.lower() == expected.lower()
        )

        state["current_iteration_data"]["planning_input_tokens"] = input_tokens
        state["current_iteration_data"]["planning_output_tokens"] = output_tokens

    return state


def action(state: GraphState) -> GraphState:
    """
    IMPROVED ACTION: LLM outputs exact actuator commands for target state.
    """
    global CURRENT_PROMPT_LEVEL, CURRENT_MODEL

    # Track timing start
    action_start = time.perf_counter()

    state["digital_twin_states"] = copy.deepcopy(state)

    curr_state = state["curr_state"]
    target_state = state["path"][1] if len(state["path"]) > 1 else curr_state

    # CRITICAL FIX: Update digital twin to target state for simulation
    state["digital_twin_states"]["curr_state"] = target_state

    fault = state.get("condition", "normal")
    fault_display = get_fault_display_name(fault)  # Fault Localisation

    # Get prompts based on current level
    prompts = get_prompts_for_level(CURRENT_PROMPT_LEVEL)
    system_msg = prompts["action_system"]
    if prompts["action_few_shot"]:
        system_msg += "\n\n" + prompts["action_few_shot"]

    error_context = ""
    if state.get("simulation_error"):
        error_msg = state["simulation_error"]

        # Check if this is an actuator error (routed back from evaluation)
        # if 'ACTUATOR_ERROR' in error_msg or 'actuator' in error_msg.lower():
        #     # Extract the relevant KG snippet for this state to include in feedback
        #     kg_hint = ""
        #     if is_kg_loaded() and _KG_ACTION_TTL:
        #         target_lower = target_state.lower()
        #         ttl_lines = _KG_ACTION_TTL.split('\n')
        #         relevant_lines = []
        #         in_state_block = False
        #         for line in ttl_lines:
        #             if target_lower in line.lower():
        #                 relevant_lines.append(line)
        #                 in_state_block = True
        #             elif in_state_block and line.strip().startswith(':'):
        #                 in_state_block = False
        #             elif in_state_block:
        #                 relevant_lines.append(line)

        #         if relevant_lines:
        #             kg_hint = "\n\nRELEVANT KG DATA for this state:\n```turtle\n" + "\n".join(relevant_lines[:20]) + "\n```"

        error_context = f"""
        
⚠️ PREVIOUS ACTUATOR CONFIGURATION RESULTED IN ERROR:
{error_msg}
 
Please re-evaluate the actuator settings for this state.
"""
        # Clear after showing
        state["simulation_error"] = None

    user_msg = f"""\
TARGET STATE: {target_state}
FAULT CONDITION: {fault_display}
{error_context}

Based on the state name and fault condition, what actuators must be set?

Return JSON with: target_state, actions (list of actuator/value pairs), reasoning
"""

    print(f"\n[ACTION] Target: {target_state}, Fault: {fault}")

    # DEBUG: Show if KG context was loaded for Action
    if is_kg_loaded():
        print(f"[ACTION KG] ✅ KG context loaded ({len(_KG_ACTION_TTL)} chars)")

        # Show how Action Agent derives actuators from KG
        print(f"[ACTION KG] Deriving actuators for '{target_state}' from KG:")
        # print(f"[ACTION KG] 1. Planning Agent decided: '{curr_state}' → '{target_state}'")
        # print(f"[ACTION KG] 2. Looking up UML:doAction in TTL for target state...")

    #     # Extract relevant TTL snippet for this state
    #     target_lower = target_state.lower()
    #     ttl_lines = _KG_ACTION_TTL.split('\n')
    #     relevant_lines = []
    #     in_state_block = False
    #     for line in ttl_lines:
    #         if target_lower in line.lower() or (in_state_block and line.startswith(' ')):
    #             relevant_lines.append(line)
    #             in_state_block = True
    #         elif in_state_block and not line.startswith(' ') and line.strip():
    #             in_state_block = False

    #     if relevant_lines:
    #         print(f"[ACTION KG] Relevant TTL for '{target_state}':")
    #         for line in relevant_lines[:15]:  # Max 15 lines
    #             print(f"[ACTION KG]   {line}")
    #         if len(relevant_lines) > 15:
    #             print(f"[ACTION KG]   ... ({len(relevant_lines)-15} more lines)")
    #     else:
    #         print(f"[ACTION KG] ⚠️ No direct TTL match found - using state name pattern")

    #     print(f"[ACTION KG] 3. LLM will derive actuators from UML:doAction → CPSMod:isChangedByActuator")
    # else:
    #     print(f"[ACTION KG] ⚠️ No KG context available - using static prompts only")

    llm = _make_llm()  # Uses global CURRENT_MODEL
    action_llm = llm.with_structured_output(ActionPlan, include_raw=True)

    t0 = time.perf_counter()
    raw_response = action_llm.invoke([("system", system_msg), ("user", user_msg)])
    dt = time.perf_counter() - t0

    response = raw_response["parsed"]

    # Token-Usage
    token_usage = raw_response["raw"].usage_metadata or {}
    input_tokens = token_usage.get("input_tokens", 0)
    output_tokens = token_usage.get("output_tokens", 0)

    state["m_action_input_tokens"] += input_tokens
    state["m_action_output_tokens"] += output_tokens
    state["m_llm_action_calls"] += 1
    state["m_llm_action_latency_s"] += dt

    print(f"[ACTION] Tokens: {input_tokens} in, {output_tokens} out")

    print(f"[ACTION] LLM response: {len(response.actions)} actuators")
    for a in response.actions:
        print(f"  - {a.actuator} = {a.value}")
    print(f"[ACTION] Reasoning: {response.reasoning}")

    # Track raw LLM output and warnings BEFORE apply_action_plan
    raw_output_str = ", ".join([f"{a.actuator}={a.value}" for a in response.actions])
    warnings_list = []
    for a in response.actions:
        if normalize_actuator(a.actuator) is None:
            warnings_list.append(a.actuator)

    # Apply actions to digital twin state
    state["digital_twin_states"] = apply_action_plan(
        state["digital_twin_states"], response
    )

    # Log
    state["log_actions_raw"].append(
        {
            "itr": state.get("itr", 0),
            "target": target_state,
            "fault": fault,
            "actions": [
                {"actuator": a.actuator, "value": a.value} for a in response.actions
            ],
            "reasoning": response.reasoning,
            "latency_s": dt,
        }
    )

    # Track timing end and store action data for detailed logging
    action_end = time.perf_counter()
    if "current_iteration_data" in state and state["current_iteration_data"]:
        state["current_iteration_data"]["action_time_s"] = action_end - action_start
        state["current_iteration_data"]["action_target_state"] = target_state
        state["current_iteration_data"]["action_raw_output"] = raw_output_str
        state["current_iteration_data"]["action_warnings"] = ", ".join(warnings_list)
        state["current_iteration_data"]["action_reasoning"] = (
            response.reasoning[:500] if response.reasoning else ""
        )
        state["current_iteration_data"]["action_input_tokens"] = input_tokens
        state["current_iteration_data"]["action_output_tokens"] = output_tokens

        # Store IS values (what was actually applied to digital twin)
        twin = state["digital_twin_states"]
        state["current_iteration_data"]["ist_valve_in_B201"] = twin.get(
            "valve_in_B201", 0
        )
        state["current_iteration_data"]["ist_valve_in_B202"] = twin.get(
            "valve_in_B202", 0
        )
        state["current_iteration_data"]["ist_valve_in_B203"] = twin.get(
            "valve_in_B203", 0
        )
        state["current_iteration_data"]["ist_valve_in_B204"] = twin.get(
            "valve_in_B204", 0
        )
        state["current_iteration_data"]["ist_valve_out_B201"] = twin.get(
            "valve_out_B201", 0
        )
        state["current_iteration_data"]["ist_valve_out_B202"] = twin.get(
            "valve_out_B202", 0
        )
        state["current_iteration_data"]["ist_valve_out_B203"] = twin.get(
            "valve_out_B203", 0
        )
        state["current_iteration_data"]["ist_valve_out_B204"] = twin.get(
            "valve_out_B204", 0
        )
        state["current_iteration_data"]["ist_pump_power"] = twin.get("pump_power", 0.0)
        state["current_iteration_data"]["ist_pump_power_alt"] = twin.get(
            "pump_power_alt", 0.0
        )

        # Store SOLL values (expected for the CHOSEN state from planning agent)
        chosen_state = state["current_iteration_data"].get(
            "planning_chosen_state", target_state
        )
        chosen_lower = chosen_state.lower()
        expected_config = None
        for state_name, config in EXPECTED_ACTUATORS.items():
            if state_name.lower() == chosen_lower:
                expected_config = config
                break

        if expected_config:
            state["current_iteration_data"]["soll_valve_in_B201"] = expected_config.get(
                "valve_in_B201", 0
            )
            state["current_iteration_data"]["soll_valve_in_B202"] = expected_config.get(
                "valve_in_B202", 0
            )
            state["current_iteration_data"]["soll_valve_in_B203"] = expected_config.get(
                "valve_in_B203", 0
            )
            state["current_iteration_data"]["soll_valve_in_B204"] = expected_config.get(
                "valve_in_B204", 0
            )
            state["current_iteration_data"]["soll_valve_out_B201"] = (
                expected_config.get("valve_out_B201", 0)
            )
            state["current_iteration_data"]["soll_valve_out_B202"] = (
                expected_config.get("valve_out_B202", 0)
            )
            state["current_iteration_data"]["soll_valve_out_B203"] = (
                expected_config.get("valve_out_B203", 0)
            )
            state["current_iteration_data"]["soll_valve_out_B204"] = (
                expected_config.get("valve_out_B204", 0)
            )
            state["current_iteration_data"]["soll_pump_power"] = expected_config.get(
                "pump_power", 0.0
            )
            state["current_iteration_data"]["soll_pump_power_alt"] = (
                expected_config.get("pump_power_alt", 0.0)
            )

    # Clear the error after processing (so it doesn't persist to next iteration)
    if state.get("simulation_error") and "ACTUATOR_ERROR" in str(
        state.get("simulation_error", "")
    ):
        state["simulation_error"] = None

    return state


def simulation(state: GraphState) -> GraphState:
    """Run digital twin simulation to validate actuator configuration."""
    simulation_start = time.perf_counter()

    print("\n[SIMULATION] Running digital twin...")

    twin = state["digital_twin_states"]
    sim = Simulation(state=twin)

    curr = twin["curr_state"]

    print(f"[SIM] State: {curr}")
    print(
        f"[SIM] Valves IN: [{twin['valve_in_B201']}, {twin['valve_in_B202']}, {twin['valve_in_B203']}, {twin['valve_in_B204']}]"
    )
    print(
        f"[SIM] Valves OUT: [{twin['valve_out_B201']}, {twin['valve_out_B202']}, {twin['valve_out_B203']}, {twin['valve_out_B204']}]"
    )
    print(f"[SIM] Pumps: [P101={twin['pump_power']}, P102={twin['pump_power_alt']}]")

    try:
        # Run appropriate simulation based on state
        if curr == "state_filling_tank_B201":
            sim.state_filling_tank_B201()
            twin["level_B201"] = sim.level_B201
            twin["tank_B201_state"] = sim.tank_B201_state
        elif curr == "state_filling_tank_B202":
            sim.state_filling_tank_B202()
            twin["level_B202"] = sim.level_B202
            twin["tank_B202_state"] = sim.tank_B202_state
        elif curr == "state_filling_tank_B203":
            sim.state_filling_tank_B203()
            twin["level_B203"] = sim.level_B203
            twin["tank_B203_state"] = sim.tank_B203_state
        elif curr in ["state_emptying_tank_B201", "state_bypass_emptying_tank_b201"]:
            sim.state_emptying_tank_B201()
            twin["level_B201"] = sim.level_B201
            twin["tank_B201_state"] = sim.tank_B201_state
            twin["level_B204"] = sim.level_B204
            twin["tank_B204_state"] = sim.tank_B204_state
        elif curr in ["state_emptying_tank_B202", "state_bypass_emptying_tank_b202"]:
            sim.state_emptying_tank_B202()
            twin["level_B202"] = sim.level_B202
            twin["tank_B202_state"] = sim.tank_B202_state
            twin["level_B204"] = sim.level_B204
            twin["tank_B204_state"] = sim.tank_B204_state
        elif curr in ["state_emptying_tank_B203", "state_bypass_emptying_tank_b203"]:
            sim.state_emptying_tank_B203()
            twin["level_B203"] = sim.level_B203
            twin["tank_B203_state"] = sim.tank_B203_state
            twin["level_B204"] = sim.level_B204
            twin["tank_B204_state"] = sim.tank_B204_state
        elif curr == "state_emptying_tank_B204":
            sim.state_emptying_tank_B204()
            twin["level_B204"] = sim.level_B204
            twin["tank_B204_state"] = sim.tank_B204_state

        twin["sim_time"] = sim.time
        state["digital_twin_states"] = twin
        state["simulation_error"] = None

        # Neutral output - evaluation happens in evaluation() node
        print(
            f"[SIM] Completed - Levels: B201={twin['level_B201']:.4f}, B202={twin['level_B202']:.4f}, B203={twin['level_B203']:.4f}, B204={twin['level_B204']:.4f}"
        )

    except ValueError as e:
        error_msg = str(e)
        print(f"[SIM] ❌ ERROR: {error_msg}")
        state["simulation_error"] = error_msg
        state["should_reprompt"] = True

    # Track simulation timing
    simulation_end = time.perf_counter()
    if "current_iteration_data" in state and state["current_iteration_data"]:
        state["current_iteration_data"]["simulation_time_s"] = (
            simulation_end - simulation_start
        )

    return state


def evaluation(state: GraphState) -> GraphState:
    """
    ENHANCED EVALUATION with ROLLBACK on planning errors.

    Key concept: Check LLM's decision against the levels IT SAW (decision_levels),
    not the levels after simulation. If wrong, ROLLBACK to decision_levels.
    """
    evaluation_start = time.perf_counter()

    print("\n" + "=" * 50)
    print("[EVALUATION]")
    print("=" * 50)

    # =========================================================================
    # DEBUG: Print decision_levels immediately
    # =========================================================================
    if "decision_levels" in state:
        dl = state["decision_levels"]
        print(f"[DEBUG] decision_levels from state: B201={dl['level_B201']:.4f}")
    else:
        print(f"[DEBUG] decision_levels NOT SET in state!")
    print(f"[DEBUG] state['level_B201'] = {state['level_B201']:.4f}")

    # =========================================================================
    # DEFENSIVE INITIALIZATION
    # =========================================================================
    state.setdefault("m_planning_correct", 0)
    state.setdefault("m_planning_total", 0)
    state.setdefault("m_missed_transitions", 0)
    state.setdefault("m_physical_violations", 0)
    state.setdefault("m_actuator_correct", 0)
    state.setdefault("m_actuator_total", 0)
    state.setdefault("log_planning_checks", [])
    state.setdefault("log_actuator_checks", [])
    state.setdefault("log_physical_checks", [])

    # Get target state from path
    path = state.get("path", [])
    target_state = path[1] if len(path) > 1 else state["curr_state"]

    # =========================================================================
    # 1. SIMULATION ERROR CHECK (from simulation node)
    # =========================================================================
    if state.get("simulation_error") is not None:
        print(f"[SIMULATION] ❌ ERROR: {state['simulation_error']}")
        # state['should_reprompt'] = True
        path = state.get("path", [])
        target_state = path[1] if len(path) > 1 else state["curr_state"]
        twin = state.get("digital_twin_states", state)
        actuator_check = check_actuators(twin, target_state)

        # Track actuator accuracy even on simulation failure
        state["m_actuator_correct"] += actuator_check["correct"]
        state["m_actuator_total"] += actuator_check["total"]

        if actuator_check.get("errors"):
            # Simulation failed BECAUSE of actuator error!
            print(
                f"[ACTUATOR CHECK] ❌ {actuator_check['correct']}/{actuator_check['total']} correct - THIS CAUSED THE SIMULATION FAILURE"
            )
            for err in actuator_check["errors"][:5]:  # Show first 5 errors
                print(f"  ❌ {err}")

            # Mark as ACTUATOR_ERROR so reprompt goes to action agent
            state["simulation_error"] = (
                f"ACTUATOR_ERROR: Simulation failed because actuator configuration is wrong. "
                f"Your actuator configuration does not match the expected behavior for state '{target_state}'. "
                f"Re-analyze the Knowledge Graph: find the UML:doAction linked to the target state"
                f"then follow CPSMod:isChangedByActuator to find the correct actuators."
                f"even if due to a fault condition you think another actuator combination might be the right one, stick with the version from the ttl and the Knowledge graph, it is always true!!"
            )
        else:
            # Simulation failed but actuators were correct - unusual case
            print(
                f"[ACTUATOR CHECK] ✅ {actuator_check['correct']}/{actuator_check['total']} correct - but simulation still failed"
            )
            # Keep original error message (might be a planning issue)

        state["should_reprompt"] = True
        return state

    # =========================================================================
    # 2. GET BOTH LEVEL SETS
    # =========================================================================
    twin = state["digital_twin_states"]

    # NEW levels (after simulation)
    new_levels = {
        "level_B201": twin["level_B201"],
        "level_B202": twin["level_B202"],
        "level_B203": twin["level_B203"],
        "level_B204": twin["level_B204"],
    }

    # OLD levels (what LLM saw when deciding)
    decision_levels = state.get("decision_levels", new_levels)

    print(
        f"[LEVELS] At decision time: B201={decision_levels['level_B201']:.4f}, B202={decision_levels['level_B202']:.4f}, B203={decision_levels['level_B203']:.4f}, B204={decision_levels['level_B204']:.4f}"
    )
    print(
        f"[LEVELS] After simulation:  B201={new_levels['level_B201']:.4f}, B202={new_levels['level_B202']:.4f}, B203={new_levels['level_B203']:.4f}, B204={new_levels['level_B204']:.4f}"
    )

    # =========================================================================
    # 3. PLANNING CHECK - Was decision correct based on what LLM saw?
    # =========================================================================
    planning_check = check_planning_decision(state, use_decision_levels=True)
    state["m_planning_total"] += 1

    if planning_check["planning_correct"]:
        state["m_planning_correct"] += 1
        print(f"[PLANNING CHECK] ✅ CORRECT (based on levels at decision time)")
        print(f"  Expected: {planning_check['expected_next']}")
        print(f"  LLM chose: {planning_check['llm_chose']}")
    else:
        print(f"[PLANNING CHECK] ❌ INCORRECT (based on levels at decision time)")
        print(f"  Expected: {planning_check['expected_next']}")
        print(f"  LLM chose: {planning_check['llm_chose']}")
        print(f"  Error type: {planning_check['error_type']}")

        if planning_check["error_type"] == "MISSED_TRANSITION":
            state["m_missed_transitions"] += 1

    # Log planning check
    state["log_planning_checks"].append(
        {
            "itr": state.get("itr", 0),
            "curr_state": state["curr_state"],
            "decision_levels": decision_levels,
            "expected": planning_check["expected_next"],
            "llm_chose": planning_check["llm_chose"],
            "correct": planning_check["planning_correct"],
            "error_type": planning_check["error_type"],
            "exit_condition_met": planning_check["exit_condition_met"],
        }
    )

    # =========================================================================
    # 4. ACTUATOR CHECK - ALWAYS performed (regardless of planning correctness)
    #    This ensures actuator accuracy is tracked even when planning fails
    # =========================================================================
    actuator_check = check_actuators(twin, target_state)
    state["m_actuator_correct"] += actuator_check["correct"]
    state["m_actuator_total"] += actuator_check["total"]

    if actuator_check["errors"]:
        print(
            f"[ACTUATOR CHECK] ⚠️  {actuator_check['correct']}/{actuator_check['total']} correct ({actuator_check['accuracy']*100:.0f}%)"
        )
        for err in actuator_check["errors"]:
            print(f"  ❌ {err}")
    else:
        print(
            f"[ACTUATOR CHECK] ✅ {actuator_check['correct']}/{actuator_check['total']} correct"
        )

    # Log actuator check (always, with planning context)
    state["log_actuator_checks"].append(
        {
            "itr": state.get("itr", 0),
            "target_state": target_state,
            "correct": actuator_check["correct"],
            "total": actuator_check["total"],
            "accuracy": actuator_check["accuracy"],
            "errors": actuator_check["errors"],
            "planning_was_correct": planning_check["planning_correct"],
        }
    )

    # =========================================================================
    # 5. DECISION: ROLLBACK or PROCEED?
    # =========================================================================

    if not planning_check["planning_correct"]:
        # =====================================================================
        # ROLLBACK: Decision was wrong! Reset to decision_levels, reprompt.
        # =====================================================================
        print(f"\n[ROLLBACK] Decision was WRONG based on levels at decision time!")
        print(
            f"  Resetting levels to: B201={decision_levels['level_B201']:.4f}, B202={decision_levels['level_B202']:.4f}, ..."
        )
        print(f"  LLM will get another chance with the same starting state.")

        # Restore levels to what they were BEFORE simulation
        state["level_B201"] = decision_levels["level_B201"]
        state["level_B202"] = decision_levels["level_B202"]
        state["level_B203"] = decision_levels["level_B203"]
        state["level_B204"] = decision_levels["level_B204"]

        # Build feedback message - generic base hint + specific error context
        base_hint = (
            "Your state transition decision did not match the expected system behavior. "
            "Re-analyze the Knowledge Graph data."
        )

        error_type = planning_check["error_type"]
        if error_type == "MISSED_TRANSITION":
            reprompt_reason = (
                f"{base_hint} "
                f"MISSED_TRANSITION: You chose to stay in '{state['curr_state']}', but the exit condition was already met. "
                f"Re-analyze the current UML:State and the UML:transitionGuard conditions. "
                f"You need to transition to the UML:targetState connected to the current UML:sourceState."
            )
        elif error_type == "PREMATURE_TRANSITION":
            reprompt_reason = (
                f"{base_hint} "
                f"PREMATURE_TRANSITION: You chose to transition to '{planning_check['llm_chose']}', but the exit condition was NOT met yet. "
                f"Re-analyze the current UML:State and the UML:transitionGuard conditions before transitioning."
            )
        else:  # WRONG_PATH_SHOULD_BYPASS, WRONG_PATH_UNNECESSARY_BYPASS, WRONG_STATE
            reprompt_reason = (
                f"{base_hint} "
                f"WRONG_STATE: Incorrect state transition chosen. "
                f"Re-analyze the state machine logic: check the current UML:State (UML:sourceState), "
                f"the possible UML:targetStates, and the UML:transitionGuard conditions. "
                f"Read the rdfs:comment on EACH transition - it contains OPERATIONAL REQUIREMENTS "
                f"that specify when each path should be used."
                f"Choose the correct transition with regard to the fault condition that needs to be addressed."
            )

        state["should_reprompt"] = True
        state["simulation_error"] = reprompt_reason

        print(f"\n[EVALUATION] ❌ REPROMPT TRIGGERED!")
        print(f"  Reason: {reprompt_reason}")

    else:
        # =====================================================================
        # PROCEED: Decision was correct! Apply new levels.
        # =====================================================================

        # Copy new levels from simulation
        state["level_B201"] = new_levels["level_B201"]
        state["level_B202"] = new_levels["level_B202"]
        state["level_B203"] = new_levels["level_B203"]
        state["level_B204"] = new_levels["level_B204"]

        # Copy other state from twin
        state["tank_B201_state"] = twin["tank_B201_state"]
        state["tank_B202_state"] = twin["tank_B202_state"]
        state["tank_B203_state"] = twin["tank_B203_state"]
        state["tank_B204_state"] = twin["tank_B204_state"]
        state["valve_in_B201"] = twin["valve_in_B201"]
        state["valve_in_B202"] = twin["valve_in_B202"]
        state["valve_in_B203"] = twin["valve_in_B203"]
        state["valve_in_B204"] = twin["valve_in_B204"]
        state["valve_out_B201"] = twin["valve_out_B201"]
        state["valve_out_B202"] = twin["valve_out_B202"]
        state["valve_out_B203"] = twin["valve_out_B203"]
        state["valve_out_B204"] = twin["valve_out_B204"]
        state["pump_power"] = twin["pump_power"]
        state["pump_power_alt"] = twin["pump_power_alt"]
        state["sim_time"] = twin["sim_time"]

        # Check for physical violations with NEW levels
        physical_check = check_physical_outcome(state, {})

        if not physical_check["valid"]:
            state["m_physical_violations"] += physical_check["violation_count"]
            print(
                f"[PHYSICAL CHECK] ❌ {physical_check['violation_count']} violations:"
            )
            for v in physical_check["violations"]:
                print(f"  ⚠️  {v}")
        else:
            print(f"[PHYSICAL CHECK] ✅ OK")

        # Note: Actuator check already done above (before if/else branch)

        # Check if we should reprompt due to physical violations or actuator errors
        should_reprompt = False
        reprompt_reason = None

        if not physical_check["valid"]:
            # This shouldn't happen if planning was correct, but catch it anyway
            should_reprompt = True
            reprompt_reason = f"PHYSICAL VIOLATION after correct planning: {physical_check['violations']}"
        elif actuator_check.get("errors"):
            # STRICT MODE: ANY actuator error triggers reprompt
            should_reprompt = True
            # error_details = '; '.join(actuator_check['errors'])
            reprompt_reason = (
                f"ACTUATOR_ERROR: Simulation failed because actuator configuration is wrong. "
                f"Your actuator configuration does not match the expected behavior for state '{target_state}'. "
                f"Re-analyze the Knowledge Graph: find the UML:doAction linked to the target state"
                f"then follow CPSMod:isChangedByActuator to find the correct actuators."
                f"Even if due to a fault condition you think another actuator combination might be the right one, stick with the version from the ttl and the Knowledge graph, it is always true!!"
            )

        if should_reprompt:
            state["should_reprompt"] = True
            state["simulation_error"] = reprompt_reason
            print(f"\n[EVALUATION] ❌ REPROMPT TRIGGERED!")
            print(f"  Reason: {reprompt_reason}")
        else:
            state["should_reprompt"] = False

            # UPDATE STATE based on LLM's decision
            if target_state != state["curr_state"]:
                print(f"\n[STATE TRANSITION] {state['curr_state']} → {target_state}")
                state["curr_state"] = target_state

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n[EVALUATION SUMMARY]")
    print(
        f"  Planning: {'✅' if planning_check['planning_correct'] else '❌ ' + str(planning_check['error_type'])}"
    )
    print(
        f"  Actuator: {'✅' if not actuator_check.get('errors') else '❌'} {actuator_check['correct']}/{actuator_check['total']} ({actuator_check['accuracy']*100:.0f}%)"
    )
    print(f"  Reprompt: {'YES' if state.get('should_reprompt') else 'NO'}")
    print("=" * 50 + "\n")

    # =========================================================================
    # FINALIZE AND STORE ITERATION DATA
    # =========================================================================
    evaluation_end = time.perf_counter()
    if "current_iteration_data" in state and state["current_iteration_data"]:
        itr_data = state["current_iteration_data"]
        itr_data["evaluation_time_s"] = evaluation_end - evaluation_start

        # Calculate actuator accuracy for this iteration
        twin = state.get("digital_twin_states", state)
        chosen_state = itr_data.get("planning_chosen_state", "")
        if chosen_state:
            act_check = check_actuators(twin, chosen_state)
            itr_data["actuators_correct"] = act_check["correct"]
            itr_data["actuators_total"] = act_check["total"]
            itr_data["actuators_accuracy"] = act_check["accuracy"]

        # Calculate total iteration time
        itr_start = itr_data.get("iteration_start", evaluation_end)
        itr_data["iteration_total_time_s"] = evaluation_end - itr_start

        # Mark if reprompted
        itr_data["was_reprompted"] = state.get("should_reprompt", False)
        if state.get("should_reprompt"):
            itr_data["reprompt_reason"] = (
                state.get("simulation_error", "")[:200]
                if state.get("simulation_error")
                else ""
            )

        # Store in log (make a copy to avoid reference issues)
        state["log_iteration_details"].append(dict(itr_data))

    return state


def reprompting(state: GraphState) -> GraphState:
    """Handle reprompting after failed simulation."""
    reprompt_start = time.perf_counter()

    print("\n[REPROMPT] Previous action failed, retrying...")
    state["m_reprompt_count"] += 1
    state["log_reprompts"].append(
        {
            "itr": state.get("itr", 0),
            "curr_state": state.get("curr_state"),
            "error": state.get("simulation_error"),
        }
    )

    # Track reprompt timing
    reprompt_end = time.perf_counter()
    if "current_iteration_data" in state and state["current_iteration_data"]:
        state["current_iteration_data"]["reprompt_time_s"] += (
            reprompt_end - reprompt_start
        )

    return state


def plant(state: GraphState) -> GraphState:
    """
    SIMPLIFIED PLANT: Only runs physics, NO state transitions.
    State transitions are now handled by the LLM in evaluation().
    """
    print("\n[PLANT] Running physics simulation...")

    sim = Simulation(state=state)
    curr = state["curr_state"]

    # Run physics based on current state
    if curr == "state_filling_tank_B201":
        sim.state_filling_tank_B201()
        state["level_B201"] = sim.level_B201
        state["tank_B201_state"] = sim.tank_B201_state
    elif curr == "state_filling_tank_B202":
        sim.state_filling_tank_B202()
        state["level_B202"] = sim.level_B202
        state["tank_B202_state"] = sim.tank_B202_state
    elif curr == "state_filling_tank_B203":
        sim.state_filling_tank_B203()
        state["level_B203"] = sim.level_B203
        state["tank_B203_state"] = sim.tank_B203_state
    elif curr in ["state_emptying_tank_B201", "state_bypass_emptying_tank_b201"]:
        sim.state_emptying_tank_B201()
        state["level_B201"] = sim.level_B201
        state["tank_B201_state"] = sim.tank_B201_state
        state["level_B204"] = sim.level_B204
        state["tank_B204_state"] = sim.tank_B204_state
    elif curr in ["state_emptying_tank_B202", "state_bypass_emptying_tank_b202"]:
        sim.state_emptying_tank_B202()
        state["level_B202"] = sim.level_B202
        state["tank_B202_state"] = sim.tank_B202_state
        state["level_B204"] = sim.level_B204
        state["tank_B204_state"] = sim.tank_B204_state
    elif curr in ["state_emptying_tank_B203", "state_bypass_emptying_tank_b203"]:
        sim.state_emptying_tank_B203()
        state["level_B203"] = sim.level_B203
        state["tank_B203_state"] = sim.tank_B203_state
        state["level_B204"] = sim.level_B204
        state["tank_B204_state"] = sim.tank_B204_state
    elif curr == "state_emptying_tank_B204":
        sim.state_emptying_tank_B204()
        state["level_B204"] = sim.level_B204
        state["tank_B204_state"] = sim.tank_B204_state

    state["sim_time"] = sim.time

    print(
        f"[PLANT] Levels: B201={state['level_B201']:.4f}, B202={state['level_B202']:.4f}, B203={state['level_B203']:.4f}, B204={state['level_B204']:.4f}"
    )

    return state


def monitoring(state: GraphState) -> GraphState:
    """Monitor system and update metrics."""
    # Start timing for this iteration
    iteration_start = time.perf_counter()

    print(f"\n{'='*60}")
    print(f"[MONITORING] Iteration {state['itr'] + 1}")
    print(f"{'='*60}")

    state["itr"] += 1
    state["timer"] += state["sim_time"]
    state["log_time"].append(state["timer"])

    logging_current_state(state)

    print(f"Current State: {state['curr_state']}")
    print(f"Fault: {state['condition']}")
    print(
        f"Levels: B201={state['level_B201']:.4f}, B202={state['level_B202']:.4f}, B203={state['level_B203']:.4f}, B204={state['level_B204']:.4f}"
    )
    print(f"Time: {state['timer']:.3f}")
    print(f"Iteration: {state['itr']}")

    # ==========================================================================
    # SAVE DECISION LEVELS: These are the levels the LLM sees when deciding!
    # Used later in evaluation() to check if decision was correct.
    # ==========================================================================
    state["decision_levels"] = {
        "level_B201": state["level_B201"],
        "level_B202": state["level_B202"],
        "level_B203": state["level_B203"],
        "level_B204": state["level_B204"],
    }
    print(
        f"[DEBUG] Saved decision_levels: B201={state['decision_levels']['level_B201']:.4f}"
    )

    # Update metrics
    state["m_step_count"] += 1
    state["m_invalid_actuator_count"] += _validate_actuators(state)

    # Check for success
    if (
        state["curr_state"].lower() == "state_emptying_tank_b204"
        and not state["m_success"]
    ):
        state["m_success"] = True
        state["m_success_itr"] = state["itr"]
        state["m_success_sim_time"] = state["timer"]

    state["m_last_state"] = state["curr_state"]
    state["should_act"] = True

    # Initialize current iteration data for detailed logging
    monitoring_end = time.perf_counter()
    state["current_iteration_data"] = {
        # Metadata (to be filled by run_single_experiment)
        "iteration": state["itr"],
        # Timing
        "iteration_start": iteration_start,
        "monitoring_time_s": monitoring_end - iteration_start,
        "planning_time_s": 0.0,
        "action_time_s": 0.0,
        "simulation_time_s": 0.0,
        "evaluation_time_s": 0.0,
        "reprompt_time_s": 0.0,
        "iteration_total_time_s": 0.0,
        # Planning Agent outputs
        "planning_curr_state": state["curr_state"],
        "planning_chosen_state": "",
        "planning_expected_state": "",
        "planning_correct": False,
        "planning_reasoning": "",
        # Action Agent outputs
        "action_target_state": "",
        "action_raw_output": "",  # Raw LLM output as string
        "action_warnings": "",  # Unknown actuator warnings
        "action_reasoning": "",
        # Actuator ToBe values (expected for chosen state)
        "soll_valve_in_B201": 0,
        "soll_valve_in_B202": 0,
        "soll_valve_in_B203": 0,
        "soll_valve_in_B204": 0,
        "soll_valve_out_B201": 0,
        "soll_valve_out_B202": 0,
        "soll_valve_out_B203": 0,
        "soll_valve_out_B204": 0,
        "soll_pump_power": 0.0,
        "soll_pump_power_alt": 0.0,
        # Actuator IS values (what was actually applied after mapping)
        "ist_valve_in_B201": 0,
        "ist_valve_in_B202": 0,
        "ist_valve_in_B203": 0,
        "ist_valve_in_B204": 0,
        "ist_valve_out_B201": 0,
        "ist_valve_out_B202": 0,
        "ist_valve_out_B203": 0,
        "ist_valve_out_B204": 0,
        "ist_pump_power": 0.0,
        "ist_pump_power_alt": 0.0,
        # Actuator accuracy
        "actuators_correct": 0,
        "actuators_total": 10,
        "actuators_accuracy": 0.0,
        # Reprompt info
        "was_reprompted": False,
        "reprompt_reason": "",
        # Token usage
        "planning_input_tokens": 0,
        "planning_output_tokens": 0,
        "action_input_tokens": 0,
        "action_output_tokens": 0,
    }

    return state


# =============================================================================
# CONDITIONAL EDGE FUNCTIONS
# =============================================================================


def should_reprompt(state: GraphState):
    """Decide whether to reprompt, proceed to plant, or END (on reprompt limit)."""
    MAX_REPROMPTS_PER_ITERATION = 5

    if state["should_reprompt"]:
        current_itr = state.get("itr", 0)
        reprompts_this_itr = len(
            [r for r in state.get("log_reprompts", []) if r.get("itr") == current_itr]
        )

        if reprompts_this_itr >= MAX_REPROMPTS_PER_ITERATION:
            print(
                f"\n❌ [REPROMPT LIMIT] Max {MAX_REPROMPTS_PER_ITERATION} reprompts reached for iteration {current_itr}"
            )
            print(f"   Aborting experiment - marking as FAILED")

            # *** NEU: Letzte Iteration-Daten speichern bevor END ***
            if "current_iteration_data" in state and state["current_iteration_data"]:
                itr_data = state["current_iteration_data"]
                itr_data["was_reprompted"] = True
                itr_data["reprompt_reason"] = (
                    f"REPROMPT_LIMIT_REACHED: {reprompts_this_itr} attempts"
                )
                itr_data["aborted"] = True
                state["log_iteration_details"].append(dict(itr_data))

            state["should_reprompt"] = False
            state["m_success"] = False
            state["m_abort_reason"] = "reprompt_limit"
            _print_metrics(state)
            return END

        return "reprompting"
    return "plant"


def reprompt_target(state: GraphState):
    """
    Decide where to route the reprompt: planning or action?

    - Planning errors (wrong state choice) → go to planning
    - Action errors (wrong actuators) → go to action
    - Simulation errors (cannot empty/fill) → go to action (actuator problem!)
    """
    error = state.get("simulation_error", "") or ""
    error_lower = error.lower().replace("_", " ")

    # ACTUATOR ERRORS → Action Agent
    # This includes simulation failures caused by wrong actuator config
    if "actuator_error" in error_lower or "actuator error" in error_lower:
        print(f"[REPROMPT] Routing to ACTION (actuator configuration error)")
        return "action"

    # SIMULATION FAILURES are typically actuator errors!
    # "cannot empty" = pump not on or wrong valve
    # "cannot fill" = inlet valve not open
    if "cannot empty" in error_lower or "cannot fill" in error_lower:
        print(
            f"[REPROMPT] Routing to ACTION (simulation failed - likely actuator config error)"
        )
        return "action"

    # PLANNING ERRORS → Planning Agent
    if any(
        keyword in error_lower
        for keyword in [
            "planning error",
            "missed transition",
            "premature transition",
            "wrong path",
            "already full",
            "already empty",
            "physical",
            "overflow",
            "invalid state",
            "not a valid state",
            "exit condition",
            "re-analyze the state machine",
        ]
    ):
        print(f"[REPROMPT] Routing to PLANNING (state decision error)")
        return "planning"

    # Default: assume action error (more common case)
    print(f"[REPROMPT] Routing to ACTION (default)")
    return "action"


def check_transition_conditions(state: GraphState) -> bool:
    """
    Check if the current state's exit conditions are met.

    Returns True if a transition should occur (tank full/empty).
    This determines whether to ask the LLM for the next state.
    """
    curr = state["curr_state"].lower()

    # INITIAL STATE: Always transition to start the process
    if "initial" in curr or curr == "state_initialstep":
        return True

    # Filling states - exit when tank is full
    if curr == "state_filling_tank_b201":
        return state["level_B201"] >= state["max_height_B201"]
    elif curr == "state_filling_tank_b202":
        return state["level_B202"] >= state["max_height_B202"]
    elif curr == "state_filling_tank_b203":
        return state["level_B203"] >= state["max_height_B203"]

    # Emptying states - exit when tank is empty
    elif curr in ["state_emptying_tank_b201", "state_bypass_emptying_tank_b201"]:
        return state["level_B201"] <= state["min_height_B201"]
    elif curr in ["state_emptying_tank_b202", "state_bypass_emptying_tank_b202"]:
        return state["level_B202"] <= state["min_height_B202"]
    elif curr in ["state_emptying_tank_b203", "state_bypass_emptying_tank_b203"]:
        return state["level_B203"] <= state["min_height_B203"]
    elif curr == "state_emptying_tank_b204":
        return state["level_B204"] <= state["min_height_B204"]

    return False


def should_act(state: GraphState):
    """Decide whether to plan, continue, or end."""
    curr = state["curr_state"]
    curr_lower = curr.lower()

    print(f"Iteration: {state['itr']}")

    # End condition: B204 is empty
    if (
        curr_lower == "state_emptying_tank_b204"
        and state["level_B204"] <= state["min_height_B204"]
    ):
        print("\n🏁 [END] Process complete - B204 is empty ✅")
        state["m_abort_reason"] = "success"
        _print_metrics(state)
        return END

    # Safety limit
    if state["itr"] > 30:
        print("\n⚠️  [END] Max iterations reached ❌")
        state["m_abort_reason"] = "max_iterations"
        _print_metrics(state)
        return END

    # Check if transition conditions are met - if so, need to plan next state
    if check_transition_conditions(state):
        print(f"[TRANSITION CHECK] Conditions met for {curr}, planning next state...")
        return "planning"

    # If should_act flag is set (from monitoring), plan next action
    if state.get("should_act", False):
        return "planning"

    # Otherwise just run physics (no LLM needed)
    return "plant"


# =============================================================================
# GRAPH BUILDER
# =============================================================================


def build_graph():
    """Build and compile the state graph."""
    builder = StateGraph(GraphState)

    # Add nodes
    builder.add_node("initializing", initializing)
    builder.add_node("monitoring", monitoring)
    builder.add_node("planning", planning)
    builder.add_node("action", action)
    builder.add_node("simulation", simulation)
    builder.add_node("evaluation", evaluation)
    builder.add_node("reprompting", reprompting)
    builder.add_node("plant", plant)

    # Add edges
    builder.add_edge(START, "initializing")
    builder.add_edge("initializing", "monitoring")
    builder.add_edge("planning", "action")
    builder.add_edge("action", "simulation")
    builder.add_edge("simulation", "evaluation")
    builder.add_edge("plant", "monitoring")

    # Conditional edges
    builder.add_conditional_edges(
        "evaluation", should_reprompt, ["reprompting", "plant", END]
    )
    builder.add_conditional_edges(
        "reprompting", reprompt_target, ["planning", "action"]
    )
    builder.add_conditional_edges("monitoring", should_act, ["planning", END])

    return builder.compile()


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_sequence(actual_sequence: List[str], fault: str) -> dict:
    """
    Validate if the state sequence is correct for the given fault.
    """
    # Determine expected sequence based on fault
    expected_bypass = EXPECTED_PATHS.get(fault, "normal") == "bypass"
    expected = EXPECTED_SEQUENCE_BYPASS if expected_bypass else EXPECTED_SEQUENCE_NORMAL

    # Remove consecutive duplicates (staying in same state)
    # Also filter out state_initialStep as it's not part of the actual process
    unique_states = []
    for s in actual_sequence:
        s_lower = s.lower()
        # Skip initial step - it's not part of the process sequence
        if "initial" in s_lower:
            continue
        if not unique_states or s_lower != unique_states[-1]:
            unique_states.append(s_lower)

    # Normalize expected to lowercase for comparison
    expected_lower = [s.lower() for s in expected]

    # Check exact match
    sequence_correct = unique_states == expected_lower

    # Check individual phases
    filling_states = unique_states[:3] if len(unique_states) >= 3 else unique_states
    expected_filling = expected_lower[:3]
    filling_correct = filling_states == expected_filling

    emptying_states = unique_states[3:] if len(unique_states) > 3 else []
    expected_emptying = expected_lower[3:]
    emptying_correct = emptying_states == expected_emptying

    # Find missing and extra states
    missing_states = set(expected_lower) - set(unique_states)
    extra_states = set(unique_states) - set(expected_lower)

    # Calculate similarity score (how many states match in order)
    match_count = 0
    for i, exp_state in enumerate(expected_lower):
        if i < len(unique_states) and unique_states[i] == exp_state:
            match_count += 1
    sequence_similarity = match_count / len(expected_lower) if expected_lower else 0.0

    return {
        "sequence_correct": sequence_correct,
        "filling_correct": filling_correct,
        "emptying_correct": emptying_correct,
        "sequence_similarity": sequence_similarity,
        "expected": expected,
        "actual": unique_states,
        "expected_count": len(expected),
        "actual_count": len(unique_states),
        "missing_states": list(missing_states),
        "extra_states": list(extra_states),
        "skipped_states": len(missing_states),
    }


# =============================================================================
# SINGLE EXPERIMENT
# =============================================================================


def run_single_experiment(graph, fault: str, verbose: bool = True) -> dict:
    """Run a single experiment with given fault condition."""
    global CURRENT_MODEL, CURRENT_PROMPT_LEVEL

    if verbose:
        print("\n" + "=" * 60)
        print(f"RUNNING EXPERIMENT: {fault}")
        print(f"Model: {CURRENT_MODEL}, Prompt Level: {CURRENT_PROMPT_LEVEL}")
        print("=" * 60 + "\n")

    inputs: GraphState = {
        "condition": fault,
    }

    # Measure wall time for the entire experiment
    start_time = time.perf_counter()
    result = graph.invoke(input=inputs, config={"recursion_limit": 500})
    wall_time = time.perf_counter() - start_time

    # Determine abort_reason from result (conditional edges don't persist state changes!)
    final_state = result.get("curr_state", "").lower()
    iterations = result.get("itr", 0)
    level_b204 = result.get("level_B204", 1.0)
    min_b204 = result.get("min_height_B204", 0.022)

    # Check termination condition
    if "b204" in final_state and level_b204 <= min_b204:
        abort_reason = "success"
    elif iterations > 30:
        abort_reason = "max_iterations"
    else:
        # Check if we hit reprompt limit (indicated by not reaching end state)
        # and having reprompts
        reprompts = result.get("m_reprompt_count", 0)
        if reprompts > 0 and "b204" not in final_state:
            abort_reason = "reprompt_limit"
        else:
            abort_reason = "unknown"

    # Extract state sequence and validate
    state_sequence = result.get("log_curr_state", [])
    seq_validation = validate_sequence(state_sequence, fault)

    # Extract metrics
    metrics = {
        "model": CURRENT_MODEL,
        "prompt_level": CURRENT_PROMPT_LEVEL,
        "fault": fault,
        "final_state": result.get("curr_state", ""),
        "iterations": result.get("itr", 0),
        "sim_time": result.get("timer", 0.0),
        "reprompts": result.get("m_reprompt_count", 0),
        "invalid_states": result.get("m_invalid_states", 0),
        "planning_calls": result.get("m_llm_planning_calls", 0),
        "action_calls": result.get("m_llm_action_calls", 0),
        "planning_latency_total": result.get("m_llm_planning_latency_s", 0.0),
        "action_latency_total": result.get("m_llm_action_latency_s", 0.0),
        "wall_time": wall_time,  # Measured in run_single_experiment
        "state_sequence": state_sequence,
        # Actuator accuracy metrics
        "actuator_correct": result.get("m_actuator_correct", 0),
        "actuator_total": result.get("m_actuator_total", 0),
        "actuator_checks": result.get("log_actuator_checks", []),
        # Planning accuracy metrics
        "planning_correct": result.get("m_planning_correct", 0),
        "planning_total": result.get("m_planning_total", 0),
        "missed_transitions": result.get("m_missed_transitions", 0),
        "physical_violations": result.get("m_physical_violations", 0),
        # Sequence validation metrics
        "sequence_correct": 1 if seq_validation["sequence_correct"] else 0,
        "filling_correct": 1 if seq_validation["filling_correct"] else 0,
        "emptying_correct": 1 if seq_validation["emptying_correct"] else 0,
        "sequence_similarity": seq_validation["sequence_similarity"],
        "skipped_states": seq_validation["skipped_states"],
        "missing_states": seq_validation["missing_states"],
        "extra_states": seq_validation["extra_states"],
        "expected_path": EXPECTED_PATHS.get(fault, "unknown"),
        "abort_reason": abort_reason,  # Computed from result state
    }

    # First-Try-Rate for planning agent
    iteration_details = result.get("log_iteration_details", [])
    unique_iterations = set()
    first_try_correct_planning = 0
    first_try_correct_action = 0

    for detail in iteration_details:
        itr = detail.get("iteration", detail.get("itr", 0))
        if itr not in unique_iterations:
            unique_iterations.add(itr)
            # First try for Iteration
            if detail.get("planning_correct", 0) == 1:
                first_try_correct_planning += 1
            if detail.get("actuators_accuracy", 0) == 1.0:
                first_try_correct_action += 1

    transitions_attempted = len(unique_iterations)
    metrics["transitions_attempted"] = transitions_attempted
    metrics["first_try_rate_planning"] = first_try_correct_planning / max(
        1, transitions_attempted
    )
    metrics["first_try_rate_action"] = first_try_correct_action / max(
        1, transitions_attempted
    )

    # Attempts per each transition (efficiency)
    metrics["attempts_per_transition"] = metrics["planning_calls"] / max(
        1, transitions_attempted
    )

    # Process Completion
    expected_transitions = 7  # Fixend mixing module
    metrics["process_completion"] = transitions_attempted / expected_transitions

    # Calculate averages
    p_calls = max(1, metrics["planning_calls"])
    a_calls = max(1, metrics["action_calls"])
    metrics["planning_latency_avg"] = metrics["planning_latency_total"] / p_calls
    metrics["action_latency_avg"] = metrics["action_latency_total"] / a_calls

    # Calculate accuracies
    if metrics["actuator_total"] > 0:
        metrics["actuator_accuracy"] = (
            metrics["actuator_correct"] / metrics["actuator_total"]
        )
    else:
        metrics["actuator_accuracy"] = 1.0

    if metrics["planning_total"] > 0:
        metrics["planning_accuracy"] = (
            metrics["planning_correct"] / metrics["planning_total"]
        )
    else:
        metrics["planning_accuracy"] = 1.0

    # Token metrics
    metrics["planning_input_tokens"] = result.get("m_planning_input_tokens", 0)
    metrics["planning_output_tokens"] = result.get("m_planning_output_tokens", 0)
    metrics["action_input_tokens"] = result.get("m_action_input_tokens", 0)
    metrics["action_output_tokens"] = result.get("m_action_output_tokens", 0)
    metrics["total_input_tokens"] = (
        metrics["planning_input_tokens"] + metrics["action_input_tokens"]
    )
    metrics["total_output_tokens"] = (
        metrics["planning_output_tokens"] + metrics["action_output_tokens"]
    )
    metrics["total_tokens"] = (
        metrics["total_input_tokens"] + metrics["total_output_tokens"]
    )

    # Check if correct path was taken (bypass vs normal)
    took_bypass = any("bypass" in s.lower() for s in state_sequence)
    expected_bypass = EXPECTED_PATHS.get(fault) == "bypass"
    metrics["correct_path"] = 1 if (took_bypass == expected_bypass) else 0

    # STRICT SUCCESS CRITERIA
    reached_end = result.get("m_success", False)
    metrics["reached_end"] = 1 if reached_end else 0
    metrics["success"] = (
        1 if (reached_end and seq_validation["sequence_correct"]) else 0
    )

    # Include detailed iteration data for CSV export
    metrics["iteration_details"] = result.get("log_iteration_details", [])

    if verbose:
        print("\n[STATE SEQUENCE]")
        print(f"  Expected ({len(seq_validation['expected'])} states):")
        for i, s in enumerate(seq_validation["expected"], 1):
            print(f"    {i}. {s}")
        print(f"  Actual ({len(seq_validation['actual'])} unique states):")
        for i, s in enumerate(seq_validation["actual"], 1):
            marker = (
                "✓"
                if i <= len(seq_validation["expected"])
                and s == seq_validation["expected"][i - 1].lower()
                else "✗"
            )
            print(f"    {i}. {s} {marker}")

        print(f"\n[SEQUENCE VALIDATION]")
        print(
            f"  Sequence correct: {'✅ YES' if seq_validation['sequence_correct'] else '❌ NO'}"
        )
        print(
            f"  Filling phase correct: {'✅' if seq_validation['filling_correct'] else '❌'}"
        )
        print(
            f"  Emptying phase correct: {'✅' if seq_validation['emptying_correct'] else '❌'}"
        )
        print(f"  Similarity: {seq_validation['sequence_similarity']*100:.0f}%")

        if seq_validation["missing_states"]:
            print(f"  ⚠️  Missing states: {seq_validation['missing_states']}")
        if seq_validation["extra_states"]:
            print(f"  ⚠️  Extra states: {seq_validation['extra_states']}")

        print(f"\n[ENHANCED METRICS]")
        print(
            f"  Planning accuracy: {metrics['planning_correct']}/{metrics['planning_total']} = {metrics['planning_accuracy']*100:.1f}%"
        )
        print(
            f"  Actuator accuracy: {metrics['actuator_correct']}/{metrics['actuator_total']} = {metrics['actuator_accuracy']*100:.1f}%"
        )
        print(f"  Missed transitions: {metrics['missed_transitions']}")
        print(f"  Physical violations: {metrics['physical_violations']}")

        print(f"\n[FINAL RESULT]")
        print(f"  Reached end state: {'✅' if reached_end else '❌'}")
        print(
            f"  Sequence correct: {'✅' if seq_validation['sequence_correct'] else '❌'}"
        )
        print(f"  SUCCESS: {'✅ YES' if metrics['success'] else '❌ NO'}")

    return metrics


# =============================================================================
# ABLATION STUDY
# =============================================================================


def run_ablation_study(
    graph, faults: List[str], n_runs: int = 3, output_dir: str = "./results"
) -> pd.DataFrame:
    """Run ablation study across multiple fault types and runs."""
    global CURRENT_MODEL, CURRENT_PROMPT_LEVEL

    import os
    from datetime import datetime

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = []

    print("\n" + "=" * 60)
    print("STARTING ABLATION STUDY")
    print("=" * 60)
    print(f"Model: {CURRENT_MODEL}")
    print(f"Prompt Level: {CURRENT_PROMPT_LEVEL}")
    print(f"Faults: {faults}")
    print(f"Runs per fault: {n_runs}")
    print(f"Total experiments: {len(faults) * n_runs}")
    print("=" * 60 + "\n")

    for fault in faults:
        print(f"\n{'='*40}")
        print(f"FAULT TYPE: {fault}")
        print(f"Expected path: {EXPECTED_PATHS.get(fault, 'unknown')}")
        print(f"{'='*40}")

        for run in range(n_runs):
            print(f"\n--- Run {run + 1}/{n_runs} ---")

            try:
                metrics = run_single_experiment(graph, fault, verbose=False)
                metrics["run"] = run

                all_results.append(metrics)

                path_taken = (
                    "bypass"
                    if any("bypass" in s.lower() for s in metrics["state_sequence"])
                    else "normal"
                )
                path_status = "✓" if metrics["correct_path"] else "✗"
                seq_status = "✓" if metrics["sequence_correct"] else "✗"
                act_acc = metrics.get("actuator_accuracy", 0) * 100
                plan_acc = metrics.get("planning_accuracy", 0) * 100

                print(
                    f"  Success: {metrics['success']}, "
                    f"Seq: {seq_status} ({metrics['sequence_similarity']*100:.0f}%), "
                    f"Path: {path_taken} {path_status}, "
                    f"Plan: {plan_acc:.0f}%, "
                    f"Act: {act_acc:.0f}%, "
                    f"Time: {metrics['wall_time']:.1f}s"
                )

            except Exception as e:
                print(f"  ERROR: {e}")
                all_results.append(
                    {
                        "model": CURRENT_MODEL,
                        "prompt_level": CURRENT_PROMPT_LEVEL,
                        "fault": fault,
                        "run": run,
                        "success": 0,
                        "sequence_correct": 0,
                        "filling_correct": 0,
                        "emptying_correct": 0,
                        "sequence_similarity": 0.0,
                        "correct_path": 0,
                        "planning_accuracy": 0.0,
                        "actuator_accuracy": 0.0,
                        "abort_reason": "exception",
                        "error": str(e),
                        "iteration_details": [],  # Empty list for exceptions
                    }
                )

    df = pd.DataFrame(all_results)

    model_short = CURRENT_MODEL.replace("gpt-", "").replace("-", "")
    filename_base = f"ablation_{model_short}_{CURRENT_PROMPT_LEVEL}_{timestamp}"

    csv_path = os.path.join(output_dir, f"{filename_base}_runs.csv")

    csv_columns = [
        "model",
        "prompt_level",
        "fault",
        "run",
        "success",
        "reached_end",
        "sequence_correct",
        "filling_correct",
        "emptying_correct",
        "sequence_similarity",
        "skipped_states",
        "correct_path",
        "expected_path",
        "planning_accuracy",
        "planning_correct",
        "planning_total",
        "missed_transitions",
        "actuator_accuracy",
        "actuator_correct",
        "actuator_total",
        "physical_violations",
        "reprompts",
        "invalid_states",
        "iterations",
        "planning_calls",
        "action_calls",
        "planning_latency_avg",
        "action_latency_avg",
        "wall_time",
        "final_state",
        "abort_reason",
        "sim_time",
        # === NEW ===
        "transitions_attempted",
        "process_completion",
        "first_try_rate_planning",
        "first_try_rate_action",
        "attempts_per_transition",
        "planning_input_tokens",
        "planning_output_tokens",
        "action_input_tokens",
        "action_output_tokens",
        "total_input_tokens",
        "total_output_tokens",
        "total_tokens",
    ]

    csv_columns = [c for c in csv_columns if c in df.columns]

    df[csv_columns].to_csv(csv_path, index=False)
    print(f"\n✓ Detailed results saved to: {csv_path}")

    # =========================================================================
    # EXPORT DETAILED ITERATIONS CSV
    # =========================================================================
    all_iterations = []
    for result in all_results:
        run_num = result.get("run", 0)
        fault = result.get("fault", "unknown")
        iteration_details = result.get("iteration_details", [])

        for itr_data in iteration_details:
            row = {
                # Metadata
                "model": CURRENT_MODEL,
                "prompt_level": CURRENT_PROMPT_LEVEL,
                "fault": fault,
                "run": run_num,
                "iteration": itr_data.get("iteration", 0),
                # Timing
                "iteration_total_time_s": itr_data.get("iteration_total_time_s", 0.0),
                "monitoring_time_s": itr_data.get("monitoring_time_s", 0.0),
                "planning_time_s": itr_data.get("planning_time_s", 0.0),
                "action_time_s": itr_data.get("action_time_s", 0.0),
                "simulation_time_s": itr_data.get("simulation_time_s", 0.0),
                "evaluation_time_s": itr_data.get("evaluation_time_s", 0.0),
                "reprompt_time_s": itr_data.get("reprompt_time_s", 0.0),
                # Planning Agent
                "planning_curr_state": itr_data.get("planning_curr_state", ""),
                "planning_chosen_state": itr_data.get("planning_chosen_state", ""),
                "planning_expected_state": itr_data.get("planning_expected_state", ""),
                "planning_correct": 1 if itr_data.get("planning_correct", False) else 0,
                "planning_reasoning": itr_data.get("planning_reasoning", ""),
                # Action Agent
                "action_target_state": itr_data.get("action_target_state", ""),
                "action_raw_output": itr_data.get("action_raw_output", ""),
                "action_warnings": itr_data.get("action_warnings", ""),
                "action_reasoning": itr_data.get("action_reasoning", ""),
                # Actuator SOLL (expected for chosen state)
                "soll_valve_in_B201": itr_data.get("soll_valve_in_B201", 0),
                "soll_valve_in_B202": itr_data.get("soll_valve_in_B202", 0),
                "soll_valve_in_B203": itr_data.get("soll_valve_in_B203", 0),
                "soll_valve_in_B204": itr_data.get("soll_valve_in_B204", 0),
                "soll_valve_out_B201": itr_data.get("soll_valve_out_B201", 0),
                "soll_valve_out_B202": itr_data.get("soll_valve_out_B202", 0),
                "soll_valve_out_B203": itr_data.get("soll_valve_out_B203", 0),
                "soll_valve_out_B204": itr_data.get("soll_valve_out_B204", 0),
                "soll_pump_power": itr_data.get("soll_pump_power", 0.0),
                "soll_pump_power_alt": itr_data.get("soll_pump_power_alt", 0.0),
                # Actuator IS (what was actually applied)
                "ist_valve_in_B201": itr_data.get("ist_valve_in_B201", 0),
                "ist_valve_in_B202": itr_data.get("ist_valve_in_B202", 0),
                "ist_valve_in_B203": itr_data.get("ist_valve_in_B203", 0),
                "ist_valve_in_B204": itr_data.get("ist_valve_in_B204", 0),
                "ist_valve_out_B201": itr_data.get("ist_valve_out_B201", 0),
                "ist_valve_out_B202": itr_data.get("ist_valve_out_B202", 0),
                "ist_valve_out_B203": itr_data.get("ist_valve_out_B203", 0),
                "ist_valve_out_B204": itr_data.get("ist_valve_out_B204", 0),
                "ist_pump_power": itr_data.get("ist_pump_power", 0.0),
                "ist_pump_power_alt": itr_data.get("ist_pump_power_alt", 0.0),
                # Actuator accuracy
                "actuators_correct": itr_data.get("actuators_correct", 0),
                "actuators_total": itr_data.get("actuators_total", 10),
                "actuators_accuracy": itr_data.get("actuators_accuracy", 0.0),
                # Reprompt info
                "was_reprompted": 1 if itr_data.get("was_reprompted", False) else 0,
                "reprompt_reason": itr_data.get("reprompt_reason", ""),
                # Token usage
                "planning_input_tokens": itr_data.get("planning_input_tokens", 0),
                "planning_output_tokens": itr_data.get("planning_output_tokens", 0),
                "action_input_tokens": itr_data.get("action_input_tokens", 0),
                "action_output_tokens": itr_data.get("action_output_tokens", 0),
            }
            all_iterations.append(row)

    if all_iterations:
        iterations_df = pd.DataFrame(all_iterations)
        iterations_csv_path = os.path.join(
            output_dir, f"{filename_base}_iterations.csv"
        )
        iterations_df.to_csv(iterations_csv_path, index=False)
        print(f"✓ Iterations detail saved to: {iterations_csv_path}")

    # Summary
    summary_rows = []
    for fault in faults:
        fault_df = df[df["fault"] == fault]
        if len(fault_df) > 0:
            row = {
                "model": CURRENT_MODEL,
                "prompt_level": CURRENT_PROMPT_LEVEL,
                "fault": fault,
                "success_mean": fault_df["success"].mean(),
                "success_std": fault_df["success"].std() if len(fault_df) > 1 else None,
                "success_sum": fault_df["success"].sum(),
                "reached_end_mean": (
                    fault_df["reached_end"].mean()
                    if "reached_end" in fault_df
                    else None
                ),
                "reached_end_sum": (
                    fault_df["reached_end"].sum() if "reached_end" in fault_df else None
                ),
                "sequence_correct_mean": (
                    fault_df["sequence_correct"].mean()
                    if "sequence_correct" in fault_df
                    else None
                ),
                "correct_path_mean": (
                    fault_df["correct_path"].mean()
                    if "correct_path" in fault_df
                    else None
                ),
                "planning_accuracy_mean": (
                    fault_df["planning_accuracy"].mean()
                    if "planning_accuracy" in fault_df
                    else None
                ),
                "planning_accuracy_std": (
                    fault_df["planning_accuracy"].std()
                    if "planning_accuracy" in fault_df and len(fault_df) > 1
                    else None
                ),
                "actuator_accuracy_mean": (
                    fault_df["actuator_accuracy"].mean()
                    if "actuator_accuracy" in fault_df
                    else None
                ),
                "missed_transitions_sum": (
                    fault_df["missed_transitions"].sum()
                    if "missed_transitions" in fault_df
                    else None
                ),
                "physical_violations_sum": (
                    fault_df["physical_violations"].sum()
                    if "physical_violations" in fault_df
                    else None
                ),
                "reprompts_mean": (
                    fault_df["reprompts"].mean() if "reprompts" in fault_df else None
                ),
                "wall_time_mean": (
                    fault_df["wall_time"].mean() if "wall_time" in fault_df else None
                ),
                "wall_time_std": (
                    fault_df["wall_time"].std()
                    if "wall_time" in fault_df and len(fault_df) > 1
                    else None
                ),
                "process_completion_mean": (
                    fault_df["process_completion"].mean()
                    if "process_completion" in fault_df
                    else None
                ),
                "first_try_rate_planning_mean": (
                    fault_df["first_try_rate_planning"].mean()
                    if "first_try_rate_planning" in fault_df
                    else None
                ),
                "first_try_rate_action_mean": (
                    fault_df["first_try_rate_action"].mean()
                    if "first_try_rate_action" in fault_df
                    else None
                ),
                "attempts_per_transition_mean": (
                    fault_df["attempts_per_transition"].mean()
                    if "attempts_per_transition" in fault_df
                    else None
                ),
                # Token usage
                "planning_input_tokens_sum": (
                    fault_df["planning_input_tokens"].sum()
                    if "planning_input_tokens" in fault_df
                    else None
                ),
                "planning_output_tokens_sum": (
                    fault_df["planning_output_tokens"].sum()
                    if "planning_output_tokens" in fault_df
                    else None
                ),
                "action_input_tokens_sum": (
                    fault_df["action_input_tokens"].sum()
                    if "action_input_tokens" in fault_df
                    else None
                ),
                "action_output_tokens_sum": (
                    fault_df["action_output_tokens"].sum()
                    if "action_output_tokens" in fault_df
                    else None
                ),
                "total_tokens_sum": (
                    fault_df["total_tokens"].sum()
                    if "total_tokens" in fault_df
                    else None
                ),
                # Abort reason counts
                "aborted_success": (
                    (fault_df["abort_reason"] == "success").sum()
                    if "abort_reason" in fault_df
                    else None
                ),
                "aborted_reprompt_limit": (
                    (fault_df["abort_reason"] == "reprompt_limit").sum()
                    if "abort_reason" in fault_df
                    else None
                ),
                "aborted_max_iterations": (
                    (fault_df["abort_reason"] == "max_iterations").sum()
                    if "abort_reason" in fault_df
                    else None
                ),
            }
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(output_dir, f"{filename_base}_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"✓ Summary results saved to: {summary_csv_path}")
    print("\n" + "=" * 60)

    print("\nPer-Fault Summary:")
    print("-" * 90)
    print(
        f"{'Fault':<18} {'Success':>8} {'Compl%':>8} {'Plan%':>8} {'1stTry%':>8} {'Act%':>8} {'Att/Tr':>8}"
    )
    print("-" * 90)

    for fault in faults:
        fault_df = df[df["fault"] == fault]
        if len(fault_df) > 0:
            success_rate = fault_df["success"].mean() * 100
            compl_rate = (
                fault_df["process_completion"].mean() * 100
                if "process_completion" in fault_df
                else 0
            )
            plan_acc = (
                fault_df["planning_accuracy"].mean() * 100
                if "planning_accuracy" in fault_df
                else 0
            )
            first_try = (
                fault_df["first_try_rate_planning"].mean() * 100
                if "first_try_rate_planning" in fault_df
                else 0
            )
            act_acc = (
                fault_df["actuator_accuracy"].mean() * 100
                if "actuator_accuracy" in fault_df
                else 0
            )
            att_per_tr = (
                fault_df["attempts_per_transition"].mean()
                if "attempts_per_transition" in fault_df
                else 0
            )

            print(
                f"{fault:<18} {success_rate:>7.0f}% {compl_rate:>7.0f}% {plan_acc:>7.0f}% {first_try:>7.0f}% {act_acc:>7.0f}% {att_per_tr:>7.1f}"
            )

    print("-" * 90)

    print(f"\n📊 OVERALL:")
    print(f"  SUCCESS: {df['success'].mean()*100:.1f}%")
    if "planning_accuracy" in df.columns:
        print(f"  Planning accuracy: {df['planning_accuracy'].mean()*100:.1f}%")
    if "actuator_accuracy" in df.columns:
        print(f"  Actuator accuracy: {df['actuator_accuracy'].mean()*100:.1f}%")
    if "missed_transitions" in df.columns:
        print(f"  Total missed transitions: {df['missed_transitions'].sum()}")
    if "physical_violations" in df.columns:
        print(f"  Total physical violations: {df['physical_violations'].sum()}")

    return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM-controlled Mixer System with Enhanced Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mixer_case_v2.py -f pump_failure
  python mixer_case_v2.py -f all -n 3
  python mixer_case_v2.py -f all -n 3 --model gpt-4o-mini
  python mixer_case_v2.py -f leak -m gpt-4o-mini -p minimal
        """,
    )
    parser.add_argument(
        "--fault",
        "-f",
        type=str,
        default=None,
        choices=FAULT_TYPES + ["all"],
        help='Fault type to test (or "all" for ablation study)',
    )
    parser.add_argument(
        "--runs",
        "-n",
        type=int,
        default=3,
        help="Number of runs per fault (for ablation study)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt-4o",
        choices=AVAILABLE_MODELS,
        help=f"LLM model to use (default: gpt-4o)",
    )
    parser.add_argument(
        "--prompt-level",
        "-p",
        type=str,
        default="full",
        choices=PROMPT_LEVELS,
        help="Prompt difficulty level",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Reduce output verbosity"
    )

    args = parser.parse_args()

    CURRENT_MODEL = args.model
    CURRENT_PROMPT_LEVEL = args.prompt_level

    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"Model: {CURRENT_MODEL}")
    print(f"Prompt Level: {CURRENT_PROMPT_LEVEL}")
    print("=" * 60)

    print("\nLoading Knowledge Graph context...")
    if SPARQL_AVAILABLE:
        kg_loaded = load_kg_context()
        if kg_loaded:
            print("[KG] ✅ Knowledge Graph context loaded successfully")
        else:
            print("[KG] ⚠️ GraphDB not available - using static prompts only")
    else:
        print("[KG] ⚠️ SPARQL client not available - using static prompts only")

    print("\nCreating the state graph...")
    graph = build_graph()

    if args.fault is None:
        print("\n" + "-" * 40)
        print(
            f"Running default test: pump_failure (model={CURRENT_MODEL}, level={CURRENT_PROMPT_LEVEL})"
        )
        print("-" * 40)
        # Also save CSV for single default run
        df = run_ablation_study(
            graph, faults=["pump_failure"], n_runs=1, output_dir=args.output
        )

    elif args.fault == "all":
        df = run_ablation_study(
            graph, faults=FAULT_TYPES, n_runs=args.runs, output_dir=args.output
        )

    else:
        # Always use run_ablation_study to get CSV output
        df = run_ablation_study(
            graph, faults=[args.fault], n_runs=args.runs, output_dir=args.output
        )
