# ⌨️🤖 crtl-alt-recover
**From Detection to Action: LLM Agents for Fault-Tolerant Control**

Ctrl + Alt + Recover — validated recovery actions, not blind resets.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Type](https://img.shields.io/badge/type-research-orange)
![Control](https://img.shields.io/badge/LLM-supervisory-green)

---

## Overview

**crtl-alt-recover** is a hybrid **research artifact written as executable software** that investigates how **Large Language Model (LLM) agents** can support **active fault-tolerant control** in cyber-physical systems.

The repository implements a **supervisory control loop** in which LLM-based agents transform fault detection outputs into **validated recovery actions**, rather than directly issuing low-level control commands. All proposed actions are **tested against a digital twin** before execution to ensure feasibility and safety.

At the core of the framework are:
- a **multi-agent workflow** (monitoring, planning, action synthesis, simulation, validation, and reprompting),
- a **Digital Process Plant Twin (DPPT)** that provides executable plant models and validation services, and
- a **Graph Retrieval-Augmented Generation (Graph RAG)** layer grounded in the **CPSMod ontology**, which supplies plant-specific structure, behavior, control context, and fault semantics to the agents.

The approach is evaluated on two complementary case studies:
- a **discrete batch Mixing Module**, where recovery is formulated as valid state-machine paths and actuator configurations, and
- a **continuous Stirred-Tank Reactor (CSTR)** under PID regulation, where recovery is achieved through validated setpoint adaptations under degraded operating conditions.

---

## What happens when you run this repo?

When you run one of the provided case-study scripts, the code executes a **closed-loop supervisory fault-recovery cycle** driven by LLM agents and validated through simulation.

At a high level, each run proceeds as follows:

1. A **digital twin** of the plant (discrete Mixing Module or continuous CSTR) is initialized in nominal operation  
2. One or more **faults are injected at runtime**, based on command-line arguments  
3. A monitoring loop detects abnormal behavior or constraint violations  
4. An LLM-based planning agent proposes one or more recovery actions  
5. Proposed actions are **validated via simulation rollout** on the digital twin  
6. Validated actions are applied to the plant model  
7. Invalid or unsafe actions trigger **reprompting and correction**  
8. The run terminates after successful recovery or a safety fallback  

The scripts support both **single-run execution** (`n = 1`) and **batched experiments** (`n > 1`) for statistical evaluation.  
Fault scenarios can be selected individually or executed as predefined subsets directly from the command line, without modifying the source code.

---

## Knowledge Graph and Graph RAG

The supervisory agents in **crtl-alt-recover** rely on a **knowledge graph–grounded retrieval mechanism** to reason about admissible recovery actions in a plant-specific and constraint-aware manner.

The repository uses a **Graph Retrieval-Augmented Generation (Graph RAG)** approach backed by a **GraphDB** instance. Plant knowledge is encoded using the **CPSMod ontology**, which captures structural, functional, behavioral, and fault-related aspects of the system in a unified semantic representation.

During execution, the LLM agents:
- query the knowledge graph using **SPARQL** to retrieve task-relevant subgraphs,
- obtain explicit information about valid actions, state transitions, constraints, and fault dependencies,
- and condition their planning and action synthesis on this structured context.

GraphDB is **mandatory** for the current version of the code.  
Without graph-grounded retrieval, the agents may hallucinate invalid actions or violate system constraints.

### Knowledge graph setup (required)

1. Start a local **GraphDB** instance (e.g., Ontotext GraphDB)  
2. Create repositories for the provided knowledge graphs  
3. Import the supplied **JSON-LD** files encoding the Mixing Module and CSTR models  
4. Ensure the SPARQL endpoint and repository names match those specified in the code  

---

## Case Studies

### Discrete Supervisory Control — Mixing Module

The **Mixing Module** used for discrete supervisory control is implemented in a **separate repository** and must be imported before running the discrete case study.

This repository provides the **LLM-based supervisory logic**, while the Mixing Module repository provides:
- the discrete plant simulation,
- the state-machine definition,
- actuator models and fault injection logic.

To run the discrete case study:
1. Clone the Mixing Module repository
2. Ensure it is available on the Python path (e.g., via editable install or local import)
3. Run the discrete supervisory script in this repository

Once integrated, the Mixing Module is modeled as a **finite state machine** with explicitly defined states, transitions, guards, and actuator configurations.

- Recovery is formulated as **valid state-machine recovery paths**
- Actions correspond to **discrete actuator commands** (valves, pumps, bypass routing)
- Planning is constrained by admissible transitions and fault-dependent logic retrieved via the knowledge graph

![Discrete supervisory recovery](assets/mixer_case.gif)

This case study emphasizes state-machine correctness, actuator validity, and robustness under discrete faults such as pump degradation, clogging, leakage, and sensor faults.

---

### Continuous Supervisory Control — CSTR

The continuous case study considers a **Continuous Stirred-Tank Reactor (CSTR)** operating under closed-loop **PID regulation** with nonlinear dynamics.

- Fault scenarios include cooling fouling, pump degradation, and cooling saturation
- LLM agents propose **high-level setpoint adaptations** rather than direct actuator commands
- All proposals are validated via simulation to ensure safety constraints are respected

![Continuous supervisory control architecture](assets/cstr_case.png)

The supervisory agent frequently exhibits throughput–safety trade-offs, such as reducing inlet flow under degraded cooling to maintain temperature limits.

---

## Results Summary

### Discrete Mixing Module
- Successful recovery in the majority of evaluated fault scenarios
- Differences between models primarily affect **reprompting frequency** and **action efficiency**
- Knowledge graph grounding significantly reduces invalid actuator proposals

### Continuous CSTR
- Recovery achieved primarily through **setpoint adaptation**
- Cooling fouling scenarios consistently require **throughput reduction** to maintain thermal safety
- Simulation-based validation prevents unsafe supervisory actions from being applied

Across both case studies, the framework demonstrates robust and interpretable recovery behavior with bounded reprompting and strong constraint adherence.

---

## Running the Experiments

### Prerequisites

- Python 3.10+
- A running **GraphDB** instance with required knowledge graphs
- Any **OpenAI-compatible LLM** or local models via **Ollama**

```bash
export OPENAI_API_KEY=YOUR_API_KEY


Discrete Mixing Module 
python mixer_case.py --runs 1
python mixer_case.py --runs 10
python mixer_case.py --fault pump_degradation
python mixer_case.py --faults pump_degradation clogging leakage

Continuous CSTR 
python cstr_case.py --runs 1
python cstr_case.py --runs 10
python cstr_case.py --fault cooling_fouling
python cstr_case.py --faults cooling_fouling pump_degradation cooling_saturation

Note: When using local models via Ollama, model names must not contain : characters, as this interferes with CSV result logging.

