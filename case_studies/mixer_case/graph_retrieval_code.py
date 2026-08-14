#!/usr/bin/env python3

import requests

import sys

from textwrap import dedent


# GraphDB SPARQL endpoint (adjust repository if needed)

ENDPOINT = "http://localhost:7200/repositories/MixingModuleExample"


# SPARQL CONSTRUCT query

QUERY_PLANNING = dedent(
    """
PREFIX :        <http://example.org/mixer#>
PREFIX UML:     <http://www.hsu-ifa.de/ontologies/UMLStateMachine#>
PREFIX VDI2206: <http://www.w3id.org/hsu-aut/VDI2206#>
PREFIX VDI3682: <http://www.w3id.org/hsu-aut/VDI3682#>
PREFIX CPSMod:  <http://www.hsu-ifa.de/ontologies/CPSMod#>
PREFIX rdf:     <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:    <http://www.w3.org/2000/01/rdf-schema#>
 
CONSTRUCT {
  # Seed-Kette (Module → ProcessOperator → StateMachine)
  :mixerModule1234 a VDI2206:Module ;
                   rdfs:label ?cpsL .
  :mixing1234 a VDI3682:ProcessOperator ;
              VDI3682:isAssignedto :mixerModule1234 ;
              CPSMod:processRealizesBehavior :MixerStateMachine ;
              rdfs:label ?poL .
  :MixerStateMachine a UML:StateMachine ;
                     rdfs:label ?smL .
 
  # UML States
  ?s a UML:State ;
     rdfs:label ?sL .
  ?t a UML:State ;
     rdfs:label ?tL .
 
  # UML Transitions WITH rdfs:comment (Domain Knowledge!)
  ?tr a UML:Transition ;
      UML:sourceState ?s ;
      UML:targetState ?t ;
      UML:transitionEvent ?e ;
      UML:transitionGuard ?guard ;
      rdfs:label ?trL ;
      rdfs:comment ?trComment .
  
  # Events
  ?e a UML:Event ;
     rdfs:label ?eL .
 
  # Actions (only true UML:Action)
  ?s UML:doAction ?a .
  ?a a UML:Action ;
     rdfs:label ?aL .
}
WHERE {
  GRAPH <http://www.hsu-ifa.de/graphs/mixerModuleImperial_new> {
    # --- Seed ---
    :mixerModule1234 a VDI2206:Module .
    :mixing1234 a VDI3682:ProcessOperator ;
                VDI3682:isAssignedto :mixerModule1234 ;
                CPSMod:processRealizesBehavior :MixerStateMachine .
    OPTIONAL { :mixerModule1234 rdfs:label ?cpsL }
    OPTIONAL { :mixing1234 rdfs:label ?poL }
    OPTIONAL { :MixerStateMachine rdfs:label ?smL }
 
    # Initial State (fallback to :state_initialStep)
    OPTIONAL { :MixerStateMachine UML:initialState ?init }
    BIND(COALESCE(?init, :state_initialStep) AS ?start)
 
    # All reachable States from start
    ?start ( (^UML:sourceState|^UML:targetState) / (UML:sourceState|UML:targetState) )* ?s .
    
    # Transitions between reachable States
    ?tr a UML:Transition ;
        UML:sourceState ?s ;
        UML:targetState ?t .
    ?start ( (^UML:sourceState|^UML:targetState) / (UML:sourceState|UML:targetState) )* ?t .
 
    # Transition metadata
    OPTIONAL { ?tr UML:transitionEvent ?e . OPTIONAL { ?e rdfs:label ?eL } }
    OPTIONAL { ?tr UML:transitionGuard ?guard }
    OPTIONAL { ?tr rdfs:label ?trL }
    OPTIONAL { ?tr rdfs:comment ?trComment }   # <-- DOMAIN KNOWLEDGE COMMENTS
    OPTIONAL { ?s rdfs:label ?sL }
    OPTIONAL { ?t rdfs:label ?tL }
 
    # Actions (only UML:Action typed)
    OPTIONAL {
      ?s UML:doAction ?a .
      ?a a UML:Action .
      OPTIONAL { ?a rdfs:label ?aL }
    }
  }
}
"""
).strip()


QUERY_ACTION = dedent(
    """
PREFIX :        <http://example.org/mixer#>
PREFIX UML:     <http://www.hsu-ifa.de/ontologies/UMLStateMachine#>
PREFIX VDI2206: <http://www.w3id.org/hsu-aut/VDI2206#>
PREFIX VDI3682: <http://www.w3id.org/hsu-aut/VDI3682#>
PREFIX CPSMod:  <http://www.hsu-ifa.de/ontologies/CPSMod#>
PREFIX rdfs:    <http://www.w3.org/2000/01/rdf-schema#>

CONSTRUCT {
  # Seed (context)
  :mixerModule1234 a VDI2206:Module .
  :mixing1234 a VDI3682:ProcessOperator ;
              VDI3682:isAssignedto :mixerModule1234 ;
              CPSMod:processRealizesBehavior :MixerStateMachine .

  # States with labels and local names
  ?state a UML:State ;
         rdfs:label ?state_label ;
         :localName ?state_id .

  # Actions (only true UML:Action)
  ?state UML:doAction ?action .
  ?action a UML:Action ;
          rdfs:label ?action_label ;
          :localName ?action_id .

  # Actuators linked to Actions
  ?action CPSMod:isChangedByActuator ?actuator .
  ?actuator a VDI2206:Actuator ;
            rdfs:label ?actuator_label ;
            :localName ?actuator_id .
}
WHERE {
  GRAPH <http://www.hsu-ifa.de/graphs/mixerModuleImperial_new> {

    # Seed
    :mixerModule1234 a VDI2206:Module .
    :mixing1234 a VDI3682:ProcessOperator ;
                VDI3682:isAssignedto :mixerModule1234 ;
                CPSMod:processRealizesBehavior :MixerStateMachine .

    # Initial State (fallback)
    OPTIONAL { :MixerStateMachine UML:initialState ?init }
    BIND(COALESCE(?init, :state_initialStep) AS ?start)

    # All reachable States
    ?start ( (^UML:sourceState|^UML:targetState) / (UML:sourceState|UML:targetState) )* ?state .
    ?state a UML:State .
    OPTIONAL { ?state rdfs:label ?state_label }
    BIND(REPLACE(STR(?state), ".*[/#]", "") AS ?state_id)

    # Actions (doAction)
    OPTIONAL {
      ?state UML:doAction ?action .
      ?action a UML:Action .
      OPTIONAL { ?action rdfs:label ?action_label }
      BIND(REPLACE(STR(?action), ".*[/#]", "") AS ?action_id)

      # Actuators (both directions for robustness)
      OPTIONAL {
        { ?action CPSMod:isChangedByActuator ?actuator }
        UNION
        { ?actuator CPSMod:isChangedByActuator ?action }
        ?actuator a VDI2206:Actuator .
      }
      OPTIONAL { ?actuator rdfs:label ?actuator_label }
      OPTIONAL { BIND(REPLACE(STR(?actuator), ".*[/#]", "") AS ?actuator_id) }
    }
  }
}
"""
).strip()

QUERY_CSTR = dedent(
    """
              # ****************************************************************************
#
#   QUERY 1: CSTR PROCESS OVERVIEW (CONSTRUCT)
#
# ****************************************************************************
#
# PURPOSE:
# --------
# This CONSTRUCT query generates an RDF subgraph containing all relevant
# process information for an LLM agent, including:
#
#   1. ProcessOperators with their labels
#   2. Input/Output relationships
#   3. MathematicalModels with equations
#   4. Parameters (Observable/Actuatable)
#   5. Setpoint values
#   6. Module and Component assignments
#
# ****************************************************************************

PREFIX : <http://example.org/cstr#>
PREFIX VDI3682: <http://www.w3id.org/hsu-aut/VDI3682#>
PREFIX VDI2206: <http://www.w3id.org/hsu-aut/VDI2206#>
PREFIX CPSMod: <http://www.hsu-ifa.de/ontologies/CPSMod#>
PREFIX DIN17359: <http://www.hsu-ifa.de/ontologies/DIN17359#>
PREFIX DINEN61360: <http://www.hsu-ifa.de/ontologies/DINEN61360#>
PREFIX sosa: <http://www.w3.org/ns/sosa/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

CONSTRUCT {
    # ========================================================================
    # PROCESS OPERATOR: Core entity with label
    # ========================================================================
    ?processOp rdf:type VDI3682:ProcessOperator .
    ?processOp rdfs:label ?poLabel .
    
    # ========================================================================
    # INPUTS AND OUTPUTS: Material, energy, information flows
    # ========================================================================
    ?processOp VDI3682:hasInput ?input .
    ?processOp VDI3682:hasOutput ?output .
    
    # ========================================================================
    # MATHEMATICAL MODELS: Behavior equations for causal reasoning
    # ========================================================================
    ?processOp CPSMod:processRealizesBehavior ?mathModel .
    ?mathModel rdf:type VDI2206:MathematicalModel .
    ?mathModel rdfs:label ?mathModelLabel .
    
    # ========================================================================
    # PARAMETERS: Observable and Actuatable properties
    # ========================================================================
    ?info CPSMod:isCharacterizedByParameter ?parameter .
    ?parameter rdf:type DIN17359:Parameter .
    ?parameter rdf:type ?parameterType .
    
    # ========================================================================
    # SETPOINTS: Target values with reference values
    # ========================================================================
    ?spInfo owl:sameAs ?refValue .
    ?refValue DIN17359:referenceValue ?spValue .
    
    # ========================================================================
    # MODULES: Physical subsystem grouping
    # ========================================================================
    ?processOp VDI3682:isAssignedTo ?module .
    ?module rdf:type VDI2206:Module .
    
    # ========================================================================
    # COMPONENTS: Hardware elements (Sensors, Actuators, Controllers)
    # ========================================================================
    ?module VDI2206:ModuleConsistsOfComponent ?component .
    ?component rdf:type ?componentType .
}

FROM <http://www.hsu-ifa.de/graphs/CSTR_Simulation>

WHERE {
    # ========================================================================
    # BASE: Get all ProcessOperators (VDI 3682)
    # ========================================================================
    ?processOp a VDI3682:ProcessOperator .
    OPTIONAL { ?processOp rdfs:label ?poLabel }
    
    # ========================================================================
    # INPUTS: Material, energy, and information flows entering the PO
    # ========================================================================
    OPTIONAL { ?processOp VDI3682:hasInput ?input }
    
    # ========================================================================
    # OUTPUTS: Material, energy, and information flows leaving the PO
    # ========================================================================
    OPTIONAL { ?processOp VDI3682:hasOutput ?output }
    
    # ========================================================================
    # MATHEMATICAL MODELS: Equations describing the PO behavior
    # ========================================================================
    OPTIONAL { 
        ?processOp CPSMod:processRealizesBehavior ?mathModel .
        ?mathModel rdfs:label ?mathModelLabel .
    }
    
    # ========================================================================
    # PARAMETERS: Observable and Actuatable properties (DIN 17359 + SOSA)
    # ========================================================================
    OPTIONAL {
        {
            ?processOp VDI3682:hasInput ?info .
            ?info CPSMod:isCharacterizedByParameter ?parameter .
        }
        UNION
        {
            ?processOp VDI3682:hasOutput ?info .
            ?info CPSMod:isCharacterizedByParameter ?parameter .
        }
        
        ?parameter a DIN17359:Parameter .
        
        # Get the specific parameter type (Observable or Actuatable)
        OPTIONAL { 
            ?parameter a sosa:ObservableProperty .
            BIND(sosa:ObservableProperty AS ?parameterType)
        }
        OPTIONAL { 
            ?parameter a sosa:ActuatableProperty .
            BIND(sosa:ActuatableProperty AS ?parameterType)
        }
    }
    
    # ========================================================================
    # SETPOINTS: Target values for PID controllers
    # ========================================================================
    OPTIONAL {
        ?processOp VDI3682:hasInput ?spInfo .
        FILTER(CONTAINS(STR(?spInfo), "_sp"))
        ?spInfo owl:sameAs ?refValue .
        ?refValue DIN17359:referenceValue ?spValue .
    }
    
    # ========================================================================
    # MODULES: Physical grouping of components (VDI 2206)
    # ========================================================================
    OPTIONAL {
        ?processOp VDI3682:isAssignedTo ?module .
        ?module a VDI2206:Module .
    }
    
    # ========================================================================
    # COMPONENTS: Hardware elements within modules
    # ========================================================================
    OPTIONAL {
        ?processOp VDI3682:isAssignedTo ?mod .
        ?mod VDI2206:ModuleConsistsOfComponent ?component .
        
        # Get the specific component type
        OPTIONAL { 
            ?component a VDI2206:Sensor .
            BIND(VDI2206:Sensor AS ?componentType)
        }
        OPTIONAL { 
            ?component a VDI2206:Actuator .
            BIND(VDI2206:Actuator AS ?componentType)
        }
        OPTIONAL { 
            ?component a VDI2206:InformationProcessing .
            BIND(VDI2206:InformationProcessing AS ?componentType)
        }
    }
}

# ============================================================================
# QUERY 1 - EXPECTED OUTPUT (RDF TRIPLES)
# ============================================================================
#
# Example output triples for PO7_TempPID:
#
# :PO7_TempPID rdf:type VDI3682:ProcessOperator .
# :PO7_TempPID rdfs:label "PO7 – Temperature PID Controller"@en .
# :PO7_TempPID VDI3682:hasInput :information_T_meas .
# :PO7_TempPID VDI3682:hasInput :information_T_sp .
# :PO7_TempPID VDI3682:hasOutput :information_u_cool_cmd .
# :PO7_TempPID CPSMod:processRealizesBehavior :Equation_PO7_TempPID .
# :Equation_PO7_TempPID rdf:type VDI2206:MathematicalModel .
# :Equation_PO7_TempPID rdfs:label "u_cool = Kp_T × e + Ki_T × ∫e dt..."@en .
# :information_T_meas CPSMod:isCharacterizedByParameter :obs_param_T_meas .
# :obs_param_T_meas rdf:type DIN17359:Parameter .
# :obs_param_T_meas rdf:type sosa:ObservableProperty .
# :information_T_sp owl:sameAs :ref_T_sp .
# :ref_T_sp DIN17359:referenceValue 3.1E2 .
# :PO7_TempPID VDI3682:isAssignedTo :module_Cooling .
# :module_Cooling VDI2206:ModuleConsistsOfComponent :controller_PID_T .
# :controller_PID_T rdf:type VDI2206:InformationProcessing .
#
# ============================================================================
"""
).strip()


def run_construct(query: str) -> str:
    """

    Execute the SPARQL CONSTRUCT query against the GraphDB endpoint

    and return the result in Turtle format.

    """

    headers = {"Accept": "text/turtle"}  # Request result as Turtle serialization

    try:

        resp = requests.post(
            ENDPOINT, data={"query": query}, headers=headers, timeout=30
        )

    except requests.exceptions.RequestException as e:

        sys.stderr.write(f"[Error] Could not reach the endpoint: {e}\n")

        sys.exit(2)

    if resp.status_code != 200:

        sys.stderr.write(
            f"[Error] HTTP {resp.status_code} returned by SPARQL endpoint.\n"
            f"Response:\n{resp.text}\n"
        )

        sys.exit(1)

    return resp.text  # Return Turtle string


if __name__ == "__main__":

    # Run query and print the result to the terminal

    ttl = run_construct(QUERY_CSTR)

    print(ttl)

    # Optionally save result to a file

    # with open("construct_result.ttl", "w", encoding="utf-8") as f:

    #     f.write(ttl)

    # print("→ Result saved in construct_result.ttl")
