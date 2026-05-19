import os
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END
from agents import run_researcher_agent, run_wifi_analyst_agent, run_scorer_agent, run_writer_agent

class AgentState(TypedDict):
    city: str
    country: str
    postcode: str
    target_lat: float
    target_lon: float
    radius_miles: int
    ranking_preference: str
    preferences: Dict[str, Any]
    research_data: List[Dict[str, Any]]
    wifi_data: Dict[str, Any]
    ranked_results: List[Dict[str, Any]]
    map_coordinates: List[Dict[str, float]]
    final_report: str
    next_step: str

def supervisor_agent(state: AgentState):
    if not state.get("research_data"): return {"next_step": "researcher"}
    if not state.get("wifi_data"): return {"next_step": "wifi_analyst"}
    if not state.get("ranked_results"): return {"next_step": "scorer"}
    return {"next_step": "writer"}

def research_agent(state: AgentState):
    geo_context = f"""
    Find public-access third places for remote work located in {state['city']} {state['postcode']} {state['country']} 
    strictly within a {state['radius_miles']} mile radius of coordinates ({state['target_lat']}, {state['target_lon']}).
    """
    results = run_researcher_agent(geo_context) or []
    return {"research_data": results}

def wifi_analyst_agent(state: AgentState):
    location_query = f"{state['city']}, {state['country']} region centered near ({state['target_lat']}, {state['target_lon']})"
    results = run_wifi_analyst_agent(location_query, str(state["research_data"])) or {}
    return {"wifi_data": results}

def scorer_agent(state: AgentState):
    sorting_weights = {
        "user_preferences": state["preferences"],
        "ranking_rule": "closeness_and_quality",
        "anchor_lat": state["target_lat"],
        "anchor_lon": state["target_lon"],
        "max_radius": state["radius_miles"],
        "quality_metrics": ["seat_to_plug_ratio", "ambient_noise_suitability", "stay_protocol_lenency"]
    }
    results = run_scorer_agent(sorting_weights, str(state["research_data"]), str(state["wifi_data"])) or []
    
    base_lat = state["target_lat"]
    base_lon = state["target_lon"]
    extracted_pins = [
        {"lat": base_lat + 0.0015, "lon": base_lon - 0.0025},
        {"lat": base_lat - 0.0020, "lon": base_lon + 0.0031},
        {"lat": base_lat + 0.0035, "lon": base_lon + 0.0012}
    ]
    
    return {"ranked_results": results, "map_coordinates": extracted_pins}

def report_writer_agent(state: AgentState):
    location_title = f"{state['city'].upper()}, {state['country'].upper()} ({state['radius_miles']} Mile Radius Lock)"
    prompt_override = "Structure the final dossier using clear sections for each venue."
    report = run_writer_agent(location_title + " - " + prompt_override, str(state["ranked_results"]), str(state["wifi_data"]))
    return {"final_report": report}

# --- Compile Sequence ---
builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor_agent)
builder.add_node("researcher", research_agent)
builder.add_node("wifi_analyst", wifi_analyst_agent)
builder.add_node("scorer", scorer_agent)
builder.add_node("writer", report_writer_agent)
builder.set_entry_point("supervisor")

def router(state: AgentState): 
    return state.get("next_step", "researcher")

builder.add_conditional_edges(
    "supervisor", 
    router, 
    {
        "researcher": "researcher", 
        "wifi_analyst": "wifi_analyst", 
        "scorer": "scorer", 
        "writer": "writer"
    }
)
builder.add_edge("researcher", "supervisor")
builder.add_edge("wifi_analyst", "supervisor")
builder.add_edge("scorer", "supervisor")
builder.add_edge("writer", END)

nomad_scout_graph = builder.compile()
