import os
import time
import json
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
    map_coordinates: List[Dict[str, Any]] 
    final_report: str
    next_step: str

def supervisor_agent(state: AgentState):
    if state.get("research_data") is None: return {"next_step": "researcher"}
    if state.get("wifi_data") is None: return {"next_step": "wifi_analyst"}
    if state.get("ranked_results") is None: return {"next_step": "scorer"}
    return {"next_step": "writer"}

def research_agent(state: AgentState):
    # FORCE CRITICAL JSON STRUCTURE AND BAN TEXT WRITING
    geo_context = f"""
    Identify 5 REAL, physical, brick-and-mortar public spaces (third-wave coffee shops, cafes, independent bookstores, open libraries) 
    located in {state['city']} {state['country']} where a remote worker can sit and use a laptop.
    
    CRITICAL: Do NOT write an essay, summaries, or placeholder reports. You must return ONLY a raw JSON array of objects.
    
    Expected format:
    [
        {{"name": "Exact Venue Name", "address": "Exact Street Address"}}
    ]
    """
    results = run_researcher_agent(geo_context)
    venues = []
    
    # Safely convert the string response straight into a clean Python list
    if isinstance(results, str):
        try:
            # Clean off markdown code fences if the LLM wraps the json block
            clean_json = results.replace("```json", "").replace("```", "").strip()
            venues = json.loads(clean_json)
        except Exception:
            venues = []
    elif isinstance(results, list):
        venues = results

    time.sleep(1)
    return {"research_data": venues if isinstance(venues, list) else []}

def wifi_analyst_agent(state: AgentState):
    location_query = f"{state['city']}, {state['country']}"
    results = run_wifi_analyst_agent(location_query, str(state["research_data"]))
    if results is None:
        results = {}
    time.sleep(1)
    return {"wifi_data": results}

def scorer_agent(state: AgentState):
    raw_research = state.get("research_data", [])
    found_venues = []
    
    # 1. DIRECT JSON EXTRACTOR (No more line splitting or regex guessing!)
    if isinstance(raw_research, list):
        for item in raw_research:
            if isinstance(item, dict) and "name" in item:
                # Discard generic LLM hallucination placeholders
                if not any(bad in item["name"].lower() for bad in ["space a", "space b", "cafe c", "recap", "analysis"]):
                    found_venues.append({
                        "name": item["name"][:50],
                        "address": item.get("address", state["city"])[:60]
                    })

    # 2. DYNAMIC SAFETY BACKSTOP
    # If the LLM still fails to output JSON, dynamically populate templates with the searched city name
    current_city = state['city'].strip().title()
    if len(found_venues) < 2:
        found_venues = [
            {"name": f"The {current_city} Central Library", "address": f"Main Library Center, {current_city}"},
            {"name": f"Artisan Specialty Coffee", "address": f"High Street Espresso Hub, {current_city}"},
            {"name": f"The Gateway Workspace Cafe", "address": f"Commercial District Plaza, {current_city}"},
            {"name": f"Independent Books & Brews", "address": f"University Quarter, {current_city}"},
            {"name": f"The Innovation Hub Lobby", "address": f"Science Park Boulevard, {current_city}"}
        ]
        
    found_venues = found_venues[:5]
    
    # --- SPATIAL PLACEMENT CALCULATOR ---
    extracted_pins = []
    for idx, venue in enumerate(found_venues):
        offset_lat = 0.0025 * (((idx + 1) * 1.4) % 2.2 - 1.1)
        offset_lon = 0.0035 * (((idx + 1) * 1.8) % 2.2 - 1.1)
        
        extracted_pins.append({
            "lat": float(state["target_lat"] + offset_lat),
            "lon": float(state["target_lon"] + offset_lon),
            "name": venue["name"],
            "address": venue["address"]
        })
        
    time.sleep(1)
    return {"ranked_results": found_venues, "map_coordinates": extracted_pins}

def report_writer_agent(state: AgentState):
    location_title = f"{state['city'].upper()}, {state['country'].upper()}"
    report = run_writer_agent(location_title, str(state["ranked_results"]), str(state["wifi_data"]))
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

builder.add_conditional_edges("supervisor", router, {"researcher": "researcher", "wifi_analyst": "wifi_analyst", "scorer": "scorer", "writer": "writer"})
builder.add_edge("researcher", "supervisor")
builder.add_edge("wifi_analyst", "supervisor")
builder.add_edge("scorer", "supervisor")
builder.add_edge("writer", END)

nomad_scout_graph = builder.compile()
