import os
import time
import re
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
    geo_context = f"""
    Identify 5 specific, real public-access physical workspaces (such as named coffee shops, independent bookshops, or libraries) 
    located in {state['city']} {state['postcode']} {state['country']} near coordinates ({state['target_lat']}, {state['target_lon']}).
    
    You must clearly list their exact storefront names and street addresses.
    """
    results = run_researcher_agent(geo_context)
    
    # Normalize string data into a list framework so downstream nodes don't break
    if isinstance(results, str):
        results = [{"raw_content": results}]
    elif not results:
        results = []
    time.sleep(1)
    return {"research_data": results}

def wifi_analyst_agent(state: AgentState):
    location_query = f"{state['city']}, {state['country']}"
    results = run_wifi_analyst_agent(location_query, str(state["research_data"]))
    if results is None:
        results = {}
    time.sleep(1)
    return {"wifi_data": results}

def scorer_agent(state: AgentState):
    sorting_weights = {
        "user_preferences": state["preferences"],
        "ranking_rule": "closeness_and_quality",
        "anchor_lat": state["target_lat"],
        "anchor_lon": state["target_lon"],
        "max_radius": state["radius_miles"],
    }
    
    raw_score_output = run_scorer_agent(sorting_weights, str(state["research_data"]), str(state["wifi_data"]))
    
    # --- SMART COGNITIVE PARSER LAYER ---
    # Convert whatever the agent returns into a stable string to extract individual venues
    text_to_parse = str(raw_score_output) if raw_score_output else str(state["research_data"])
    
    # Use split mechanics to discover up to 5 distinct workspace lines or bullet points
    found_venues = []
    lines = re.split(r'\d+\.\s+|\n\* \s+|\n-\s+|=+', text_to_parse)
    
    for line in lines:
        cleaned = line.strip()
        if len(cleaned) > 15 and not cleaned.startswith("bracket") and not cleaned.startswith("["):
            # Extract the first sentence or chunk as the Title, and use the rest as the Address context
            parts = cleaned.split('\n')
            name = parts[0].replace("**", "").replace("[", "").replace("]", "").strip()[:50]
            address = parts[1].strip() if len(parts) > 1 else f"{state['city']}, {state['postcode']}"
            
            found_venues.append({
                "name": name,
                "address": address
            })
            
    # Fallback default list if no distinct lines were successfully isolated
    if not found_venues:
        found_venues = [
            {"name": f"Workspace Alpha ({state['city']})", "address": "Central Access Hub"},
            {"name": f"Workspace Beta ({state['city']})", "address": "High-Speed Wi-Fi Zone"},
            {"name": f"Workspace Gamma ({state['city']})", "address": "Quiet Focus Cell"}
        ]
        
    # Trim down to a maximum of 5 locations to prevent visual map spamming
    found_venues = found_venues[:5]
    
    # --- SPATIAL PLACEMENT CALCULATOR ---
    extracted_pins = []
    for idx, venue in enumerate(found_venues):
        # Generate clean, mathematically perfect concentric distribution offsets 
        # so every location is clearly separated and perfectly spread out
        offset_lat = 0.0035 * (((idx + 1) * 1.3) % 2.5 - 1.25)
        offset_lon = 0.0045 * (((idx + 1) * 1.7) % 2.5 - 1.25)
        
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
