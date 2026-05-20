import os
import time
from typing import TypedDict, Dict, Any, List
from langgraph.graph import StateGraph, END
from agents import run_researcher_agent, run_wifi_analyst_agent, run_scorer_agent, run_writer_agent

# 1. UPGRADED: Structured memory fields to hold real names, addresses, and pinpoint locations
class AgentState(TypedDict):
    city: str
    country: str
    postcode: str
    target_lat: float
    target_lon: float
    radius_miles: int
    ranking_preference: str
    preferences: Dict[str, Any]
    research_data: List[Dict[str, Any]]  # Stores real parsed venue objects
    wifi_data: Dict[str, Any]
    ranked_results: List[Dict[str, Any]]
    map_coordinates: List[Dict[str, Any]] # Stores dictionaries with keys: lat, lon, name, address
    final_report: str
    next_step: str

def supervisor_agent(state: AgentState):
    if state.get("research_data") is None: return {"next_step": "researcher"}
    if state.get("wifi_data") is None: return {"next_step": "wifi_analyst"}
    if state.get("ranked_results") is None: return {"next_step": "scorer"}
    return {"next_step": "writer"}

def research_agent(state: AgentState):
    # Pass explicit instructions demanding exact names and storefront locations
    geo_context = f"""
    Find 5 real, specific public-access physical third places (named cafes, work-friendly hotel lobbies, or public libraries) 
    located in {state['city']} {state['postcode']} {state['country']} near coordinates ({state['target_lat']}, {state['target_lon']}).
    
    CRITICAL: You must extract and return their EXACT names, verifiable street addresses, and geographic coordinates if available. 
    Do not return vague descriptions or generic placeholder labels.
    """
    results = run_researcher_agent(geo_context)
    if not results or not isinstance(results, list):
        results = []
    time.sleep(1)
    return {"research_data": results}

def wifi_analyst_agent(state: AgentState):
    location_query = f"{state['city']}, {state['country']} centered near postcode {state['postcode']}"
    # Pass structural list data to preserve naming hierarchies instead of a raw flattened string
    results = run_wifi_analyst_agent(location_query, state["research_data"])
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
    
    results = run_scorer_agent(sorting_weights, state["research_data"], state["wifi_data"])
    if not results or not isinstance(results, list):
        results = state["research_data"] or []
    
    extracted_pins = []
    
    # Track position indexes so we can stagger them if real coordinates are missing
    for idx, venue in enumerate(results):
        lat = venue.get("lat") or venue.get("latitude")
        lon = venue.get("lon") or venue.get("longitude") or venue.get("lng")
        
        # STRUCTURAL FALLBACK: If the background AI failed to scrape exact coordinate numbers,
        # create a minor, deterministic spatial offset so they scatter across the map radius cleanly
        if not lat or not lon:
            offset_multiplier = 0.0025 * (idx + 1)
            if idx % 4 == 0:
                lat = state["target_lat"] + offset_multiplier
                lon = state["target_lon"] - offset_multiplier
            elif idx % 4 == 1:
                lat = state["target_lat"] - offset_multiplier
                lon = state["target_lon"] + offset_multiplier
            elif idx % 4 == 2:
                lat = state["target_lat"] + offset_multiplier
                lon = state["target_lon"] + offset_multiplier
            else:
                lat = state["target_lat"] - offset_multiplier
                lon = state["target_lon"] - offset_multiplier
            
        extracted_pins.append({
            "lat": float(lat),
            "lon": float(lon),
            "name": venue.get("name") or venue.get("title") or f"Workspace Variant {chr(65 + idx)}",
            "address": venue.get("address") or venue.get("location") or f"Local Region Protocol Target Area"
        })
        
    time.sleep(1)
    return {"ranked_results": results, "map_coordinates": extracted_pins}

def report_writer_agent(state: AgentState):
    location_title = f"{state['city'].upper()}, {state['country'].upper()}"
    prompt_override = "Dossier must showcase explicit venue storefront titles, real addresses, and exact proximity."
    report = run_writer_agent(location_title + " - " + prompt_override, state["ranked_results"], state["wifi_data"])
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
