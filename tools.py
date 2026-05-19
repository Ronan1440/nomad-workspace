import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

# Load environment variables (API keys)
load_dotenv()

# Initialize the official Tavily Search tool
try:
    # This automatically looks for TAVILY_API_KEY in your .env file
    tavily_tool = TavilySearch(max_results=5)
except Exception as e:
    print(f"Warning: TavilySearch failed to initialize. Error: {e}")
    tavily_tool = None

@tool
def scout_coworking_spaces(city: str) -> str:
    """
    Searches the web for top-rated coworking spaces, coliving spots, 
    and work-friendly cafes in a specified city. 
    Use this to gather names, general vibes, and neighborhood locations.
    """
    if not tavily_tool:
        return "Search tool is currently unavailable due to missing API configuration."
        
    query = f"best coworking spaces coliving digital nomads in {city} reviews"
    print(f"🔍 [Tool Call] Searching web for spaces in {city}...")
    
    try:
        # Use the standard invoke method built into the official tool
        results = tavily_tool.invoke({"query": query})
        return str(results)
    except Exception as e:
        return f"Error executing coworking search: {str(e)}"

@tool
def analyze_city_wifi(city: str, space_names: list) -> str:
    """
    Searches for internet speed data, WiFi stability reviews, SIM card recommendations, 
    and telecom reliability (e.g., fiber internet availability, power outages) for a city.
    """
    if not tavily_tool:
        return "Search tool is currently unavailable due to missing API configuration."
        
    spaces_str = ", ".join(space_names) if space_names else "coworking spaces"
    query = f"{city} average internet speed reliability SIM card nomad {spaces_str} wifi speed reviews"
    print(f"⚡ [Tool Call] Searching web for technical infrastructure metrics in {city}...")
    
    try:
        results = tavily_tool.invoke({"query": query})
        return str(results)
    except Exception as e:
        return f"Error executing WiFi infrastructure search: {str(e)}"