import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tools import scout_coworking_spaces, analyze_city_wifi

# Load environment variables
load_dotenv()

# Initialize our primary LLM 
# gpt-4o-mini 
try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
except Exception as e:
    print(f"Warning: Failed to initialize ChatOpenAI. Check your OPENAI_API_KEY. Error: {e}")
    llm = None

def run_researcher_agent(city: str) -> list:
    """Executes the research phase using web search tools."""
    print(f"🕵️‍♂️ [LLM Agent] Researcher processing {city}...")
    
    # 1. Invoke the live tool to pull data down
    raw_web_data = scout_coworking_spaces.invoke({"city": city})
    
    # 2. Use the LLM to structure the messy web data into clean JSON-like dictionaries
    system_prompt = """You are an expert digital nomad research assistant. 
    Your job is to look at raw web search results and extract the top 3-5 coworking spaces, coliving spots, or work cafes.
    
    For each location, extract:
    - Name
    - General vibe (e.g., social, quiet, party, networking)
    - Specific neighborhood location
    
    Return your response strictly as a clean, structured summary."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Analyze this raw search data and extract the key spaces:\n\n{data}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"data": raw_web_data})
    
    # Pass the LLM's structured breakdown back to the graph state
    return [{"raw_summary": response.content}]

def run_wifi_analyst_agent(city: str, research_summary: str) -> dict:
    """Analyzes technical telemetry, speed reviews, and cellular infrastructure."""
    print("⚡ [LLM Agent] WiFi Analyst processing infrastructure...")
    
    # 1. Run the custom WiFi tool
    raw_wifi_data = analyze_city_wifi.invoke({"city": city, "space_names": []})
    
    system_prompt = """You are a highly technical Network Engineer and remote work strategist.
    Analyze the raw infrastructure data and break down:
    1. Expected internet speeds (download/upload averages) for the city's workspaces.
    2. Telecom provider recommendations (best local SIM cards/eSIMs for backup hotspots).
    3. Technical red flags (e.g., rolling blackouts, frequent fiber dropouts, areas with poor cell signal).
    
    Be brutally honest. Nomads rely on this to keep their jobs."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Analyze this telemetry data for {city}:\n\n{data}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"city": city, "data": raw_wifi_data})
    
    return {"telemetry_report": response.content}

def run_scorer_agent(preferences: dict, research: str, wifi: str) -> list:
    """Ranks and filters options based on custom user dials."""
    print("⚖️ [LLM Agent] Scorer tailoring rankings to preferences...")
    
    system_prompt = """You are a lifestyle matchmaking algorithm for remote workers. 
    Compare the gathered workspace data and the technical WiFi data against the user's explicit preferences.
    
    User Preferences:
    - Target Vibe: {vibe}
    - Requires High Upload Speed: {video_calls}
    - Needs Strong AC/Outlets: {ac}
    - Indoor/Sun Avoidance: {indoor_only}
    - Bonus Social/Dating Focus: {social_notes}
    
    Rank the spaces and explain exactly *why* they match or miss these custom criteria."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Review this workspace data:\n\n{research}\n\nAnd this WiFi telemetry:\n\n{wifi}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "vibe": preferences.get("vibe", "Balanced"),
        "video_calls": preferences.get("video_calls", False),
        "ac": preferences.get("ac", False),
        "indoor_only": preferences.get("indoor_only", False),
        "social_notes": preferences.get("social_notes", False),
        "research": research,
        "wifi": wifi
    })
    
    return [{"ranking_breakdown": response.content}]

def run_writer_agent(city: str, ranked_data: str, wifi_data: str) -> str:
    """Takes all structured steps and crafts a beautiful markdown report."""
    print("✍️ [LLM Agent] Writer crafting pristine markdown output...")
    
    system_prompt = """You are a premium travel and tech journalist editing a newsletter for high-earning digital nomads.
    Your task is to take raw workspace matches, ranking metrics, and technical connectivity data, and compile it into a flawless, publication-ready Markdown report.
    
    Ensure you use clear headers, clean bullet points, bold key phrases, and warning blocks for red flags. 
    Make it highly readable and instantly scannable."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Generate the ultimate Nomad Scout Report for {city}.\n\nRankings Context:\n{ranked}\n\nWiFi Context:\n{wifi}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"city": city, "ranked": ranked_data, "wifi": wifi_data})
    
    return response.content
