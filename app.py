import streamlit as st
import os
import pandas as pd
import pydeck as pdk
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
from graph import nomad_scout_graph

load_dotenv()
geolocator = Nominatim(user_agent="nomad_workspace_scout_portfolio")

st.set_page_config(
    page_title="Nomad Workspace Scout", 
    page_icon="📡", 
    layout="wide"
)

# Custom CSS targeting the Gaming Inspired Purple Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Azeret+Mono:wght=400;700&family=Chakra+Petch:wght=500;700&display=swap');
    
    .stApp {
        background-color: #0c0814; 
        color: #d8b4fe; 
        font-family: 'Azeret Mono', monospace;
    }

    [data-testid="stSidebar"] {
        background-color: #05030a;
        border-right: 2px solid #a855f7;
    }
    
    h1, h2, h3 {
        font-family: 'Chakra Petch', sans-serif;
        text-transform: uppercase;
        color: #c084fc;
        text-shadow: 0 0 10px #a855f7;
        letter-spacing: 2px;
    }

    .stButton>button {
        background-color: #a855f7 !important;
        color: #ffffff !important;
        font-weight: bold;
        border: none;
        border-radius: 0px;
        box-shadow: 4px 4px 0px #000000;
        transition: 0.2s;
        width: 100%;
        text-transform: uppercase;
        margin-top: 10px;
    }
    
    .stButton>button:hover {
        background-color: #c084fc !important;
        transform: translate(-2px, -2px);
        box-shadow: 6px 6px 0px #d8b4fe;
    }

    input {
        background-color: #05030a !important;
        color: #e9d5ff !important;
        border: 1px solid #a855f7 !important;
    }

    .stMarkdown div {
        background: transparent !important;
        padding: 0px !important;
        border-left: none !important;
    }

    .report-card {
        background: rgba(168, 85, 247, 0.05) !important;
        padding: 20px !important;
        border-left: 3px solid #a855f7 !important;
    }

    .stAlert {
        background-color: #05030a;
        color: #e9d5ff;
        border: 1px dashed #a855f7;
    }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER SECTION ---
header_col1, header_col2 = st.columns([1, 12])
with header_col1:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=75)
    else:
        st.write("📡")

with header_col2:
    st.markdown("""
        <div style="margin-left: -20px; margin-top: 5px;">
            <h1 style='margin: 0px;'>Nomad Workspace Scout</h1>
            <p style='margin: 5px 0 0 0; font-size: 14px; color: #d8b4fe;'>
                <strong>SYSTEM AREA:</strong> WORKSPACE INTELLIGENCE NETWORK<br>
                <strong>CONNECTION STATUS:</strong> ACTIVE MULTI-AGENT PIPELINE
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.info(
    "💡 **SYSTEM NOTE:** This utility does *not* find remote job vacancies. "
    "It maps real physical 'Third Places' (cafes, open libraries, lobbies) for professionals "
    "who need physical infrastructure—like verified Wi-Fi, accessible plugs, and manageable soundscapes—to work out of."
)

# --- SIDEBAR INTERFACE ---
with st.sidebar:
    st.header("GEOGRAPHIC TARGET")
    country_input = st.text_input("COUNTRY", placeholder="e.g., Scotland, UK")
    city_input = st.text_input("CITY", placeholder="e.g., Galashiels, Edinburgh")
    postcode_input = st.text_input("POSTCODE / ZIP (RECOMMENDED)", placeholder="e.g., TD1 1AA")
    
    search_radius = st.slider(
        "SEARCH RADIUS", 
        min_value=1, 
        max_value=20, 
        value=5, 
        step=1,
        format="%d miles"
    )
    
    st.subheader("WORK MODE")
    vibe_dial = st.select_slider(
        "TARGET ATMOSPHERE", 
        options=["QUIET FOCUS", "MODERATE HUM", "LIVELY & SOCIAL"]
    )
    
    st.subheader("INFRASTRUCTURE REQS")
    video_calls = st.checkbox("HIGH SPEED WIFI (CALL VERIFIED)")
    good_ac = st.checkbox("ACCESSIBLE POWER PLUGS")
    indoor_work = st.checkbox("PREFER INDOOR WORKSPACES")
    
    # --- SECURITY ENFORCEMENT ---
    st.markdown("---")
    st.header("DEVELOPER ACCESS")
    access_key = st.text_input("ENTER SYSTEM PASSKEY", type="password", placeholder="••••••••")
    
    st.markdown("---")
    st.header("ARCHITECTURE STACK")
    st.markdown("""
        <div style='font-size: 12px; font-family: "Azeret Mono", monospace; color: #c084fc; line-height: 1.6;'>
        • LangGraph (State Machine)<br>
        • Geopy Spatial Resolution<br>
        • Tavily AI Real-Time API<br>
        • GPT-4o-Mini Engine
        </div>
    """, unsafe_allow_html=True)

# --- EXECUTION ENGINE ---
if st.button("RUN SCOUT AGENTS"):
    if access_key != "pchose9903":
        st.error("ACCESS DENIED: INVALID SYSTEM PASSKEY. UNABLE TO INITIATE AGENT METRICS.")
    elif not city_input.strip() or not country_input.strip():
        st.error("ERROR: CITY AND COUNTRY ARE REQUIRED PARAMETERS. PLEASE FILL IN BOTH FIELDS.")
    else:
        status_box = st.empty()
        with status_box.container():
            st.code("> Geocoding exact target parameters to resolve latitude and longitude...")
            
        try:
            geo_query = f"{city_input.strip()}, {postcode_input.strip()}, {country_input.strip()}" if postcode_input.strip() else f"{city_input.strip()}, {country_input.strip()}"
            location = geolocator.geocode(geo_query, timeout=10)
            
            if not location:
                location = geolocator.geocode(f"{city_input.strip()}, {country_input.strip()}", timeout=10)
                
            if not location:
                st.error("GEOGRAPHIC ERROR: UNABLE TO RESOLVE COORDINATES FOR SPECIFIED TARGET. CHECK SPELLING.")
            else:
                target_lat = location.latitude
                target_lon = location.longitude
                
                with status_box.container():
                    st.code(f"> Resolved Coordinate Matrix: Lat {target_lat} / Lon {target_lon}\n> Confirmed Region Lock: {location.address}\n> Deploying Public Third-Place Search Loop...")
                
                inputs = {
                    "city": city_input.strip(),
                    "country": country_input.strip(),
                    "postcode": postcode_input.strip(),
                    "target_lat": target_lat,
                    "target_lon": target_lon,
                    "radius_miles": search_radius,
                    "ranking_preference": "closeness_and_quality",
                    "preferences": {
                        "vibe": vibe_dial,
                        "video_calls": video_calls,
                        "ac": good_ac,
                        "indoor_only": indoor_work
                    }
                }
                
                thread_key = f"{city_input.lower()}_{country_input.lower()}".replace(' ', '_')
                config = {"configurable": {"thread_id": f"scout_{thread_key}"}}
                
                final_state = nomad_scout_graph.invoke(inputs, config)
                status_box.empty()
                
                st.success(f"REPORT COMPILED FOR {city_input.upper()} SECURED BY SPATIAL LAT/LON LIMITS")
                
                # --- DISPLAY LAYER: INTERACTIVE DASHBOARD ---
                col_left, col_right = st.columns([1, 1])
                
                with col_left:
                    st.subheader("📡 SPATIAL MAPPING CORE")
                    map_venues = final_state.get("map_coordinates", [])
                    
                    map_data = []
                    directory_data = []
                    
                    # Always establish Center Pin (Cyan)
                    map_data.append({
                        "lat": target_lat,
                        "lon": target_lon,
                        "name": "🎯 SEARCH CENTER MIDPOINT",
                        "color_r": 0, "color_g": 240, "color_b": 255
                    })
                    
                    for idx, venue in enumerate(map_venues):
                        v_name = venue.get("name", f"Workspace {chr(65+idx)}")
                        v_address = venue.get("address", "Local Area Address Pool")
                        
                        map_data.append({
                            "lat": venue["lat"],
                            "lon": venue["lon"],
                            "name": f"📌 {v_name}",
                            "color_r": 168, "color_g": 85, "color_b": 247
                        })
                        
                        directory_data.append({
                            "ID": f"VENUE {chr(65+idx)}",
                            "Workspace Venue": v_name,
                            "Street Address": v_address
                        })
                    
                    # VALIDATION ACCORDANCE: Ensure dataframe contains items before rendering canvas view
                    if len(map_data) > 0:
                        map_df = pd.DataFrame(map_data)
                        view_state = pdk.ViewState(
                            latitude=target_lat,
                            longitude=target_lon,
                            zoom=13,
                            pitch=0
                        )
                        layer = pdk.Layer(
                            "ScatterplotLayer",
                            map_df,
                            get_position="[lon, lat]",
                            get_color="[color_r, color_g, color_b]",
                            get_radius=110, # Increased radius for better visual tracking
                            pickable=True
                        )
                        st.pydeck_chart(pdk.Deck(
                            layers=[layer],
                            initial_view_state=view_state,
                            tooltip={"text": "{name}"},
                            map_style="mapbox://styles/mapbox/dark-v11"
                        ))
                        st.caption("🔵 Cyan Pin: Search Center Point | 🟣 Purple Pins: Discovered Workspaces (Hover to see names)")
                    else:
                        st.warning("⚠️ Spatial plotting data array empty. Check search preferences framework.")
                    
                    # --- INTERACTIVE VISUAL DIRECTORY INDEX ---
                    st.markdown("### 🗺️ LOCATION KEY DIRECTORY")
                    if directory_data:
                        for item in directory_data:
                            st.markdown(
                                f"""<div style='padding: 10px; margin-bottom: 8px; background: rgba(168, 85, 247, 0.08); border-left: 4px solid #a855f7;'>
                                    <strong style='color:#c084fc; font-family:"Chakra Petch", sans-serif;'>{item['ID']}: {item['Workspace Venue']}</strong><br>
                                    <span style='font-size:12px; color:#d8b4fe;'>📍 {item['Street Address']}</span>
                                </div>""", 
                                unsafe_allow_html=True
                            )
                    else:
                        st.caption("No address entries parsed for this index query.")
                
                with col_right:
                    st.subheader("📋 INTELLIGENCE DOSSIER SUMMARY")
                    report_content = final_state.get("final_report", "ERROR: NO DATA RETURNED BY GRAPH CORE.")
                    st.markdown(f'<div class="report-card">{report_content}</div>', unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.download_button(
                        label="DOWNLOAD SUMMARY DOSSIER (.TXT)",
                        data=report_content,
                        file_name=f"nomad_report_{city_input.lower().replace(' ', '_')}.txt",
                        mime="text/plain"
                    )
            
        except Exception as e:
            status_box.empty()
            st.error(f"SYSTEM EXECUTION FAILURE: {str(e)}")
