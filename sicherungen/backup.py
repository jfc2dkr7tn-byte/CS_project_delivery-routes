import streamlit as st
import requests
import pandas as pd 
import pydeck as pdk
from urllib.parse import quote
import itertools

api_key = 'jCj3v16W0cJMzI4m2GtkQ32lwmkmUVki'

# -------- CONFIGURATIONS
st.set_page_config(
    page_title="Route Planner",
    layout="wide",
)
st.markdown("""
<style>

/* Deletes complete Header-Element */
header[data-testid="stHeader"] {
    display: none !important;
}

/* Deletes the toolbar menu at the top right (three dots) */
div[data-testid="stToolbar"] {
    display: none !important;
}

/* Deletes the "Deploy" banner in some versions */
[data-testid="stDecoration"] {
    display: none !important;
}

/* Deletes the space left by the header */
div.block-container {
    padding-top: 1rem !important;
}

</style>
""", unsafe_allow_html=True)

# Color Theme

NEW_THEME_CSS = """
<style>
    /* ========== FONTS ========== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');
    
    /* ========== GLOBAL SETTINGS ========== */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    
    /* ========== COLOR PALETTE ========== */
    :root {
        --midnight-blue: #1C2E4A;
        --dusty-blue: #52677D;
        --ivory: #BDC4D4;
        --deep-navy: #0F1A2B;
        --buttercream: #D1CFC9;
        --white: #FFFFFF;
        --text-primary: #1C2E4A;
        --text-secondary: #52677D;
        --border-light: #BDC4D4;
    }
    
    /* ========== REMOVE STREAMLIT HEADER ========== */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    div[data-testid="stToolbar"] {
        display: none !important;
    }
    
    [data-testid="stDecoration"] {
        display: none !important;
    }
    
    /* ========== MAIN LAYOUT ========== */
    .stApp {
        background: linear-gradient(135deg, #D1CFC9 0%, #BDC4D4 100%);
    }
    
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* ========== SIDEBAR ========== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F1A2B 0%, #1C2E4A 100%);
        border-right: none;
        box-shadow: 4px 0 20px rgba(15, 26, 43, 0.15);
    }
    
    section[data-testid="stSidebar"] > div {
        padding: 2rem 1.5rem;
    }
    
    /* Sidebar text colors */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #D1CFC9 !important;
    }
    
    /* Sidebar dividers */
    section[data-testid="stSidebar"] hr {
        margin: 1.5rem 0;
        border: none;
        height: 1px;
        background: rgba(189, 196, 212, 0.2);
    }
    
    /* Fix dropdown visibility */
    section[data-testid="stSidebar"] div[data-baseweb="select"] ul li {
        color: #1C2E4A !important;
    }
    
    /* ========== TYPOGRAPHY ========== */
    h1 {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1C2E4A;
        margin-bottom: 1rem;
    }
    
    h2 {
        font-size: 1.75rem;
        font-weight: 600;
        color: #1C2E4A;
        margin-bottom: 1rem;
    }
    
    h3 {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1C2E4A;
        margin-bottom: 0.75rem;
    }
    
    p, span, div, label {
        color: #1C2E4A;
        line-height: 1.6;
    }
    
    .stCaption {
        color: #52677D !important;
        font-size: 0.875rem;
    }
    
    /* ========== BUTTONS ========== */
    /* Primary Button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1C2E4A 0%, #0F1A2B 100%);
        color: #D1CFC9;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 12px rgba(28, 46, 74, 0.25);
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0F1A2B 0%, #1C2E4A 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(28, 46, 74, 0.35);
    }
    
    /* Secondary Button */
    .stButton > button:not([kind="primary"]) {
        background: #FFFFFF;
        color: #1C2E4A;
        border: 2px solid #BDC4D4;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stButton > button:not([kind="primary"]):hover {
        border-color: #52677D;
        background: #F8F9FA;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(82, 103, 125, 0.15);
    }
    
    /* ========== DOWNLOAD BUTTONS ========== */
    .stDownloadButton > button {
        background: #FFFFFF;
        color: #1C2E4A !important;
        border: 2px solid #BDC4D4;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
    }
    
    .stDownloadButton > button:hover {
        border-color: #52677D;
        background: #F8F9FA;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(82, 103, 125, 0.15);
    }
    
    /* ========== INPUT FIELDS ========== */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        border: 2px solid #BDC4D4;
        border-radius: 10px;
        background: #FFFFFF;
        color: #1C2E4A;
        padding: 0.75rem 1rem;
        font-size: 0.95rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div:focus-within,
    .stNumberInput > div > div > input:focus {
        outline: none;
        border-color: #52677D;
        box-shadow: 0 0 0 3px rgba(82, 103, 125, 0.1);
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #52677D;
        opacity: 0.6;
    }
    
    /* Input Labels */
    .stTextInput label,
    .stSelectbox label,
    .stNumberInput label {
        font-weight: 600;
        color: #1C2E4A !important;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    /* ========== RADIO BUTTONS ========== */
    div[role="radiogroup"] {
        gap: 1rem;
    }
    
    div[role="radiogroup"] > label {
        background: #FFFFFF;
        border: 2px solid #BDC4D4;
        border-radius: 10px;
        padding: 0.75rem 1.25rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    
    div[role="radiogroup"] > label:hover {
        border-color: #52677D;
        background: #F8F9FA;
        transform: translateY(-1px);
    }
    
    div[role="radiogroup"] > label > div:first-child {
        border-color: #52677D !important;
        background-color: #FFFFFF !important;
    }
    
    div[role="radiogroup"] > label > div:first-child > div {
        background-color: #52677D !important;
    }
    
    /* ========== CHECKBOXES ========== */
    div[data-baseweb="checkbox"] {
        background: #FFFFFF;
        border: 2px solid #BDC4D4;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    div[data-baseweb="checkbox"]:hover {
        border-color: #52677D;
        background: #F8F9FA;
    }
    
    div[data-baseweb="checkbox"] > div:first-child {
        border-color: #52677D !important;
        background-color: #FFFFFF !important;
    }
    
    div[data-baseweb="checkbox"] > div:first-child > div {
        background-color: #52677D !important;
    }
    
    /* ========== TABS ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 2px solid #BDC4D4;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #52677D;
        border: none;
        padding: 0.75rem 1.25rem;
        font-weight: 500;
        font-size: 0.95rem;
        border-radius: 8px 8px 0 0;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(189, 196, 212, 0.1);
        color: #1C2E4A;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #1C2E4A !important;
        font-weight: 600;
        background: transparent !important;
        border-bottom: 3px solid #52677D !important;
    }
    
    /* ========== ALERT BOXES ========== */
    .stAlert {
        background: #FFFFFF;
        border: 2px solid #BDC4D4;
        border-left: 4px solid #52677D;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        box-shadow: 0 2px 8px rgba(28, 46, 74, 0.08);
    }
    
    /* ========== SPINNER ========== */
    .stSpinner > div {
        border-top-color: #52677D !important;
    }
    
    /* ========== EXPANDER ========== */
    .streamlit-expanderHeader {
        background: #FFFFFF;
        border: 2px solid #BDC4D4;
        border-radius: 10px;
        font-weight: 600;
        color: #1C2E4A;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #52677D;
        background: #F8F9FA;
    }
    
    /* ========== SELECT BOX ========== */
    div[data-baseweb="select"] {
        border-radius: 10px;
    }
    
    /* ========== MOBILE RESPONSIVENESS ========== */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 0.5rem !important;
        }
        
        section[data-testid="stSidebar"] {
            width: 100% !important;
        }
        
        h1 {
            font-size: 2rem;
        }
        
        h2 {
            font-size: 1.5rem;
        }
    }
    
    /* ========== SCROLLBAR ========== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #D1CFC9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #52677D;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #1C2E4A;
    }
    
    /* ========== ANIMATIONS ========== */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .element-container {
        animation: fadeIn 0.4s ease-out;
    }
</style>
"""


# Adapt to mobile responsiveness
st.markdown("""
<style>
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 0.5rem;
        }
        section[data-testid="stSidebar"] {
            width: 100% !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# -------- Color THEME

st.markdown("""
<style>
    /* Import  fonts */
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600&family=Source+Sans+Pro:wght@400;600&display=swap');

    /* Set font settings */
    html, body, [class*="css"] {
        font-family: 'Source Sans Pro', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Oswald', sans-serif;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Main background */
    .stApp {
        background-color: #D1CFC9 !important;
    }
    
    /* Sidebar  */
    section[data-testid="stSidebar"] {
        background: #0F1a2b !important;
    }
    
    /* All text in sidebar white */
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    /* Fix for dropdown items inside sidebar being visible when clicked */
    section[data-testid="stSidebar"] div[data-baseweb="select"] ul li {
        color: black !important;
    }
    
    /* Headerlines  */
    h1, h2, h3 {
        color: #010139 !important;
    }
    
    /* Primary buttons  */
    .stButton > button[kind="primary"] {
        background: #010139;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-family: 'Oswald', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: #02b4ff; /* Bright Royal Blue on hover */
        box-shadow: 0 4px 12px rgba(2, 180, 255, 0.3);
        transform: translateY(-1px);
    }
    
    /* Secondary buttons (Outline) */
    .stButton > button:not([kind="primary"]) {
        background: #dbeafe;
        color: #010139;
        border: 2px solid #010139;
        border-radius: 8px;
        font-weight: 600;
        font-family: 'Oswald', sans-serif;
    }
    .stButton > button:not([kind="primary"]):hover {
        background: #010139;
        color: white;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stNumberInput > div > div > input {
        border: 2px solid #02b4ff; /* Royal Blue Border */
        border-radius: 6px;
        background-color: white;
        color: #010139;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 2px solid #dbeafe;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #64748b;
        border: none;
        padding: 12px 24px;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #010139 !important;
        font-weight: 600;
        border-bottom: 3px solid #010139 !important;
        background: transparent !important;
    }

    /* Selection widgets */
    div[role="radiogroup"] > label > div:first-child {
        border-color: #010139 !important;
        background-color: #e4ecf7 !important;
    }
    div[role="radiogroup"] > label > div:first-child > div {
        background-color: #010139 !important;
    }
    div[data-baseweb="checkbox"] > div:first-child {
        border-color: #010139 !important;
        background-color: white !important;
    }
    div[data-baseweb="checkbox"] > div:first-child > div {
        background-color: #010139   !important;
    }

    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
</style>
""", unsafe_allow_html=True)

# -------- MANUAL OPTIMIZATION FUNCTION
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def call_route_api_with_optimization(start_coords, waypoint_coords, engine_type, weight, max_speed, is_round_trip=True):
    '''
    Calculate the optimal order by testing different permutations
    Returns: travel_time_sec, length_m, optimized_order, route_geometry
    '''
    if 'optimization_progress' not in st.session_state:
        st.session_state.optimization_progress = 0
    
    waypoint_indices = list(range(len(waypoint_coords)))
    
    
    # --- Helper to construct coordinate list based on trip type ---
    def build_coord_list(start, waypoints_ordered):
        coords = [start] + waypoints_ordered
        if is_round_trip:
            coords.append(start)
        return coords
    # -------------------------------------------------------------

    if len(waypoint_coords) <= 1: 
        best_order = waypoint_indices
        
        # Calculate route
        ordered_waypoints = [waypoint_coords[i] for i in best_order]
        all_coords = build_coord_list(start_coords, ordered_waypoints)
        
        coords_str = ":".join(all_coords)
    
        url_route = f"https://api.tomtom.com/routing/1/calculateRoute/{coords_str}/json"
        params = {
            "key": api_key, 
            'routeType': "fastest", 
            'traffic': 'true', 
            "vehicleEngineType": engine_type,
            "vehicleWeight": int(weight), 
            "vehicleMaxSpeed": int(max_speed),
        }
        
        r_route = requests.get(url_route, params=params)
        data_route = r_route.json()
        
        if "routes" not in data_route or not data_route["routes"]:
            raise ValueError('No route returned from API!')
        
        route = data_route["routes"][0]
        summary = route["summary"]
        best_time = summary['travelTimeInSeconds']
        best_distance = summary['lengthInMeters']
        route_geometry = extract_route_geometry(route)
        
        # optimized order: Start (0) + Waypoints
        optimized_order = [0] + [i+1 for i in best_order]
        # Only add return index if round trip
        if is_round_trip:
            optimized_order.append(len(waypoint_coords) + 1)
        
        return best_time, best_distance, optimized_order, route_geometry
    
    # For multiple waypoints, find the optimal order
    best_time = float('inf')
    best_distance = 0
    best_order = None
    best_route_geometry = None
    
    max_permutations = 24
    permutations_tested = 0
    
    for perm in itertools.permutations(waypoint_indices):
        if permutations_tested >= max_permutations:
            break
        
        st.session_state.optimization_progress = (permutations_tested + 1) / max_permutations
            
        ordered_waypoints = [waypoint_coords[i] for i in perm]
        all_coords = build_coord_list(start_coords, ordered_waypoints)
        coords_str = ":".join(all_coords)
        
        url_route = f"https://api.tomtom.com/routing/1/calculateRoute/{coords_str}/json"
        params = {
            "key": api_key, 
            'routeType': "fastest", 
            'traffic': 'true', 
            "vehicleEngineType": engine_type,
            "vehicleWeight": int(weight), 
            "vehicleMaxSpeed": int(max_speed),
        }
        
        try:
            r_route = requests.get(url_route, params=params)
            data_route = r_route.json()
            
            if "routes" in data_route and data_route["routes"]:
                route = data_route["routes"][0]
                travel_time = route["summary"]['travelTimeInSeconds']
                
                if travel_time < best_time:
                    best_time = travel_time
                    best_distance = route["summary"]['lengthInMeters']
                    best_order = list(perm)
                    best_route_geometry = extract_route_geometry(route)
                    
            permutations_tested += 1
            
        except Exception as e:
            continue
    
    if best_order is None: 
        best_order = waypoint_indices 
        
        ordered_waypoints = [waypoint_coords[i] for i in best_order]
        all_coords = build_coord_list(start_coords, ordered_waypoints)
        coords_str = ":".join(all_coords)
        
        url_route = f"https://api.tomtom.com/routing/1/calculateRoute/{coords_str}/json"
        params = {
            "key": api_key, 'routeType': "fastest", 'traffic': 'true', 
            "vehicleEngineType": engine_type, "vehicleWeight": int(weight), 
            "vehicleMaxSpeed": int(max_speed),
        }
        
        r_route = requests.get(url_route, params=params)
        data_route = r_route.json()
        
        if "routes" in data_route and data_route["routes"]:
            route = data_route["routes"][0]
            best_time = route["summary"]['travelTimeInSeconds']
            best_distance = route["summary"]['lengthInMeters']
            best_route_geometry = extract_route_geometry(route)
    
    # Build optimized order list
    optimized_order = [0] + [i+1 for i in best_order]
    if is_round_trip:
        optimized_order.append(len(waypoint_coords) + 1)
    
    return best_time, best_distance, optimized_order, best_route_geometry

# -------- EXTRACT ROUTE GEOMETRY FOR MAP
def extract_route_geometry(route):
    # Extract the route line coordinates from TomTom API response
    # Returns a list of [lon, lat] coordinates for the entire route
    route_geometry = []
    
    # TomTom API returns route points in 'legs'
    if 'legs' in route:
        for leg in route['legs']:
            if 'points' in leg:
                # points are in format {'latitude': xx, 'longitude': yy}
                for point in leg['points']:
                    route_geometry.append([point['longitude'], point['latitude']])
    
    return route_geometry

# -------- Search API Function
@st.cache_data(ttl=3600)  # Cache for 1 hour
def call_search_api(location_str):
    try:
        encoded = quote(location_str)
        url_search = f"https://api.tomtom.com/search/2/geocode/{encoded}.json"
        params = {"key": api_key}
        
        r_search = requests.get(url_search, params=params) # get the entrys where the entries are the same as in the api
        r_search.raise_for_status() # raise error if request fails
        data_search = r_search.json()

        results = data_search.get('results', []) # store it directly in a list
        if not results:
            raise ValueError(f"No results found for '{location_str}'") # raise error with location if no results found
        
        position = results[0]['position']
        lat = position['lat']
        lon = position['lon']
        
        return lat, lon
    
    except requests.exceptions.Timeout:
        st.error(f"Request timed out for location: {location_str}")
        raise
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        raise
    
    position = results[0]['position'] # select first row
    
    lat = position['lat'] # assign the coordinates
    lon = position['lon']
    
    st.session_state.lat = lat
    st.session_state.lon = lon

    return lat, lon

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Route Specifications'
if 'stopovers' not in st.session_state:

    st.session_state.stopovers = [""]

# -------- SIDEBAR
# We only want to show the sidebar info on non-results pages
if st.session_state.page != "Results": 
    st.sidebar.write('''
        ## About _routefinder_
        Built for truck drivers.
        **Find the safest, fastest, and most efficient route with vehicle-specific insights!**

        ## Resources
        - [TomTom API](https://api.tomtom.com/search/2/search/)
        ''')

    with st.sidebar.expander('About us', expanded=False):
        st.markdown("""
        **Motivation:**
        We saw a lack of Vehicle Specification and stopover optimization in exisitng applications.
                    
        **Usecase:**
        - Planning of truck routes for international logistics.
        - *(works also: Optimal route for dropping friends/familiy off after events.)*

        **Group members:**
        - Patrick Stoffel  
        - Tim Rütsche  
        - Valerie Pieringer  
        - Gloria Tatzer  
        - Nils Stadler
        """)  

# -------- PAGE 1: Route Specifications 
def show_route_specs_page():
    col_left, col_center, col_right = st.columns([1, 2, 1])
    
    with col_center:
        st.title("Route Specifications")
        
        # Add a radio button for trip type
        trip_type = st.radio(
            "Trip Type:", 
            options=["Round Trip (Return to Start)", "One Way"],
            horizontal=True
        )
        
        st.subheader("Starting Point")
        start_loc = st.text_input(
            "Please enter starting point:",
            placeholder="Format: Zeusa 1a, 80-180 Kowale, Polen",
            key="start_input"
        )
        
        st.subheader("Stopovers")
        for i in range(len(st.session_state.stopovers)):
            # Create two columns: Wide for input, narrow for the delete button
            col_input, col_delete = st.columns([7, 1])
            
            with col_input:
                st.session_state.stopovers[i] = st.text_input(
                    f"Stopover {i+1}:",
                    value=st.session_state.stopovers[i],
                    placeholder="Format: Rothusstrasse 88, 3065 Bolligen, Schweiz",
                    key=f"stopover_{i}"
                )
            
            with col_delete:
                # Adds a little spacing so the button aligns with the text box
                st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
                # The Delete Button
                if st.button("Remove", key=f"del_{i}", help="Remove this stopover"):
                    st.session_state.stopovers.pop(i) # Remove from list
                    st.rerun() # Refresh page immediately
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button(" Add Stopover"):
                st.session_state.stopovers.append("")
                st.rerun()
        with col2:
            if st.button("Continue", type="primary"):
                if not start_loc:
                    st.error("Please enter a starting point")
                elif all(s.strip() == "" for s in st.session_state.stopovers):
                    st.error("Please add at least one stopover")
                else:
                    st.session_state.start_loc = start_loc
                
                    st.session_state.is_round_trip = (trip_type == "Round Trip (Return to Start)")
                    st.session_state.via_list = [s for s in st.session_state.stopovers if s.strip()]
                    st.session_state.page = "Vehicle Specifications"
                    st.rerun()

                    
# -------- PAGE 2: Vehicle Specifications
def show_vehicle_specs_page():
    # 1. Create 3 columns again
    col_left, col_center, col_right = st.columns([1, 2, 1])
    
    # 2. Put content in center
    with col_center:
        st.title("Vehicle Specifications")
        
        vehicleEngineType = st.selectbox(
            "Engine Type:",
            ["combustion", "electric", "hydrogen"],
            index=0
        )
        
        # Map Streamlit engine selection to ML categories
        engine_map = {
            "combustion": "diesel",
            "electric": "electric",
            "hydrogen": "hydrogen"
        }

        ml_engine = engine_map[vehicleEngineType]
        st.session_state.ml_engine = ml_engine

        vehicleWeight = st.number_input(
            "Vehicle Weight (kg):",
            min_value=500,
            max_value=50000,
            value=2000,
            step=100
        )
        # Implementing buttons for users to preselct common speeds
        most_common_speed = st.radio(
            "Quick Select Maximum Speed (km/h):",
            options=[80, 90, 100, 130, 150],
            horizontal=True,
            format_func=lambda x: f"{x} km/h"
        )
        
        vehicleMaxSpeed = st.number_input(
            "Maximum Speed (km/h):",
            min_value=20,
            max_value=200,
            value=most_common_speed,
            step=5
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back"):
                st.session_state.page = "Route Specifications"
                st.rerun()
        
        with col2:
            if st.button("Calculate optimized Route", type="primary"):
                st.session_state.vehicleEngineType = vehicleEngineType
                st.session_state.vehicleWeight = vehicleWeight
                st.session_state.vehicleMaxSpeed = vehicleMaxSpeed
                st.session_state.page = "Results"
                st.rerun()

# -------- PAGE 3: Calculation and Results

def show_calculation_page():
    st.markdown("""
        <style>
            .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
                padding-left: 0rem;
                padding-right: 0rem;
            }
            section[data-testid="stSidebar"] {
                width: 400px !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # State for toggling between Map and Full List view
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'map'

    def toggle_view_mode():
        st.session_state.view_mode = 'list' if st.session_state.view_mode == 'map' else 'map'

    # Calculate Route
    with st.spinner("Calculating optimal route..."):
        try:
            # 1. Geocode
            lat_start, lon_start = call_search_api(st.session_state.start_loc)
            start_coords = f"{lat_start},{lon_start}"
            
            stop_names = st.session_state.get("via_list", [])
            stop_coords = []
            for name in stop_names:
                lat, lon = call_search_api(name)
                stop_coords.append(f"{lat},{lon}")
            
            # 2. Get trip type BEFORE optimization
            is_round_trip = st.session_state.get('is_round_trip', True)
            
            # 3. Optimize (ONLY ONCE!)
            travel_time_sec, length_m, optimized_order, route_geometry = call_route_api_with_optimization(
                start_coords, 
                stop_coords, 
                st.session_state.vehicleEngineType,
                st.session_state.vehicleWeight, 
                st.session_state.vehicleMaxSpeed,
                is_round_trip=is_round_trip 
            )
            
            # 4. ML Correction
            from ml_model import predict_corrected_time
            
            corrected_time_min = predict_corrected_time(
                distance_km=length_m / 1000, 
                stops=len(stop_coords),
                weight_kg=st.session_state.vehicleWeight, 
                max_speed=st.session_state.vehicleMaxSpeed,
                engine=st.session_state.ml_engine, 
                elevation_gain=100, 
                traffic=1.0
            )

            # 5. Build Locations
            all_locations = []
            # Add Start
            all_locations.append({"type": "Start", "name": st.session_state.start_loc, "lat": lat_start, "lon": lon_start})
            
            # Add Stopovers
            for i, (name, coord_str) in enumerate(zip(stop_names, stop_coords), 1):
                lat, lon = map(float, coord_str.split(","))
                all_locations.append({"type": f"Stopover", "name": name, "lat": lat, "lon": lon})
            
            # Add Return ONLY if round trip
            if is_round_trip:
                all_locations.append({"type": "Return", "name": st.session_state.start_loc, "lat": lat_start, "lon": lon_start})
            
            def generate_route_csv(all_locations, optimized_order, length_m, corrected_time_min):
                """Generate CSV export of route"""
                df_data = []
                for seq, idx in enumerate(optimized_order, 1):
                    loc = all_locations[idx]
                    df_data.append({
                        'Sequence': seq,
                        'Type': loc['type'],
                        'Location': loc['name'],
                        'Latitude': loc['lat'],
                        'Longitude': loc['lon']
                    })
                
                df = pd.DataFrame(df_data)
                return df
            
            # SIDEBAR
            with st.sidebar:
                # Start New Route Button at top
                if st.button("Start New Route", use_container_width=True, type="primary"):
                    st.session_state.page = "Route Specifications"
                    st.rerun()
                
                st.divider()
                
                # Adding estimated time with ML correction factor
                st.title("Route Summary")
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1C2E4A 0%, #0F1A2B 100%); 
                            padding: 1.5rem; border-radius: 12px; 
                            box-shadow: 0 4px 16px rgba(28, 46, 74, 0.3); 
                            margin-bottom: 1.5rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <h2 style="margin:0; color: #D1CFC9; font-size: 2rem;">{length_m/1000:.1f}</h2>
                            <p style="margin:0; color: #BDC4D4; font-size: 0.85rem; margin-top: 0.25rem;">kilometers</p>
                        </div>
                        <div style="text-align: right;">
                            <h2 style="margin:0; color: #D1CFC9; font-size: 2rem;">{corrected_time_min/60:.2f}</h2>
                            <p style="margin:0; color: #BDC4D4; font-size: 0.85rem; margin-top: 0.25rem;">hours</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Quick Navigation
                st.subheader("Route Details")
                for seq, idx in enumerate(optimized_order, 1):
                    loc = all_locations[idx]
                    icon = "🏁" if loc['type'] == "Start" else "🔄" if loc['type'] == "Return" else "📍"
                    st.markdown(f"{icon} **{seq}.** {loc['name'].split(',')[0]}")
                
                st.divider()
                
                # Export Options
                st.subheader("Export Options")
                col1, col2 = st.columns(2)
                
                df = generate_route_csv(all_locations, optimized_order, length_m, corrected_time_min)
                csv = df.to_csv(index=False)
                
                with col1:
                    st.download_button(
                        label="CSV",
                        data=csv,
                        file_name="route_details.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                route_text = f"Route Summary\nDistance: {length_m/1000:.1f} km\nTime: {corrected_time_min/60:.2f} h\n\n"
                for seq, idx in enumerate(optimized_order, 1):
                    loc = all_locations[idx]
                    route_text += f"{seq}. {loc['type']}: {loc['name']}\n"
                
                with col2:
                    st.download_button(
                        label="TXT",
                        data=route_text,
                        file_name="route_summary.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                st.divider()
                
                # Toggle Button
                btn_label = "⛶ View Full Details" if st.session_state.view_mode == 'map' else "Back to Map"
                st.button(btn_label, on_click=toggle_view_mode, use_container_width=True)

                # Map Controls 
                if st.session_state.view_mode == 'map':
                    st.subheader("Map Layers")
                    style_options = {"Light": "mapbox://styles/mapbox/light-v10", "Dark": "mapbox://styles/mapbox/dark-v10", "Satellite": "mapbox://styles/mapbox/satellite-v9"}
                    selected_style = st.selectbox("Map Style", list(style_options.keys()), label_visibility="collapsed", index=0, placeholder="Select Map Style")
                    show_traffic = st.checkbox("Show Real-time Traffic", value=False)
                    
                    st.divider()
                    st.caption("Quick Navigation")
                    # Mini List for quick clicks
                    for seq, idx in enumerate(optimized_order, 1):
                        loc = all_locations[idx]
                        st.markdown(f"**{seq}.** {loc['name'].split(',')[0]}")

                # New Route Button
                st.divider()
                if st.button("Start New Route", use_container_width=True):
                    st.session_state.page = "Route Specifications"
                    st.rerun()

            # VIEW detailed LIST
            if st.session_state.view_mode == 'list':
                st.markdown("<h1 style='text-align: center; color: #1C2E4A; margin-bottom: 2rem;'>Detailed Itinerary</h1>", unsafe_allow_html=True)
                
                for seq, idx in enumerate(optimized_order, 1):
                    loc = all_locations[idx]
                    
                    # Different colors for different location types
                    if loc['type'] == 'Start':
                        border_color = '#52677D'
                        bg_gradient = 'linear-gradient(135deg, #F8F9FA 0%, #FFFFFF 100%)'
                    elif loc['type'] == 'Return':
                        border_color = '#52677D'
                        bg_gradient = 'linear-gradient(135deg, #F8F9FA 0%, #FFFFFF 100%)'
                    else:
                        border_color = '#BDC4D4'
                        bg_gradient = '#FFFFFF'
                    
                    gmaps_url = f"https://www.google.com/maps/search/?api=1&query={loc['lat']},{loc['lon']}"
                    
                    
                    st.markdown(f"""
                    <div style="background: {bg_gradient}; 
                                padding: 1.5rem; 
                                border-radius: 12px; 
                                margin-bottom: 1rem; 
                                border-left: 5px solid {border_color}; 
                                box-shadow: 0 2px 8px rgba(28, 46, 74, 0.08);
                                transition: all 0.3s ease;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                            <h3 style="margin: 0; color: #1C2E4A; font-size: 1.25rem;">
                                {seq}. {loc['type']}
                            </h3>
                            <a href="{gmaps_url}" 
                               target="_blank" 
                               style="text-decoration: none; 
                                      font-weight: 600; 
                                      background: linear-gradient(135deg, #52677D 0%, #1C2E4A 100%); 
                                      color: #D1CFC9; 
                                      padding: 0.5rem 1rem; 
                                      border-radius: 8px; 
                                      font-size: 0.9rem;
                                      transition: all 0.3s ease;">
                                View on Map →
                            </a>
                        </div>
                        <p style="margin: 0; color: #52677D; font-size: 1rem; line-height: 1.5;">
                            {loc['name']}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)


            # VIEW B: FULL SCREEN MAP 
            else:
                # Map controls in top right
                col_map, col_controls = st.columns([5, 1])
                
                with col_controls:
                    st.markdown("### Map Settings")
                    style_options = {
                        "Light": "mapbox://styles/mapbox/light-v10", 
                        "Dark": "mapbox://styles/mapbox/dark-v10", 
                        "Satellite": "mapbox://styles/mapbox/satellite-v9"
                    }
                    selected_style = st.selectbox(
                        "Map Style", 
                        list(style_options.keys()), 
                        index=0
                    )
                # Prepare Map Data
                map_points = []
                seen_coords = {}
                
                for seq, idx in enumerate(optimized_order, 1):
                    loc = all_locations[idx]
                    coord_key = f"{loc['lat']},{loc['lon']}"
                    
                    # Offset logic to separate overlapping points (Start/Return)
                    lat_offset = 0
                    lon_offset = 0
                    if coord_key in seen_coords:
                        seen_coords[coord_key] += 1
                        offset_factor = seen_coords[coord_key]
                        lat_offset = 0.0003 * offset_factor 
                        lon_offset = 0.0003 * offset_factor
                    else:
                        seen_coords[coord_key] = 0
                        
                    map_points.append({
                        "lon": loc["lon"] + lon_offset, 
                        "lat": loc["lat"] + lat_offset, 
                        "name": loc['name'], "type": loc['type'], 
                        "order": str(seq)
                    })
                
                if map_points:
                    avg_lat = sum(p["lat"] for p in map_points) / len(map_points)
                    avg_lon = sum(p["lon"] for p in map_points) / len(map_points)

                layers = []
                
                # Traffic
                if show_traffic:
                    traffic_layer = pdk.Layer(
                        "TileLayer", data=None,
                        get_tile_data=f"https://api.tomtom.com/traffic/map/4/tile/flow/relative-delay/{{z}}/{{x}}/{{y}}.png?key={api_key}",
                        min_zoom=0, max_zoom=22, tileSize=256, opacity=0.9
                    )
                    layers.append(traffic_layer)

                # Route Line
                if selected_style in ["Satellite", "Dark"]:
                     line_color = [2, 180, 255, 200] 
                     point_fill = [255, 255, 255, 255]
                     point_line = [255, 255, 255, 255] 
                     text_color = [0, 0, 0] 
                else:
                    line_color = [1, 1, 57, 200]      
                    point_fill = [1, 1, 57, 255]      
                    point_line = [255, 255, 255, 255] 
                    text_color = [255, 255, 255]
                     
                if route_geometry:
                    layers.append(pdk.Layer(
                        "PathLayer", data=[{"path": route_geometry}],
                        get_path="path", get_color=[255, 255, 255, 120],
                        width_min_pixels=6, get_width=10, pickable=True
                    ))
                    # Main Line
                    layers.append(pdk.Layer(
                        "PathLayer", data=[{"path": route_geometry}],
                        get_path="path", get_color=line_color,
                        width_min_pixels=3, get_width=6, pickable=True
                    ))

                    
                # Points
                layers.append(pdk.Layer(
                    "ScatterplotLayer", data=map_points,
                    get_position="[lon, lat]", 
                    get_radius=100,
                    radius_scale=1,
                    radius_min_pixels=15,
                    radius_max_pixels=40,      
                    get_fill_color=point_fill, 
                    get_line_color=point_line, 
                    get_line_width=30,
                    line_width_min_pixels=2,
                    pickable=True,
                    auto_highlight=True
                ))
                
                # Numbers
                layers.append(pdk.Layer(
                    "TextLayer", data=map_points,
                    get_position="[lon, lat]", get_text="order",
                    get_size=18, size_min_pixels=16, get_color=text_color,
                    get_text_anchor="'middle'",
                    get_alignment_baseline="'center'",
                    font_family="'Oswald', sans-serif",
                    font_weight=700,
                    get_background_color=[28, 46, 74, 220], background_padding=[5,5]
                ))
                
                

                deck = pdk.Deck(
                    map_style=style_options[selected_style],
                    initial_view_state=pdk.ViewState(latitude=avg_lat, longitude=avg_lon, zoom=11, pitch=0, transition_duration=1000, controller=True),
                    layers=layers,
                    tooltip={"html": "<div style='background: #010139; color: white; padding: 10px; border-radius: 5px; border: 1px solid #02b4ff;'><b>Stop {order}</b><br/>{name}</div>"
                    }
                )
                    
                # Full height map
                st.pydeck_chart(deck, use_container_width=True, height=1000)

        except Exception as e:
            st.error(f"Error: {e}")
            if st.button("Go Back"):
                st.session_state.page = "Vehicle Specifications"
                st.rerun()
                
# -------- Main App Logic
if st.session_state.page == "Route Specifications":
    show_route_specs_page()
elif st.session_state.page == "Vehicle Specifications":
    show_vehicle_specs_page()
elif st.session_state.page == "Results":
    show_calculation_page()