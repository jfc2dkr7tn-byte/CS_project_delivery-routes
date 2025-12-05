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
        color: #010139 !important, ;
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
            return position['lat'], position['lon']
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
            placeholder="e.g.: Zeusa 1a, 80-180 Kowale, Polen",
            key="start_input"
        )
        
        st.subheader("Stopovers")
        for i in range(len(st.session_state.stopovers)):
            st.session_state.stopovers[i] = st.text_input(
                f"Please enter Stopover {i+1}:",
                value=st.session_state.stopovers[i],
                placeholder="e.g.:Rothusstrasse 88, 3065 Bolligen, Schweiz",
                key=f"stopover_{i}"
            )
        
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
                padding-left: 1rem;
                padding-right: 1rem;
            }
            section[data-testid="stSidebar"] {
                width: 400px !important;
            }
            /* Change default text color to blue */
            .stApp {
                color: #0F1A2B;
            }
            p, span, div {
                color: #0F1A2B;
            }
            /* Download button text color */
            .stDownloadButton button {
                color: #0F1A2B !important;
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
            
            # 2. Optimize
            # Retrieve the setting (default to True if not set)
            is_round_trip = st.session_state.get('is_round_trip', True)
            
            travel_time_sec, length_m, optimized_order, route_geometry = call_route_api_with_optimization(
                start_coords, stop_coords, st.session_state.vehicleEngineType,
                st.session_state.vehicleWeight, st.session_state.vehicleMaxSpeed,
                is_round_trip=is_round_trip 
            )
            
            # 3. ML Correction
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

            # 4. Build Locations
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
            
            # Implement Sidebar for Results
            with st.sidebar:
                # A. Start New Route Button
                if st.button("Start New Route", use_container_width=True, type="primary"):
                    st.session_state.page = "Route Specifications"
                    st.rerun()
                
                st.divider()
                
                # B. Route Summary
                st.title("Results")
                
                # Adding estimated time with ML correction factor
                st.markdown(f"""
                <div style="background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.2); margin-bottom: 20px;">
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <h2 style="margin:0; color: white;">{length_m/1000:.1f} <span style="font-size:0.6em">km</span></h2>
                            <p style="margin:0; color: #cbd5e1; font-size: 0.8em;">Distance</p>
                        </div>
                        <div style="text-align: right;">
                            <h2 style="margin:0; color: white;">{corrected_time_min/60:.2f} <span style="font-size:0.6em">h</span></h2>
                            <p style="margin:0; color: #cbd5e1; font-size: 0.8em;">Est. Time</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # C. Quick Navigation
                st.caption("Quick Navigation")
                # Mini List for quick clicks
                for seq, idx in enumerate(optimized_order, 1):
                    loc = all_locations[idx]
                    st.markdown(f"**{seq}.** {loc['name'].split(',')[0]}")
                
                st.divider()
                
                # D. Export Options (Side by Side)
                st.subheader("Export Options")
                col1, col2 = st.columns(2)
                
                df = generate_route_csv(all_locations, optimized_order, length_m, corrected_time_min)
                csv = df.to_csv(index=False)
                
                with col1:
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="route_details.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                route_text = f"Route Summary\nTotal Distance: {length_m/1000:.1f} km\nEstimated Time: {corrected_time_min/60:.2f} h\n\n"
                for seq, idx in enumerate(optimized_order, 1):
                    loc = all_locations[idx]
                    route_text += f"{seq}. {loc['type']}: {loc['name']}\n"
                
                with col2:
                    st.download_button(
                        label="Download TXT",
                        data=route_text,
                        file_name="route_summary.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                st.divider()
                
                # E. View Toggle Button
                btn_label = "View Full Details on Route" if st.session_state.view_mode == 'map' else "Back to Map"
                st.button(btn_label, on_click=toggle_view_mode, type="secondary", use_container_width=True)

            # 
            
            # VIEW A: Detailed LIST
            if st.session_state.view_mode == 'list':
                st.header("Itinerary")
                
                # Detailed Cards
                for seq, idx in enumerate(optimized_order, 1):
                    loc = all_locations[idx]
                    border_color = '#0F1A2B' if loc['type'] in ['Start', 'Return'] else '#02b4ff'
                    gmaps_url = f"https://www.google.com/maps/search/?api=1&query={loc['lat']},{loc['lon']}"
                    
                    st.markdown(f"""
                    <div style="padding: 15px; background-color: white; border-radius: 8px; margin-bottom: 12px; border-left: 6px solid {border_color}; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h4 style="margin:0; color:#0F1A2B;">{seq}. {loc['type']}</h4>
                            <a href="{gmaps_url}" target="_blank" style="text-decoration:none; font-weight:bold; background:#D1CFC9; color:#0F1A2B; padding:6px 12px; border-radius:6px; border:1px solid #0F1A2B;">Address</a>
                        </div>
                        <p style="margin:5px 0 0 0; color: #0F1A2B; font-size:1.1em;">{loc['name']}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # VIEW B: FULL SCREEN MAP 
            else:
                # Map Controls in Top Right Corner
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
                for seq, idx in enumerate(optimized_order, 1):
                    loc = all_locations[idx]
                    map_points.append({
                        "lon": loc["lon"], "lat": loc["lat"], 
                        "name": loc['name'], "type": loc['type'], 
                        "order": str(seq)
                    })
                
                if map_points:
                    avg_lat = sum(p["lat"] for p in map_points) / len(map_points)
                    avg_lon = sum(p["lon"] for p in map_points) / len(map_points)

                layers = []

                # Route Line
                if selected_style in ["Satellite", "Dark"]:
                     line_color = [2, 180, 255, 200]
                     point_fill = [255, 255, 255, 255] 
                     point_line = [2, 180, 255, 255]
                else:
                     line_color = [15, 26, 43, 200]
                     point_fill = [15, 26, 43, 255]
                     point_line = [2, 180, 255, 255]
                     
                if route_geometry:
                    layers.append(pdk.Layer(
                        "PathLayer", data=[{"path": route_geometry}],
                        get_path="path", get_color=[0, 0, 0, 150],
                        width_min_pixels=6, get_width=15, pickable=True
                    ))
                    # Main Line
                    layers.append(pdk.Layer(
                        "PathLayer", data=[{"path": route_geometry}],
                        get_path="path", get_color=line_color,
                        width_min_pixels=3, get_width=8, pickable=True
                    ))
                    
                # Points
                layers.append(pdk.Layer(
                    "ScatterplotLayer", data=map_points,
                    get_position="[lon, lat]", 
                    get_radius=150,        
                    get_fill_color=point_fill, 
                    get_line_color=point_line, 
                    get_line_width=40,       
                    pickable=True,
                    auto_highlight=True
                ))
                
                # Numbers
                layers.append(pdk.Layer(
                    "TextLayer", data=map_points,
                    get_position="[lon, lat]", get_text="order",
                    get_size=16, get_color=[255, 255, 255],
                    get_background_color=[15, 26, 43, 200], background_padding=[4,4]
                ))

                deck = pdk.Deck(
                    map_style=style_options[selected_style],
                    initial_view_state=pdk.ViewState(latitude=avg_lat, longitude=avg_lon, zoom=11, pitch=0, transition_duration=1000, controller=True),
                    layers=layers,
                    tooltip={"html": "<div style='background: #0F1A2B; color: white; padding: 10px; border-radius: 5px; border: 1px solid #02b4ff;'><b>Stop {order}</b><br/>{name}</div>"
                    }
                )
                    
                # Full height map
                st.pydeck_chart(deck, use_container_width=True, height=800)

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