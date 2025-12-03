import streamlit as st
import requests
import pandas as pd 
import pydeck as pdk
from urllib.parse import quote
import itertools

api_key = 'jCj3v16W0cJMzI4m2GtkQ32lwmkmUVki'

# -------- MANUAL OPTIMIZATION FUNCTION
def call_route_api_with_optimization(start_coords, waypoint_coords, engine_type, weight, max_speed):
    '''
    Calculate the optimal order by testing different permutations
    Returns: travel_time_sec, length_m, optimized_order, route_geometry
    '''

    waypoint_indices = list(range(len(waypoint_coords)))
    
    if len(waypoint_coords) <= 1: # no optimization needed for 0 or 1 waypoints
        best_order = waypoint_indices
        best_time = 0
        best_distance = 0
        
        # just calculate the route
        all_coords = [start_coords] + [waypoint_coords[i] for i in best_order] + [start_coords] # make sure that it is a roundtrip by adding start_coords in the end
        coords_str = ":".join(all_coords)
    
        url_route = f"https://api.tomtom.com/routing/1/calculateRoute/{coords_str}/json" # call tomtom route API
        params = {
            "key": api_key, 
            'routeType': "fastest",
            'traffic': 'true', 
            "vehicleEngineType": engine_type,
            "vehicleWeight": int(weight), 
            "vehicleMaxSpeed": int(max_speed),
        }
        
        r_route = requests.get(url_route, params=params) # select the entry, where the key of params is the same as the real entry in the api
        data_route = r_route.json()
        
        if "routes" not in data_route or not data_route["routes"]: # error message if there is no route returned from tomtom
            raise ValueError('No route returned from API!')
        
        #extract route summary
        route = data_route["routes"][0]
        summary = route["summary"]
        best_time = summary['travelTimeInSeconds']
        best_distance = summary['lengthInMeters']
        
        route_geometry = extract_route_geometry(route) # get route geometry for map
        
        # general optimized order start + waypoints in original order + return
        optimized_order = [0] + [i+1 for i in best_order] + [len(waypoint_coords) + 1]
        
        return best_time, best_distance, optimized_order, route_geometry
    
    # For multiple waypoints, find the optimal order
    best_time = float('inf')
    best_distance = 0
    best_order = None
    best_route_geometry = None
    
    # Try different permutations (limit to avoid too many API calls)
    max_permutations = 24
    permutations_tested = 0
    
    # if there are more than 1 stopover -> test differnet waypoint orders
    for perm in itertools.permutations(waypoint_indices):
        if permutations_tested >= max_permutations: # safety if permutations bigger than max permutations -> break the loop
            break
            
        # build coordinates in the format tomtom requires: start + waypoints in this permutation + return to start
        all_coords = [start_coords] + [waypoint_coords[i] for i in perm] + [start_coords]
        coords_str = ":".join(all_coords)
        
        # call API
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
            
            if "routes" in data_route and data_route["routes"]: # check if the route even exists
                route = data_route["routes"][0]
                travel_time = route["summary"]['travelTimeInSeconds']
                
                if travel_time < best_time: # determine the fastest route
                    best_time = travel_time
                    best_distance = route["summary"]['lengthInMeters']
                    best_order = list(perm)
                    best_route_geometry = extract_route_geometry(route)
                    
            permutations_tested += 1 # count the api calls
            
        except Exception as e:
            st.warning(f"Error testing permutation: {e}")
            continue
    
    if best_order is None: 
        best_order = waypoint_indices # fallback: use original order
        
        # Calculate route with original order
        all_coords = [start_coords] + [waypoint_coords[i] for i in best_order] + [start_coords]
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
        
        if "routes" in data_route and data_route["routes"]:
            route = data_route["routes"][0]
            best_time = route["summary"]['travelTimeInSeconds']
            best_distance = route["summary"]['lengthInMeters']
            best_route_geometry = extract_route_geometry(route)
    
    # Build optimized order list: start (0), waypoints, return (last)
    optimized_order = [0] + [i+1 for i in best_order] + [len(waypoint_coords) + 1]
    
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
def call_search_api(location_str):
    encoded = quote(location_str)
    url_search = f"https://api.tomtom.com/search/2/geocode/{encoded}.json"
    params = {"key": api_key}
    
    r_search = requests.get(url_search, params=params) # get the entrys where the entries are the same as in the api
    data_search = r_search.json()

    results = data_search.get('results', []) # store it directly in a list
    if not results:
        raise ValueError(f"No results found for '{location_str}'") # raise error with location

    position = results[0]['position'] # select first row
    lat = position['lat'] # assign the coordinates
    lon = position['lon']
    
    st.session_state.lat = lat
    st.session_state.lon = lon

    return lat, lon

# -------- CONFIGURATIONS
st.set_page_config(
    page_title="Route Planner",
    layout="centered",
)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Route Specifications'
if 'stopovers' not in st.session_state:
    st.session_state.stopovers = [""]

# -------- SIDEBAR
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
    relevance, story, etc.

    **Group members:**
    - Patrick Stoffel  
    - Tim Rütsche  
    - Valerie Pieringer  
    - Gloria Tatzer  
    - Nils Stadler
    """)

# -------- PAGE 1: Route Specifications
def show_route_specs_page():
    st.title("Route Specifications")
    
    st.subheader("Starting Point")
    start_loc = st.text_input(
        "Please enter starting point:",
        placeholder="Format: Zeusa 1a, 80-180 Kowale, Polen",
        key="start_input"
    )
    
    st.subheader("Stopovers")
    for i in range(len(st.session_state.stopovers)):
        st.session_state.stopovers[i] = st.text_input(
            f"Please enter Stopover {i+1}:",
            value=st.session_state.stopovers[i],
            placeholder="Format: Rothusstrasse 88, 3065 Bolligen, Schweiz",
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
            else:
                st.session_state.start_loc = start_loc
                st.session_state.via_list = [s for s in st.session_state.stopovers if s.strip()]
                st.session_state.page = "Vehicle Specifications"
                st.rerun()

# -------- PAGE 2: Vehicle Specifications
def show_vehicle_specs_page():
    st.title("Vehicle Specifications")
    
    vehicleEngineType = st.selectbox(
        "Engine Type:",
        ["combustion", "electric", "hydrogen"],
        index=0
    )
    
    vehicleWeight = st.number_input(
        "Vehicle Weight (kg):",
        min_value=500,
        max_value=50000,
        value=2000,
        step=100
    )
    
    vehicleMaxSpeed = st.number_input(
        "Maximum Speed (km/h):",
        min_value=20,
        max_value=200,
        value=100,
        step=5
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back"):
            st.session_state.page = "Route Specifications"
            st.rerun()
    
    with col2:
        if st.button("Calculate OPTIMIZED Route", type="primary"):
            st.session_state.vehicleEngineType = vehicleEngineType
            st.session_state.vehicleWeight = vehicleWeight
            st.session_state.vehicleMaxSpeed = vehicleMaxSpeed
            st.session_state.page = "Results"
            st.rerun()

# -------- PAGE 3: Results
def show_calculation_page():
    st.title("Route Results - OPTIMIZED ORDER")
    
    # Create two tabs
    tab1, tab2 = st.tabs(["Route Details", "Map"])
    
    with st.spinner("Calculating optimal route... This may take a moment for multiple stops."): # show this spinner while calculating
        try:
            # Get coordinates for all locations
            lat_start, lon_start = call_search_api(st.session_state.start_loc)
            start_coords = f"{lat_start},{lon_start}"
            
            stop_names = st.session_state.get("via_list", [])
            stop_coords = []
            
            for name in stop_names:
                lat, lon = call_search_api(name)
                stop_coords.append(f"{lat},{lon}")
            
            # fill in the values intop the function and calculate
            travel_time_sec, length_m, optimized_order, route_geometry = call_route_api_with_optimization(
                start_coords,
                stop_coords,
                st.session_state.vehicleEngineType,
                st.session_state.vehicleWeight,
                st.session_state.vehicleMaxSpeed
            )
            
            # Store route geometry in session state for map
            st.session_state.route_geometry = route_geometry
            
            # Prepare data for display
            all_locations = []
            
            # Add start
            all_locations.append({
                "type": "Start",
                "name": st.session_state.start_loc,
                "lat": lat_start,
                "lon": lon_start
            })
            
            # Add waypoints
            for i, (name, coord_str) in enumerate(zip(stop_names, stop_coords), 1):
                lat, lon = map(float, coord_str.split(","))
                all_locations.append({
                    "type": f"Stopover",
                    "name": name,
                    "lat": lat,
                    "lon": lon
                })
            
            # Add return to start
            if stop_names:
                all_locations.append({
                    "type": "Return to Start",
                    "name": st.session_state.start_loc,
                    "lat": lat_start,
                    "lon": lon_start
                })
            
            # TAB 1: Route Details
            with tab1:
                st.subheader("Route Details")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Distance", f"{length_m/1000:.1f} km")
                with col2:
                    st.metric("Travel Time", f"{travel_time_sec/3600:.1f} hours")
                
                st.subheader("Optimized Route Order")
                st.info("Order has been optimized for shortest travel time")
                
                # Create a nice table
                data = []
                for seq, idx in enumerate(optimized_order, 1):
                    loc = all_locations[idx]
                    data.append({
                        "Order": seq,
                        "Type": loc["type"],
                        "Location": loc["name"],
                        "Coordinates": f"{loc['lat']:.4f}, {loc['lon']:.4f}"
                    })
                
                # Display in simple format
                st.subheader("Route Details")
                for seq, idx in enumerate(optimized_order, 1):
                    loc = all_locations[idx]
                    st.write(f"**{seq}. {loc['type']}**")
                    st.write(f"   {loc['name']}")
                
                # Comparison with original order
                if len(stop_names) > 1:
                    st.subheader("Optimization Benefit")
                    st.write(f"Route optimized for {len(stop_names)} stopovers")
                    st.write("The order shown above minimizes total travel time.")
            
            # TAB 2: Interactive Map
            with tab2:
                st.subheader("Route Map with Road Visualization")
                
                # Prepare map points in optimized order
                map_points = []
                for seq, idx in enumerate(optimized_order, 1):
                    loc = all_locations[idx]
                    map_points.append({
                        "lon": loc["lon"],
                        "lat": loc["lat"],
                        "name": f"{seq}. {loc['type']}: {loc['name'][:50]}...",
                        "order": str(seq)
                    })
                
                if map_points:
                    # Calculate map center
                    avg_lat = sum(p["lat"] for p in map_points) / len(map_points)
                    avg_lon = sum(p["lon"] for p in map_points) / len(map_points)
                    
                    # Create layers for the map
                    layers = []
                    
                    # 1. Add the ROUTE LINE if we have geometry data
                    if route_geometry and len(route_geometry) > 1:
                        route_line_layer = pdk.Layer(
                            "PathLayer",
                            data=[{
                                "path": route_geometry,
                                "name": "Optimized Route"
                            }],
                            get_path="path",
                            get_color=[0, 100, 255, 200],  # Blue color for route
                            width_min_pixels=3,
                            pickable=True,
                            get_width=5,
                        )
                        layers.append(route_line_layer)
                    
                    # 2. Add stop points
                    points_layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=map_points,
                        get_position="[lon, lat]",
                        get_radius=20000,
                        get_fill_color=[255, 0, 0, 180],  # Red for stops
                        pickable=True,
                    )
                    layers.append(points_layer)
                    
                    # 3. Add numbers on points
                    text_layer = pdk.Layer(
                        "TextLayer",
                        data=map_points,
                        get_position="[lon, lat]",
                        get_text="order",
                        get_size=16,
                        get_color=[255, 255, 255],
                        get_background_color=[255, 0, 0, 200],  # Red background
                        background_padding=[4, 4],
                    )
                    layers.append(text_layer)
                    
                    # Create deck
                    deck = pdk.Deck(
                        map_style="mapbox://styles/mapbox/light-v9",
                        initial_view_state=pdk.ViewState(
                            latitude=avg_lat,
                            longitude=avg_lon,
                            zoom=5,
                            pitch=0,
                        ),
                        layers=layers,
                        tooltip={"text": "{name}"},
                    )
                    
                    st.pydeck_chart(deck)
            
            # Back button at the bottom (outside tabs)
            st.divider()
            if st.button("Back to Vehicle Specifications"):
                st.session_state.page = "Vehicle Specifications"
                st.rerun()
                
        except Exception as e:
            st.error(f"Error calculating route: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
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