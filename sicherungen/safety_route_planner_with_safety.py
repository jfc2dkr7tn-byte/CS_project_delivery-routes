import streamlit as st
import requests
import pandas as pd 
import pydeck as pdk
from urllib.parse import quote
import itertools
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

api_key = 'jCj3v16W0cJMzI4m2GtkQ32lwmkmUVki'

# -------- ML MODEL FUNCTIONS --------
@st.cache_resource
def load_or_train_model():
    """Load or train the accident prediction model"""
    model_path = 'accident_model.joblib'
    encoders_path = 'label_encoders.joblib'
    scaler_path = 'scaler.joblib'
    
    # Try to load existing model
    if os.path.exists(model_path) and os.path.exists(encoders_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            label_encoders = joblib.load(encoders_path)
            scaler = joblib.load(scaler_path)
            return model, label_encoders, scaler
        except:
            st.warning("Could not load saved model. Training new model...")
    
    # Train new model
    try:
        df = pd.read_csv('cleaned_accident_data.csv')
    except:
        st.error("Could not load accident data. Using fallback data.")
        # Create fallback dataset
        data = {
            'Weather': ['Clear']*30 + ['Rainy']*25 + ['Snowy']*20 + ['Foggy']*15 + ['Stormy']*10,
            'Road_Type': ['City Road']*25 + ['Highway']*25 + ['Rural Road']*25 + ['Mountain Road']*25,
            'Time_of_Day': ['Morning']*25 + ['Afternoon']*25 + ['Evening']*25 + ['Night']*25,
            'Traffic_Density': np.random.uniform(0, 2, 100),
            'Speed_Limit': np.random.randint(30, 120, 100),
            'Accident': [0]*70 + [1]*30  # 30% accident rate
        }
        df = pd.DataFrame(data)
    
    # Prepare features
    categorical_cols = ['Weather', 'Road_Type', 'Time_of_Day']
    numerical_cols = ['Traffic_Density', 'Speed_Limit']
    
    # Encode categorical variables
    label_encoders = {}
    X_encoded = pd.DataFrame()
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Add numerical columns
    for col in numerical_cols:
        X_encoded[col] = df[col]
    
    # Handle missing values
    X_encoded = X_encoded.fillna(X_encoded.median())
    
    # Target variable
    y = df['Accident']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_scaled, y)
    
    # Save model
    joblib.dump(model, model_path)
    joblib.dump(label_encoders, encoders_path)
    joblib.dump(scaler, scaler_path)
    
    return model, label_encoders, scaler

def predict_accident_risk(weather, road_type, time_of_day, traffic_density, speed_limit):
    """Predict accident risk based on conditions"""
    # Load model
    model, label_encoders, scaler = load_or_train_model()
    
    # Encode features
    features = []
    for col in ['Weather', 'Road_Type', 'Time_of_Day']:
        le = label_encoders[col]
        value = weather if col == 'Weather' else road_type if col == 'Road_Type' else time_of_day
        
        # Handle unseen categories
        if value in le.classes_:
            features.append(le.transform([value])[0])
        else:
            # Use most common category as fallback
            features.append(0)
    
    # Add numerical features
    features.append(traffic_density)
    features.append(speed_limit)
    
    # Scale and predict
    features_scaled = scaler.transform([features])
    probability = model.predict_proba(features_scaled)[0][1]
    
    return probability

# -------- ROUTE OPTIMIZATION FUNCTIONS --------
def call_route_api_with_optimization(start_coords, waypoint_coords, engine_type, weight, max_speed):
    '''Calculate the optimal order by testing different permutations'''
    
    waypoint_indices = list(range(len(waypoint_coords)))
    
    if len(waypoint_coords) <= 1: # no optimization needed for 0 or 1 waypoints
        best_order = waypoint_indices
        best_time = 0
        best_distance = 0
        
        # just calculate the route
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
        
        if "routes" not in data_route or not data_route["routes"]:
            raise ValueError('No route returned from API!')
        
        #extract route summary
        route = data_route["routes"][0]
        summary = route["summary"]
        best_time = summary['travelTimeInSeconds']
        best_distance = summary['lengthInMeters']
        
        route_geometry = extract_route_geometry(route)
        
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
    
    for perm in itertools.permutations(waypoint_indices):
        if permutations_tested >= max_permutations:
            break
            
        # build coordinates
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
            st.warning(f"Error testing permutation: {e}")
            continue
    
    if best_order is None: 
        best_order = waypoint_indices
        
        # Calculate route with original order
        all_coords = [start_coords] + [waypoint_coords[i] for i in best_order] + [start_coords]
        coords_str = ":".join(all_coords)
        
        url_route = f"https://api.tomTom.com/routing/1/calculateRoute/{coords_str}/json"
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
    
    # Build optimized order list
    optimized_order = [0] + [i+1 for i in best_order] + [len(waypoint_coords) + 1]
    
    return best_time, best_distance, optimized_order, best_route_geometry

def extract_route_geometry(route):
    """Extract the route line coordinates from TomTom API response"""
    route_geometry = []
    
    if 'legs' in route:
        for leg in route['legs']:
            if 'points' in leg:
                for point in leg['points']:
                    route_geometry.append([point['longitude'], point['latitude']])
    
    return route_geometry

def call_search_api(location_str):
    """Get coordinates for a location"""
    encoded = quote(location_str)
    url_search = f"https://api.tomtom.com/search/2/geocode/{encoded}.json"
    params = {"key": api_key}
    
    r_search = requests.get(url_search, params=params)
    data_search = r_search.json()

    results = data_search.get('results', [])
    if not results:
        raise ValueError(f"No results found for '{location_str}'")

    position = results[0]['position']
    lat = position['lat']
    lon = position['lon']
    
    return lat, lon

def estimate_route_safety(weather, time_of_day, route_length_km, has_mountain_roads=False):
    """Estimate safety score for a route"""
    
    # Base safety score
    safety_score = 75
    
    # Weather adjustments
    weather_penalties = {
        'Clear': 0,
        'Rainy': -15,
        'Snowy': -25,
        'Foggy': -20,
        'Stormy': -30
    }
    safety_score += weather_penalties.get(weather, 0)  # Fixed: was weather_condition
    
    # Time of day adjustments
    time_penalties = {
        'Morning': -5,
        'Afternoon': 0,
        'Evening': -10,
        'Night': -20
    }
    safety_score += time_penalties.get(time_of_day, 0)
    
    # Route length penalty (longer routes = slightly more risk)
    length_penalty = min(route_length_km / 100, 10)  # Max 10% penalty
    safety_score -= length_penalty
    
    # Mountain road penalty
    if has_mountain_roads:
        safety_score -= 15
    
    # Ensure score is between 0 and 100
    safety_score = max(0, min(100, safety_score))
    
    # Get ML predictions for different road types
    road_types = ['Highway', 'City Road', 'Rural Road', 'Mountain Road']
    road_safety = {}
    
    for road_type in road_types:
        if road_type == 'Mountain Road' and not has_mountain_roads:
            continue
            
        # Estimate traffic density based on time and road type
        base_density = 1.0
        if time_of_day in ['Morning', 'Evening']:
            base_density = 1.8 if road_type == 'City Road' else 1.5
        elif time_of_day == 'Afternoon':
            base_density = 1.5 if road_type == 'City Road' else 1.2
        else:  # Night
            base_density = 0.8
        
        # Estimate speed limit
        speed_limits = {
            'Highway': 120,
            'City Road': 50,
            'Rural Road': 80,
            'Mountain Road': 60
        }
        
        # Get ML prediction - Fixed: using parameter 'weather' not 'weather_condition'
        risk = predict_accident_risk(
            weather=weather,  # Fixed: was weather_condition
            road_type=road_type,
            time_of_day=time_of_day,
            traffic_density=base_density,
            speed_limit=speed_limits.get(road_type, 80)
        )
        
        road_safety[road_type] = (1 - risk) * 100
    
    return safety_score, road_safety

# -------- STREAMLIT UI --------
st.set_page_config(
    page_title="Route Planner with Safety",
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
    **Route Safety Features:**
    - Machine Learning accident prediction
    - Weather-aware routing
    - Time-based risk assessment
    - Road-type specific safety scores

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
    
    st.subheader("Vehicle Details")
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
    
    st.subheader("Route Safety Parameters")
    col1, col2 = st.columns(2)
    with col1:
        weather_condition = st.selectbox(
            "Weather Condition:",
            ["Clear", "Rainy", "Snowy", "Foggy", "Stormy"],
            index=0
        )
    
    with col2:
        time_of_day = st.selectbox(
            "Time of Travel:",
            ["Morning", "Afternoon", "Evening", "Night"],
            index=1
        )
    
    has_mountain_roads = st.checkbox("Route includes mountain roads", value=False)
    
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
            st.session_state.weather_condition = weather_condition
            st.session_state.time_of_day = time_of_day
            st.session_state.has_mountain_roads = has_mountain_roads
            st.session_state.page = "Results"
            st.rerun()

# -------- PAGE 3: Results
def show_calculation_page():
    st.title("Route Results - OPTIMIZED ORDER")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Route Details", "Safety Analysis", "Map"])
    
    with st.spinner("Calculating optimal route with safety assessment..."):
        try:
            # Get coordinates
            lat_start, lon_start = call_search_api(st.session_state.start_loc)
            start_coords = f"{lat_start},{lon_start}"
            
            stop_names = st.session_state.get("via_list", [])
            stop_coords = []
            
            for name in stop_names:
                lat, lon = call_search_api(name)
                stop_coords.append(f"{lat},{lon}")
            
            # Calculate route
            travel_time_sec, length_m, optimized_order, route_geometry = call_route_api_with_optimization(
                start_coords,
                stop_coords,
                st.session_state.vehicleEngineType,
                st.session_state.vehicleWeight,
                st.session_state.vehicleMaxSpeed
            )
            
            # Store route geometry
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
            
            # Estimate safety score
            route_length_km = length_m / 1000
            weather_condition = st.session_state.weather_condition  # Store in local variable
            safety_score, road_safety = estimate_route_safety(
                weather_condition,  # Pass the weather condition
                st.session_state.time_of_day,
                route_length_km,
                st.session_state.has_mountain_roads
            )
            
            # TAB 1: Route Details
            with tab1:
                st.subheader("Route Details")
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Distance", f"{route_length_km:.1f} km")
                with col2:
                    hours = travel_time_sec / 3600
                    minutes = (hours % 1) * 60
                    st.metric("Travel Time", f"{int(hours)}h {int(minutes)}min")
                with col3:
                    # Color-coded safety score
                    if safety_score >= 80:
                        st.metric("Safety Score", f"{safety_score:.0f}%", delta="Excellent", 
                                 delta_color="normal")
                    elif safety_score >= 60:
                        st.metric("Safety Score", f"{safety_score:.0f}%", delta="Good", 
                                 delta_color="normal")
                    elif safety_score >= 40:
                        st.metric("Safety Score", f"{safety_score:.0f}%", delta="Moderate", 
                                 delta_color="off")
                    else:
                        st.metric("Safety Score", f"{safety_score:.0f}%", delta="Low", 
                                 delta_color="inverse")
                
                st.subheader("Optimized Route Order")
                for seq, idx in enumerate(optimized_order, 1):
                    loc = all_locations[idx]
                    st.write(f"**{seq}. {loc['type']}**")
                    st.write(f"   {loc['name']}")
                
                # Route summary
                st.subheader("Route Summary")
                summary_data = {
                    "Parameter": ["Weather", "Time of Day", "Vehicle Type", "Route Length", "Number of Stops"],
                    "Value": [
                        weather_condition,  # Use local variable
                        st.session_state.time_of_day,
                        st.session_state.vehicleEngineType,
                        f"{route_length_km:.1f} km",
                        len(stop_names)
                    ]
                }
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
            
            # TAB 2: Safety Analysis
            with tab2:
                st.subheader("Route Safety Analysis")
                
                # Safety score visualization
                st.write(f"**Overall Safety Score: {safety_score:.1f}%**")
                
                # Create a color-coded progress bar
                if safety_score >= 80:
                    color = "green"
                elif safety_score >= 60:
                    color = "lightgreen"
                elif safety_score >= 40:
                    color = "orange"
                else:
                    color = "red"
                
                st.markdown(f"""
                <div style="width: 100%; background-color: #e0e0e0; border-radius: 10px; margin: 10px 0;">
                    <div style="width: {safety_score}%; background-color: {color}; height: 20px; border-radius: 10px; text-align: center; color: white; font-weight: bold;">
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Road type safety breakdown
                st.subheader("Safety by Road Type")
                
                for road_type, score in road_safety.items():
                    col1, col2, col3 = st.columns([1, 3, 1])
                    with col1:
                        if road_type == "Highway":
                            st.write("**Highway**")
                        elif road_type == "City Road":
                            st.write("**City Road**")
                        elif road_type == "Rural Road":
                            st.write("🌾 **Rural Road**")
                        else:
                            st.write("⛰️ **Mountain Road**")
                    
                    with col2:
                        if score >= 80:
                            bar_color = "#4CAF50"
                        elif score >= 60:
                            bar_color = "#8BC34A"
                        elif score >= 40:
                            bar_color = "#FFC107"
                        else:
                            bar_color = "#F44336"
                        
                        st.markdown(f"""
                        <div style="width: 100%; background-color: #e0e0e0; border-radius: 5px; margin: 5px 0;">
                            <div style="width: {score}%; background-color: {bar_color}; height: 10px; border-radius: 5px;">
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.write(f"{score:.0f}%")
                
                # Safety recommendations
                st.subheader("Safety Recommendations")
                
                recommendations = []
                
                # Weather-based recommendations
                if weather_condition != "Clear":  # Use local variable
                    recommendations.append(f"⚠️ **Reduce speed** by 20-30% due to {weather_condition.lower()} conditions")
                
                # Time-based recommendations
                if st.session_state.time_of_day == "Night":
                    recommendations.append("🌙 **Use high beams** when appropriate and take regular breaks")
                elif st.session_state.time_of_day in ["Morning", "Evening"]:
                    recommendations.append("🚗 **Expect higher traffic** during peak hours")
                
                # Safety score based recommendations
                if safety_score < 50:
                    recommendations.append("🚨 **Consider postponing** this route due to high risk factors")
                    recommendations.append("🔄 **Explore alternative routes** with fewer risk factors")
                elif safety_score < 70:
                    recommendations.append("📱 **Stay alert** and monitor changing conditions")
                
                # Vehicle-specific recommendations
                if st.session_state.vehicleWeight > 10000:
                    recommendations.append("🚛 **Allow extra braking distance** for heavy vehicle")
                
                for rec in recommendations:
                    st.write(rec)
                
                # Risk factors
                st.subheader("Risk Factors Analysis")
                risk_factors = pd.DataFrame({
                    "Factor": ["Weather", "Time of Day", "Route Length", "Road Types"],
                    "Risk Level": [
                        "High" if weather_condition in ["Stormy", "Snowy"] else  # Use local variable
                        "Medium" if weather_condition in ["Rainy", "Foggy"] else "Low",
                        "High" if st.session_state.time_of_day == "Night" else 
                        "Medium" if st.session_state.time_of_day in ["Morning", "Evening"] else "Low",
                        "Medium" if route_length_km > 200 else "Low",
                        "Mixed" if len(road_safety) > 2 else "Uniform"
                    ],
                    "Impact": [
                        weather_condition,  # Use local variable
                        st.session_state.time_of_day,
                        f"{route_length_km:.1f} km",
                        ", ".join(list(road_safety.keys()))
                    ]
                })
                st.dataframe(risk_factors, use_container_width=True, hide_index=True)
            
            # TAB 3: Map
            with tab3:
                st.subheader("Route Map")
                
                if route_geometry:
                    # Prepare map points
                    map_points = []
                    for seq, idx in enumerate(optimized_order, 1):
                        loc = all_locations[idx]
                        map_points.append({
                            "lon": loc["lon"],
                            "lat": loc["lat"],
                            "name": f"{seq}. {loc['type']}: {loc['name'][:30]}...",
                            "order": str(seq)
                        })
                    
                    if map_points:
                        # Calculate map center
                        avg_lat = sum(p["lat"] for p in map_points) / len(map_points)
                        avg_lon = sum(p["lon"] for p in map_points) / len(map_points)
                        
                        # Create layers
                        layers = []
                        
                        # Route line
                        route_line_layer = pdk.Layer(
                            "PathLayer",
                            data=[{"path": route_geometry, "name": "Optimized Route"}],
                            get_path="path",
                            get_color=[0, 100, 255, 200],
                            width_min_pixels=3,
                            pickable=True,
                            get_width=5,
                        )
                        layers.append(route_line_layer)
                        
                        # Points
                        points_layer = pdk.Layer(
                            "ScatterplotLayer",
                            data=map_points,
                            get_position="[lon, lat]",
                            get_radius=20000,
                            get_fill_color=[255, 0, 0, 180],
                            pickable=True,
                        )
                        layers.append(points_layer)
                        
                        # Text labels
                        text_layer = pdk.Layer(
                            "TextLayer",
                            data=map_points,
                            get_position="[lon, lat]",
                            get_text="order",
                            get_size=16,
                            get_color=[255, 255, 255],
                            get_background_color=[255, 0, 0, 200],
                            background_padding=[4, 4],
                        )
                        layers.append(text_layer)
                        
                        # Create deck
                        deck = pdk.Deck(
                            map_style="light",
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
                else:
                    st.warning("Route geometry not available for map display")
            
            # Back button
            st.divider()
            if st.button("Back to Vehicle Specifications"):
                st.session_state.page = "Vehicle Specifications"
                st.rerun()
                
        except Exception as e:
            st.error(f"Error calculating route: {str(e)}")
            if st.button("Go Back"):
                st.session_state.page = "Vehicle Specifications"
                st.rerun()

# -------- MAIN APP LOGIC
if st.session_state.page == "Route Specifications":
    show_route_specs_page()
elif st.session_state.page == "Vehicle Specifications":
    show_vehicle_specs_page()
elif st.session_state.page == "Results":
    show_calculation_page()

# -------- FOOTER
st.divider()
st.caption("Route Safety Predictions powered by Machine Learning | Accident data trained on historical records")