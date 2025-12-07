import streamlit as st
import requests
import pandas as pd 
import pydeck as pdk
from urllib.parse import quote
import itertools
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import matplotlib.pyplot as plt

api_key = 'jCj3v16W0cJMzI4m2GtkQ32lwmkmUVki'


# -----------------------------------------------------------------------
# Machine Learning Functions
# -----------------------------------------------------------------------
@st.cache_resource
def load_or_train_model():
    """Train the accident prediction model fresh every time"""
    
    # STEP 1: LOAD OR CREATE TRAINING DATA
    try:
        # Load our historical accident data for training
        df = pd.read_csv('cleaned_accident_data.csv')
    except:
        st.warning("Could not load accident data. Using synthetic data for demonstration.")
        # Fallback: Create synthetic data if real data isn't available
        # This ensures the app works even without real historical data
        data = {
            'Weather': ['Clear']*30 + ['Rainy']*25 + ['Snowy']*20 + ['Foggy']*15 + ['Stormy']*10,
            'Road_Type': ['City Road']*25 + ['Highway']*25 + ['Rural Road']*25 + ['Mountain Road']*25,
            'Time_of_Day': ['Morning']*25 + ['Afternoon']*25 + ['Evening']*25 + ['Night']*25,
            'Traffic_Density': np.random.uniform(0, 2, 100),
            'Speed_Limit': np.random.randint(30, 120, 100),
            'Accident': [0]*70 + [1]*30
        }
        df = pd.DataFrame(data)
    
    # STEP 2: PREPARE DATA FOR MODEL TRAINING
    categorical_cols = ['Weather', 'Road_Type', 'Time_of_Day']  # Categorical variables: Text data that needs to be converted to numbers
    numerical_cols = ['Traffic_Density', 'Speed_Limit'] # Numerical variables: Numbers that can be used directly
    
    # Create dictionary to store our category encoders
    label_encoders = {}
    X_encoded = pd.DataFrame() # create a Pandas dataframe that holds our features
    
    # Convert text categories to numbers using Label Encoding
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Add numerical columns unchanged
    for col in numerical_cols:
        X_encoded[col] = df[col]
    
    # Handle missing values by filling in median values (intelligent treatment)
    X_encoded = X_encoded.fillna(X_encoded.median())
    
    # Definition of the target variable
    y = df['Accident']
    
    # STEP 3: SCALE THE DATA
    # Different features have different ranges (speed 30-120, traffic 0-2)
    # Scaling puts all features on the same scale so no single feature dominates
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # STEP 4: TRAIN / TEST SPLIT
    # Split data (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # STEP 5: TRAIN THE MACHINE LEARNING MODEL
    # Random Forest: Creates multiple decision trees and averages their predictions
    # n_estimators=100: Creates 100 decision trees
    # random_state=42: Makes results reproducible (same random choices each time)
    # max_depth=5: Limits how complex each tree can be (prevents overfitting)
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    
    # Train on training data
    model.fit(X_train, y_train)
    
    # STEP 6: TEST THE MODEL
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = model.score(X_test, y_test)
    
    # Store the accuracy in session state to show it only once
    if 'model_accuracy' not in st.session_state:
        st.session_state.model_accuracy = accuracy
    
    return model, label_encoders, scaler

def show_model_training_message():
    """Show the model training success message only once"""
    if 'model_training_shown' not in st.session_state:
        if 'model_accuracy' in st.session_state:
            accuracy = st.session_state.model_accuracy
            st.success(f"Model trained successfully! Test Accuracy: {accuracy:.2%}")
            st.session_state.model_training_shown = True

def predict_accident_risk(weather, time_of_day):
    """Predict accident risk based on conditions - simplified version using only weather and time"""
    # STEP 1: LOAD THE TRAINED MODEL
    model, label_encoders, scaler = load_or_train_model()
    
    # STEP 2: ENCODE CATEGORICAL FEATURES (Weather and Time of Day only)
    features = []  # List to store all feature values
    
    # Encode weather
    weather_le = label_encoders['Weather']
    try:
        weather_encoded = weather_le.transform([weather])[0]
    except ValueError:
        weather_encoded = weather_le.transform([weather_le.classes_[0]])[0]
    features.append(weather_encoded)
    
    # Encode time of day
    time_le = label_encoders['Time_of_Day']
    try:
        time_encoded = time_le.transform([time_of_day])[0]
    except ValueError:
        time_encoded = time_le.transform([time_le.classes_[0]])[0]
    features.append(time_encoded)
    
    # STEP 3: USE MEDIAN VALUES FOR OTHER FEATURES (since we're not asking for them)
    # Get median values from training data to fill missing features
    # We need to maintain the same feature order as during training
    features.append(1.0)  # Median traffic density (approximately)
    features.append(80)   # Median speed limit (approximately)
    
    # Add road type (use most common road type from training)
    road_le = label_encoders['Road_Type']
    road_encoded = road_le.transform(['City Road'])[0]  # Use City Road as default
    features.append(road_encoded)
    
    # STEP 4: SCALE THE FEATURES
    features_scaled = scaler.transform([features])
    
    # STEP 5: MAKE PREDICTION
    probability = model.predict_proba(features_scaled)[0][1]
    
    return probability  # Returns a value between 0 and 1 representing accident risk


# -----------------------------------------------------------------------
# Sustainability Score
# -----------------------------------------------------------------------

def calculate_sustainability_score(engine_type, route_length_km, travel_time_hours):
    """Calculate a simple sustainability score (0-100)"""
    
    # STEP 1: START WITH BASE SCORE
    # All journeys start with a moderate sustainability score of 70
    score = 70
    
    # STEP 2: ENGINE TYPE ADJUSTMENTS
    # Different engine types have different environmental impacts
    # Electric is most sustainable, hydrogen is good, combustion is least sustainable
    engine_scores = {
        'electric': +30,
        'hydrogen': +20,
        'combustion': -10
    }

    # Add the appropriate adjustment based on the vehicle's engine type
    score += engine_scores.get(engine_type, 0)
    
    # STEP 3: ROUTE LENGTH PENALTY
    # Shorter routes are more sustainable (less energy/fuel consumption)
    # Only penalize routes longer than 100km
    if route_length_km > 100:
        length_penalty = min((route_length_km - 100) / 10, 15)  # Max 15% penalty
        score -= length_penalty
    
    # STEP 4: TRAVEL TIME PENALTY
    # Shorter travel times are generally better (less idling, more efficient routing)
    # Only penalize trips longer than 3 hours
    if travel_time_hours > 3:
        time_penalty = min((travel_time_hours - 3) * 2, 10)  # Max 10% penalty
        score -= time_penalty
    
    # STEP 5: ENSURE SCORE IS IN VALID RANGE
    # Make sure the final score is between 0 (worst) and 100 (best)
    score = max(0, min(100, score))
    
    return score

def get_sustainability_tips(engine_type, score):
    """Provide simple sustainability tips"""
    
    tips = []
    
    # STEP 1: ENGINE-SPECIFIC TIPS
    # Provide advice based on the vehicle's engine type
    if engine_type == 'combustion':
        tips.append("Consider switching to electric or hydrogen for better sustainability")
    
    # STEP 2: SCORE-BASED TIPS
    # Provide feedback based on the calculated sustainability score
    if score < 60:
        tips.append("This route has moderate-high environmental impact")
    elif score < 80:
        tips.append("This route has moderate environmental impact")
    else:
        tips.append("This route has low environmental impact - Good choice!")
    
    # STEP 3: GENERAL TIPS
    # Always add these general sustainability tips
    tips.append("Maintain steady speed to reduce fuel consumption")
    
    return tips


# -----------------------------------------------------------------------
# Route Optimization Functions
# -----------------------------------------------------------------------
def call_route_api_with_optimization(start_coords, waypoint_coords, engine_type, weight, max_speed):
    '''Calculate the optimal order by testing different permutations'''
    
    # STEP 1: CREATE INDEX LIST FOR WAYPOINTS
    # This creates a list like [0, 1, 2] for example for 3 waypoints
    waypoint_indices = list(range(len(waypoint_coords)))
    
    # STEP 2: HANDLE SIMPLE CASES (0 or 1 waypoints)
    # If there's only 0 or 1 waypoint, no optimization needed
    if len(waypoint_coords) <= 1:
        # no optimization needed for 0 or 1 waypoints
        best_order = waypoint_indices
        best_time = 0
        best_distance = 0
        
        # Build coordinates string for API call
        # Format: start + waypoints in order + return to start
        all_coords = [start_coords] + [waypoint_coords[i] for i in best_order] + [start_coords]
        coords_str = ":".join(all_coords)

        # Prepare API URL and parameters
        url_route = f"https://api.tomtom.com/routing/1/calculateRoute/{coords_str}/json"
        params = {
            "key": api_key, 
            'routeType': "fastest",
            'traffic': 'true', 
            "vehicleEngineType": engine_type,
            "vehicleWeight": int(weight), 
            "vehicleMaxSpeed": int(max_speed),
        }
        
        # Make API call to TomTom routing service
        r_route = requests.get(url_route, params=params)
        data_route = r_route.json()
        
        # Check if we got a valid route response
        if "routes" not in data_route or not data_route["routes"]:
            raise ValueError('No route returned from API!')
        
        # STEP 3: EXTRACT ROUTE INFORMATION
        # Get the first (best) route from the response
        route = data_route["routes"][0]
        summary = route["summary"]
        best_time = summary['travelTimeInSeconds']
        best_distance = summary['lengthInMeters']
        
        # Extract the route geometry (list of coordinates) for map display
        route_geometry = extract_route_geometry(route)
        
        # STEP 4: CREATE OPTIMIZED ORDER LIST
        # Format: start(0) + waypoints(1,2,3...) + end(start again)
        optimized_order = [0] + [i+1 for i in best_order] + [len(waypoint_coords) + 1]
        
        return best_time, best_distance, optimized_order, route_geometry
    
    # STEP 5: HANDLE MULTIPLE WAYPOINTS (OPTIMIZATION NEEDED)
    # For 2+ waypoints, we need to find the best order to visit them
    best_time = float('inf')        # Start with infinite time (will get better)
    best_distance = 0               # Initialize distance
    best_order = None               # Store the best order found
    best_route_geometry = None      # Store the best route geometry
    
    # Try different permutations (orders) of waypoints
    # Limit number of permutations to avoid too many API calls
    max_permutations = 24
    permutations_tested = 0
    
    # STEP 6: TEST DIFFERENT WAYPOINT ORDERS
    # Try all possible orders (permutations) of waypoints
    for perm in itertools.permutations(waypoint_indices):
        # Stop if we've tested too many permutations
        if permutations_tested >= max_permutations:
            break
            
        # Build coordinates list for this specific order
        all_coords = [start_coords] + [waypoint_coords[i] for i in perm] + [start_coords]
        coords_str = ":".join(all_coords)
        
        # Call TomTom API for this specific route order
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
            # API request
            r_route = requests.get(url_route, params=params)
            data_route = r_route.json()
            
            # Check if we got a valid route
            if "routes" in data_route and data_route["routes"]:
                route = data_route["routes"][0]
                travel_time = route["summary"]['travelTimeInSeconds']
                
                # Check if this order is better than previous best
                if travel_time < best_time:
                    # Update best values with this better route
                    best_time = travel_time
                    best_distance = route["summary"]['lengthInMeters']
                    best_order = list(perm)
                    best_route_geometry = extract_route_geometry(route)
                    
            permutations_tested += 1 # Count this permutation
            
        except Exception as e:
            # If this permutation fails, skip it and continue
            st.warning(f"Error testing permutation: {e}")
            continue
    
    # STEP 8: BUILD FINAL OPTIMIZED ORDER LIST
    # Convert waypoint indices to route position indices
    optimized_order = [0] + [i+1 for i in best_order] + [len(waypoint_coords) + 1]
    
    return best_time, best_distance, optimized_order, best_route_geometry

def extract_route_geometry(route):
    """Extract the route line coordinates from TomTom API response"""
    route_geometry = []
    
    if 'legs' in route:
        # A route has multiple legs (segments between waypoints)
        for leg in route['legs']:
            if 'points' in leg:
                # Extract each coordinate point along this leg
                for point in leg['points']:
                    # Store as [longitude, latitude] for mapping libraries
                    route_geometry.append([point['longitude'], point['latitude']])
    
    return route_geometry

def call_search_api(location_str):
    """Get coordinates for a location"""
    # STEP 1: ENCODE LOCATION STRING
    # Convert location string to URL-safe format
    encoded = quote(location_str)

    # STEP 2: BUILD API URL
    url_search = f"https://api.tomtom.com/search/2/geocode/{encoded}.json"
    params = {"key": api_key}
    
    # STEP 3: MAKE API CALL
    r_search = requests.get(url_search, params=params)
    data_search = r_search.json()

    # STEP 4: EXTRACT COORDINATES
    results = data_search.get('results', [])
    if not results:
        raise ValueError(f"No results found for '{location_str}'")

    # Get the first (most relevant) result
    position = results[0]['position']
    lat = position['lat']
    lon = position['lon']
    
    return lat, lon


# -----------------------------------------------------------------------
# Streamlit Setup
# -----------------------------------------------------------------------

st.set_page_config(
    page_title="Route Planner with Safety & Sustainability",
    layout="centered",
)

# STEP 1: INITIALIZE SESSION STATE
# Session state is Streamlit's way of preserving data between user interactions
if 'page' not in st.session_state:
    st.session_state.page = 'Route Specifications' # Default starting tab

if 'stopovers' not in st.session_state:
    st.session_state.stopovers = [""] # Initialize with one empty stopover field

# Show model training message only once at the beginning
show_model_training_message()

# Sidebar provides static information that's always accessible
st.sidebar.write('''
    ## About _routefinder_
    Built for truck drivers.
    **Find the safest, fastest, and most sustainable route with vehicle-specific insights!**

    ## Resources
    - [TomTom API](https://api.tomtom.com/search/2/search/)
    - [Dataset on accidents](https://www.kaggle.com/datasets/denkuznetz/traffic-accident-prediction) (adjusted for simplicity)
    ''')

with st.sidebar.expander('About us', expanded=False):
    st.markdown("""
    **Group members:**
    - Patrick Stoffel  
    - Tim Rütsche  
    - Valerie Pieringer  
    - Gloria Tatzer  
    - Nils Stadler
    """)


# -----------------------------------------------------------------------
# Page 1: Route Specifications
# -----------------------------------------------------------------------
# This page handles the first step of route planning: defining the route path
# Users specify where they're starting from and any stopovers along the way
def show_route_specs_page():
    # STEP 1: PAGE TITLE
    # Display the main title for this page
    st.title("Route Specifications")
    
    # STEP 2: STARTING POINT INPUT
    # The user needs to specify where their journey begins
    st.subheader("Starting Point")
    start_loc = st.text_input(
        "Please enter starting point:",
        placeholder="Format: St. Jakobsstrasse 87, 9000 St. Gallen, Schweiz",
        key="start_input"
    )
    
    # STEP 3: STOPOVERS INPUT SECTION
    # Stopovers are intermediate points between start
    st.subheader("Stopovers")
    # Loop through all stopovers stored in session state
    # This creates a dynamic list of input fields (when a new field is added, it is still considered)
    for i in range(len(st.session_state.stopovers)):
        st.session_state.stopovers[i] = st.text_input(
            f"Please enter Stopover {i+1}:",
            value=st.session_state.stopovers[i],
            placeholder="Format: Galenusstrasse 9, 13187 Berlin, Deutschland",
            key=f"stopover_{i}"
        )
    
    # STEP 4: ACTION BUTTONS
    # Create two columns to display buttons side by side
    col1, col2 = st.columns(2) # Split the page into 2 equal columns
    with col1:
        if st.button(" Add Stopover"):
            # When clicked: Add an empty string to the stopovers list
            st.session_state.stopovers.append("")
            # Rerun the app to show the new input field immediately
            st.rerun()
    
    with col2:
        if st.button("Continue", type="primary"):
            # Primary button (colored) indicates main action

            # STEP 5: INPUT VALIDATION
            # Check if user entered a starting point
            if not start_loc:
                # Show error message if starting point is empty
                st.error("Please enter a starting point")
            else:
                # STEP 6: SAVE DATA AND NAVIGATE
                # Save valid inputs to session state for use on next page
                st.session_state.start_loc = start_loc
                 # Save non-empty stopovers only (filter out empty strings)
                st.session_state.via_list = [s for s in st.session_state.stopovers if s.strip()]
                 # Change page to next step (Vehicle Specifications)
                st.session_state.page = "Vehicle Specifications"
                st.rerun()


# -----------------------------------------------------------------------
# Page 2: Vehicle Specifications and Safety Inputs
# -----------------------------------------------------------------------
# This page collects vehicle and safety parameters that affect route calculation
# Vehicle details impact travel time, sustainability, and route recommendations
def show_vehicle_specs_page():
    # STEP 1: PAGE TITLE
    # Display the main title for this page
    st.title("Vehicle Specifications")
    
    # STEP 2: VEHICLE DETAILS SECTION
    # Collect technical specifications about vehicle
    st.subheader("Vehicle Details")
    
    # Engine type selection
    vehicleEngineType = st.selectbox(
        "Engine Type:",
        ["combustion", "electric", "hydrogen"],
        index=0
    )
    
    # Vehicle weight slider (control lever from left to right)
    st.write("Vehicle Weight:")
    vehicleWeight = st.slider(
        "Select weight (kg):",
        min_value=500,
        max_value=50000,
        value=2000,
        step=100,
        label_visibility="collapsed"  # Hide the label since we have our own above
    )
    
    # Display the selected weight value
    st.write(f"**Selected weight:** {vehicleWeight:,} kg")
    
    # Maximum speed selection buttons
    st.write("Maximum Speed:")
    
    # Create speed buttons from 60 to 120 km/h in proper order
    speed_options = [60, 70, 80, 90, 100, 110, 120]
    
    # Initialize session state for speed if not exists
    if 'selected_speed' not in st.session_state:
        st.session_state.selected_speed = 100  # Default value
    
    # Create columns for the buttons - PROPERLY ORDERED
    # Use 4 columns for better layout (2 rows: 60,70,80,90 on top row, 100,110,120 on bottom row)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("60 km/h", use_container_width=True, 
                    type="primary" if st.session_state.selected_speed == 60 else "secondary"):
            st.session_state.selected_speed = 60
            st.rerun()
    with col2:
        if st.button("70 km/h", use_container_width=True,
                    type="primary" if st.session_state.selected_speed == 70 else "secondary"):
            st.session_state.selected_speed = 70
            st.rerun()
    with col3:
        if st.button("80 km/h", use_container_width=True,
                    type="primary" if st.session_state.selected_speed == 80 else "secondary"):
            st.session_state.selected_speed = 80
            st.rerun()
    with col4:
        if st.button("90 km/h", use_container_width=True,
                    type="primary" if st.session_state.selected_speed == 90 else "secondary"):
            st.session_state.selected_speed = 90
            st.rerun()
    
    # Second row for higher speeds
    col5, col6, col7 = st.columns(3)
    
    with col5:
        if st.button("100 km/h", use_container_width=True,
                    type="primary" if st.session_state.selected_speed == 100 else "secondary"):
            st.session_state.selected_speed = 100
            st.rerun()
    with col6:
        if st.button("110 km/h", use_container_width=True,
                    type="primary" if st.session_state.selected_speed == 110 else "secondary"):
            st.session_state.selected_speed = 110
            st.rerun()
    with col7:
        if st.button("120 km/h", use_container_width=True,
                    type="primary" if st.session_state.selected_speed == 120 else "secondary"):
            st.session_state.selected_speed = 120
            st.rerun()
    
    # Display the selected speed
    st.write(f"**Selected speed:** {st.session_state.selected_speed} km/h")
    vehicleMaxSpeed = st.session_state.selected_speed
    
    # STEP 3: ROUTE SAFETY PARAMETERS SECTION
    # Collect environmental factors that affect safety and accident prediction
    st.subheader("Route Safety Parameters")

    # Create two columns to display weather and time selectors side by side
    col1, col2 = st.columns(2) # Split into two equal columns

    with col1: # Weather condition selecter
        weather_condition = st.selectbox(
            "Weather Condition:",
            ["Clear", "Rainy", "Snowy", "Foggy", "Stormy"],
            index=0
        )
    
    with col2: # Time of day selector
        time_of_day = st.selectbox(
            "Time of Travel:",
            ["Morning", "Afternoon", "Evening", "Night"],
            index=1
        )
    
    # STEP 4: NAVIGATION BUTTONS
    # Create two columns for back and calculate buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back"):
            st.session_state.page = "Route Specifications" # Return to previous page
            st.rerun()
    
    with col2:
        if st.button("Calculate OPTIMIZED Route", type="primary"):
            # STEP 5: SAVE ALL INPUTS TO SESSION STATE
            # Store vehicle specifications for use in route calculation
            st.session_state.vehicleEngineType = vehicleEngineType
            st.session_state.vehicleWeight = vehicleWeight
            st.session_state.vehicleMaxSpeed = vehicleMaxSpeed

            # Store safety parameters for accident prediction
            st.session_state.weather_condition = weather_condition
            st.session_state.time_of_day = time_of_day

            # STEP 6: NAVIGATE TO RESULTS PAGE
            st.session_state.page = "Results"
            st.rerun()


# -----------------------------------------------------------------------
# Page 3: Calculation and Results
# -----------------------------------------------------------------------
# This is the results page where all calculations and optimizations are displayed
def show_calculation_page():
    # STEP 1: PAGE SETUP
    st.title("Route Results - OPTIMIZED ORDER")
    
    # STEP 2: CREATE TABBED INTERFACE
    tab1, tab2, tab3, tab4 = st.tabs(["Route Details", "Safety Analysis", "Sustainability", "Map"])
    
    # STEP 3: SHOW LOADING INDICATOR
    with st.spinner("Calculating optimal route with safety and sustainability assessment..."):
        try:
            # STEP 4: GET COORDINATES FOR ALL LOCATIONS
            lat_start, lon_start = call_search_api(st.session_state.start_loc)
            start_coords = f"{lat_start},{lon_start}"
            
            stop_names = st.session_state.get("via_list", [])
            stop_coords = []
            
            for name in stop_names:
                lat, lon = call_search_api(name)
                stop_coords.append(f"{lat},{lon}")
            
            # STEP 5: CALCULATE OPTIMIZED ROUTE
            travel_time_sec, length_m, optimized_order, route_geometry = call_route_api_with_optimization(
                start_coords,
                stop_coords,
                st.session_state.vehicleEngineType,
                st.session_state.vehicleWeight,
                st.session_state.vehicleMaxSpeed
            )
            
            # STEP 6: STORE ROUTE DATA IN SESSION STATE
            st.session_state.route_geometry = route_geometry
            
            # STEP 7: PREPARE LOCATION DATA FOR DISPLAY
            all_locations = []
            
            # Add starting location
            all_locations.append({
                "type": "Start",
                "name": st.session_state.start_loc,
                "lat": lat_start,
                "lon": lon_start
            })
            
            # Add all waypoints/stopovers
            for i, (name, coord_str) in enumerate(zip(stop_names, stop_coords), 1):
                lat, lon = map(float, coord_str.split(","))
                all_locations.append({
                    "type": f"Stopover",
                    "name": name,
                    "lat": lat,
                    "lon": lon
                })
            
            # Add return to start location
            if stop_names:
                all_locations.append({
                    "type": "Return to Start",
                    "name": st.session_state.start_loc,
                    "lat": lat_start,
                    "lon": lon_start
                })
            
            # STEP 8: CALCULATE KEY METRICS
            route_length_km = length_m / 1000
            travel_time_hours = travel_time_sec / 3600
            
            # STEP 9: CALCULATE SAFETY SCORE USING MACHINE LEARNING MODEL
            # Get parameters from session state (user inputs from page 2)
            weather_condition = st.session_state.weather_condition
            time_of_day = st.session_state.time_of_day
            
            # STEP 9b: CALL THE SIMPLIFIED MACHINE LEARNING MODEL FOR SAFETY PREDICTION
            # Using only weather and time of day (no road type, traffic density, or speed limit)
            accident_probability = predict_accident_risk(
                weather=weather_condition,
                time_of_day=time_of_day
            )
            
            # STEP 9c: CONVERT PROBABILITY TO SAFETY SCORE (0-100%)
            safety_score = (1 - accident_probability) * 100
            
            # STEP 9d: GET SAFETY INFORMATION FOR DISPLAY
            if accident_probability < 0.3:
                road_safety = "Low Risk"
                safety_color = "green"
            elif accident_probability < 0.6:
                road_safety = "Medium Risk"
                safety_color = "orange"
            else:
                road_safety = "High Risk"
                safety_color = "red"
            
            # STEP 10: CALCULATE SUSTAINABILITY SCORE
            sustainability_score = calculate_sustainability_score(
                st.session_state.vehicleEngineType,
                route_length_km,
                travel_time_hours
            )
            
            # STEP 11: CREATE VISUALIZATION DATA
            # Prepare data for safety comparison visualization
            # Create comparison data for different weather conditions
            weather_conditions = ["Clear", "Rainy", "Snowy", "Foggy", "Stormy"]
            time_conditions = ["Morning", "Afternoon", "Evening", "Night"]
            
            # Calculate safety scores for different conditions to show comparison
            safety_comparison_data = []
            for weather in weather_conditions:
                # Calculate accident probability for this weather with current time of day
                test_prob = predict_accident_risk(weather=weather, time_of_day=time_of_day)
                test_safety = (1 - test_prob) * 100
                safety_comparison_data.append({
                    "Weather": weather,
                    "Safety_Score": test_safety,
                    "Is_Current": weather == weather_condition
                })
            
            # Prepare sustainability comparison data
            engine_types = ["combustion", "hydrogen", "electric"]
            sustainability_scores = []
            for engine in engine_types:
                score = calculate_sustainability_score(
                    engine,
                    route_length_km,
                    travel_time_hours
                )
                sustainability_scores.append({
                    "Engine_Type": engine.capitalize(),
                    "Sustainability_Score": score,
                    "Is_Current": engine == st.session_state.vehicleEngineType
                })
            
            # TAB 1: Route Details
            with tab1:
                st.subheader("Route Details")
                
                # Display key metrics in three columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Distance", f"{route_length_km:.1f} km")
                with col2:
                    hours = travel_time_sec / 3600
                    minutes = (hours % 1) * 60
                    st.metric("Travel Time", f"{int(hours)}h {int(minutes)}min")
                with col3:
                    # ML-BASED SAFETY SCORE (using only weather and time)
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
                
                # Show optimized visiting order
                st.subheader("Optimized Route Order")
                for seq, idx in enumerate(optimized_order, 1):
                    loc = all_locations[idx]
                    st.write(f"**{seq}. {loc['type']}**")
                    st.write(f"   {loc['name']}")
                
                # Route summary table - SIMPLIFIED FOR ML PREDICTION
                st.subheader("Route Summary")
                summary_data = {
                    "Parameter": [
                        "Weather Condition", 
                        "Time of Travel", 
                        "Accident Probability",
                        "Safety Level",
                        "Vehicle Type", 
                        "Vehicle Weight",
                        "Max Speed",
                        "Route Length", 
                        "Number of Stops"
                    ],
                    "Value": [
                        weather_condition,
                        time_of_day,
                        f"{accident_probability:.1%}",
                        road_safety,
                        st.session_state.vehicleEngineType,
                        f"{st.session_state.vehicleWeight:,} kg",
                        f"{st.session_state.vehicleMaxSpeed} km/h",
                        f"{route_length_km:.1f} km",
                        len(stop_names)
                    ]
                }
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
            
            # TAB 2: Safety Analysis
            with tab2:
                st.subheader("Route Safety Analysis")
                
                # Display overall safety score - ML-BASED
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Safety Score", f"{safety_score:.1f}%")
                with col2:
                    st.metric("Accident Probability", f"{accident_probability:.1%}")
                
                st.write(f"**Risk Level: {road_safety}**")

                # Display ML prediction details
                st.subheader("Safety Prediction Details")
                st.write(f"**Prediction based on:** Weather condition and time of day")
                st.write(f"**ML Model:** Random Forest Classifier trained on historical accident data")
                st.write(f"**Prediction confidence:** The model predicts a {accident_probability:.1%} probability of an accident")
                st.write(f"**Safety interpretation:** {road_safety} conditions based on your inputs")
                
                # Create a visual progress bar for safety score
                st.subheader("Safety Score Visualization")
                fig1, ax1 = plt.subplots(figsize=(10, 2))
                
                # Create a horizontal bar for safety score
                ax1.barh([0], [safety_score], color=safety_color, height=0.5)
                ax1.barh([0], [100], color='lightgray', height=0.5, alpha=0.3)
                ax1.set_xlim(0, 100)
                ax1.set_yticks([])
                ax1.set_xlabel('Safety Score (%)')
                ax1.set_title(f'Current Safety Score: {safety_score:.1f}%')
                
                # Add text annotation
                ax1.text(safety_score/2, 0, f'{safety_score:.1f}%', 
                        ha='center', va='center', color='white', fontweight='bold')
                
                # Add risk level markers
                risk_levels = [0, 40, 60, 80, 100]
                risk_labels = ['High Risk', 'Medium Risk', 'Moderate Safety', 'Good Safety', 'Excellent Safety']
                for level, label in zip(risk_levels, risk_labels):
                    ax1.axvline(x=level, color='gray', linestyle='--', alpha=0.5)
                    ax1.text(level, -0.3, label, rotation=45, ha='center', fontsize=8, alpha=0.7)
                
                st.pyplot(fig1)
                plt.close(fig1)
                
                # Generate and display safety recommendations
                st.subheader("Safety Recommendations")
                
                recommendations = []
                
                # Weather-based recommendations (from user input on page 2)
                if weather_condition != "Clear":
                    recommendations.append(f"**Reduce speed** by 20-30% due to {weather_condition.lower()} conditions")
                    if weather_condition == "Rainy":
                        recommendations.append("**Increase following distance** - wet roads double braking distance")
                    elif weather_condition == "Snowy":
                        recommendations.append("**Use snow tires** and carry chains if required")
                    elif weather_condition == "Foggy":
                        recommendations.append("**Use fog lights** and reduce speed significantly")
                    elif weather_condition == "Stormy":
                        recommendations.append("**Consider delaying trip** if severe weather is forecast")
                
                # Time-based recommendations (from user input on page 2)
                if time_of_day == "Night":
                    recommendations.append("**Use high beams** when appropriate and take regular breaks")
                    recommendations.append("**Watch for wildlife** - animal activity increases at night")
                elif time_of_day in ["Morning", "Evening"]:
                    recommendations.append("**Expect higher traffic** during peak hours")
                    recommendations.append("**Watch for pedestrians** and cyclists during commute times")
                elif time_of_day == "Afternoon":
                    recommendations.append("**Be aware of driver fatigue** - take breaks every 2 hours")
                
                # ML score based recommendations
                if safety_score < 50:
                    recommendations.append("**Consider postponing** this route due to high risk factors")
                    recommendations.append("**Explore alternative routes** or travel times")
                elif safety_score < 70:
                    recommendations.append("**Stay alert** and monitor changing conditions")
                    recommendations.append("**Plan extra travel time** for safe driving")
                
                # Vehicle-specific recommendations
                if st.session_state.vehicleWeight > 10000:
                    recommendations.append("**Allow extra braking distance** for heavy vehicle")
                
                # Display all recommendations
                for rec in recommendations:
                    st.write(f" - {rec}")
            
            # TAB 3: Sustainability
            with tab3:
                st.subheader("Sustainability Score")
                
                # Display sustainability score
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sustainability Score", f"{sustainability_score:.0f}/100")
                with col2:
                    if sustainability_score >= 80:
                        rating = "Excellent"
                    elif sustainability_score >= 60:
                        rating = "Good"
                    elif sustainability_score >= 40:
                        rating = "Moderate"
                    else:
                        rating = "Poor"
                    st.metric("Rating", rating)
                
                # Sustainability Visualization
                st.subheader("Sustainability Visualization")
                
                # Create sustainability gauge chart
                fig4, ax4 = plt.subplots(figsize=(10, 2))
                
                # Define color zones
                colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
                zones = [0, 20, 40, 60, 80, 100]
                
                # Create colored zones
                for i in range(len(zones)-1):
                    ax4.barh([0], [zones[i+1]-zones[i]], left=zones[i], 
                            color=colors[i], height=0.5, alpha=0.7)
                
                # Add current score indicator
                ax4.barh([0], [sustainability_score], color='grey', 
                        height=0.3, alpha=0.9)
                ax4.set_xlim(0, 100)
                ax4.set_yticks([])
                ax4.set_xlabel('Sustainability Score')
                ax4.set_title(f'Current Sustainability Score: {sustainability_score:.0f}/100')
                
                # Add score text
                ax4.text(sustainability_score/2, 0, f'{sustainability_score:.0f}', 
                        ha='center', va='center', color='white', fontweight='bold', fontsize=12)
                
                st.pyplot(fig4)
                plt.close(fig4)
                
                # Show key factors affecting sustainability
                st.subheader("Key Factors Analysis")
                
                factors_data = {
                    "Factor": ["Engine Type", "Route Length", "Estimated CO₂ Impact", "Travel Time"],
                    "Value": [
                        st.session_state.vehicleEngineType.title(),
                        f"{route_length_km:.1f} km",
                        "High" if st.session_state.vehicleEngineType == 'combustion' else "Low" if st.session_state.vehicleEngineType == 'hydrogen' else "Very Low",
                        f"{travel_time_hours:.1f} hours"
                    ],
                    "Impact": [
                        "Positive" if st.session_state.vehicleEngineType in ['electric', 'hydrogen'] else "Negative",
                        "High" if route_length_km > 200 else "Medium" if route_length_km > 100 else "Low",
                        f"{sustainability_score:.0f}/100",
                        "High" if travel_time_hours > 5 else "Medium" if travel_time_hours > 3 else "Low"
                    ]
                }
                
                st.dataframe(pd.DataFrame(factors_data), use_container_width=True, hide_index=True)
                
                # Display sustainability tips
                st.subheader("Sustainability Tips")
                tips = get_sustainability_tips(st.session_state.vehicleEngineType, sustainability_score)
                
                for tip in tips:
                    st.write(f" - {tip}")
            
            # TAB 4: Map (remains exactly the same)
            with tab4:
                st.subheader("Route Map")
                
                if route_geometry:
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
                        avg_lat = sum(p["lat"] for p in map_points) / len(map_points)
                        avg_lon = sum(p["lon"] for p in map_points) / len(map_points)
                        
                        layers = []
                        
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
                        
                        points_layer = pdk.Layer(
                            "ScatterplotLayer",
                            data=map_points,
                            get_position="[lon, lat]",
                            get_radius=200,
                            get_fill_color=[255, 0, 0, 180],
                            pickable=True,
                        )
                        layers.append(points_layer)
                        
                        text_layer = pdk.Layer(
                            "TextLayer",
                            data=map_points,
                            get_position="[lon, lat]",
                            get_text="order",
                            get_size=20,
                            get_color=[255, 255, 255],
                            get_background_color=[255, 0, 0, 200],
                            background_padding=[4, 4],
                        )
                        layers.append(text_layer)
                        
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
            
            # STEP 11: NAVIGATION BUTTON
            st.divider()
            if st.button("Back to Vehicle Specifications"):
                st.session_state.page = "Vehicle Specifications"
                st.rerun()
                
        except Exception as e:
            st.error(f"Error calculating route: {str(e)}")
            if st.button("Go Back"):
                st.session_state.page = "Vehicle Specifications"
                st.rerun()


# -----------------------------------------------------------------------
# Main App Logic
# -----------------------------------------------------------------------
# This section controls which page/function is displayed based on user's current step
# The app uses session_state.page to track which page the user is currently viewing
if st.session_state.page == "Route Specifications":
    show_route_specs_page()
elif st.session_state.page == "Vehicle Specifications":
    show_vehicle_specs_page()
elif st.session_state.page == "Results":
    show_calculation_page()


# -----------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------
# The footer appears at the bottom of every page in the application
# It provides branding and key feature information
st.divider()
st.caption("Route Safety Predictions powered by Machine Learning")