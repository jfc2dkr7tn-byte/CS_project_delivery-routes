import streamlit as st
import requests
import pandas as pd 
import pydeck as pdk
from urllib.parse import quote

api_key = 'jCj3v16W0cJMzI4m2GtkQ32lwmkmUVki' # personal key for TomTom API


# -------- Calling the search API
def call_search_api(location_str):
    encoded = quote(location_str) # encode the adress for the url
    url_search = f"https://api.tomtom.com/search/2/geocode/{encoded}.json"
    params = {"key": api_key} # sends the request
    
    r_search = requests.get(url_search, params = params) # choose the entry, where api_key matches entry
    data_search = r_search.json() # safe requested data in data_search

    results = data_search.get('results', [])
    if not results: # Error check
        raise ValueError(f"No results found for '{location_str}'")

    position = results[0]['position'] # select first hit in search of TomTom
    lat = position['lat']
    lon = position['lon']

    st.session_state.lat = lat # save the requests in variables
    st.session_state.lon = lon # save the request in variables

    return lat, lon


# -------- Calling the route API
def call_route_api(coords, engine_type, weight, max_speed):
    url_route = f"https://api.tomtom.com/routing/1/calculateRoute/{coords}/json"
    params = {"key": api_key, 
            'routeType': "fastest",
            'traffic': 'true', 
            'computeBestOrder': 'true',
            "vehicleEngineType": engine_type,
            "vehicleWeight": int(weight), 
            "vehicleMaxSpeed": int(max_speed)
            }
    
    r_route = requests.get(url_route, params = params)
    data_route = r_route.json()

    routes = data_route.get("routes", [])
    if not routes:
        raise ValueError('No route returned from API!') # Error check
    
    route = routes[0] # extract the first info (similar to search above)
    summary = route["summary"]

    travel_time_sec = summary ['travelTimeInSeconds']
    length_m = summary['lengthInMeters']

    # ---------- optimierte Reihenfolge der Wegpunkte ----------
    optimized_waypoints = route.get("optimizedWaypoints", [])

    # Anzahl aller Orte in coords: start + waypoints + end
    n_locations = len(coords.split(":"))
    # Anzahl echter Waypoints (alles zwischen Start und Endpunkt)
    n_waypoints = max(n_locations - 2, 0)

    # Default: ursprüngliche Reihenfolge
    optimized_order = list(range(n_locations))

    if optimized_waypoints and n_waypoints > 0:
        # optimierte Reihenfolge nur für die "Mitte"
        middle_order = [None] * n_waypoints

        for wp in optimized_waypoints:
            provided_wp_idx = wp.get("providedIndex")
            optimized_wp_idx = wp.get("optimizedIndex")

            # Indizes aus API: 0..n_waypoints-1 (nur Waypoints!)
            if (
                provided_wp_idx is None
                or optimized_wp_idx is None
                or not (0 <= provided_wp_idx < n_waypoints)
                or not (0 <= optimized_wp_idx < n_waypoints)
            ):
                continue

            # Waypoint-Index → Location-Index (Shift um +1):
            # 0 -> 1, 1 -> 2, ..., (n_waypoints-1) -> (n_locations-2)
            location_idx = provided_wp_idx + 1
            middle_order[optimized_wp_idx] = location_idx

        # Falls Lücken geblieben sind: mit noch nicht verwendeten Mittelpunkten füllen
        remaining = [
            loc_idx
            for loc_idx in range(1, n_locations - 1)  # nur echte Waypoints
            if loc_idx not in middle_order
        ]
        for i in range(n_waypoints):
            if middle_order[i] is None and remaining:
                middle_order[i] = remaining.pop(0)

        # Finale Reihenfolge: Start (0) + optimierte Mitte + Endpunkt (n_locations-1)
        optimized_order = [0] + middle_order + [n_locations - 1]

    # in session_state ablegen (falls du es anderswo brauchst)
    st.session_state.travelTimeSeconds = travel_time_sec
    st.session_state.length_m = length_m
    st.session_state.optimized_order = optimized_order

    return travel_time_sec, length_m, optimized_order


# -------- CONFIGURATIONS
st.set_page_config(
    page_title="Route Planner",
    page_icon=":route:",
    layout="centered",
    initial_sidebar_state="auto",
)


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
    text

    **Group members:**
    - Patrick Stoffel  
    - Tim Rütsche  
    - Valerie Pieringer  
    - Gloria Tatzer  
    - Nils Stadler
    """)




# -------- FIRST PAGE: ROUTE SPECS

def show_route_specs_page(): # input start and stopover location NAMES!
    st.title('Route Specifications')

    st.subheader('Starting Point')
    start_loc = st.text_input(
        'Enter starting point:', 
        placeholder = 'Zeusa 1a, 80-180 Kowale, Polen'
    )

    st.subheader('Stopovers')
    for i in range(len(st.session_state.stopovers)):
        st.session_state.stopovers[i] = st.text_input(
            f"Stopover {i}:",
            placeholder = 'Rothusstrasse 88, 3065 Bolligen, Schweiz', 
            value = st.session_state.stopovers[i], 
            key = f"fstopover_{i}"
        )

    if st.button('Add Stopover'):
        st.session_state.stopovers.append("")
        st.rerun()

    if st.button('CONTINUE'): # move on to next page and save all the inputs 
        if not start_loc:
            st.error('Please enter a starting point')
        else:
            st.session_state.start_loc = start_loc
            st.session_state.via_list = [
            s for s in st.session_state.stopovers if s.strip()
            ]
            st.session_state.page = 'Vehicle Specifications'
            st.rerun()


# -------- SECOND PAGE: VEHICLE SPECIFICATIONS

def show_vehicle_specs_page(): # input vehicle specifications
    st.title('Vehicle Specifications')

    vehicleEngineType = st.selectbox(
        'Choose your engine type:',
        ['electric', 'combustion', 'hydrogen']
    )
    
    vehicleWeight = st.number_input(
        'Enter vehicle weight (kg):',
        placeholder = 2000
    )
    
    vehicleMaxSpeed = st.number_input(
        'Enter maximal vehicle speed:',
        placeholder = 100
    )


    col1, col2 = st.columns(2)

    with col1:
        if st.button('BACK'):
            st.session_state.page = 'Route Specifications'
            st.rerun()

    with col2:
        if st.button('CALCULATE ROUTE'):
            st.session_state.vehicleEngineType = vehicleEngineType
            st.session_state.vehicleWeight = vehicleWeight
            st.session_state.vehicleMaxSpeed = vehicleMaxSpeed
            st.session_state.page = 'Results'
            st.rerun()


# -------- THIRD PAGE: CALCULATION / RESULTS

def show_calculation_page():
    try:

        lat_start, lon_start = call_search_api(st.session_state.start_loc) # start coordinates

        stop_names = st.session_state.get("via_list", [])

        stop_coords = []
        for name in stop_names:
            lat, lon = call_search_api(name) # call search api for coordinates
            stop_coords.append((lat, lon))
    
        coord_parts = [f"{lat_start},{lon_start}"] + [
            f"{lat},{lon}" for (lat, lon) in stop_coords
            ] # order the lat, lon correctly
        
        coord_parts.append(f'{lat_start},{lon_start}')

        coords_str = ":".join(coord_parts)
        st.session_state.coords = coords_str

        travel_time_sec, length_m, optimized_order = call_route_api(
            coords_str, 
            st.session_state.vehicleEngineType,
            st.session_state.vehicleWeight,
            st.session_state.vehicleMaxSpeed,
        )

    except ValueError as e:
        st.error(str(e))
        st.stop()

    tab1, tab2, tab3 = st.tabs(['Summary', 'Calculations', 'Map'])

    with tab1:
        st.subheader("Your Input")

        rows = [
            ("Start:", st.session_state.start_loc),
            (
                "Stopovers:",
                "\n".join(f"- {name}" for name in stop_names) if stop_names else "None",
            ),
            ("Engine type:", st.session_state.vehicleEngineType),
            ("Weight:", f"{st.session_state.vehicleWeight} kg"),
            ("Max speed (km/h):", f"{st.session_state.vehicleMaxSpeed} km/h"),
        ]

        for label, value in rows:
            c1, c2 = st.columns([1, 3])
            with c1:
                st.markdown(f"**{label}**")
            with c2:
                st.markdown(value)
    
    with tab2:
        st.subheader("Order of Stops (Optimized by API)")

        all_stops = [("Start", st.session_state.start_loc, lat_start, lon_start)]
        for name, (lat, lon) in zip(stop_names, stop_coords):
            all_stops.append(("Stopover", name, lat, lon))

        all_stops.append(("Return to Start", st.session_state.start_loc, lat_start, lon_start))

        if not optimized_order:
            optimized_order = st.session_state.get("optimized_order")

        # Falls API mal nichts liefert, auf Original-Reihenfolge zurückfallen
        if optimized_order is None:
            optimized_order = list(range(len(all_stops)))

        # Kopfzeile
        header_col1, header_col2 = st.columns([5, 2])
        with header_col1:
            st.markdown("**Stop**")
        with header_col2:
            st.markdown("**Coordinates (lat, lon)**")

        # Zeilen
        for seq_pos, orig_idx in enumerate(optimized_order, start=1):
            label, name, lat, lon = all_stops[orig_idx]

            col1, col2 = st.columns([5, 2])

            with col1:
                st.write(f"{seq_pos}. {label}: {name}")

            with col2:
                st.write(f"{lat:.6f}, {lon:.6f}")    


        st.subheader('Results')

        col3, col4 = st.columns([5, 2])

        with col3:
            st.write("Travel time (h):")
            st.write("Length (km):")

        with col4:
            st.write(f"{travel_time_sec / 3600:.2f}")
            st.write(f"{length_m / 1000:.2f}")


    with tab3:
        st.subheader("Map - Optimized Stop Order")

        # Rebuild full stop list (same logic as in tab2)
        all_stops = [("Start", st.session_state.start_loc, lat_start, lon_start)]
        for name, (lat, lon) in zip(stop_names, stop_coords):
            all_stops.append(("Stopover", name, lat, lon))
        all_stops.append(("Return to Start", st.session_state.start_loc, lat_start, lon_start))

        # Fallback: original order if API didn't optimize
        if optimized_order is None:
            optimized_order = list(range(len(all_stops)))

        # Build data for map: one point per stop in *optimized* order
        map_points = []
        for seq_pos, orig_idx in enumerate(optimized_order, start=1):
            label, name, lat, lon = all_stops[orig_idx]
            map_points.append(
                {
                    "lon": lon,
                    "lat": lat,
                    "order": str(seq_pos),              # number shown on map
                    "name": f"{seq_pos}. {label}: {name}"
                }
            )

        if map_points:
            # Center view on average of all points
            avg_lat = sum(p["lat"] for p in map_points) / len(map_points)
            avg_lon = sum(p["lon"] for p in map_points) / len(map_points)

            deck = pdk.Deck(
                map_style = None, 
                initial_view_state=pdk.ViewState(
                    latitude=avg_lat,
                    longitude=avg_lon,
                    zoom=5,
                    pitch=0,
                ),
                layers=[
                    # Just points
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=map_points,
                        get_position="[lon, lat]",
                        get_radius=4000,
                        get_fill_color=[255, 0, 0],
                    ),
                    # Numbers on top of the points
                    pdk.Layer(
                        "TextLayer",
                        data=map_points,
                        get_position="[lon, lat]",
                        get_text="order",
                        get_size=25,
                        get_color=[0, 0, 0],
                        get_alignment_baseline="'bottom'",
                    ),
                ],
                tooltip={"text": "{name}"},
            )

            st.pydeck_chart(deck)
        else:
            st.write("No stops to display on the map.")

        if st.button('BACK'):
            st.session_state.page = 'Vehicle Specifications'
            st.rerun()
   
        
# -------- RUN PAGES
if 'page' not in st.session_state:
    st.session_state.page = 'Route Specifications' # initialize first page

if 'stopovers' not in st.session_state:
    st.session_state.stopovers = [""]  # start with one empty stopover field

if st.session_state.page == 'Route Specifications':
    show_route_specs_page()
elif st.session_state.page == 'Vehicle Specifications':
    show_vehicle_specs_page()
elif st.session_state.page == 'Results':
    show_calculation_page()