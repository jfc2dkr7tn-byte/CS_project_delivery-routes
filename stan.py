import streamlit as st
import requests
import pandas as pd 
import pydeck as pdk

api_key = 'jCj3v16W0cJMzI4m2GtkQ32lwmkmUVki' # personal key for TomTom API


# -------- Calling the search API
def call_search_api(start_loc):
    url_search = f"https://api.tomtom.com/search/2/search/{start_loc}.json"
    params = {"key": api_key}
    
    r_search = requests.get(url_search, params = params) # choose the entry, where api_key matches entry
    data_search = r_search.json() # safe requested data in data_search

    position = data_search['results'][0]['position'] # select first hit in search of TomTom
    lat = position['lat']
    lon = position['lon']

    st.session_state.lat = lat # save the requests in variables
    st.session_state.lon = lon # save the request in variables
    return lat, lon


# -------- Calling the route API
def call_route_api(coords):
    url_route = f"https://api.tomtom.com/routing/1/calculateRoute/{coords}/json"
    params = {"key": api_key, 
            'routeType': "fastest",
            'traffic': 'true', 
            'computeBestOrder': 'true', 
            }
    r_route = requests.get(url_route, params = params)
    data_route = r_route.json()

    route = data_route["routes"][0] # extract the first info (similar to search above)
    summary = route["summary"]

    travel_time_sec = summary ['travelTimeInSeconds']
    length_m = summary['lengthInMeters']

    path_coords = []
    for leg in route.get("legs", []):
        for point in leg.get("points", []):
            # TomTom usually uses "latitude"/"longitude" or "lat"/"lon"
            lat = point.get("latitude", point.get("lat"))
            lon = point.get("longitude", point.get("lon"))
            if lat is not None and lon is not None:
                path_coords.append([lon, lat])  # pydeck wants [lon, lat]

    st.session_state.travelTimeSeconds = travel_time_sec
    st.session_state.length_m = length_m

    return travel_time_sec, length_m, path_coords


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


# -------- FIRST PAGE: ROUTE SPECS

def show_route_specs_page(): # input start and stopover location NAMES!
    st.title('Route Specifications')
    
    start_loc = st.text_input(
        'Enter starting point:', 
        placeholder = 'Berlin'
    )

    via_loc = st.text_input(
        'Enter stopovers: (comma-separated)',
        placeholder = 'Bern, Basel, Chur'
    )

    if st.button('CONTINUE'): # move on to next page and save all the inputs 
        st.session_state.start_loc = start_loc
        st.session_state.via_loc = via_loc
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
    )
    
    vehicleMaxSpeed = st.number_input(
        'Enter maximal vehicle speed:',
    )

    if st.button('CALCULATE ROUTE'): # move on to next page and save all the inputs
        st.session_state.vehicleEngineType = vehicleEngineType
        st.session_state.vehicleWeight = vehicleWeight
        st.session_state.vehicleMaxSpeed = vehicleMaxSpeed
        st.session_state.page = 'Results'
        st.rerun()


# -------- THIRD PAGE: CALCULATION / RESULTS

def show_calculation_page():
    st.title('Results')
    st.write('Here your calculation will run...')

    st.subheader('Your Input') # protocol all the inputs
    st.write(f'**Start:**{st.session_state.start_loc}')
    st.write(f'**Stopovers:**{st.session_state.via_loc}')
    st.write(f'**Engine type:**{st.session_state.vehicleEngineType}')
    st.write(f'**Weight:**{st.session_state.vehicleWeight}')
    st.write(f'**Max speed (km/h):**{st.session_state.vehicleMaxSpeed}')

    lat_start, lon_start = call_search_api(st.session_state.start_loc)

    stop_names = [
        s.strip() for s in st.session_state.via_loc.split(",") if s.strip()
    ] # create a list with all the stopovers, based on the input

    stop_coords = []
    for name in stop_names:
        lat, lon = call_search_api(name) # call search api for coordinates
        stop_coords.append((lat, lon))

    coord_parts = [f"{lat_start},{lon_start}"] + [
        f"{lat},{lon}" for (lat, lon) in stop_coords
    ] # order the lat, lon correctly

    st.session_state.coords = ":".join(coord_parts) # tomtom requires a special format with':'

    st.subheader('Geocoded Coordinates')
    st.write(f'**Start:** {lat_start}, {lon_start}')
    if stop_coords:
        for name, (lat, lon) in zip(stop_names, stop_coords):
            st.write(f'**Stopover - {name}:** {lat}, {lon}')
    else:
        st.write("No stopovers provided.")

    st.subheader('Results') # here the main results are shown
    st.write(f'**Latitude Start:**{lat_start}')
    st.write(f'**Longitude Start:**{lon_start}')

    travel_time_sec, length_m, route_path = call_route_api(st.session_state.coords)

    # --------
    st.subheader("Route on Map")

    if route_path:
        path_data = pd.DataFrame(
            {
                "path": [route_path],
                "name": ["Route"],
            }
        )

        deck = pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=lat_start,
                longitude=lon_start,
                zoom=5,
                pitch=0,
            ),
            layers=[
                pdk.Layer(
                    "PathLayer",
                    data=path_data,
                    get_path="path",
                    get_width=5,
                    get_color=[255, 0, 0],
                    pickable=False,
                ),
                pdk.Layer(
                    "ScatterplotLayer",
                    data=pd.DataFrame(
                        [{"lat": lat_start, "lon": lon_start}]
                        + [{"lat": lat, "lon": lon} for (lat, lon) in stop_coords]
                    ),
                    get_position="[lon, lat]",
                    get_radius=5000,
                    get_fill_color=[0, 0, 255],
                ),
            ],
            tooltip={"text": "{name}"},
        )

        st.pydeck_chart(deck)
    else:
        st.write("No route geometry available.")
    # --------

    st.subheader('Results') # here the calculations are shown
    st.write(f"Travel time (sec): {travel_time_sec}")
    st.write(f"Length (m): {length_m}")


# -------- RUN PAGES
if 'page' not in st.session_state:
    st.session_state.page = 'Route Specifications' # initialize first page


if st.session_state.page == 'Route Specifications':
    show_route_specs_page()
elif st.session_state.page == 'Vehicle Specifications':
    show_vehicle_specs_page()
elif st.session_state.page == 'Results':
    show_calculation_page()