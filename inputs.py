import streamlit as st
import requests

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
            'computeBestOrder': 'false', 
            }
    r_route = requests.get(url_route, params = params)
    data_route = r_route.json()

    route = data_route["routes"][0] # extract the first info (similar to search above)
    summary = route["summary"]

    travel_time_sec = summary ['travelTimeInSeconds']
    length_m = summary['lengthInMeters']

    st.session_state.travelTimeSeconds = travel_time_sec
    st.session_state.length_m = length_m

    return travel_time_sec, length_m


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
    )

    via_loc = st.text_input(
        'Enter stopovers:'
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
    lat_via, lon_via = call_search_api(st.session_state.via_loc)
    st.session_state.coords = f"{lat_start},{lon_start}:{lat_via},{lon_via}" #save the coordinates in the list that TomTom can demands

    st.subheader('Results') # here the main results are shown
    st.write(f'**Latitude Start:**{lat_start}')
    st.write(f'**Longitude Start:**{lon_start}')

    st.write(f'**Latitude via:**{lat_via}')
    st.write(f'**Longitude via:**{lon_via}')

    travel_time_sec, length_m = call_route_api(st.session_state.coords)

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