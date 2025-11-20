import math
import urllib.parse as urlparse
from datetime import datetime, date, time, timedelta
from typing import List, Optional, Tuple

import pandas as pd
import requests as rq
import streamlit as st
import folium
from streamlit_folium import st_folium

# ======================================================
# Constants
# ======================================================

MAX_STOPS = 10  # maximum number of stops between start and end (round trip)

# ⚠️ Fixed TomTom API key (as requested)
TOMTOM_API_KEY = "O1AJrDWIrxThOtN1tnTYGgF3TluX2g7H"

SESSION_KEYS = [
    "routes",
    "summary_df",
    "stops",
    "depart_dt",
    "etas_df",
    "start_coord",
    "stop_coords",
    "stop_addresses",
]

# ======================================================
# Session State initialization
# ======================================================

for key in SESSION_KEYS:
    if key not in st.session_state:
        st.session_state[key] = None

if "selected_route_idx" not in st.session_state:
    st.session_state["selected_route_idx"] = 0

# ======================================================
# Helper functions – Geocoding & Geometry
# ======================================================

def search_city_coords(city: str, key: str) -> str:
    """
    Find coordinates (lat, lon) of an address/city using TomTom Search API.
    Returns a string 'lat,lon' that can be used in calculateRoute.
    """
    base_url = "https://api.tomtom.com/search/2/search/" + urlparse.quote(city) + ".json"
    params = {
        "minFuzzyLevel": 1,
        "maxFuzzyLevel": 2,
        "view": "Unified",
        "relatedPois": "off",
        "key": key,
    }

    response = rq.get(base_url, params=params, timeout=10)
    response.raise_for_status()
    json_response = response.json()

    try:
        latitude = json_response["results"][0]["position"]["lat"]
        longitude = json_response["results"][0]["position"]["lon"]
    except (IndexError, KeyError):
        raise ValueError(f"No coordinates found for '{city}'.")

    return f"{latitude},{longitude}"


def build_roundtrip_locations_string(start_coord: str, stop_coords: List[str]) -> str:
    """
    Build the locations string for calculateRoute as a roundtrip:
    Start -> Stops -> Start
    'latS,lonS:lat1,lon1:...:latN,lonN:latS,lonS'
    """
    locations = [start_coord] + stop_coords + [start_coord]
    return ":".join(locations)


def parse_coord(coord_str: str) -> Tuple[float, float]:
    lat, lon = map(float, coord_str.split(","))
    return lat, lon


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """
    Approximate distance in km between two lat/lon points (Haversine formula).
    """
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def optimize_stop_order_nn(start_coord: str, stop_coords: List[str], stop_addresses: List[str]):
    """
    Very simple nearest-neighbor optimization of stop order based on
    straight-line distance (no extra API calls).

    Returns:
        reordered_stop_coords, reordered_stop_addresses
    """
    if len(stop_coords) <= 1:
        return stop_coords, stop_addresses

    s_lat, s_lon = parse_coord(start_coord)
    coords = [parse_coord(c) for c in stop_coords]

    remaining = list(range(len(coords)))  # indices
    order = []
    curr_lat, curr_lon = s_lat, s_lon

    while remaining:
        best_idx = None
        best_dist = float("inf")
        for i in remaining:
            lat, lon = coords[i]
            d = haversine_km(curr_lat, curr_lon, lat, lon)
            if d < best_dist:
                best_dist = d
                best_idx = i
        order.append(best_idx)
        curr_lat, curr_lon = coords[best_idx]
        remaining.remove(best_idx)

    reordered_coords = [stop_coords[i] for i in order]
    reordered_addresses = [stop_addresses[i] for i in order]
    return reordered_coords, reordered_addresses

# ======================================================
# Helper – Vehicle Parameters
# ======================================================

def build_vehicle_params(
    engine_type: str,
    vehicle_weight_kg: float,
    extra_load_kg: float,
    max_speed_kmh: float,
) -> dict:
    """
    Build TomTom vehicle-related query parameters.

    engine_type (UI): 'Combustion', 'Diesel', 'Hybrid', 'Electric'
    TomTom Routing API allows only: 'combustion' or 'electric' for vehicleEngineType.
    """
    total_weight = vehicle_weight_kg + extra_load_kg

    # Map UI selection to allowed TomTom values
    # -> Diesel & Hybrid werden intern wie Combustion behandelt
    if engine_type == "Electric":
        vehicle_engine_type = "electric"
    else:
        vehicle_engine_type = "combustion"

    params = {
        "vehicleWeight": int(total_weight),
        "vehicleEngineType": vehicle_engine_type,
        # IMPORTANT: correct parameter name according to TomTom docs
        "vehicleMaxSpeed": int(max_speed_kmh),
    }

    return params

# ======================================================
# Routing & Scoring
# ======================================================

def normalize_series(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val):
        return pd.Series([0.0] * len(series), index=series.index)
    if max_val == min_val:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - min_val) / (max_val - min_val)


def get_routes_info(
    locations_str: str,
    depart_at: str,
    key: str,
    consumption_l_per_100km: Optional[float] = None,
    fuel_price_per_l: Optional[float] = None,
    co2_per_l: Optional[float] = None,
    time_weight: float = 1.0,
    cost_weight: float = 1.0,
    co2_weight: float = 1.0,
    vehicle_params: Optional[dict] = None,
    route_type: str = "fastest",
):
    """
    Call TomTom calculateRoute ONCE and return:
    - summary_df for ALL routes (including multi-criteria scores)
    - routes_list sorted by best multiScore (lowest = best)

    Multi-criteria score based on:
    - travel time
    - cost
    - CO₂
    with configurable weights.
    """
    base_url = f"https://api.tomtom.com/routing/1/calculateRoute/{locations_str}/json"

    params = {
        "maxAlternatives": 3,           # up to 4 routes (1 main + 3 alternatives)
        "instructionsType": "text",
        "departAt": depart_at,
        "traffic": "true",
        "routeType": route_type,        # 'fastest', 'shortest', 'eco'
        "travelMode": "car",
        "key": key,
    }

    if vehicle_params:
        params.update(vehicle_params)

    response = rq.get(base_url, params=params, timeout=20)
    response.raise_for_status()
    json_response = response.json()

    routes = json_response.get("routes", [])
    if not routes:
        raise ValueError("TomTom did not return any routes.")

    summaries = [r["summary"] for r in routes]
    df = pd.DataFrame(summaries)
    df["route_index"] = list(range(len(routes)))

    if "travelTimeInSeconds" in df.columns:
        df["travelTimeMinutes"] = (df["travelTimeInSeconds"] / 60).round(1)
    else:
        df["travelTimeMinutes"] = float("nan")

    if "lengthInMeters" in df.columns:
        df["distanceKm"] = (df["lengthInMeters"] / 1000).round(2)
    else:
        df["distanceKm"] = float("nan")

    # Check if we have enough data for full multi-criteria scoring
    has_full_data = (
        consumption_l_per_100km is not None
        and fuel_price_per_l is not None
        and co2_per_l is not None
        and "distanceKm" in df.columns
        and df["distanceKm"].notna().any()
        and "travelTimeInSeconds" in df.columns
        and df["travelTimeInSeconds"].notna().any()
    )

    if has_full_data:
        dist_km = df["distanceKm"].astype(float)
        fuel_used_l = dist_km * (consumption_l_per_100km / 100.0)
        cost_chf = fuel_used_l * fuel_price_per_l
        co2_route = fuel_used_l * co2_per_l

        df["fuelUsedL"] = fuel_used_l
        df["costCHF"] = cost_chf
        df["co2Kg"] = co2_route

        time_norm = normalize_series(df["travelTimeInSeconds"])
        cost_norm = normalize_series(df["costCHF"])
        co2_norm = normalize_series(df["co2Kg"])

        # Handle weights: if all zero, fall back to equal weights
        w_time = float(time_weight)
        w_cost = float(cost_weight)
        w_co2 = float(co2_weight)
        total_w = w_time + w_cost + w_co2
        if total_w <= 0:
            w_time = w_cost = w_co2 = 1.0
            total_w = 3.0

        w_time /= total_w
        w_cost /= total_w
        w_co2 /= total_w

        df["multiScore"] = w_time * time_norm + w_cost * cost_norm + w_co2 * co2_norm

        df = df.sort_values("multiScore").reset_index(drop=True)
    else:
        # Fallback: sort by travel time only
        if "travelTimeInSeconds" in df.columns:
            df = df.sort_values("travelTimeInSeconds").reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
        df["fuelUsedL"] = float("nan")
        df["costCHF"] = float("nan")
        df["co2Kg"] = float("nan")
        df["multiScore"] = float("nan")

    routes_sorted = [routes[i] for i in df["route_index"]]
    return df, routes_sorted

# ======================================================
# Traffic Incidents
# ======================================================

def extract_poly_points(route) -> List[Tuple[float, float]]:
    """Extract polyline points (lat, lon) from a route object."""
    poly_points = []
    for leg in route.get("legs", []):
        for p in leg.get("points", []):
            poly_points.append((p["latitude"], p["longitude"]))
    return poly_points


def get_traffic_incidents_for_route(poly_points: List[Tuple[float, float]], api_key: str) -> List[dict]:
    """
    Fetch traffic incidents within a bounding box around the route using
    TomTom Traffic Incident Details (v5).
    """
    if not poly_points:
        return []

    lats = [p[0] for p in poly_points]
    lons = [p[1] for p in poly_points]

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"

    base_url = "https://api.tomtom.com/traffic/services/5/incidentDetails"
    params = {
        "key": api_key,
        "bbox": bbox,
        "timeValidityFilter": "present",
        "language": "en-GB",
    }

    try:
        r = rq.get(base_url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("incidents", [])
    except Exception:
        # In case of any error, just return no incidents; we don't want to
        # break the whole app because of traffic.
        return []


def create_route_map(
    route,
    api_key: str,
    start_coord: Optional[str] = None,
    stop_coords: Optional[List[str]] = None,
    stop_addresses: Optional[List[str]] = None,
):
    """
    Create a Folium map with:
    - route geometry
    - markers for start/end and stops
    - optional traffic incidents overlay
    """
    poly_points = extract_poly_points(route)
    if not poly_points:
        return None

    # Center map
    avg_lat = sum(p[0] for p in poly_points) / len(poly_points)
    avg_lon = sum(p[1] for p in poly_points) / len(poly_points)

    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=11)

    # Start marker
    if start_coord is not None:
        try:
            s_lat, s_lon = parse_coord(start_coord)
        except Exception:
            s_lat, s_lon = poly_points[0]
    else:
        s_lat, s_lon = poly_points[0]

    folium.Marker(
        location=[s_lat, s_lon],
        tooltip="Start / End",
        icon=folium.Icon(color="green"),
    ).add_to(m)

    # Stop markers
    if stop_coords:
        for i, coord_str in enumerate(stop_coords):
            try:
                lat, lon = parse_coord(coord_str)
            except Exception:
                continue

            if stop_addresses and i < len(stop_addresses):
                label = f"Stop {i + 1}: {stop_addresses[i]}"
            else:
                label = f"Stop {i + 1}"

            folium.Marker(
                location=[lat, lon],
                tooltip=label,
                icon=folium.Icon(color="blue", icon="flag"),
            ).add_to(m)

    # Route line
    folium.PolyLine(poly_points, weight=6, opacity=0.8).add_to(m)

    # Traffic incidents
    incidents = get_traffic_incidents_for_route(poly_points, api_key)
    for inc in incidents:
        geom = inc.get("geometry", {})
        coords = geom.get("coordinates", [])
        lat = lon = None

        # Geometry can be Point [lon, lat] or LineString [[lon, lat], ...]
        if geom.get("type") == "Point" and isinstance(coords, list) and len(coords) == 2:
            lon, lat = coords
        elif isinstance(coords, list) and coords and isinstance(coords[0], list):
            lon, lat = coords[0][0], coords[0][1]

        if lat is None or lon is None:
            continue

        props = inc.get("properties", {})
        desc = props.get("description", "Traffic incident")

        folium.Marker(
            location=[lat, lon],
            tooltip=desc,
            icon=folium.Icon(color="red", icon="exclamation-sign"),
        ).add_to(m)

    return m

# ======================================================
# ETAs and Costs
# ======================================================

def compute_etas_for_stops(route, depart_dt: datetime, stop_names: List[str]) -> pd.DataFrame:
    """
    Calculate ETA for each leg based on the legs and the departure time.
    stop_names: list including Start, each stop, and back to Start (round trip!)
    Returns a DataFrame with From, To, Leg travel time, ETA.
    """
    legs = route.get("legs", [])
    if not legs:
        return pd.DataFrame()

    records = []
    current_time = depart_dt

    # legs[i] goes from stop_names[i] to stop_names[i+1]
    for i, leg in enumerate(legs):
        leg_summary = leg.get("summary", {})
        travel_sec = leg_summary.get("travelTimeInSeconds", 0)
        current_time = current_time + timedelta(seconds=travel_sec)
        if i + 1 < len(stop_names):
            records.append(
                {
                    "From": stop_names[i],
                    "To": stop_names[i + 1],
                    "LegTravelTimeMin": round(travel_sec / 60, 1),
                    "ETA_Arrival": current_time,
                }
            )

    return pd.DataFrame(records)


def compute_costs_and_co2(
    summary_row: pd.Series,
    consumption_l_per_100km: float,
    fuel_price_per_l: float,
    co2_per_l: float,
):
    """
    Estimate distance, fuel usage, cost and CO₂ for a single route summary row.
    """
    if "distanceKm" not in summary_row or pd.isna(summary_row["distanceKm"]):
        return None, None, None, None

    dist_km = float(summary_row["distanceKm"])
    fuel_used_l = dist_km * (consumption_l_per_100km / 100.0)
    cost = fuel_used_l * fuel_price_per_l
    co2 = fuel_used_l * co2_per_l
    return dist_km, fuel_used_l, cost, co2

# ======================================================
# Streamlit App
# ======================================================

st.set_page_config(page_title="Delivery Round Trip (TomTom)", layout="wide")

st.title("Delivery Route Optimizer")
st.caption(
    "Round trip with fixed start/end, limited stops, one routing call per calculation. "
    "Alternative routes, traffic incidents, and vehicle-aware, weighted cost & CO₂ estimation."
)

# ---------------- Sidebar: Configuration ----------------

st.sidebar.header("Vehicle")

st.sidebar.subheader("Cost model")

consumption = st.sidebar.number_input(
    "Base fuel consumption (l/100 km)",
    min_value=1.0,
    max_value=100.0,
    value=8.0,
    step=0.5,
)

fuel_price = st.sidebar.number_input(
    "Fuel price (CHF per liter)",
    min_value=0.0,
    max_value=5.0,
    value=1.9,
    step=0.1,
)

co2_per_l = st.sidebar.number_input(
    "CO₂ per liter (kg)",
    min_value=0.0,
    max_value=5.0,
    value=2.32,
    step=0.01,
)

st.sidebar.subheader("Vehicle details (advanced)")

engine_type = st.sidebar.selectbox(
    "Engine type",
    ["Combustion", "Diesel", "Hybrid", "Electric"],
)

vehicle_weight = st.sidebar.number_input(
    "Vehicle curb weight (kg)",
    min_value=500,
    max_value=40000,
    value=1800,
    step=100,
)

extra_load = st.sidebar.number_input(
    "Additional load (kg)",
    min_value=0,
    max_value=30000,
    value=200,
    step=50,
)

max_speed = st.sidebar.number_input(
    "Max speed (km/h)",
    min_value=50,
    max_value=160,
    value=120,
    step=10,
)

st.sidebar.subheader("Multi-criteria weighting")

time_weight = st.sidebar.slider(
    "Weight: travel time",
    min_value=0.0,
    max_value=1.0,
    value=0.4,
    step=0.05,
)

cost_weight = st.sidebar.slider(
    "Weight: cost",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05,
)

co2_weight = st.sidebar.slider(
    "Weight: CO₂",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05,
)

st.sidebar.subheader("Routing base type")

routing_base_type = st.sidebar.selectbox(
    "Routing base type",
    ["fastest", "eco", "shortest"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Maximum of {} stops between start and end. "
    "Each calculation executes exactly one TomTom routing call (plus one traffic call).".format(
        MAX_STOPS
    )
)

# ---------------- Main inputs ----------------

st.markdown("## 🧭 Round Trip Input")

col_start, col_stops = st.columns(2)

with col_start:
    start_address = st.text_input("Start / End address or city", value="Kecskemét, Belsőnyír 150, 6000 Ungarn")

with col_stops:
    st.write("Intermediate stops (one address per line, max {}):".format(MAX_STOPS))
    stops_text = st.text_area(
        "Stops",
        value="Hofenstrasse 75, 3032 Wohlen bei Bern\nTschudistrasse 13, 9000 St. Gallen\nNormannenstrasse 19, 3018 Bern\nRütistrasse 12, 8953 Dietikon\nDuffstrasse 15, 8153 Rümlang",
    )

st.markdown("### Departure time")
col_d, col_t = st.columns(2)
with col_d:
    dep_date = st.date_input("Date", value=date.today())
with col_t:
    dep_time = st.time_input("Time", value=time(8, 0))

depart_dt = datetime.combine(dep_date, dep_time)
depart_iso = depart_dt.isoformat(timespec="seconds")

st.markdown("### Optional: latest arrival back at start")
use_deadline = st.checkbox("Set deadline for return to start?")
deadline_dt = None
if use_deadline:
    col_dd, col_dt = st.columns(2)
    with col_dd:
        deadline_date = st.date_input("Deadline date", value=date.today())
    with col_dt:
        deadline_time = st.time_input("Deadline time", value=time(12, 0))
    deadline_dt = datetime.combine(deadline_date, deadline_time)

st.markdown("### Route options")
st.info("Die Reihenfolge der Stopps wird automatisch optimiert (Nearest-Neighbour).")

# ------------------------------------------------------
# Trigger routing
# ------------------------------------------------------

if st.button("Calculate round trip"):
    try:
        with st.spinner("Geocoding and route calculation in progress…"):

            stop_addresses = [
                line.strip() for line in stops_text.split("\n") if line.strip()
            ]

            if not start_address:
                st.error("Please enter a start/end address.")
                st.stop()

            if len(stop_addresses) == 0:
                st.error("Please enter at least one intermediate stop.")
                st.stop()

            if len(stop_addresses) > MAX_STOPS:
                st.error(
                    f"A maximum of {MAX_STOPS} stops is allowed. "
                    f"You entered {len(stop_addresses)}."
                )
                st.stop()

            # Geocode start & stops
            start_coord = search_city_coords(start_address, TOMTOM_API_KEY)
            stop_coords = [
                search_city_coords(addr, TOMTOM_API_KEY) for addr in stop_addresses
            ]

            # Always optimize stop order (no extra TomTom calls)
            stop_coords, stop_addresses = optimize_stop_order_nn(
                start_coord, stop_coords, stop_addresses
            )

            # Build locations string for calculateRoute (round trip, in optimized order)
            loc_str = build_roundtrip_locations_string(start_coord, stop_coords)

            # Vehicle params for TomTom
            vehicle_params = build_vehicle_params(
                engine_type=engine_type,
                vehicle_weight_kg=vehicle_weight,
                extra_load_kg=extra_load,
                max_speed_kmh=max_speed,
            )

            # Routing (ONE call, with alternatives)
            summary_df, routes = get_routes_info(
                locations_str=loc_str,
                depart_at=depart_iso,
                key=TOMTOM_API_KEY,
                consumption_l_per_100km=consumption,
                fuel_price_per_l=fuel_price,
                co2_per_l=co2_per_l,
                time_weight=time_weight,
                cost_weight=cost_weight,
                co2_weight=co2_weight,
                vehicle_params=vehicle_params,
                route_type=routing_base_type,
            )

            # Default to best (first after sorting by multiScore)
            best_route = routes[0]

            # Stop names for ETA: Start, stops (in final / optimized order), back to Start
            stop_names = (
                ["Start/End"]
                + [f"Stop {i + 1}: {addr}" for i, addr in enumerate(stop_addresses)]
                + ["Back to Start/End"]
            )

            # ETAs for best route (we will recompute if user selects another route)
            etas_df = compute_etas_for_stops(best_route, depart_dt, stop_names)

            # Store everything in session state
            st.session_state["routes"] = routes
            st.session_state["summary_df"] = summary_df
            st.session_state["stops"] = stop_names
            st.session_state["depart_dt"] = depart_dt
            st.session_state["etas_df"] = etas_df
            st.session_state["start_coord"] = start_coord
            st.session_state["stop_coords"] = stop_coords
            st.session_state["stop_addresses"] = stop_addresses
            st.session_state["selected_route_idx"] = 0

    except Exception as e:
        st.error(f"Error during calculation: {e}")

# ------------------------------------------------------
# Result view, if data is present
# ------------------------------------------------------

routes = st.session_state["routes"]
summary_df = st.session_state["summary_df"]
stops = st.session_state["stops"]
depart_dt_state = st.session_state["depart_dt"]
etas_df_state = st.session_state["etas_df"]
start_coord_state = st.session_state["start_coord"]
stop_coords_state = st.session_state["stop_coords"]
stop_addresses_state = st.session_state["stop_addresses"]

if routes is not None and summary_df is not None:

    tab_route, tab_details = st.tabs(["🗺️ Route & Key Figures", "📋 Details & ETAs"])

    # ------- Tab: Route & Key Figures -------
    with tab_route:
        st.markdown("## Route variants & key figures")

        # Route selection
        def format_route_option(i: int) -> str:
            row = summary_df.loc[i]
            tt = row.get("travelTimeMinutes", None)
            dist = row.get("distanceKm", None)
            cost = row.get("costCHF", None)
            co2_val = row.get("co2Kg", None)
            parts = [f"Variant {i + 1}"]
            if tt is not None and not pd.isna(tt):
                parts.append(f"{tt:.1f} min")
            if dist is not None and not pd.isna(dist):
                parts.append(f"{dist:.1f} km")
            if cost is not None and not pd.isna(cost):
                parts.append(f"{cost:.2f} CHF")
            if co2_val is not None and not pd.isna(co2_val):
                parts.append(f"{co2_val:.1f} kg CO₂")
            return " | ".join(parts)

        selected_idx = st.radio(
            "Select route variant (sorted by weighted multi-score)",
            options=summary_df.index.tolist(),
            index=st.session_state.get("selected_route_idx", 0),
            format_func=format_route_option,
        )

        st.session_state["selected_route_idx"] = selected_idx
        selected_route = routes[selected_idx]
        route_row = summary_df.loc[selected_idx]

        # Key metrics
        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        with col_k1:
            tt_min = route_row.get("travelTimeMinutes", None)
            st.metric(
                "Total driving time",
                f"{tt_min:.1f} min" if tt_min is not None and not pd.isna(tt_min) else "–",
            )

        with col_k2:
            dist_km, fuel_used_l, cost, co2_val = compute_costs_and_co2(
                route_row, consumption, fuel_price, co2_per_l
            )
            st.metric(
                "Round trip distance",
                f"{dist_km:.1f} km" if dist_km is not None else "–",
            )

        with col_k3:
            if cost is not None:
                st.metric("Estimated cost", f"{cost:.2f} CHF")
            else:
                st.metric("Estimated cost", "–")

        with col_k4:
            if co2_val is not None:
                st.metric("Estimated CO₂", f"{co2_val:.1f} kg")
            else:
                st.metric("Estimated CO₂", "–")

        # Deadline check for return to start
        if deadline_dt is not None and tt_min is not None and not pd.isna(tt_min):
            arrival_last = depart_dt_state + timedelta(minutes=float(tt_min))
            diff_sec = (arrival_last - deadline_dt).total_seconds()
            if diff_sec > 0:
                st.error(
                    f"⚠️ Expected return to start: "
                    f"{arrival_last.strftime('%d.%m.%Y %H:%M')} "
                    f"(delay of {round(diff_sec / 60)} min)"
                )
            else:
                st.success(
                    f"✅ Expected return to start: "
                    f"{arrival_last.strftime('%d.%m.%Y %H:%M')} "
                    f"(buffer of {round(-diff_sec / 60)} min)"
                )

        st.markdown("### Map of selected round trip (with traffic incidents)")

        if start_coord_state is not None:
            m = create_route_map(
                selected_route,
                api_key=TOMTOM_API_KEY,
                start_coord=start_coord_state,
                stop_coords=stop_coords_state,
                stop_addresses=stop_addresses_state,
            )
            if m is not None:
                st_folium(m, width=1000, height=600)
            else:
                st.info("No geometry data found for the selected route.")
        else:
            st.info("No start coordinates available to draw the map.")

    # ------- Tab: Details & ETAs -------
    with tab_details:
        st.markdown("## Route summaries (all variants)")
        st.dataframe(summary_df)

        st.markdown("## ETAs per leg (for selected route)")

        # Recompute ETAs for the selected route to reflect correct timings
        if stops is not None and depart_dt_state is not None:
            etas_df_selected = compute_etas_for_stops(
                selected_route, depart_dt_state, stops
            )
            if etas_df_selected is not None and not etas_df_selected.empty:
                df_show = etas_df_selected.copy()
                df_show["ETA_Arrival"] = df_show["ETA_Arrival"].dt.strftime(
                    "%d.%m.%Y %H:%M"
                )
                st.dataframe(df_show)
            else:
                st.info("No ETA data available for the selected route.")
        else:
            # fall back to previous best route ETAs if available
            if etas_df_state is not None and not etas_df_state.empty:
                df_show = etas_df_state.copy()
                df_show["ETA_Arrival"] = df_show["ETA_Arrival"].dt.strftime(
                    "%d.%m.%Y %H:%M"
                )
                st.dataframe(df_show)
            else:
                st.info("No ETA data available.")

else:
    st.info(
        "Please enter start/end address and stops, "
        "then press «Calculate round trip»."
    )
