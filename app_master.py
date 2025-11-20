import urllib.parse as urlparse
import requests as rq
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from datetime import datetime, date, time, timedelta
from typing import List, Optional

# ======================================================
# Konstanten
# ======================================================

MAX_STOPPS = 10  # maximale Anzahl Stopps zwischen Start und Ziel (Rundtour)

# ======================================================
# Session State initialisieren
# ======================================================

if "routes" not in st.session_state:
    st.session_state["routes"] = None
    st.session_state["summary_df"] = None
    st.session_state["stops"] = None
    st.session_state["depart_dt"] = None
    st.session_state["etas_df"] = None
    st.session_state["start_coord"] = None
    st.session_state["stop_coords"] = None
    st.session_state["stop_addresses"] = None

# ======================================================
# Helper-Funktionen
# ======================================================

def search_city_coords(city: str, key: str) -> str:
    """
    Sucht die Koordinaten (lat, lon) einer Adresse/Stadt ueber die TomTom Search API.
    Gibt einen String 'lat,lon' zurueck, der direkt in calculateRoute verwendet werden kann.
    """
    base_url = "https://api.tomtom.com/search/2/search/" + urlparse.quote(city) + ".json"
    params = {
        "minFuzzyLevel": 1,
        "maxFuzzyLevel": 2,
        "view": "Unified",
        "relatedPois": "off",
        "key": key
    }

    response = rq.get(base_url, params=params)
    response.raise_for_status()
    json_response = response.json()

    try:
        latitude = json_response["results"][0]["position"]["lat"]
        longitude = json_response["results"][0]["position"]["lon"]
    except (IndexError, KeyError):
        raise ValueError(f"Keine Koordinaten fuer '{city}' gefunden.")

    return f"{latitude},{longitude}"


def build_roundtrip_locations_string(start_coord: str, stop_coords: List[str]) -> str:
    """
    Baut die locations-String fuer calculateRoute als Rundtour:
    Start -> Stopps -> Start
    'latS,lonS:lat1,lon1:...:latN,lonN:latS,lonS'
    """
    locations = [start_coord] + stop_coords + [start_coord]
    return ":".join(locations)


def get_routes_info(
    locations_str: str,
    depart_at: str,
    key: str,
    consumption_l_per_100km: Optional[float] = None,
    fuel_price_per_l: Optional[float] = None,
):
    """
    Ruft TomTom calculateRoute EINMAL auf und liefert:
    - summary_df mit den wichtigsten Metriken der besten Route
    - routes (Liste mit genau dieser besten Route)

    Es werden Alternativrouten angefragt und anschliessend anhand
    Fahrzeit, Verbrauch und Kosten bewertet.
    """
    base_url = f"https://api.tomtom.com/routing/1/calculateRoute/{locations_str}/json"
    params = {
        "maxAlternatives": 3,           # bis zu 4 Routen (1+3 Alternativen)
        "instructionsType": "text",
        "departAt": depart_at,          # RFC-3339 Datum/Zeit, Timezone optional 
        "traffic": "true",
        "routeType": "fastest",         # Optimierung nach Fahrzeit 
        "travelMode": "car",
        "key": key,
    }

    response = rq.get(base_url, params=params)
    response.raise_for_status()
    json_response = response.json()

    routes = json_response.get("routes", [])
    if not routes:
        raise ValueError("TomTom hat keine Routen zurueckgegeben.")

    summaries = [r["summary"] for r in routes]
    df = pd.DataFrame(summaries)

    # Fahrzeit in Minuten und Distanz in km hinzufuegen
    if "travelTimeInSeconds" in df.columns:
        df["travelTimeMinutes"] = (df["travelTimeInSeconds"] / 60).round(1)
    if "lengthInMeters" in df.columns:
        df["distanceKm"] = (df["lengthInMeters"] / 1000).round(2)

    # Falls keine Verbrauch/Kosten-Angaben vorhanden sind, nimm einfach erste Route
    if consumption_l_per_100km is None or fuel_price_per_l is None or "distanceKm" not in df.columns:
        df = df.iloc[[0]].reset_index(drop=True)
        routes = [routes[0]]
        return df, routes

    # Verbrauch und Kosten pro Route berechnen
    dist_km = df["distanceKm"]
    fuel_used_l = dist_km * (consumption_l_per_100km / 100.0)
    cost_chf = fuel_used_l * fuel_price_per_l

    df["fuelUsedL"] = fuel_used_l
    df["costCHF"] = cost_chf

    # Normalisierung fuer Multi-Kriterium-Score
    def normalize(series: pd.Series) -> pd.Series:
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            # alle Werte gleich -> alle 0
            return pd.Series([0.0] * len(series), index=series.index)
        return (series - min_val) / (max_val - min_val)

    time_norm = normalize(df["travelTimeInSeconds"])
    fuel_norm = normalize(df["fuelUsedL"])
    cost_norm = normalize(df["costCHF"])

    # Einfaches gleichgewichtetes Multi-Kriterium:
    # kleinere Werte = besser
    df["multiScore"] = time_norm + fuel_norm + cost_norm

    best_idx = df["multiScore"].idxmin()

    # Nur beste Route zurueckgeben
    best_df = df.loc[[best_idx]].reset_index(drop=True)
    best_route = [routes[best_idx]]

    return best_df, best_route


def create_route_map(
    routes,
    start_coord: Optional[str] = None,
    stop_coords: Optional[List[str]] = None,
    stop_addresses: Optional[List[str]] = None
):
    """
    Erstellt eine Folium-Karte mit der Geometrie der (einzigen) Route
    und Markern fuer Start/Ziel sowie alle Zwischenstopps.
    """
    if routes is None or len(routes) == 0:
        return None

    route = routes[0]

    # Alle Punkte (lat, lon) sammeln
    poly_points = []
    for leg in route.get("legs", []):
        for p in leg.get("points", []):
            poly_points.append((p["latitude"], p["longitude"]))

    if not poly_points:
        return None

    # Karte zentrieren (Mittelwert der Koordinaten)
    avg_lat = sum(p[0] for p in poly_points) / len(poly_points)
    avg_lon = sum(p[1] for p in poly_points) / len(poly_points)

    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=11)

    # Start-/Zielmarker
    if start_coord is not None:
        try:
            s_lat, s_lon = map(float, start_coord.split(","))
        except Exception:
            s_lat, s_lon = poly_points[0]
    else:
        s_lat, s_lon = poly_points[0]

    folium.Marker(
        location=[s_lat, s_lon],
        tooltip="Start / Ziel",
        icon=folium.Icon(color="green")
    ).add_to(m)

    # Zwischenstopps markieren
    if stop_coords:
        for i, coord_str in enumerate(stop_coords):
            try:
                lat, lon = map(float, coord_str.split(","))
            except Exception:
                continue

            if stop_addresses and i < len(stop_addresses):
                label = f"Stopp {i+1}: {stop_addresses[i]}"
            else:
                label = f"Stopp {i+1}"

            folium.Marker(
                location=[lat, lon],
                tooltip=label,
                icon=folium.Icon(color="blue", icon="flag")
            ).add_to(m)

    # Route als Linie einzeichnen
    folium.PolyLine(poly_points, weight=6, opacity=0.8).add_to(m)

    return m


def compute_etas_for_stops(route, depart_dt: datetime, stop_names: List[str]) -> pd.DataFrame:
    """
    Berechnet ETA (Ankunftszeiten) pro Leg anhand der Legs und der Abfahrtszeit.
    stop_names: Liste inkl. Start, Stopps und wieder Start (Rundtour!)
    Rueckgabe: DataFrame mit Von, Nach, Fahrzeit, ETA.
    """
    legs = route.get("legs", [])
    if not legs:
        return pd.DataFrame()

    records = []
    current_time = depart_dt

    # legs[i] fuehrt von stop_names[i] zu stop_names[i+1]
    for i, leg in enumerate(legs):
        leg_summary = leg.get("summary", {})
        travel_sec = leg_summary.get("travelTimeInSeconds", 0)
        current_time = current_time + timedelta(seconds=travel_sec)
        if i + 1 < len(stop_names):
            records.append({
                "Von": stop_names[i],
                "Nach": stop_names[i + 1],
                "FahrzeitLegMin": round(travel_sec / 60, 1),
                "ETA_Ankunft": current_time
            })

    return pd.DataFrame(records)


def compute_costs_and_co2(summary_row: pd.Series, consumption_l_per_100km: float,
                          fuel_price_per_l: float, co2_per_l: float):
    """
    Schaetzt Kosten und CO2 anhand Distanz und Verbraeuchen.
    """
    if "distanceKm" not in summary_row:
        return None, None, None, None

    dist_km = summary_row["distanceKm"]
    fuel_used_l = dist_km * (consumption_l_per_100km / 100.0)
    cost = fuel_used_l * fuel_price_per_l
    co2 = fuel_used_l * co2_per_l
    return dist_km, fuel_used_l, cost, co2

# ======================================================
# Streamlit App
# ======================================================

st.set_page_config(page_title="Delivery Rundtour mit TomTom", layout="wide")

st.title("🚚 Delivery Rundtour (TomTom API)")
st.caption("Start und Ziel sind gleich, Stopps begrenzt, nur ein TomTom-Call pro Berechnung.")

# ---------------- Sidebar: Konfiguration ----------------

st.sidebar.header("🔑 API & Fahrzeug")

api_key = st.sidebar.text_input("TomTom API Key", type="password")

st.sidebar.subheader("Fahrzeug / Kosten")
consumption = st.sidebar.number_input(
    "Verbrauch (l/100 km)",
    min_value=1.0,
    max_value=100.0,  # auf 100 l/100 km erhoeht
    value=8.0,
    step=0.5
)
fuel_price = st.sidebar.number_input("Treibstoffpreis (CHF pro Liter)", min_value=0.0, max_value=5.0, value=1.9, step=0.1)
co2_per_l = st.sidebar.number_input("CO₂ je Liter (kg)", min_value=0.0, max_value=5.0, value=2.32, step=0.01)

st.sidebar.markdown("---")
st.sidebar.info("Maximal {} Stopps zwischen Start und Ziel. Jede Berechnung macht genau einen TomTom-Request."
                .format(MAX_STOPPS))

# ---------------- Haupteingaben ----------------

st.markdown("## 🧭 Tour-Eingabe (Rundtour)")

col_start, col_stops = st.columns(2)

with col_start:
    start_address = st.text_input("Start-/Ziel-Adresse / -Stadt", value="Zuerich HB")

with col_stops:
    st.write("Zwischenstopps (eine Adresse pro Zeile, maximal {}):".format(MAX_STOPPS))
    stops_text = st.text_area(
        "Stopps",
        value="ETH Zuerich\nBahnhof Winterthur\nFlughafen Zuerich"
    )

st.markdown("### Abfahrtszeit")
col_d, col_t = st.columns(2)
with col_d:
    dep_date = st.date_input("Datum", value=date.today())
with col_t:
    dep_time = st.time_input("Uhrzeit", value=time(8, 0))

depart_dt = datetime.combine(dep_date, dep_time)
depart_iso = depart_dt.isoformat(timespec="seconds")

st.markdown("### Optional: Spaeteste Rueckkehr zum Start")
use_deadline = st.checkbox("Deadline fuer Rueckkehr setzen?")
deadline_dt = None
if use_deadline:
    col_dd, col_dt = st.columns(2)
    with col_dd:
        deadline_date = st.date_input("Deadline-Datum", value=date.today())
    with col_dt:
        deadline_time = st.time_input("Deadline-Uhrzeit", value=time(12, 0))
    deadline_dt = datetime.combine(deadline_date, deadline_time)

# ------------------------------------------------------
# Routing ausloesen
# ------------------------------------------------------

if st.button("Rundtour berechnen"):
    if not api_key:
        st.error("Bitte gib zuerst deinen TomTom API Key ein.")
    else:
        try:
            with st.spinner("Geocoding und Routenberechnung laufen…"):

                stop_addresses = [line.strip() for line in stops_text.split("\n") if line.strip()]
                if not start_address:
                    st.error("Bitte eine Start-/Zieladresse angeben.")
                    st.stop()

                if len(stop_addresses) == 0:
                    st.error("Bitte mindestens einen Zwischenstopp angeben.")
                    st.stop()

                if len(stop_addresses) > MAX_STOPPS:
                    st.error(f"Es sind maximal {MAX_STOPPS} Stopps erlaubt. "
                             f"Du hast {len(stop_addresses)} angegeben.")
                    st.stop()

                # Start & Stopps geokodieren
                start_coord = search_city_coords(start_address, api_key)
                stop_coords = [search_city_coords(addr, api_key) for addr in stop_addresses]

                # locations-String fuer calculateRoute (Rundtour, Reihenfolge wie eingegeben)
                loc_str = build_roundtrip_locations_string(start_coord, stop_coords)

                # Routing (genau EIN Call, mit Alternativen)
                summary_df, routes = get_routes_info(
                    locations_str=loc_str,
                    depart_at=depart_iso,
                    key=api_key,
                    consumption_l_per_100km=consumption,
                    fuel_price_per_l=fuel_price
                )

                best_route = routes[0]

                # Stop-Namen-Liste fuer ETA: Start, Stopps (wie eingegeben), wieder Start
                stop_names = (
                    ["Start/Ziel"]
                    + [f"Stopp {i+1}: {addr}" for i, addr in enumerate(stop_addresses)]
                    + ["Zurueck zu Start/Ziel"]
                )

                # ETAs
                etas_df = compute_etas_for_stops(best_route, depart_dt, stop_names)

                # Alles im Session State speichern
                st.session_state["routes"] = routes
                st.session_state["summary_df"] = summary_df
                st.session_state["stops"] = stop_names
                st.session_state["depart_dt"] = depart_dt
                st.session_state["etas_df"] = etas_df
                st.session_state["start_coord"] = start_coord
                st.session_state["stop_coords"] = stop_coords
                st.session_state["stop_addresses"] = stop_addresses

        except Exception as e:
            st.error(f"Fehler bei der Berechnung: {e}")

# ------------------------------------------------------
# Ergebnis-Ansicht, wenn bereits Daten vorhanden sind
# ------------------------------------------------------

routes = st.session_state["routes"]
summary_df = st.session_state["summary_df"]
stops = st.session_state["stops"]
depart_dt_state = st.session_state["depart_dt"]
etas_df = st.session_state["etas_df"]

if routes is not None and summary_df is not None:

    tab_route, tab_details = st.tabs(["🗺️ Route & Kennzahlen", "📋 Details & ETAs"])

    # ------- Tab Route & Kennzahlen -------
    with tab_route:
        st.markdown("## Route & Kennzahlen (Rundtour)")

        route_row = summary_df.loc[0]

        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        with col_k1:
            tt_min = route_row.get("travelTimeMinutes", None)
            st.metric("Gesamt-Fahrzeit", f"{tt_min} min" if tt_min is not None else "–")
        with col_k2:
            dist_km, fuel_used_l, cost, co2 = compute_costs_and_co2(
                route_row, consumption, fuel_price, co2_per_l
            )
            st.metric("Distanz Rundtour", f"{dist_km} km" if dist_km is not None else "–")
        with col_k3:
            if cost is not None:
                st.metric("Schaetzung Kosten", f"{cost:.2f} CHF")
            else:
                st.metric("Schaetzung Kosten", "–")
        with col_k4:
            if co2 is not None:
                st.metric("CO₂ geschaetzt", f"{co2:.1f} kg")
            else:
                st.metric("CO₂ geschaetzt", "–")

        # Deadline-Bewertung fuer Rueckkehr zum Start
        if deadline_dt is not None and tt_min is not None:
            arrival_last = depart_dt_state + timedelta(minutes=float(tt_min))
            diff_sec = (arrival_last - deadline_dt).total_seconds()
            if diff_sec > 0:
                st.error(
                    f"⚠️ Voraussichtliche Rueckkehr zum Start: "
                    f"{arrival_last.strftime('%d.%m.%Y %H:%M')} (Verspaetung {round(diff_sec/60)} min)"
                )
            else:
                st.success(
                    f"✅ Voraussichtliche Rueckkehr zum Start: "
                    f"{arrival_last.strftime('%d.%m.%Y %H:%M')} (Puffer {round(-diff_sec/60)} min)"
                )

        st.markdown("### Karte der Rundtour")
        m = create_route_map(
            routes,
            start_coord=st.session_state.get("start_coord"),
            stop_coords=st.session_state.get("stop_coords"),
            stop_addresses=st.session_state.get("stop_addresses")
        )
        if m is not None:
            st_folium(m, width=1000, height=600)
        else:
            st.info("Keine Geometrie-Daten fuer die Route gefunden.")

    # ------- Tab Details & ETAs -------
    with tab_details:
        st.markdown("## Routen-Summary")
        st.dataframe(summary_df)

        st.markdown("## ETAs pro Leg")
        if etas_df is not None and not etas_df.empty:
            df_show = etas_df.copy()
            df_show["ETA_Ankunft"] = df_show["ETA_Ankunft"].dt.strftime("%d.%m.%Y %H:%M")
            st.dataframe(df_show)
        else:
            st.info("Keine ETA-Daten verfuegbar.")

else:
    st.info("Bitte gib API Key, Start/Ziel und Stopps ein und klicke auf «Rundtour berechnen».")
