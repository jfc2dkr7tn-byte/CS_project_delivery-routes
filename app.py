import urllib.parse as urlparse
import requests as rq
import pandas as pd
import streamlit as st
from typing import List, Tuple, Dict
import time

# ------------------------------------------------------
# Globale Cache-Struktur für Fahrzeiten
# ------------------------------------------------------
# Key: (start_coord, stop_coord, depart_at), Value: travelTimeInSeconds
travel_time_cache: Dict[Tuple[str, str, str], int] = {}

# ------------------------------------------------------
# Helper-Funktionen
# ------------------------------------------------------

def search_address_coords(address: str, key: str) -> str:
    """
    Sucht die Koordinaten (lat, lon) einer Adresse über die TomTom Search API.
    Beispiel:
        'Bahnhofstrasse 1, 8001 Zürich'
        '1600 Amphitheatre Parkway, Mountain View, CA'
    """
    base_url = "https://api.tomtom.com/search/2/search/" + urlparse.quote(address) + ".json"
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
        raise ValueError(f"Keine Koordinaten für Adresse '{address}' gefunden.")

    return f"{latitude},{longitude}"


def get_travel_time_seconds_between(
    start: str,
    stop: str,
    depart_at: str,
    key: str,
    max_retries: int = 3
) -> int:
    """
    Holt die Fahrzeit in Sekunden zwischen zwei Koordinaten (lat,lon)
    über die TomTom Routing API (nur eine Route, keine Alternativen),
    mit einfachem Caching und 429-Handling.
    """
    cache_key = (start, stop, depart_at)
    if cache_key in travel_time_cache:
        return travel_time_cache[cache_key]

    base_url = f"https://api.tomtom.com/routing/1/calculateRoute/{start}:{stop}/json"
    params = {
        "maxAlternatives": 0,
        "instructionsType": "text",
        "departAt": depart_at,
        "key": key
    }

    last_error = None
    for attempt in range(max_retries):
        response = rq.get(base_url, params=params)
        if response.status_code == 429:
            # Zu viele Anfragen -> kurz warten und nochmal probieren
            wait_seconds = 0.5 * (attempt + 1)
            time.sleep(wait_seconds)
            last_error = rq.exceptions.HTTPError("429 Too Many Requests")
            continue

        # andere Fehler normal behandeln
        response.raise_for_status()
        json_response = response.json()

        routes = json_response.get("routes", [])
        if not routes:
            raise ValueError("TomTom hat keine Route zwischen zwei Punkten zurückgegeben.")

        summary = routes[0]["summary"]
        travel_time = summary["travelTimeInSeconds"]
        travel_time_cache[cache_key] = travel_time
        return travel_time

    # wenn wir hier landen, haben alle Versuche 429 geliefert
    raise rq.exceptions.HTTPError(
        "TomTom API: Zu viele Anfragen (429). Bitte kurz warten oder weniger Stopps verwenden."
    ) from last_error


def build_greedy_route(
    start_coord: str,
    via_coords: List[str],
    end_coord: str,
    depart_at: str,
    key: str
) -> Tuple[List[int], List[int]]:
    """
    Greedy-Heuristik mit *TomTom-Fahrzeiten*:
    - Start bei start_coord
    - Wähle immer den Zwischenhalt, der von der aktuellen Position die geringste Fahrzeit (laut API) hat
    - Am Schluss noch die Fahrzeit vom letzten Zwischenhalt zum Endpunkt

    Gibt:
        - order_indices: Reihenfolge der Indizes der via_coords
        - leg_times: Liste der Fahrzeiten (Sekunden) je Abschnitt
                     (Start -> Stop1, Stop1 -> Stop2, ..., letzter Stop -> Endpunkt)
    """
    if not via_coords:
        # Keine Zwischenhalte, nur ein direkter Leg Start -> End
        only_leg_time = get_travel_time_seconds_between(start_coord, end_coord, depart_at, key)
        return [], [only_leg_time]

    remaining_indices = list(range(len(via_coords)))
    current_coord = start_coord
    order_indices: List[int] = []
    leg_times: List[int] = []

    # Solange noch Zwischenhalte übrig sind, immer den "nächst-schnellen" nehmen
    while remaining_indices:
        best_idx = None
        best_time = None

        for idx in remaining_indices:
            candidate_coord = via_coords[idx]
            t_sec = get_travel_time_seconds_between(
                current_coord, candidate_coord, depart_at, key
            )
            if (best_time is None) or (t_sec < best_time):
                best_time = t_sec
                best_idx = idx

        order_indices.append(best_idx)
        leg_times.append(best_time)
        current_coord = via_coords[best_idx]
        remaining_indices.remove(best_idx)

    # Letzter Leg: vom letzten Zwischenhalt zum Endpunkt
    last_leg_time = get_travel_time_seconds_between(current_coord, end_coord, depart_at, key)
    leg_times.append(last_leg_time)

    return order_indices, leg_times


# Kleine Hilfsfunktion, um "lat,lon"-Strings in Floats zu parsen
def parse_coord(coord_str: str) -> Tuple[float, float]:
    lat_str, lon_str = coord_str.split(",")
    return float(lat_str), float(lon_str)


# ------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------

st.title("🚚 Optimierte Routenempfehlung (beliebige Stopps mit TomTom-API)")
st.write(
    "Gib Start, Ziel und beliebig viele Stopps **in beliebiger Reihenfolge** ein. "
    "Die App verwendet die TomTom Routing API, um eine heuristisch schnellste Reihenfolge "
    "zu bestimmen (Greedy auf Basis der API-Fahrzeiten)."
)

# --- Eingaben ---
api_key = st.text_input("TomTom API Key", type="password")

st.subheader("🔁 Start- und Zieladresse")
start_address = st.text_input(
    "Startadresse",
    value="Bahnhofstrasse 1, 8001 Zürich",
    help="Gib eine vollständige Adresse ein, z.B. 'Bahnhofstrasse 1, 8001 Zürich'."
)
end_address = st.text_input(
    "Zieladresse",
    value="Hardstrasse 201, 8005 Zürich",
    help="Gib eine vollständige Adresse ein."
)

st.subheader("📍 Zwischenhalte (optional, beliebige Reihenfolge)")
via_addresses_input = st.text_area(
    "Zwischenhalte (eine Adresse pro Zeile, Reihenfolge egal)",
    value="",
    help="Beispiel:\nLimmatquai 1, 8001 Zürich\nEuropaallee 1, 8004 Zürich"
)

# -----------------------------------------------
# SCHÖNE DATUMS- UND ZEITEINGABE
# -----------------------------------------------
st.subheader("⏰ Abfahrtszeit")

col1, col2 = st.columns(2)

with col1:
    departure_date = st.date_input(
        "Datum",
        help="Wähle das Abfahrtsdatum aus"
    )

with col2:
    departure_time_input = st.time_input(
        "Zeit",
        help="Wähle die Abfahrtszeit aus"
    )

# ISO-Format für TomTom erzeugen (YYYY-MM-DDTHH:MM:SS)
departure_time = f"{departure_date.isoformat()}T{departure_time_input.strftime('%H:%M:%S')}"

# Begrenzung der Zwischenhalte, damit die Anzahl API-Calls überschaubar bleibt
MAX_VIAS = 6  # kannst du anpassen


# --- Button ---
if st.button("Schnellste Reihenfolge & Route berechnen"):

    if not api_key:
        st.error("Bitte gib zuerst deinen TomTom API Key ein.")
    else:
        try:
            with st.spinner("Berechne optimierte Route mit TomTom…"):

                # Koordinaten Start und Ziel
                start_coords = search_address_coords(start_address, api_key)
                end_coords = search_address_coords(end_address, api_key)

                # Koordinaten & Labels für Zwischenstopps (Reihenfolge egal)
                via_coords: List[str] = []
                via_labels: List[str] = []
                if via_addresses_input.strip():
                    via_lines = [
                        line.strip()
                        for line in via_addresses_input.split("\n")
                        if line.strip()
                    ]

                    if len(via_lines) > MAX_VIAS:
                        st.warning(
                            f"Es wurden {len(via_lines)} Zwischenhalte eingegeben. "
                            f"Aus API-Gründen werden nur die ersten {MAX_VIAS} berücksichtigt."
                        )
                        via_lines = via_lines[:MAX_VIAS]

                    for addr in via_lines:
                        coord = search_address_coords(addr, api_key)
                        via_coords.append(coord)
                        via_labels.append(addr)

                # Greedy-Route auf Basis der *API-Fahrzeiten*
                order_indices, leg_times_sec = build_greedy_route(
                    start_coord=start_coords,
                    via_coords=via_coords,
                    end_coord=end_coords,
                    depart_at=departure_time,
                    key=api_key
                )

                # Labels in optimierte Reihenfolge bringen
                if via_coords:
                    optimized_via_labels = [via_labels[i] for i in order_indices]
                else:
                    optimized_via_labels = []

                # Gesamtzeit
                total_time_sec = sum(leg_times_sec)
                total_time_min = total_time_sec / 60

                # Route als Label-Liste
                route_labels: List[str] = [start_address]
                route_labels.extend(optimized_via_labels)
                route_labels.append(end_address)

                # DataFrame der einzelnen Legs bauen
                leg_rows = []
                current_label = start_address
                leg_index = 0

                if via_coords:
                    # Legs Start -> Stop1 ... StopN
                    for idx in order_indices:
                        next_label = via_labels[idx]
                        leg_rows.append({
                            "Leg": leg_index + 1,
                            "Von": current_label,
                            "Nach": next_label,
                            "Fahrzeit (Minuten)": round(leg_times_sec[leg_index] / 60, 1)
                        })
                        current_label = next_label
                        leg_index += 1

                    # Letzter Leg StopN -> Ziel
                    leg_rows.append({
                        "Leg": leg_index + 1,
                        "Von": current_label,
                        "Nach": end_address,
                        "Fahrzeit (Minuten)": round(leg_times_sec[leg_index] / 60, 1)
                    })
                else:
                    # Nur ein Leg: Start -> Ziel
                    leg_rows.append({
                        "Leg": 1,
                        "Von": start_address,
                        "Nach": end_address,
                        "Fahrzeit (Minuten)": round(leg_times_sec[0] / 60, 1)
                    })

                df_legs = pd.DataFrame(leg_rows)

                # -------- Anzeige --------
                st.subheader("🚀 Optimierte Reihenfolge der Stopps (TomTom-basiert)")
                for i, label in enumerate(route_labels):
                    if i == 0:
                        st.write(f"Start: {label}")
                    elif i == len(route_labels) - 1:
                        st.write(f"Ziel:  {label}")
                    else:
                        st.write(f"Stopp {i}: {label}")

                st.markdown(f"**Geschätzte Gesamtfahrzeit:** {total_time_min:.1f} Minuten")

                # 🗺️ Route-Map
                st.subheader("🗺️ Karte der Route")

                route_points: List[Dict[str, float]] = []

                # Startpunkt
                lat, lon = parse_coord(start_coords)
                route_points.append({
                    "name": f"Start: {start_address}",
                    "lat": lat,
                    "lon": lon
                })

                # Zwischenstopps in optimierter Reihenfolge
                for idx in order_indices:
                    lat, lon = parse_coord(via_coords[idx])
                    route_points.append({
                        "name": f"Stopp: {via_labels[idx]}",
                        "lat": lat,
                        "lon": lon
                    })

                # Zielpunkt
                lat, lon = parse_coord(end_coords)
                route_points.append({
                    "name": f"Ziel: {end_address}",
                    "lat": lat,
                    "lon": lon
                })

                df_route = pd.DataFrame(route_points)

                # einfache Punkt-Karte (Route in Punktreihenfolge sichtbar)
                st.map(df_route, latitude="lat", longitude="lon")

                st.subheader("🧾 Details zu den einzelnen Abschnitten")
                st.dataframe(df_legs)

        except rq.exceptions.HTTPError as e:
            st.error(f"HTTP-Fehler: {e}")
        except Exception as e:
            st.error(f"Fehler: {e}")