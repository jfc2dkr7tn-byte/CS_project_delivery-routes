import urllib.parse as urlparse
import requests as rq
import pandas as pd
import streamlit as st

# ------------------------------------------------------
# Helper-Funktionen
# ------------------------------------------------------

def search_city_coords(city: str, key: str) -> str:
    """
    Sucht die Koordinaten (lat, lon) einer Stadt über die TomTom Search API.
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
        raise ValueError(f"Keine Koordinaten für Stadt '{city}' gefunden.")

    return f"{latitude},{longitude}"


def get_routes_info(start: str, stop: str, depart_at: str, key: str) -> pd.DataFrame:
    """
    Ruft alternative Routen zwischen start und stop ab und gibt
    eine DataFrame mit den Summary-Infos zurück.
    """
    base_url = f"https://api.tomtom.com/routing/1/calculateRoute/{start}:{stop}/json"
    params = {
        "maxAlternatives": 5,
        "instructionsType": "text",
        "departAt": depart_at,
        "key": key
    }

    response = rq.get(base_url, params=params)
    response.raise_for_status()
    json_response = response.json()

    routes = json_response.get("routes", [])
    if not routes:
        raise ValueError("TomTom hat keine Routen zurückgegeben.")

    summaries = [r["summary"] for r in routes]
    df = pd.DataFrame(summaries)

    # Fahrzeit in Minuten hinzufügen
    if "travelTimeInSeconds" in df.columns:
        df["travelTimeMinutes"] = (df["travelTimeInSeconds"] / 60).round(1)

    return df


# ------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------

st.title("🚚 Routenempfehlung für Paketboten (TomTom API)")
st.write(
    "Diese App verwendet die TomTom Routing API, um einem Paketboten die "
    "schnellste Route zwischen zwei Städten vorzuschlagen."
)

# --- Eingaben ---
api_key = st.text_input("TomTom API Key", type="password")

col1, col2 = st.columns(2)
with col1:
    departure_point = st.text_input("Start (Stadt)", value="Los Angeles")
with col2:
    delivery_point = st.text_input("Ziel (Stadt)", value="Irvine")

departure_time = st.text_input(
    "Abfahrtszeit (ISO-Format)",
    value="2022-02-10T08:00:00"
)

# --- Button ---
if st.button("Beste Route berechnen"):

    if not api_key:
        st.error("Bitte gib zuerst deinen TomTom API Key ein.")
    else:
        try:
            with st.spinner("Berechne Routen…"):

                # Koordinaten bestimmen
                dep_coords = search_city_coords(departure_point, api_key)
                del_coords = search_city_coords(delivery_point, api_key)

                # Routing abrufen
                summary_df = get_routes_info(dep_coords, del_coords, departure_time, api_key)

                # Schnellste Route bestimmen
                time_col = (
                    "travelTimeInSeconds"
                    if "travelTimeInSeconds" in summary_df.columns
                    else None
                )

                if time_col:
                    fastest_idx = summary_df[time_col].idxmin()
                    fastest_route = summary_df.loc[fastest_idx]

                    st.subheader("🚀 Schnellste Route")
                    st.write(f"**Route Nr. {fastest_idx+1}**")
                    st.write(f"**Fahrzeit:** {fastest_route['travelTimeMinutes']} Minuten")

                # Ganze Tabelle anzeigen
                st.subheader("🔎 Alle Routen-Alternativen")
                st.dataframe(summary_df)

        except Exception as e:
            st.error(f"Fehler: {e}")
