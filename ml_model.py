import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import numpy as np

# ---------- DATASET ----------
def generate_dataset(n=3000, random_state=42):
    np.random.seed(random_state)

    distance_km = np.random.uniform(5, 1200, n)
    stops = np.random.randint(0, 6, n)
    weight_kg = np.random.uniform(1800, 40000, n)
    max_speed = np.random.uniform(70, 130, n)
    engine = np.random.choice(["diesel", "electric", "hydrogen"], n, p=[0.7, 0.2, 0.1])
    elevation_gain = np.random.uniform(0, 1200, n)
    traffic = np.random.uniform(0.8, 1.6, n)

    base_time_min = (distance_km / 80) * 60

    weight_factor = 1 + (weight_kg / 40000) * 0.18
    elevation_factor = 1 + (elevation_gain / 1200) * 0.25
    speed_factor = 90 / max_speed
    engine_factor = np.array([
        {"diesel": 1.00, "electric": 1.08, "hydrogen": 1.03}[e] for e in engine
    ])
    stop_delay_min = stops * np.random.uniform(4, 12, n)

    travel_time_min = (
        base_time_min *
        weight_factor *
        elevation_factor *
        speed_factor *
        engine_factor *
        traffic
        + stop_delay_min
    )

    df = pd.DataFrame({
        "distance_km": distance_km,
        "stops": stops,
        "weight_kg": weight_kg,
        "max_speed": max_speed,
        "engine": engine,
        "elevation_gain": elevation_gain,
        "traffic": traffic,
        "base_time_min": base_time_min,
        "travel_time_min": travel_time_min
    })

    return df


# ---------- TRAIN MODEL ----------
def train_model(df):
    X = df[["distance_km", "stops", "weight_kg", "max_speed", "engine",
            "elevation_gain", "traffic"]]
    y = df["travel_time_min"]

    transformer = ColumnTransformer([
        ("engine", OneHotEncoder(), ["engine"])
    ], remainder="passthrough")

    model = Pipeline([
        ("prep", transformer),
        ("rf", RandomForestRegressor(n_estimators=300, random_state=42))
    ])

    model.fit(X, y)
    joblib.dump(model, "travel_time_model.pkl")
    print("MODEL SAVED: travel_time_model.pkl")
    return model


# ---------- LOAD MODEL ----------
def load_model():
    return joblib.load("travel_time_model.pkl")


# ---------- PREDICT ----------
def predict_corrected_time(distance_km, stops, weight_kg,
                           max_speed, engine, elevation_gain, traffic):

    model = load_model()

    X = pd.DataFrame([{
        "distance_km": distance_km,
        "stops": stops,
        "weight_kg": weight_kg,
        "max_speed": max_speed,
        "engine": engine,
        "elevation_gain": elevation_gain,
        "traffic": traffic
    }])

    return float(model.predict(X)[0])