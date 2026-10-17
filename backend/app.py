from flask import Flask, request, jsonify
from flask_cors import CORS
import calendar
import os
import traceback
from datetime import date, timedelta
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# ─────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

BASE_DIR  = os.path.join(os.path.dirname(__file__), "state_models")
DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset", "energy_dataset.csv")

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

TIME_STEPS = 24
TARGETS    = ["wind_energy", "solar_energy", "other_renewable_energy"]

FEATURES: list[str] = [
    "wind_energy", "solar_energy", "other_renewable_energy",
    "month", "quarter", "year", "dayofyear", "season",
]
for _t in TARGETS:
    for _lag in [1, 2, 3, 6, 12]:
        FEATURES.append(f"{_t}_lag_{_lag}")
    for _win in [3, 6, 12]:
        FEATURES.append(f"{_t}_roll_{_win}")
# Total: 32 features

TARGET_COL_MAP: dict[str, str] = {
    "wind":            "wind_energy",
    "solar":           "solar_energy",
    "other_renewable": "other_renewable_energy",
}
TARGET_IDX_MAP: dict[str, int] = {k: FEATURES.index(v) for k, v in TARGET_COL_MAP.items()}

LAST_DATA_DATE = date(2025, 4, 27)

MAX_MONTHS = 12
MAX_DAYS   = 365

# ─────────────────────────────────────────────────────────────
# DATASET PRE-LOADING
# ─────────────────────────────────────────────────────────────

_state_cache: dict[str, pd.DataFrame] = {}
_state_lookup: dict[str, str] = {}

def _normalize(s: str) -> str:
    """Case- and whitespace-insensitive key for matching state names."""
    return " ".join(s.replace("_", " ").split()).strip().lower()

def _compute_features(grp: pd.DataFrame) -> pd.DataFrame:
    """Add temporal + lag + rolling features to a single-state daily DataFrame."""
    grp = grp.copy().sort_values("date").reset_index(drop=True)
    grp["month"]     = grp["date"].dt.month
    grp["quarter"]   = grp["date"].dt.quarter
    grp["year"]      = grp["date"].dt.year
    grp["dayofyear"] = grp["date"].dt.dayofyear
    grp["season"]    = ((grp["month"] % 12) // 3) + 1
    for t in TARGETS:
        for lag in [1, 2, 3, 6, 12]:
            grp[f"{t}_lag_{lag}"] = grp[t].shift(lag)
        for win in [3, 6, 12]:
            grp[f"{t}_roll_{win}"] = grp[t].shift(1).rolling(win).mean()
    return grp

def _load_dataset() -> None:
    global _state_cache, _state_lookup
    if not os.path.exists(DATA_PATH):
        print(f"[WARN] Dataset not found at {DATA_PATH}. Predictions will fail.")
        return

    raw = pd.read_csv(DATA_PATH)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values(["state_name", "date"])

    for state_name, grp in raw.groupby("state_name"):
        grp = _compute_features(grp)
        grp = grp.dropna(subset=FEATURES).reset_index(drop=True)
        if len(grp) >= TIME_STEPS:
            _state_cache[state_name] = grp

    _state_lookup = {_normalize(name): name for name in _state_cache}

    print(f"[INFO] Dataset loaded – {len(_state_cache)} states ready.")
    print(f"[INFO] State names found in dataset: {sorted(_state_cache.keys())}")

def _resolve_state(state_key: str) -> str | None:
    """Map an incoming key like 'Tamil_Nadu' to the exact CSV state_name,
    regardless of case or spacing differences (FIX #7)."""
    return _state_lookup.get(_normalize(state_key))

_load_dataset()

# ─────────────────────────────────────────────────────────────
# MODEL / SCALER HELPERS
# ─────────────────────────────────────────────────────────────

def _load_model(state_key: str, energy: str):
    folder_state = state_key.replace(" ", "_")

    path = os.path.join(
        BASE_DIR,
        folder_state,
        energy,
        "best_model.keras"
    )

    print("Model Path :", path)

    if not os.path.exists(path):
        print("Model NOT FOUND")
        return None

    print("Model FOUND")
    return tf.keras.models.load_model(path)

def _load_scaler(state_key: str, energy: str):
    folder_state = state_key.replace(" ", "_")

    path = os.path.join(
        BASE_DIR,
        folder_state,
        energy,
        "scaler.pkl"
    )

    print("Scaler Path :", path)

    if not os.path.exists(path):
        print("Scaler NOT FOUND")
        return None

    print("Scaler FOUND")
    return joblib.load(path)

# ─────────────────────────────────────────────────────────────
# DATE FEATURE HELPER
# ─────────────────────────────────────────────────────────────

def _date_features(d: date) -> dict:
    m = d.month
    return {
        "month":     m,
        "quarter":   (m - 1) // 3 + 1,
        "year":      d.year,
        "dayofyear": d.timetuple().tm_yday,
        "season":    ((m % 12) // 3) + 1,
    }

# ─────────────────────────────────────────────────────────────
# SHARED DAY-BY-DAY SIMULATION CORE
# ─────────────────────────────────────────────────────────────

def _simulate_days(model, scaler, state_df: pd.DataFrame, target_col: str,
                    target_idx: int, n_features: int, total_days: int):
    """Runs the recursive day-by-day forecast for `total_days` steps,
    starting the day after LAST_DATA_DATE. Returns a list of (date, value)."""

    last_window = state_df.tail(TIME_STEPS)
    X = scaler.transform(last_window[FEATURES].values) \
              .reshape(1, TIME_STEPS, n_features)

    history: dict[str, list[float]] = {
        col: list(state_df[col].values) for col in TARGETS
    }

    daily_preds: list[tuple[date, float]] = []
    current_date = LAST_DATA_DATE

    for _ in range(total_days):
        pred_scaled = float(model.predict(X, verbose=0)[0][0])

        dummy = np.zeros((1, n_features))
        dummy[0, target_idx] = pred_scaled
        pred_real = max(0.0, float(scaler.inverse_transform(dummy)[0, target_idx]))

        next_date = current_date + timedelta(days=1)
        current_date = next_date
        daily_preds.append((next_date, pred_real))

        df_row: dict[str, float] = _date_features(next_date)
        df_row["wind_energy"]            = pred_real if target_col == "wind_energy"            else history["wind_energy"][-1]
        df_row["solar_energy"]           = pred_real if target_col == "solar_energy"           else history["solar_energy"][-1]
        df_row["other_renewable_energy"] = pred_real if target_col == "other_renewable_energy" else history["other_renewable_energy"][-1]

        for col in TARGETS:
            h = history[col]
            for lag in [1, 2, 3, 6, 12]:
                df_row[f"{col}_lag_{lag}"] = h[-lag] if len(h) >= lag else h[0]
            for win in [3, 6, 12]:
                df_row[f"{col}_roll_{win}"] = float(np.mean(h[-win:] if len(h) >= win else h))

        for col in TARGETS:
            history[col].append(df_row[col])

        new_vec    = np.array([[df_row[f] for f in FEATURES]], dtype=float)
        new_scaled = scaler.transform(new_vec).reshape(1, 1, n_features)
        X = np.concatenate([X[:, 1:, :], new_scaled], axis=1)

    return daily_preds

# ─────────────────────────────────────────────────────────────
# PREDICT ENDPOINT
# ─────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True) or {}

        state_key = payload.get("state")
        energy    = payload.get("energy")
        unit      = payload.get("unit", "months")
        horizon   = payload.get("horizon", payload.get("months"))

        if state_key is None or energy is None or horizon is None:
            return jsonify({"error": "Missing required field(s): state, energy, horizon"}), 400

        try:
            horizon = int(horizon)
        except (TypeError, ValueError):
            return jsonify({"error": "horizon must be an integer"}), 400

        if unit not in ("months", "days"):
            return jsonify({"error": "unit must be 'months' or 'days'"}), 400

        if energy not in TARGET_COL_MAP:
            return jsonify({"error": f"Unknown energy type: {energy}"}), 400

        if unit == "months":
            if horizon < 1 or horizon > MAX_MONTHS:
                return jsonify({"error": f"horizon must be between 1 and {MAX_MONTHS} months"}), 400
        else:
            if horizon < 1 or horizon > MAX_DAYS:
                return jsonify({"error": f"horizon must be between 1 and {MAX_DAYS} days"}), 400
        resolved_state = _resolve_state(state_key)
        if resolved_state is None:
            return jsonify({
                "error": f"No historical data found for state: {state_key!r}",
                "available_states": sorted(_state_cache.keys()),
            }), 400

        target_col = TARGET_COL_MAP[energy]
        target_idx = TARGET_IDX_MAP[energy]
        print("\n")
        print("=" * 70)
        print("Requested State :", state_key)
        print("Requested Energy:", energy)
        print("Resolved State  :", resolved_state)
        print("=" * 70)

        model = _load_model(state_key, energy)
        scaler = _load_scaler(state_key, energy)

        if model is None or scaler is None:
            return jsonify({
        "status": "under_development",
        "title": "Forecast Under Development",
        "message": (
            "Forecasting for the selected state and energy type is currently "
            "under development. At this stage, EcoPredict supports predictions "
            "only for selected state and energy combinations."
        )
    }), 404

        n_features = scaler.n_features_in_

        state_df = _state_cache[resolved_state]

        if unit == "months":
            total_days = 3
            for i in range(horizon):
                abs_month = 5 + i
                yr = LAST_DATA_DATE.year + (abs_month - 1) // 12
                mo = ((abs_month - 1) % 12) + 1
                total_days += calendar.monthrange(yr, mo)[1]
        else:
            total_days = horizon

        daily_preds = _simulate_days(model, scaler, state_df, target_col,
                                      target_idx, n_features, total_days)

        pred_df = pd.DataFrame(daily_preds, columns=["date", "value"])
        pred_df["date"] = pd.to_datetime(pred_df["date"])

        if unit == "months":
            pred_df["period"] = pred_df["date"].dt.to_period("M")
            pred_df = pred_df[pred_df["period"] >= pd.Period("2025-05")]
            agg = (
                pred_df.groupby("period")["value"]
                .mean()
                .reset_index()
                .head(horizon)
            )
            values = [round(float(r["value"]), 3) for _, r in agg.iterrows()]
            labels = [r["period"].to_timestamp().strftime("%b %Y") for _, r in agg.iterrows()]
            unit_label = "MU / day (daily avg per month)"
        else:
            agg = pred_df.head(horizon)
            values = [round(float(r["value"]), 3) for _, r in agg.iterrows()]
            labels = [r["date"].strftime("%d %b %Y") for _, r in agg.iterrows()]
            unit_label = "MU / day"

        return jsonify({
            "state":         state_key,
            "energy":        energy,
            "unit":          unit,
            "horizon":       horizon,
            "prediction":    round(values[-1], 2) if values else 0,
            "values":        values,
            "labels":        labels,
            "unit_label":    unit_label,
            "forecast_from": labels[0] if labels else "",
            "forecast_to":   labels[-1] if labels else "",
        })

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

# ─────────────────────────────────────────────────────────────
# DEBUG / HEALTH ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "ok",
        "message": "Renewable Energy Forecast API is running. POST to /predict.",
        "states_loaded": len(_state_cache),
    })

@app.route("/available", methods=["GET"])
def available():

    result = []

    if not os.path.isdir(BASE_DIR):
        return jsonify(result)
    for state_folder in os.listdir(BASE_DIR):
        state_path = os.path.join(BASE_DIR, state_folder)
        if not os.path.isdir(state_path):
            continue
        for eng in TARGET_COL_MAP:
            model_path = os.path.join(
                state_path,
                eng,
                "best_model.keras"
            )
            print(model_path)
            if os.path.exists(model_path):
                result.append({
                    "state": state_folder,
                    "energy": eng
                })
    return jsonify(result)

@app.route("/meta", methods=["GET"])
def meta():
    """Combines data availability + model availability so the frontend
    (or you, debugging) can see exactly what's usable right now."""
    models = set()
    if os.path.isdir(BASE_DIR):
        for state_key in os.listdir(BASE_DIR):
            for eng in TARGET_COL_MAP:
                if os.path.exists(os.path.join(BASE_DIR, state_key, eng, "best_model.keras")):
                    models.add((state_key, eng))

    combos = []
    for state_key, eng in sorted(models):
        has_data = _resolve_state(state_key) is not None
        combos.append({
            "state_key":    state_key,
            "energy":       eng,
            "has_data":     has_data,
            "ready":        has_data,
        })

    return jsonify({
        "max_months":      MAX_MONTHS,
        "max_days":         MAX_DAYS,
        "last_data_date":  LAST_DATA_DATE.isoformat(),
        "dataset_states":  sorted(_state_cache.keys()),
        "combos":          combos,
    })

if __name__ == "__main__":
    app.run(debug=False, port=5000)