# ===========================================================
# 📦 Dependencies & Setup
# ===========================================================
import os, math, re, requests, joblib, datetime as dt
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# ===========================================================
# ⚙️ CONFIG
# ===========================================================
API_KEY = "my_secret_key_123"
OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_PATH = "best_model1.h5"
SCALER_FEATURE_PATH = "feature_scaler.save"
SCALER_TARGET_PATH = "target_scaler.save"

LAT, LON = 18.5204, 73.8567        # Pune (you can change)
SEQ_LEN = 168                      # must match training
TARGETS = ["temperature_2m", "relativehumidity_2m", "pressure_msl"]
FEATURES_CORE = ["temperature_2m", "relativehumidity_2m", "pressure_msl", "precipitation"]
INPUT_TIME_FEATS = ["hour_sin", "hour_cos", "day_sin", "day_cos"]
INPUT_FEATURES = FEATURES_CORE + INPUT_TIME_FEATS

TRAIN_SCALER_REF_START = "2024-01-01"
TRAIN_SCALER_REF_END   = "2024-12-31"

# ===========================================================
# 🧠 Load Model + Scalers
# ===========================================================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Upload {MODEL_PATH} first!")

model = load_model(MODEL_PATH, custom_objects={
    "mse": MeanSquaredError(),
    "MeanSquaredError": MeanSquaredError()
})
print("✅ Model loaded:", MODEL_PATH)

def fetch_open_meteo(start_date, end_date):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={LAT}&longitude={LON}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relativehumidity_2m,pressure_msl,precipitation"
        f"&timezone=Asia/Kolkata"
    )
    r = requests.get(url)
    r.raise_for_status()
    j = r.json()
    if "hourly" not in j:
        return pd.DataFrame()
    df = pd.DataFrame(j["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    for c in FEATURES_CORE:
        if c not in df.columns:
            df[c] = 0.0
    return df

def build_or_load_scalers():
    if os.path.exists(SCALER_FEATURE_PATH) and os.path.exists(SCALER_TARGET_PATH):
        return joblib.load(SCALER_FEATURE_PATH), joblib.load(SCALER_TARGET_PATH)

    print("⚠️ Building scalers...")
    ref_df = fetch_open_meteo(TRAIN_SCALER_REF_START, TRAIN_SCALER_REF_END)
    ref_df["hour"] = ref_df["time"].dt.hour
    ref_df["dayofyear"] = ref_df["time"].dt.dayofyear
    ref_df["hour_sin"] = np.sin(2*np.pi*ref_df["hour"]/24)
    ref_df["hour_cos"] = np.cos(2*np.pi*ref_df["hour"]/24)
    ref_df["day_sin"] = np.sin(2*np.pi*ref_df["dayofyear"]/365)
    ref_df["day_cos"] = np.cos(2*np.pi*ref_df["dayofyear"]/365)

    feat_df = ref_df[INPUT_FEATURES].astype(float)
    targ_df = ref_df[TARGETS].astype(float)

    feat_scaler = MinMaxScaler().fit(feat_df)
    targ_scaler = MinMaxScaler().fit(targ_df)
    joblib.dump(feat_scaler, SCALER_FEATURE_PATH)
    joblib.dump(targ_scaler, SCALER_TARGET_PATH)
    print("✅ Scalers built and saved.")
    return feat_scaler, targ_scaler

feature_scaler, target_scaler = build_or_load_scalers()

def make_input_features(df):
    df["hour"] = df["time"].dt.hour
    df["dayofyear"] = df["time"].dt.dayofyear
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["day_sin"] = np.sin(2*np.pi*df["dayofyear"]/365)
    df["day_cos"] = np.cos(2*np.pi*df["dayofyear"]/365)
    return df[INPUT_FEATURES]

def predict_for_timestamp(ts: pd.Timestamp):
    start_dt = (ts - pd.Timedelta(hours=SEQ_LEN)).date().strftime("%Y-%m-%d")
    end_dt = ts.date().strftime("%Y-%m-%d")
    df = fetch_open_meteo(start_dt, end_dt)
    if df.empty:
        return None
    df = df[(df["time"] > ts - pd.Timedelta(hours=SEQ_LEN)) & (df["time"] <= ts)]
    if len(df) < SEQ_LEN:
        return None

    feat_df = make_input_features(df)
    feat_scaled = feature_scaler.transform(feat_df)
    X = feat_scaled.reshape(1, SEQ_LEN, feat_scaled.shape[1])
    y_scaled = model.predict(X, verbose=0)
    y_pred = target_scaler.inverse_transform(y_scaled)[0]

    return {
        "temperature": float(y_pred[0]),
        "humidity": float(y_pred[1]),
        "pressure": float(y_pred[2])
    }

# ===========================================================
# 🌤️ Flask App
# ===========================================================
app = Flask(__name__)

def extract_date_from_prompt(prompt: str):
    """Extract a date from text like '28 March', 'on 5 April 2025', etc."""
    prompt = prompt.lower()
    match = re.search(r"(\d{1,2})\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)", prompt)
    if match:
        day = int(match.group(1))
        month_str = match.group(2)
        month_map = {
            "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
            "jul":7,"aug":8,"sep":9,"sept":9,"oct":10,"nov":11,"dec":12
        }
        month = month_map[month_str]
        year = 2025
        return pd.Timestamp(year, month, day, 12)
    return None

@app.route("/generate", methods=["POST"])
def generate_text():
    # --- Auth check ---
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer ") or auth_header.split(" ")[1] != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt'"}), 400

    prompt = data["prompt"]
    print(f"🧠 Received prompt: {prompt}")

    # --- Extract date ---
    ts = extract_date_from_prompt(prompt)
    if ts is None:
        return jsonify({"error": "Could not extract date. Try like '28 March 2025'."}), 400

    # --- Predict ---
    pred = predict_for_timestamp(ts)
    if pred is None:
        return jsonify({"error": "Could not generate prediction."}), 500

    # --- Compose response text for Gemma ---
    model_reply = (
        f"The predicted weather for {ts.strftime('%d %B %Y, 12:00 PM')} in Pune is:\n"
        f"🌡️ Temperature: {pred['temperature']:.2f}°C\n"
        f"💧 Humidity: {pred['humidity']:.2f}%\n"
        f"⏱️ Pressure: {pred['pressure']:.2f} hPa.\n"
        f"Please respond naturally."
    )

    # --- Send to Ollama/Gemma ---
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": "gemma3:1b", "prompt": model_reply, "stream": False},
            timeout=120
        )

        if response.status_code == 200:
            ollama_data = response.json()
            return jsonify({
                "raw_prediction": pred,
                "gemma_response": ollama_data.get("response", "").strip()
            })
        else:
            return jsonify({
                "error": f"Ollama error {response.status_code}",
                "details": response.text
            }), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===========================================================
# 🏁 Run Flask
# ===========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
