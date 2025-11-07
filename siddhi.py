# ===========================================================
# 🌦️ Offline AI Weather Chatbot — Flask + TensorFlow + Ollama
# ===========================================================
import os, re, joblib, numpy as np, pandas as pd, requests
from flask import Flask, request, jsonify, render_template_string
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
LOCAL_DATA_PATH = "weather_2024_hourly_pune (2).csv"

SEQ_LEN = 168
TARGETS = ["temperature_2m", "relativehumidity_2m", "pressure_msl"]
INPUT_FEATURES = ["temperature_2m", "relativehumidity_2m", "pressure_msl", "precipitation",
                  "hour_sin", "hour_cos", "day_sin", "day_cos"]

# ===========================================================
# 🧠 Load Model + Scalers
# ===========================================================
model = load_model(MODEL_PATH, custom_objects={
    "mse": MeanSquaredError(),
    "MeanSquaredError": MeanSquaredError()
})
feature_scaler = joblib.load(SCALER_FEATURE_PATH)
target_scaler = joblib.load(SCALER_TARGET_PATH)

print("✅ Model and scalers loaded successfully!")

# ===========================================================
# 📊 Load Local Data
# ===========================================================
df = pd.read_csv(LOCAL_DATA_PATH, parse_dates=["time"])
df["hour"] = df["time"].dt.hour
df["dayofyear"] = df["time"].dt.dayofyear
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["day_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
df["day_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)

# ===========================================================
# ⚡ Prediction Function
# ===========================================================
def predict_from_local(timestamp):
    # Find the closest timestamp in your local data
    target_time = pd.Timestamp(timestamp)
    end_idx = df.index[df["time"] <= target_time].max()
    if end_idx < SEQ_LEN:
        return None

    subset = df.iloc[end_idx - SEQ_LEN:end_idx]
    feat_scaled = feature_scaler.transform(subset[INPUT_FEATURES])
    X = feat_scaled.reshape(1, SEQ_LEN, len(INPUT_FEATURES))

    y_scaled = model.predict(X, verbose=0)
    y_pred = target_scaler.inverse_transform(y_scaled)[0]

    return {
        "temperature": float(y_pred[0]),
        "humidity": float(y_pred[1]),
        "pressure": float(y_pred[2])
    }

# ===========================================================
# 🧠 Date Extractor from Prompt
# ===========================================================
def extract_date_from_prompt(prompt: str):
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
        return pd.Timestamp(2024, month, day, 12)
    return None

# ===========================================================
# 💬 Flask App
# ===========================================================
app = Flask(__name__)

@app.route("/")
def home():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
<title>🌦️ Offline AI Weather Chatbot</title>
<style>
body { font-family: 'Segoe UI', sans-serif; background:#f3f4f7; display:flex; flex-direction:column; align-items:center; justify-content:center; height:100vh; }
#chatbox { width: 500px; max-width:90%; background:white; border-radius:12px; padding:20px; box-shadow:0 0 15px rgba(0,0,0,0.1); overflow-y:auto; height:70vh; }
.message { margin:10px 0; padding:10px 15px; border-radius:10px; max-width:80%; }
.user { background:#0078ff; color:white; align-self:flex-end; margin-left:auto; }
.bot { background:#e5e5ea; color:black; align-self:flex-start; }
input { width: 400px; padding:10px; border-radius:8px; border:1px solid #ccc; }
button { padding:10px 15px; border:none; background:#0078ff; color:white; border-radius:8px; cursor:pointer; margin-left:10px; }
button:hover { background:#005fcc; }
</style>
</head>
<body>
<h2>🌦️ Offline AI Weather Chatbot</h2>
<div id="chatbox"></div>
<div style="margin-top:20px;">
  <input type="text" id="prompt" placeholder="Ask: What will be the weather on 28 March 2024?" />
  <button onclick="sendMessage()">Send</button>
</div>

<script>
async function sendMessage() {
  const input = document.getElementById("prompt");
  const chatbox = document.getElementById("chatbox");
  const text = input.value.trim();
  if (!text) return;

  const userMsg = document.createElement("div");
  userMsg.className = "message user";
  userMsg.textContent = text;
  chatbox.appendChild(userMsg);
  input.value = "";

  const botMsg = document.createElement("div");
  botMsg.className = "message bot";
  botMsg.textContent = "Thinking...";
  chatbox.appendChild(botMsg);
  chatbox.scrollTop = chatbox.scrollHeight;

  try {
    const response = await fetch("/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer my_secret_key_123"
      },
      body: JSON.stringify({ prompt: text })
    });
    const data = await response.json();
    botMsg.textContent = data.gemma_response || data.error || "Error occurred.";
  } catch (err) {
    botMsg.textContent = "⚠️ " + err;
  }
  chatbox.scrollTop = chatbox.scrollHeight;
}
</script>
</body>
</html>
""")

@app.route("/generate", methods=["POST"])
def generate_text():
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer ") or auth_header.split(" ")[1] != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json()
    prompt = data.get("prompt", "")
    ts = extract_date_from_prompt(prompt)
    if ts is None:
        return jsonify({"error": "Please include a valid date (e.g. 28 March 2024)."}), 400

    pred = predict_from_local(ts)
    if pred is None:
        return jsonify({"error": "Not enough data for prediction."}), 500

    model_reply = (
        f"The predicted weather for {ts.strftime('%d %B %Y')} is:\n"
        f"🌡️ Temperature: {pred['temperature']:.2f}°C\n"
        f"💧 Humidity: {pred['humidity']:.2f}%\n"
        f"⏱️ Pressure: {pred['pressure']:.2f} hPa.\n"
        f"Please write this nicely."
    )

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
# 🏁 Run
# ===========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
