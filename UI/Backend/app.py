from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import traceback
import random

# -------------------------
# APP SETUP
# -------------------------
app = Flask(__name__)
CORS(app)

# -------------------------
# PATHS
# -------------------------
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[2]

CSV_PATH = PROJECT_ROOT / "Model" / "DataVisal" / "Traffic_Collisions_Open_Data_3719442797094142699.csv"
MODEL_PATH = PROJECT_ROOT / "Model" / "Best_traffic_model.joblib"

# -------------------------
# LOAD DATA + MODEL
# -------------------------
# -------------------------
# LOAD DATA (OPTIONAL) + MODEL
# -------------------------
TFM = None

if CSV_PATH.exists():
    print("CSV found, loading dataset...")
    TFM = pd.read_csv(CSV_PATH)
    TFM.replace(["N/R", "NSA"], np.nan, inplace=True)
else:
    print("CSV not found. Running in model-only mode.")

config = joblib.load(MODEL_PATH, mmap_mode="r")
model = config["model"]


# -------------------------
# HELPERS
# -------------------------
def get_rate(df, condition, column, default=0.5):
    val = df.loc[condition, column].mean()
    return default if pd.isna(val) else val

TEMPLATE_DATA = {k: 0 for k in model.feature_names_in_}
TEMPLATE_DATA.update({"OCC_YEAR": 2025, "DIVISION": "D14", "season": "Summer"})

# -------------------------
# ROUTES
# -------------------------
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "csv_loaded": TFM is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        month_map = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12
        }

        dow_map = {
            "Monday": 1, "Tuesday": 2, "Wednesday": 3,
            "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7
        }

        time_map = {"Morning": 10, "Afternoon": 16, "Evening": 20, "Night": 23}

        month = month_map.get(data.get("month", "July"), 7)
        dow = dow_map.get(data.get("dayOfWeek", "Monday"), 1)
        hour = time_map.get(data.get("timeOfDay", "Afternoon"), 16)

        input_data = TEMPLATE_DATA.copy()
        input_data.update({
            "OCC_DOW_sin": np.sin(2 * np.pi * dow / 7),
            "OCC_DOW_cos": np.cos(2 * np.pi * dow / 7),
            "OCC_MONTH_sin": np.sin(2 * np.pi * month / 12),
            "OCC_MONTH_cos": np.cos(2 * np.pi * month / 12),
            "OCC_HOUR_sin": np.sin(2 * np.pi * hour / 24),
            "OCC_HOUR_cos": np.cos(2 * np.pi * hour / 24),
            "is_weekend": int(dow in [6, 7]),
        })

        df = pd.DataFrame([input_data])[model.feature_names_in_]
        prob = model.predict_proba(df)[0, 1]


        return jsonify({
            "probability": float(prob),
            "classification": "Fatality" if prob >= 0.5 else "Non-Fatality",
            "success": True
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# -------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)
