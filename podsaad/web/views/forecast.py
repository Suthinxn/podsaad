from flask import Blueprint, jsonify, current_app, request
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from podsaad.web.utils.get_data_pm25 import fetch_data, get_raw_data, filter_by_station
from joblib import load
from mongoengine import Document, StringField, FloatField


module = Blueprint("forecast", __name__, url_prefix="/forecast")

LOOKBACK = 60
HORIZON = 14
FEATURES = [
    "PM_2_5", "PM_1", "PM_0_1",
    "humidity", "pressure", "temperature",
    "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos"
]


# ---------------- โมเดล ----------------
class ConvBiLSTM(nn.Module):
    def __init__(self, input_dim, lookback, horizon):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, batch_first=True, bidirectional=True, dropout=0.1)
        self.fc = nn.Linear(32 * 2, horizon)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


# --- Load scalers ---
feature_scaler = load("models_pytorch/feature_scaler.pkl")
target_scaler = load("models_pytorch/target_scaler.pkl")

# --- Load model ---
model = ConvBiLSTM(input_dim=len(FEATURES), lookback=LOOKBACK, horizon=HORIZON)
model.load_state_dict(torch.load("models_pytorch/best_model.pt", map_location="cpu"))
model.eval()


@module.route("/test", methods=["GET"])
def forecast_test():
    stations = [
    "119t","118t","93t","89t","62t","121t","o73","120t","43t",
    "63t","78t","o70","44t","o28","42t",
    ]
    
    end_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    dates = pd.date_range(end=end_date, periods=LOOKBACK, freq="D")

    all_results = []

    for station_code in stations:
        df_daily = pd.DataFrame({
            "timestamp": dates,
            "PM_2_5": np.random.uniform(10, 100, size=LOOKBACK),
            "PM_1": np.random.uniform(5, 80, size=LOOKBACK),
            "PM_0_1": np.random.uniform(1, 50, size=LOOKBACK),
            "humidity": np.random.uniform(40, 90, size=LOOKBACK),
            "pressure": np.random.uniform(950, 1020, size=LOOKBACK),
            "temperature": np.random.uniform(20, 35, size=LOOKBACK),
            "day_of_week_sin": np.random.uniform(-1, 1, size=LOOKBACK),
            "day_of_week_cos": np.random.uniform(-1, 1, size=LOOKBACK),
            "month_sin": np.random.uniform(-1, 1, size=LOOKBACK),
            "month_cos": np.random.uniform(-1, 1, size=LOOKBACK),
        }).set_index("timestamp")

        # --- Scale input features ---
        last_window_scaled = feature_scaler.transform(df_daily[FEATURES])

        # --- Prepare tensor ---
        input_tensor = torch.tensor(last_window_scaled.astype(np.float32), dtype=torch.float32).unsqueeze(0)

        # --- Prediction ---
        with torch.no_grad():
            prediction_scaled = model(input_tensor).cpu().numpy().flatten()

        # --- Inverse transform prediction (target only) ---
        forecast_real = target_scaler.inverse_transform(prediction_scaled.reshape(1, -1)).flatten()

        all_results.append({
            "station_code": station_code,
            "forecast_days": HORIZON,
            "forecast": forecast_real.tolist(),
        })

    return jsonify({"result": all_results})



def get_collection_class(station_code):
    """คืน class MongoEngine ของแต่ละ collection"""
    class_attrs = {
        "meta": {"collection": f"pm25_interpolated_{station_code}", "strict": False},
        "timestamp": StringField(),
        "station_name": StringField(),
        "station_code": StringField(),
        "lat": FloatField(),
        "lon": FloatField(),
        "PM_2_5": FloatField(),
        "PM_1": FloatField(),
        "PM_0_1": FloatField(),
        "temperature": FloatField(),
        "humidity": FloatField(),
        "pressure": FloatField(),
    }
    return type(f"PM25Interpolated_{station_code}", (Document,), class_attrs)

@module.route("/predict", methods=["GET"])
def forecast():
    all_results = []

    stations = [
    "119t","118t","93t","89t","62t","121t","o73","120t","43t",
    "63t","78t","o70","44t","o28","42t",
    ]

    for station_code in stations:
        CollectionClass = get_collection_class(station_code)
        # print(f"Debug Collection {CollectionClass}")
        # --- ดึงข้อมูลล่าสุด 30 วัน (เรียงจากเก่า → ใหม่) ---
        records = list(
            CollectionClass.objects().order_by("timestamp").limit(LOOKBACK)
        )
        if len(records) < LOOKBACK:
            continue  # ข้ามถ้าข้อมูลไม่ครบ 30 วัน

        # print(f"Debug {records}")
        # --- สร้าง DataFrame ---
        df_daily = pd.DataFrame([{f: getattr(r, f) for f in FEATURES + ["timestamp"]} for r in records])
        df_daily["timestamp"] = pd.to_datetime(df_daily["timestamp"])
        df_daily = df_daily.sort_values("timestamp").set_index("timestamp")

        # --- ตรวจสอบ NaN ---
        if df_daily[FEATURES].isnull().any().any():
            df_daily[FEATURES] = df_daily[FEATURES].interpolate(method='linear').ffill().bfill()

        # --- Scale input features ---
        try:
            last_window_scaled = scaler.transform(df_daily[FEATURES])
        except Exception as e:
            print(f"Error scaling data for {station_code}: {e}")
            continue

        # --- Prepare tensor ---
        input_tensor = torch.tensor(last_window_scaled.astype(np.float32), dtype=torch.float32).unsqueeze(0)

        # --- Prediction ---
        with torch.no_grad():
            prediction_scaled = model(input_tensor).cpu().numpy().flatten()

        # --- Inverse transform prediction ---
        dummy_features = np.zeros((HORIZON, len(FEATURES)))
        dummy_features[:, FEATURES.index("PM_2_5")] = prediction_scaled
        forecast_real = scaler.inverse_transform(dummy_features)[:, FEATURES.index("PM_2_5")]

        all_results.append({
            "station_code": station_code,
            "forecast_days": HORIZON,
            "forecast": forecast_real.tolist(),
        })

    return jsonify({"result": all_results})