from flask import Blueprint, jsonify, current_app, request
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from podsaad.web.utils.get_data_pm25 import fetch_data, get_raw_data, filter_by_station

module = Blueprint("forecast", __name__, url_prefix="/forecast")

LOOKBACK = 30
HORIZON = 14
FEATURES = ['PM_2_5','PM_1','PM_0_1','humidity','pressure','temperature']


# ---------------- โมเดล ----------------
class ConvBiLSTM(nn.Module):
    def __init__(self, input_dim, lookback, horizon):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64*2, horizon)

    def forward(self, x):
        x = x.permute(0,2,1)                # [B, input_dim, lookback]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(0,2,1)                # [B, lookback, 32]
        out, _ = self.lstm(x)
        out = out[:, -1, :]                 # last timestep
        out = self.fc(out)
        return out


# โหลดโมเดลแค่ครั้งเดียวตอนแอปรัน
model = ConvBiLSTM(input_dim=len(FEATURES), lookback=LOOKBACK, horizon=HORIZON)
model.load_state_dict(torch.load("models_pytorch/best_model.pt", map_location="cpu"))
model.eval()

# ถ้ามี scaler
# from joblib import load
# scaler = load("scaler.pkl")

@module.route("/", methods=["GET"])
def forecast_pm25():
    """
    เรียกแบบ:  GET /forecast?station_code=119t
    """
    station_code = request.args.get("station_code", "119t")

    config = {
        "API_DHARA": current_app.config.get("API_DHARA"),
        "SOURCE": current_app.config.get("SOURCE"),
    }

    # ดึงข้อมูลย้อนหลัง 60 วันเพื่อความปลอดภัย
    end_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=60)

    api_data = fetch_data(start_date, end_date, config)
    # df = get_raw_data(api_data)
    df = filter_by_station(df)

    # เลือกเฉพาะสถานีที่ต้องการ
    df = df[df["station_code"] == station_code].copy()
    if df.empty:
        return jsonify({"error": "ไม่พบข้อมูลสถานี"}), 404

    # เตรียมข้อมูล
    df = df.sort_values("timestamp").set_index("timestamp")
    df_daily = df.resample("D")[FEATURES].mean()
    df_daily = df_daily.interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")

    if len(df_daily) < LOOKBACK:
        return jsonify({"error": f"ต้องการข้อมูลย้อนหลัง {LOOKBACK} วัน"}), 400

    last_window = df_daily[-LOOKBACK:].values.astype(np.float32)
    # last_window = scaler.transform(last_window)   # ถ้ามี scaler

    input_tensor = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction = model(input_tensor)

    forecast = prediction.numpy().flatten().tolist()

    print("[FORECAST] ---", forecast)

    return jsonify({
        "station_code": station_code,
        "forecast_days": HORIZON,
        # "forecast": forecast
    })
