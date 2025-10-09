from flask import Blueprint, render_template, jsonify, current_app
from flask_login import login_required
from datetime import datetime

from podsaad import models

module = Blueprint("dashboard", __name__)


# --- ฟังก์ชันกำหนดสีตามค่า PM2.5 ---
def get_pm25_color(value):
    if value <= 25:
        return "#0066FF"  # น้ำเงิน - คุณภาพดีมาก
    elif value <= 37:
        return "#00CC00"  # เขียว - คุณภาพดี
    elif value <= 50:
        return "#FFFF00"  # เหลือง - คุณภาพปานกลาง
    elif value <= 75:
        return "#FF9900"  # ส้ม - มีผลกระทบต่อสุขภาพ
    else:
        return "#FF0000"  # แดง - มีผลกระทบต่อสุขภาพมาก


def get_pm25_level(value):
    """ส่งกลับระดับคุณภาพอากาศ"""
    if value <= 25:
        return "คุณภาพดีมาก"
    elif value <= 37:
        return "คุณภาพดี"
    elif value <= 50:
        return "คุณภาพปานกลาง"
    elif value <= 75:
        return "มีผลกระทบต่อสุขภาพ"
    else:
        return "มีผลกระทบต่อสุขภาพมาก"


@module.route("/")
def index():

    return render_template("/dashboard/index.html")


def pm25_to_intensity(value):
    if value <= 15:
        return 0.2  # ฟ้า/เขียวอ่อน
    elif value <= 35:
        return 0.4  # เขียว-เหลือง
    elif value <= 55:
        return 0.6  # ส้ม
    elif value <= 75:
        return 0.8  # แดงอ่อน
    else:
        return 1.0  # แดงเข้ม


@module.route("/data")
def data():
    mapped = [[d[0], d[1], pm25_to_intensity(d[2])] for d in pm25_data]
    return jsonify(mapped)

@module.route("/get_data")
def get_data():
    raw_data = None

    # Call API
    config = {
        "API_DHARA" : current_app.config.get("API_DHARA"),
        "SOURCE" : current_app.config.get("SOURCE"),
    }

    today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today + timedelta(days=1)

    api_data = fetch_data(today, tomorrow, config)
    df = get_raw_data(api_data)
    data_by_station = filter_by_station(df)

    print(">>>\n", data_by_station)

    return data_by_station


@module.route("/top10_province", methods=["GET", "POST"])
def top10_province():
    return render_template("/dashboard/top10_province.html")


@module.route("/graph_infomation", methods=["GET", "POST"])
def graph_infomation():
    return render_template("/dashboard/graph_infomation.html")