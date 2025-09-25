from flask import Blueprint, render_template, jsonify
from flask_login import login_required
from datetime import datetime

from podsaad import models

module = Blueprint("dashboard", __name__)

pm25_data = [
    [13.7563, 100.5018, 80],  # Bangkok
    [18.7883, 98.9853, 55],  # Chiang Mai
    [15.8700, 100.9925, 30],  # Central Thailand
    [16.4419, 102.8350, 70],  # Khon Kaen
    [7.0060, 100.4980, 45],  # Hat Yai
]


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


@module.route("/top10_province", methods=["GET", "POST"])
def top10_province():
    return render_template("/dashboard/top10_province.html")


@module.route("/graph_infomation", methods=["GET", "POST"])
def graph_infomation():
    return render_template("/dashboard/graph_infomation.html")
