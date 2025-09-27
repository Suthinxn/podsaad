from flask import Blueprint, render_template, jsonify
from flask_login import login_required
from datetime import datetime

from podsaad import models

module = Blueprint("dashboard", __name__)

pm25_data = [
    [7.0060, 100.4980, 45],  # หาดใหญ่ (สงขลา)
    [6.8664, 101.2501, 40],  # ปัตตานี
    [8.4325, 99.9631, 35],  # นครศรีธรรมราช
    [9.1382, 99.3215, 28],  # สุราษฎร์ธานี
    [7.5636, 99.6114, 32],  # ตรัง
    [8.0863, 98.9063, 30],  # ภูเก็ต
    [6.4673, 100.1867, 38],  # สตูล
    [6.4241, 101.8213, 50],  # ยะลา
    [6.5423, 101.2800, 42],  # นราธิวาส
]


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


@module.route("/")
def index():

    return render_template("/dashboard/index.html")


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
