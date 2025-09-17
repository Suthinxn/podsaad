from flask import Blueprint, render_template, jsonify
from flask_login import login_required
from datetime import datetime

from dustboy import models

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


@module.route("/data")
def data():
    return jsonify(pm25_data)
