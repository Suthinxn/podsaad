from flask import Blueprint, render_template, jsonify
from flask_login import login_required
from datetime import datetime
from io import BytesIO
import folium
from folium.plugins import HeatMap, MarkerCluster
import requests
import numpy as np
import pandas as pd
import branca.colormap as cm

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

    # --- โหลด GeoJSON ขอบเขตจังหวัดไทย ---
    state_geo = requests.get(
        "https://raw.githubusercontent.com/apisit/thailand.json/master/thailand.json"
    ).json()

    # --- ข้อมูล PM2.5 ที่มี (lat, lon, ค่า, จังหวัด) ---
    pm25_data = [
        [7.0060, 100.4980, 45, "Songkhla"],
        [6.8664, 101.2501, 40, "Pattani"],
        [8.4325, 99.9631, 35, "Nakhon Si Thammarat"],
        [9.1382, 99.3215, 28, "Surat Thani"],
        [7.5636, 99.6114, 32, "Trang"],
        [8.0863, 98.9063, 30, "Phuket"],
        [6.4673, 100.1867, 38, "Satun"],
        [6.4241, 101.8213, 60, "Yala"],
        [6.5423, 101.2800, 42, "Narathiwat"],
    ]
    df = pd.DataFrame(pm25_data, columns=["lat", "lon", "PM25", "province"])

    # --- กรอง GeoJSON เฉพาะจังหวัดที่มีข้อมูล ---
    provinces_with_data = df["province"].tolist()
    filtered_state_geo = {"type": "FeatureCollection", "features": []}
    for feature in state_geo["features"]:
        if feature["properties"]["name"] in provinces_with_data:
            filtered_state_geo["features"].append(feature)

    # --- สร้างแผนที่พื้นฐาน ---
    m = folium.Map(location=[7.5, 100], zoom_start=7, tiles="OpenStreetMap")

    # --- สร้าง HeatMap จากพิกัดจริง ---
    heat_data = [[row["lat"], row["lon"], row["PM25"]] for _, row in df.iterrows()]
    HeatMap(
        heat_data,
        min_opacity=0.4,
        radius=60,
        blur=35,
        max_zoom=10,
        gradient={0.0: "blue", 0.3: "lime", 0.5: "yellow", 0.7: "orange", 1.0: "red"},
    ).add_to(m)

    # --- ระบายจังหวัดพื้นหลัง ---
    folium.GeoJson(
        filtered_state_geo,
        name="ขอบเขตจังหวัด",
        style_function=lambda feature: {
            "fillColor": "transparent",
            "color": "#666666",
            "weight": 1.5,
            "dashArray": "5, 5",
        },
    ).add_to(m)

    # --- สร้าง FeatureGroup สำหรับ marker ---
    marker_group = folium.FeatureGroup("ค่า PM2.5").add_to(m)

    # --- วนลูปสร้าง Marker ที่รวมวงกลมและตัวเลขเข้าด้วยกัน ---
    for _, row in df.iterrows():
        pm25_value = row["PM25"]
        pm25_color = get_pm25_color(pm25_value)
        pm25_level = get_pm25_level(pm25_value)

        # สร้าง Marker เดียวที่มีทั้งวงกลมและตัวเลข
        folium.map.Marker(
            [row["lat"], row["lon"]],
            icon=folium.DivIcon(
                html=f"""
                <div style="
                    width: 50px;
                    height: 50px;
                    background-color: {pm25_color};
                    border: 3px solid white;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.4);
                    cursor: pointer;
                ">
                    <span style="
                        color: white;
                        font-weight: bold;
                        font-size: 16px;
                        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
                        font-family: 'Arial', sans-serif;
                    ">
                        {pm25_value}
                    </span>
                </div>
                """
            ),
            popup=folium.Popup(
                f"""
                <div style="font-family: 'Sarabun', sans-serif; min-width: 180px;">
                    <h4 style="margin: 5px 0; color: {pm25_color};">
                        จังหวัด{row['province']}
                    </h4>
                    <hr style="margin: 5px 0;">
                    <p style="margin: 3px 0;">
                        <b>ค่า PM2.5:</b> <span style="color: {pm25_color}; font-size: 16px; font-weight: bold;">
                        {pm25_value}</span> µg/m³
                    </p>
                    <p style="margin: 3px 0;">
                        <b>ระดับ:</b> <span style="color: {pm25_color};">{pm25_level}</span>
                    </p>
                </div>
                """,
                max_width=250,
            ),
        ).add_to(marker_group)

    # --- สร้าง Custom Legend ---
    legend_html = """
    <div style="
        position: fixed; 
        bottom: 50px; 
        right: 50px; 
        width: 200px; 
        background-color: white; 
        border: 2px solid grey; 
        z-index: 9999; 
        font-size: 14px;
        padding: 10px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        font-family: 'Sarabun', sans-serif;
    ">
        <h4 style="margin: 0 0 10px 0; text-align: center;">ระดับ PM2.5</h4>
        <div style="margin: 5px 0;">
            <span style="background-color: #0066FF; padding: 3px 8px; border-radius: 3px; color: white;">
                0-25
            </span> คุณภาพดีมาก
        </div>
        <div style="margin: 5px 0;">
            <span style="background-color: #00CC00; padding: 3px 8px; border-radius: 3px; color: white;">
                26-37
            </span> คุณภาพดี
        </div>
        <div style="margin: 5px 0;">
            <span style="background-color: #FFFF00; padding: 3px 8px; border-radius: 3px; color: black;">
                38-50
            </span> ปานกลาง
        </div>
        <div style="margin: 5px 0;">
            <span style="background-color: #FF9900; padding: 3px 8px; border-radius: 3px; color: white;">
                51-75
            </span> มีผลต่อสุขภาพ
        </div>
        <div style="margin: 5px 0;">
            <span style="background-color: #FF0000; padding: 3px 8px; border-radius: 3px; color: white;">
                76+
            </span> มีผลมาก
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # --- เพิ่ม LayerControl ---
    folium.LayerControl().add_to(m)

    return render_template("dashboard/index.html", map_html=m._repr_html_())


@module.route("/top10_province", methods=["GET", "POST"])
def top10_province():
    return render_template("/dashboard/top10_province.html")


@module.route("/graph_infomation", methods=["GET", "POST"])
def graph_infomation():
    return render_template("/dashboard/graph_infomation.html")
