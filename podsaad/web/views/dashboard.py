from flask import Blueprint, render_template, jsonify
from flask_login import login_required
from datetime import datetime
from io import BytesIO
import folium
from folium.plugins import HeatMap
import numpy as np


from podsaad import models

module = Blueprint("dashboard", __name__)


def get_pm25_color(value):
    if value <= 25:
        return "blue"  # ดีมาก
    elif value <= 37:
        return "green"  # ดี
    elif value <= 50:
        return "yellow"  # ปานกลาง
    elif value <= 75:
        return "orange"  # เริ่มมีผล
    else:
        return "red"  # มีผลต่อสุขภาพ


@module.route("/")
def index():
    import folium
    from folium.plugins import HeatMap

    pm25_data = [
        [7.0060, 100.4980, 45, "สงขลา"],
        [6.8664, 101.2501, 40, "ปัตตานี"],
        [8.4325, 99.9631, 35, "นครศรีธรรมราช"],
        [9.1382, 99.3215, 28, "สุราษฎร์ธานี"],
        [7.5636, 99.6114, 32, "ตรัง"],
        [8.0863, 98.9063, 30, "ภูเก็ต"],
        [6.4673, 100.1867, 38, "สตูล"],
        [6.4241, 101.8213, 50, "ยะลา"],
        [6.5423, 101.2800, 42, "นราธิวาส"],
    ]

    # สร้างแผนที่
    m = folium.Map(location=[7.5, 100], zoom_start=7, tiles="OpenStreetMap")

    # HeatMap layer
    heat_data = [[lat, lon, value] for lat, lon, value, _ in pm25_data]
    HeatMap(heat_data, radius=25, blur=15, max_zoom=10).add_to(m)

    # Marker ตามค่า PM2.5
    for lat, lon, value, province in pm25_data:
        color = get_pm25_color(value)
        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            popup=f"{province}: {value} µg/m³",
        ).add_to(m)

    map_html = m._repr_html_()
    return render_template("dashboard/index.html", map_html=map_html)


# @module.route("/")
# def index():
#     # ใช้แผนที่สมจริง (Esri Satellite + OSM)
#     m = folium.Map(location=[7.5, 100], zoom_start=7, tiles="OpenStreetMap")

#     # เพิ่ม tile layers พร้อม attribution
#     folium.TileLayer(
#         tiles="https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg",
#         attr="Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors",
#         name="Stamen Terrain",
#     ).add_to(m)

#     folium.TileLayer(
#         tiles="https://stamen-tiles.a.ssl.fastly.net/toner/{z}/{x}/{y}.png",
#         attr="Map tiles by Stamen Design, CC BY 3.0 — Map data © OpenStreetMap contributors",
#         name="Stamen Toner",
#     ).add_to(m)

#     folium.TileLayer(
#         tiles="https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}{r}.png",
#         attr="© OpenStreetMap contributors © CARTO",
#         name="CartoDB Positron",
#     ).add_to(m)

#     folium.TileLayer(
#         tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
#         attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
#         name="Esri Satellite",
#     ).add_to(m)

#     # วงกลม + marker
#     for lat, lon, val, province in pm25_data:
#         # folium.CircleMarker(
#         #     location=[lat, lon],
#         #     radius=8,
#         #     color=pm25_to_color(val),
#         #     fill=True,
#         #     fill_opacity=0.7,
#         #     popup=f"<b>{province}</b><br>PM2.5: {val} µg/m³",
#         # ).add_to(m)

#         # Marker icon (pin)
#         folium.Marker(
#             location=[lat, lon],
#             popup=f"<b>{province}</b><br>PM2.5: {val} µg/m³",
#             icon=folium.Icon(color="blue", icon="cloud"),
#         ).add_to(m)

#     HeatMap(
#         [[lat, lon, val] for lat, lon, val, _ in pm25_data],
#         min_opacity=0.3,
#         radius=25,
#         blur=25,
#         max_zoom=8,
#     ).add_to(m)

#     folium.LayerControl().add_to(m)

#     # Render HTML
#     map_html = m._repr_html_()
#     return render_template("dashboard/index.html", map_html=map_html)


# @module.route("/data")
# def data():
#     mapped = [[d[0], d[1], pm25_to_intensity(d[2])] for d in pm25_data]
#     return jsonify(mapped)


@module.route("/top10_province", methods=["GET", "POST"])
def top10_province():
    return render_template("/dashboard/top10_province.html")


@module.route("/graph_infomation", methods=["GET", "POST"])
def graph_infomation():
    return render_template("/dashboard/graph_infomation.html")
