from flask import Blueprint, render_template, jsonify, current_app
from flask_login import login_required
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from matplotlib.path import Path
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import requests
import folium
from folium.raster_layers import ImageOverlay
import branca.colormap as cm
from matplotlib.colors import LinearSegmentedColormap
import base64
from io import BytesIO
import warnings

warnings.filterwarnings("ignore")


from podsaad import models

module = Blueprint("dashboard", __name__)


def get_pm25_level(value):
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


def get_pm25_color(value):
    if value <= 25:
        return "#0066FF"
    elif value <= 37:
        return "#00CC00"
    elif value <= 50:
        return "#FFFF00"
    elif value <= 75:
        return "#FF9900"
    else:
        return "#FF0000"


def generate_heatmap_image():
    """สร้างภาพ heatmap และ return เป็น base64"""
    # โหลด GeoJSON
    url = "https://raw.githubusercontent.com/apisit/thailand.json/master/thailand.json"
    thailand_geo = requests.get(url).json()

    southern_provinces = [
        "Songkhla",
        "Pattani",
        "Nakhon Si Thammarat",
        "Surat Thani",
        "Trang",
        "Phuket",
        "Satun",
        "Yala",
        "Narathiwat",
        "Chumphon",
        "Ranong",
        "Phangnga",
        "Krabi",
        "Phatthalung",
    ]

    southern_features = [
        f
        for f in thailand_geo["features"]
        if f["properties"]["name"] in southern_provinces
    ]
    gdf_south = gpd.GeoDataFrame.from_features(southern_features)
    mask_geom = gdf_south.unary_union

    # ข้อมูล PM2.5
    df_pm25 = pd.DataFrame(
        [
            [8.052856870972867, 98.91866399571485, 37.42, "Krabi"],
            [10.49595443351772, 99.18869286937533, 21.67, "Chumphon"],
            [7.569948469925628, 99.58568656274488, 44.81, "Trang"],
            [8.246763400521472, 99.88254292558815, 28.34, "Nakhon Si Thammarat"],
            [6.4258031744613655, 101.7928157990397, 32.75, "Narathiwat"],
            [6.891036422471899, 101.25096924050584, 17.63, "Pattani"],
            [8.426219039086634, 98.5294494386699, 46.58, "Phangnga"],
            [7.616883312560487, 100.0485906828405, 24.91, "Phatthalung"],
            [7.885638909394165, 98.3916465144643, 39.06, "Phuket"],
            [6.545268471622002, 101.28175395399666, 15.28, "Yala"],
            [5.764744028287182, 101.06739886748494, 41.12, "Yala"],
            [9.96216409415826, 98.6388418999343, 19.47, "Ranong"],
            [7.015948866031788, 100.49172833785089, 33.85, "Songkhla"],
            [6.667066168736155, 100.32681389447792, 28.92, "Songkhla"],
            [9.125613772682643, 99.32449412333452, 25.66, "Surat Thani"],
        ],
        columns=["Lat", "Lon", "PM25", "province"],
    )

    # สร้าง IDW Interpolation
    lats, lons, values = (
        df_pm25["Lat"].values,
        df_pm25["Lon"].values,
        df_pm25["PM25"].values,
    )
    minx, miny, maxx, maxy = mask_geom.bounds
    GRID_SIZE = 800
    nx = ny = GRID_SIZE

    lon_bins = np.linspace(minx, maxx, nx + 1)
    lat_bins = np.linspace(miny, maxy, ny + 1)
    lon_centers = 0.5 * (lon_bins[:-1] + lon_bins[1:])
    lat_centers = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    Xc, Yc = np.meshgrid(lon_centers, lat_centers)

    power, epsilon = 2, 1e-10
    heat_grid = np.zeros_like(Xc)
    weight_sum = np.zeros_like(Xc)

    for lon, lat, val in zip(lons, lats, values):
        dist = np.sqrt((Xc - lon) ** 2 + (Yc - lat) ** 2)
        w = 1.0 / (dist**power + epsilon)
        heat_grid += w * val
        weight_sum += w

    heat_interp = heat_grid / weight_sum
    heat_smooth = gaussian_filter(heat_interp, sigma=3)

    # สร้าง Mask
    points = np.column_stack((Xc.ravel(), Yc.ravel()))
    mask_flat = np.zeros(points.shape[0], dtype=bool)
    polys = [mask_geom] if isinstance(mask_geom, Polygon) else list(mask_geom.geoms)

    for poly in polys:
        ext_path = Path(np.array(poly.exterior.coords))
        mask_ext = ext_path.contains_points(points)
        mask_flat |= mask_ext

    mask_grid = mask_flat.reshape((ny, nx))

    # Normalize และสร้างภาพ RGBA
    actual_min, actual_max = values.min(), values.max()
    norm = (heat_smooth - actual_min) / (actual_max - actual_min + 5)
    norm = np.clip(norm, 0, 1)
    norm_masked = np.where(mask_grid, norm, np.nan)

    pm25_colormap = LinearSegmentedColormap.from_list(
        "pm25",
        [
            (0.0, "#0066FF"),
            (0.333, "#00CC00"),
            (0.493, "#FFFF00"),
            (0.667, "#FF9900"),
            (1.0, "#FF0000"),
        ],
    )

    rgba = pm25_colormap(np.nan_to_num(norm_masked, nan=0.0))
    rgba[..., 3] = np.where(np.isnan(norm_masked), 0.0, 0.7)

    # แปลงเป็น base64
    buffer = BytesIO()
    plt.imsave(buffer, rgba, origin="lower", dpi=150, format="png")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode()

    return (
        f"data:image/png;base64,{img_base64}",
        southern_features,
        df_pm25,
        (minx, miny, maxx, maxy),
        actual_min,
        actual_max,
    )


@module.route("/")
def index():
    # สร้างแผนที่
    img_data, southern_features, df_pm25, bounds, vmin, vmax = generate_heatmap_image()
    minx, miny, maxx, maxy = bounds

    southern_geo = {"type": "FeatureCollection", "features": southern_features}
    center_lat, center_lon = (miny + maxy) / 2, (minx + maxx) / 2

    m = folium.Map(
        location=[center_lat, center_lon], zoom_start=7, tiles="OpenStreetMap"
    )

    # ใช้ base64 image แทนไฟล์
    ImageOverlay(
        name="PM2.5 Heatmap",
        image=img_data,
        bounds=[[miny, minx], [maxy, maxx]],
        opacity=0.7,
    ).add_to(m)

    folium.GeoJson(
        southern_geo,
        name="ขอบเขตจังหวัดภาคใต้",
        style_function=lambda f: {
            "color": "#333",
            "weight": 2,
            "fillOpacity": 0,
            "dashArray": "5,5",
        },
        tooltip=folium.GeoJsonTooltip(fields=["name"], aliases=["จังหวัด: "]),
    ).add_to(m)

    # Markers
    for _, row in df_pm25.iterrows():
        color = get_pm25_color(row["PM25"])
        level = get_pm25_level(row["PM25"])
        folium.map.Marker(
            [row["Lat"], row["Lon"]],
            icon=folium.DivIcon(
                html=f"""
                <div style="width:50px;height:50px;background:{color};
                            border-radius:50%;border:3px solid white;
                            display:flex;align-items:center;justify-content:center;
                            box-shadow:0 2px 6px rgba(0,0,0,0.4);">
                    <span style="color:white;font-weight:bold;font-size:16px;">{row["PM25"]}</span>
                </div>
            """
            ),
            popup=folium.Popup(
                f"""
                <div style="font-family:Sarabun,sans-serif;min-width:180px;">
                    <h4 style="color:{color};">จังหวัด{row['province']}</h4>
                    <b>ค่า PM2.5:</b> <span style="color:{color};">{row["PM25"]}</span> µg/m³<br>
                    <b>ระดับ:</b> <span style="color:{color};">{level}</span>
                </div>
            """,
                max_width=250,
            ),
        ).add_to(m)

    cm.LinearColormap(
        colors=["#0066FF", "#00CC00", "#FFFF00", "#FF9900", "#FF0000"],
        vmin=vmin,
        vmax=vmax,
        caption="PM2.5 (µg/m³)",
    ).add_to(m)

    folium.LayerControl().add_to(m)

    # แปลง map เป็น HTML
    map_html = m._repr_html_()

    return render_template("dashboard/index.html", map_html=map_html)


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


@module.route("/get_data")
def get_data():
    raw_data = None

    # Call API
    config = {
        "API_DHARA": current_app.config.get("API_DHARA"),
        "SOURCE": current_app.config.get("SOURCE"),
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
