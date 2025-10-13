from flask import Blueprint, render_template, jsonify, current_app, request, redirect
from flask_login import login_required
from datetime import date, datetime, timedelta



from podsaad.web.utils.get_heatmap import get_pm25_level, get_pm25_color, create_map, get_station_dexcriptions 
# from podsaad.web.utils.get_sarimax_heatmap import get_pm25_level, get_pm25_color, create_sarimax_map

from podsaad import models
import json
import warnings

warnings.filterwarnings("ignore")
module = Blueprint("dashboard", __name__)



@module.route("/")
def index():
    current_day = request.args.get("select_day", default=0, type=int)
    latest_date = datetime.now().strftime("%Y-%m-%d")

    today = datetime.today()
    select_day = int(request.args.get("select_day", 0))  # วันที่เลือกจาก query param
    selected_date = today + timedelta(days=select_day)
    selected_date_label = selected_date.strftime("%d %b %Y")  # เช่น 13 Oct 2025


    models_list = []
    stations = [
    "119t","118t","93t","89t","62t","121t","o73","120t","43t",
    "63t","78t","o70","44t","o28","42t",
    ]

    # print(f"DEBUG TIME {str(date.today())}")
    for i in stations:
        collection_name = f"PM25Interpolated{i}"
        model_class = getattr(models, collection_name, None)    

        # Check None Model
        if model_class is None:
            continue

        latest_record = model_class.objects(timestamp="2025-10-11").first() 
        if latest_record:
            models_list.append(latest_record)
            # print(f"DEBUG Collection : {model_class}")

    pm25_data = []
    for model in models_list:
        pm25_data.append(model.PM_2_5)

    avg_pm25 = round(sum(pm25_data) / len(pm25_data), 2)
    # print(f"DEBUG AVG PM25{avg_pm25}")
    avg_pm25 = 25

    print(f"DEBUG models list {models_list}")

    # Heatmap
    select_day = int(request.args.get("select_day", 0))
    select_model = request.args.get("select_model", "Conv1D+BiLSTM")
    
    map_html = None
    map_html = create_map(select_day, select_model)

    return render_template("dashboard/index.html", 
                           map_html=map_html,
                           today=datetime.today(),
                           current_day=current_day,
                           timedelta=timedelta, 
                           models_list=models_list, 
                           station_descriptions=get_station_dexcriptions(),
                           latest_date=latest_date,
                           avg_pm25=avg_pm25,
                           select_model=select_model,
                           selected_date_label=selected_date_label)



@module.route("/top10_province", methods=["GET", "POST"])
def top10_province():
    return render_template("/dashboard/top10_province.html")


@module.route("/graph_infomation/<station>", methods=["GET", "POST"])
def graph_infomation(station):
    print(f"Station : {station}")

    today = datetime.today()    
    days_ago = today - timedelta(days=14)

    today_str = today.strftime("%Y-%m-%d")
    days_ago_str = days_ago.strftime("%Y-%m-%d")

    collection_name = f"PM25Interpolated{station}"
    model = getattr(models, collection_name, None)

    data_last_7_days = model.objects(timestamp__gte=days_ago_str,timestamp__lte=today_str).order_by('timestamp')

    timestamps = []


    for data in data_last_7_days:
        timestamps.append(data.timestamp)
    
    pm25_list = []
    for data in data_last_7_days:
        pm25_list.append(round(data.PM_2_5, 2))

    chart_data = {
        'series': [{
            'name': 'PM2.5',
            'data': pm25_list
        }],
        'categories': timestamps
    }

    json_data = json.dumps(chart_data)

    model = model.objects().first()

    return render_template("/dashboard/graph_infomation.html", chart_json=json_data, model=model, station_descriptions=get_station_dexcriptions())
