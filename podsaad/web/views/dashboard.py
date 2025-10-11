from flask import Blueprint, render_template, jsonify, current_app, request
from flask_login import login_required
from datetime import date, datetime, timedelta



from podsaad.web.utils.get_heatmap import get_pm25_level, get_pm25_color, create_map
from podsaad import models

import warnings

warnings.filterwarnings("ignore")
module = Blueprint("dashboard", __name__)



@module.route("/")
def index():
    current_day = request.args.get("select_day", default=0, type=int)

    station_descriptions = {
    "119t": "สวนสาธารณะธารา ต.ปากน้ำ อ.เมือง, กระบี่",
    "118t": "สนามกีฬา จ.ชุมพร ต.ท่าตะเภา อ.เมือง, จ.ชุมพร",
    "93t": "วิทยาลัยสารพัดช่างตรัง ต.นาตาล่วง อ.เมือง, ตรัง",
    "89t": "ศูนย์ฟื้นฟูสุขภาพผู้สูงอายุ ต.คลัง อ.เมือง, นครศรีธรรมราช",
    "62t": "ศาลากลางจังหวัดนราธิวาส ต.บางนาค อ.เมือง, นราธิวาส",
    "121t": "ศูนย์ป่าไม้จังหวัดปัตตานี ต.รูสะมิแล อ.เมือง, ปัตตานี",
    "o73": "ศูนย์ราชการจังหวัดพังงา ต.ถ้ำน้ำผุด อ.เมือง, พังงา",
    "120t": "สนามกีฬากลางจังหวัดพัทลุง ต.เขาเจียก อ.เมือง, พัทลุง",
    "43t": "ศูนย์บริการสาธารณสุขเทศบาลภูเก็ต ต.ตลาดใหญ่ อ.เมือง, ภูเก็ต",
    "63t": "สนามโรงพิธีช้างเผือก ต.สะเตง อ.เมือง, ยะลา",
    "78t": "ศูนย์พัฒนาเด็กเล็กเทศบาลเมืองเบตง ต.เบตง อ.เบตง, ยะลา",
    "o70": "สำนักงานเทศบาลเมืองระนอง ต.เขานิเวศน์ อ.เมือง, ระนอง",
    "44t": "เทศบาลนครหาดใหญ่ ต.หาดใหญ่ อ.หาดใหญ่, สงขลา",
    "o28": "โรงเรียนชุมชนบ้านปาดัง ต.ปาดังเบซาร์ อ.สะเดา, จ.สงขลา",
    "42t": "สำนักงานสิ่งแวดล้อมภาคที่ 14 ต.มะขามเตี้ย อ.เมือง, สุราษฎร์ธานี",
    }



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

        latest_record = model_class.objects(timestamp="2025-09-06").first() 
        if latest_record:
            models_list.append(latest_record)
            # print(f"DEBUG Collection : {model_class}")

    print(f"DEBUG models list {models_list}")

    # Heatmap
    select_day = int(request.args.get("select_day", 0))
    map_html = create_map(select_day)

    return render_template("dashboard/index.html", map_html=map_html,today=datetime.today(),current_day=current_day,timedelta=timedelta, models_list=models_list, station_descriptions=station_descriptions)


# def pm25_to_intensity(value):
#     if value <= 15:
#         return 0.2  # ฟ้า/เขียวอ่อน
#     elif value <= 35:
#         return 0.4  # เขียว-เหลือง
#     elif value <= 55:
#         return 0.6  # ส้ม
#     elif value <= 75:
#         return 0.8  # แดงอ่อน
#     else:
#         return 1.0  # แดงเข้ม


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
