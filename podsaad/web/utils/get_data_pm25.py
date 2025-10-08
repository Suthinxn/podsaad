from datetime import datetime
import requests
import pandas as pd


def fetch_data(start, end, config):
    source = config["SOURCE"]
    base_url = config["API_DHARA"]

    params = {
        "source": source,
        "started_datetime": start.strftime("%Y-%m-%d"),
        "ended_datetime": end.strftime("%Y-%m-%d"),
    }
    resp = requests.get(base_url, params=params)
    resp.raise_for_status()
    return resp.json()


def get_raw_data(data):
    rows = []
    for station in data.get("stations", []):
        station_info = {
            "station_name": station["name"],
            "station_name_th": station["name_th"],
            "station_code": station["code"],
            "source": station["source"],
            "lat": station["coordinates"]["coordinates"][1],
            "lon": station["coordinates"]["coordinates"][0],
            "status": station["status"],
        }

        for climate in station.get("climates", []):
            row = station_info.copy()
            row.update(
                {
                    "sensor_type": climate["sensor_type"],
                    "value": climate["value"],
                    "timestamp": climate["timestamp"],
                }
            )
            rows.append(row)

    df = pd.DataFrame(rows)

    return df


def filter_by_station(df):
    stations = [
        "119t",
        "118t",
        "93t",
        "89t",
        "62t",
        "121t",
        "o73",
        "120t",
        "43t",
        "63t",
        "78t",
        "o70",
        "44t",
        "o28",
        "42t",
    ]

    filtered_df = df[df["station_code"].isin(stations)]
    print("[DEBUG filtered]", len(filtered_df), "rows after filtering by stations")

    pivot_df = filtered_df.pivot_table(
        index=[
            "timestamp",
            "station_name",
            "station_name_th",
            "station_code",
            "source",
            "lat",
            "lon",
            "status",
        ],
        columns="sensor_type",
        values="value",
        aggfunc="first",
    ).reset_index()

    return pivot_df
