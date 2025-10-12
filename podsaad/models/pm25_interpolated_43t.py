from mongoengine import Document, StringField, FloatField, DateTimeField

class PM25Interpolated43t(Document):
    meta = {
        "collection": "pm25_interpolated_43t",
        "strict": False
    }
    timestamp = StringField()
    station_name = StringField()
    station_code = StringField()
    lat = FloatField()
    lon = FloatField()
    PM_2_5 = FloatField()
    PM_1 = FloatField()
    PM_0_1 = FloatField()
    temperature = FloatField()
    humidity = FloatField()
    pressure = FloatField()
    day_of_week_sin = FloatField()
    day_of_week_cos = FloatField()
    month_sin = FloatField()
    month_cos = FloatField()