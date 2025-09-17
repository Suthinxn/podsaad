import mongoengine as me
from flask_login import UserMixin
import datetime
from bson import ObjectId, DBRef


USER_ROLES = [
    ("user", "ผู้ใช้"),
    ("admin", "ผู้ดูแล"),
]


class User(me.Document, UserMixin):
    username = me.StringField(required=True, min_length=3, max_length=64)
    password = me.StringField(required=True, default="")
    first_name = me.StringField(required=True, max_length=128)
    last_name = me.StringField(required=True, max_length=128)

    email = me.StringField(max_length=128)
    phone = me.StringField(max_length=11, required=True)

    status = me.StringField(required=True, default="active")
    role = me.StringField(default="admin", choices=USER_ROLES, required=True)

    created_date = me.DateTimeField(required=True, default=datetime.datetime.now)
    updated_date = me.DateTimeField(
        required=True, default=datetime.datetime.now, auto_now=True
    )
    last_login_date = me.DateTimeField(
        required=True, default=datetime.datetime.now, auto_now=True
    )

    meta = {"collection": "users", "indexes": ["first_name", "last_name"]}

    def set_password(self, password):
        from werkzeug.security import generate_password_hash

        self.password = generate_password_hash(password)

    def check_password(self, password):
        from werkzeug.security import check_password_hash

        if check_password_hash(self.password, password):
            return True
        return False

    def get_fullname(self):
        return f"{self.first_name} {self.last_name}"

    def display_role(self):
        if "admin" in self.role:
            return "ผู้ดูแล"
        return "พนักงาน"

    def display_status(self):
        if "disactivate" in self.status:
            return "ยกเลิก"
        return "เปิดใช้งาน"