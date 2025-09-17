from flask_wtf import FlaskForm
from wtforms import PasswordField, validators, StringField, HiddenField
from flask_mongoengine.wtf import model_form
from podsaad.web import models


BaseUserForm = model_form(
    models.User,
    FlaskForm,
    exclude=[
        "status",
        "created_date",
        "updated_date",
        "password",
        "last_login_date",
    ],
    field_args={
        "username": {"label": "ชื่อบัญชี"},
        "first_name": {"label": "ชื่อจริง"},
        "last_name": {"label": "นามสกุล"},
        "email": {"label": "อีเมล"},
        "phone": {"label": "เบอร์โทรศัพท์"},
        "role": {"label": "ตำแหน่ง"},
    },
)


class UserForm(BaseUserForm):
    password = PasswordField(label="รหัสผ่าน", validators=[validators.InputRequired()])
    confirm_password = PasswordField(
        label="ยืนยันรหัสผ่าน", validators=[validators.InputRequired()]
    )


class EditUserForm(BaseUserForm):
    username = StringField(label="ชื่อบัญชี", render_kw={"readonly": True})


class SearchUserForm(FlaskForm):
    name = StringField(label="ชื่อ - นามสกุล")
