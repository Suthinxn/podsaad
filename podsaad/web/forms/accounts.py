from flask_wtf import FlaskForm
from wtforms import PasswordField, validators, StringField, SelectField
from flask_mongoengine.wtf import model_form
from podsaad.web import models

BaseLoginForm = model_form(
    models.User,
    FlaskForm,
    field_args={"username": {"label": "Username"}},
    only=["username", "password"],
)


class LoginForm(BaseLoginForm):
    password = PasswordField(
        "Password", validators=[validators.InputRequired(), validators.Length(min=6)]
    )


class SetupPasswordForm(FlaskForm):
    password = PasswordField("รหัสผ่านใหม่", validators=[validators.InputRequired()])
    confirm_password = PasswordField(
        "ยืนยันรหัสผ่านใหม่",
        validators=[
            validators.InputRequired(),
            validators.Length(min=6),
            validators.EqualTo("password", message="รหัสผ่านไม่ตรงกัน"),
        ],
    )


BaseEditProfileForm = model_form(
    models.User,
    FlaskForm,
    only=[
        "first_name",
        "last_name",
        "email",
        "phone",
        "role",
    ],
    field_args={
        "first_name": {"label": "ชื่อจริง"},
        "last_name": {"label": "นามสกุล"},
        "email": {"label": "อีเมล"},
        "phone": {"label": "เบอร์โทรศัพย์"},
        "role": {"label": "ตำแหน่ง"},
    },
)


class EditProfileForm(BaseEditProfileForm):
    pass
