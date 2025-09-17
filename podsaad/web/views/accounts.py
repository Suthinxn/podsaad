import datetime
import mongoengine as me

from flask import (
    Blueprint,
    render_template,
    url_for,
    request,
    session,
    redirect,
)

from flask_login import login_user, logout_user, login_required, current_user

from podsaad import models
from podsaad.web import forms

from flask_bcrypt import Bcrypt

bcrypt = Bcrypt()

module = Blueprint("accounts", __name__)


@module.route("/login", methods=["GET", "POST"])
def login():
    # form = forms.accounts.LoginForm()
    # if not form.validate_on_submit():
    #     error_msg = form.errors
    #     if form.errors == {"password": ["Field must be at least 6 characters long."]}:
    #         error_msg = "ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง"
    #     if form.errors == {"password": ["This field is required."]}:
    #         error_msg = "กรุณากรอกรหัสผ่าน"

    return render_template("/accounts/login.html")

    # user = models.User.objects(username=form.username.data).first()
    # if not user or not user.check_password(form.password.data):
    #     error_msg = "ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง"
    #     return render_template("/accounts/login.html", form=form, error_msg=error_msg)

    # if user.status == "disactive":
    #     error_msg = "บัญชีของท่านถูกลบออกจากระบบ"
    #     return render_template("/accounts/login.html", form=form, error_msg=error_msg)

    # login_user(user)
    # user.last_login_date = datetime.datetime.now()
    # user.save()
    # next = request.args.get("next")
    # if next:
    #     return redirect(next)

    # return redirect(url_for("dashboard.index"))
