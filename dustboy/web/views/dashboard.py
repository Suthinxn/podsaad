from flask import Blueprint, render_template
from flask_login import login_required
from datetime import datetime

from dustboy import models

module = Blueprint("dashboard", __name__)


@module.route("/")
def index():

    return render_template("/dashboard/index.html")
