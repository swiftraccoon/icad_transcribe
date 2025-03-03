import logging

from flask import Blueprint, request, jsonify, flash, redirect, url_for, render_template, current_app, session

from lib.system_module import get_systems
from routes.decorators import login_required

admin = Blueprint('admin', __name__)
module_logger = logging.getLogger('icad_transcribe.admin')

@admin.route('/dashboard', methods=['GET'])
@login_required
def admin_dashboard():
    return render_template("admin_dashboard.html")

@admin.route('/configurations', methods=['GET'])
@login_required
def admin_configurations():
    return render_template("admin_transcribe.html")

@admin.route('/api_tokens', methods=['GET'])
@login_required
def admin_api_tokens():
    return render_template("admin_api_tokens.html")

# @admin.route('/systems', methods=['GET'])
# @login_required
# def admin_systems():
#     return render_template("admin_systems.html")
#
# @admin.route('/talkgroups/<int:radio_system_id>', methods=['GET'])
# @login_required
# def admin_system_talkgroups(radio_system_id):
#     if radio_system_id <= 0:
#         flash("Invalid Radio System ID.")
#         return redirect(url_for('admin.admin_systems'))
#
#     radio_system_data = get_systems(current_app.config['db'], radio_system_id=radio_system_id)
#
#     if radio_system_data.get('success'):
#         radio_system_data = radio_system_data.get('result')[0]
#         if not radio_system_data:
#             flash("Error getting Radio System Data.")
#             return redirect(url_for('admin.admin_systems'))
#     else:
#         flash("Error Getting Radio System Data.")
#         return redirect(url_for('admin.admin_systems'))
#
#     return render_template("admin_talkgroups.html", radio_system_id=radio_system_id, radio_system_name=radio_system_data.get('system_name'))