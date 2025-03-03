import logging
import uuid

from flask import Blueprint, request, jsonify, flash, redirect, url_for, render_template, current_app, session

from lib.token_module import get_tokens, add_token, delete_token, update_token
from lib.user_module import authenticate_user, user_change_password, get_users
from routes.decorators import login_required, csrf_protect_pre_login, csrf_protect_ajax

auth = Blueprint('auth', __name__)
module_logger = logging.getLogger('icad_transcribe.auth')


@auth.route('/login', methods=['POST'])
@csrf_protect_pre_login
def auth_login():
    username = request.form['username']
    password = request.form['password']
    if not username or not password:
        flash('Username and Password Required', 'danger')
        return redirect(url_for('base_site.base_site_index'))

    auth_result = authenticate_user(current_app.config['db'], username, password)
    flash(auth_result["message"], 'success' if auth_result["success"] else 'danger')
    return redirect(
        url_for('admin.admin_dashboard') if auth_result["success"] else url_for('base_site.base_site_index'))


@auth.route("/logout", methods=['GET'])
def auth_logout():
    session.clear()
    return redirect(url_for('base_site.base_site_index'))


@auth.route("/change_password", methods=['POST'])
@login_required
def auth_change_password():
    try:
        # Extract JSON data from request
        request_form = request.form
        if not request_form:
            message = "No data provided."
            module_logger.error(message)
            flash(message, 'danger')
            return redirect(url_for('admin.admin_dashboard'))

        current_password = request_form.get("currentPassword")
        new_password = request_form.get("newPassword")

        if not new_password or not new_password:
            message = "No password provided or empty string."
            module_logger.error(message)
            flash(message, 'danger')
            return redirect(url_for('admin.admin_dashboard'))

        result = user_change_password(current_app.config['db'], "admin", current_password, new_password)

        module_logger.debug(result.get("message"))
        flash(result.get("message"), 'success' if result.get("success") else 'danger')
        return redirect(url_for('admin.admin_dashboard'))

    except Exception as e:
        message = f"Unexpected error: {e}"
        module_logger.error(message)
        flash(message, 'danger')
        return redirect(url_for('admin.admin_dashboard'))

@auth.route("/token/add", methods=['POST'])
@login_required
@csrf_protect_ajax
def auth_add_token():
    user_id = request.form.get("user_id") # User ID requesting token.
    token_name = request.form.get("token_name") # Token name
    ip_address = request.form.get("token_ip_address") # Comma Seperated string of IP Addresses

    # Initialize IP address list
    ip_address_list = []

    # Add provided IP addresses if they exist
    if ip_address:
        if isinstance(ip_address, list):
            ip_address_list.extend(ip_address)
        elif isinstance(ip_address, str):
            # If given as a comma-separated string, split and add
            ip_address_list.extend([ip.strip() for ip in ip_address.split(",")])

    # Ensure there are no duplicate IPs and all entries are valid
    ip_address_list = list(set(ip_address_list))

    if len(ip_address_list) < 1:
        ip_address_list = ["*"]

    # Validate User ID
    user_result = get_users(current_app.config["db"], user_id=user_id)
    if not user_result.get('success'):
        return {
            "success": False,
            "message": f"Unexpected error: {user_result.get('message')}",
            "result": []
        }, 500

    if not user_result.get('result'):
        return {
            "success": False,
            "message": f"User not found.",
            "result": []
        }, 401

    # Validate Token Name
    token_result = get_tokens(current_app.config["db"], token_name=token_name)
    if not token_result.get('success'):
        return {
            "success": False,
            "message": f"Unexpected error: {token_result.get('message')}",
            "result": []
        }

    if token_result.get('result'):
        return {
            "success": False,
            "message": f"Token with that that name already exists.",
            "result": []
        }, 201


    # Generate Token
    token = str(uuid.uuid4())

    # Insert token database
    add_result = add_token(current_app.config["db"], token, token_name, ip_address_list, user_id)

    return jsonify(add_result)

@auth.route('/token/get' , methods=['GET'])
@login_required
@csrf_protect_ajax
def auth_get_token():
    token_id = request.args.get('token_id')
    token_name = request.args.get('token_name')
    user_id = request.args.get('user_id')

    result = get_tokens(current_app.config['db'], token_id=token_id, token_name=token_name, user_id=user_id)

    return jsonify(result), 200

@auth.route('/token/delete', methods=['POST'])
@login_required
@csrf_protect_ajax
def auth_delete_token():
    token_id = request.form.get('token_id')

    if not token_id:
        return jsonify({"success": False, "message": "Invalid Request", "result": []}), 400

    delete_result = delete_token(current_app.config['db'], token_id)
    return jsonify(delete_result), 200

@auth.route('/token/update', methods=['POST'])
@login_required
@csrf_protect_ajax
def auth_update_token():
    token_id = request.form.get('token_id')

    if not token_id:
        return jsonify({"success": False, "message": "Invalid Request", "result": []}), 400

    update_result = update_token(current_app.config['db'], request.form)

    return jsonify(update_result), 200