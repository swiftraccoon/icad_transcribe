from functools import wraps

import jwt
from flask import redirect, url_for, session, request, jsonify, current_app, render_template, flash

from lib.token_module import get_tokens


def csrf_protect_pre_login(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if request.method in ['POST', 'PUT', 'DELETE']:
            token = session.get('_csrf_token', None)
            if not token or token != request.form.get('_csrf_token'):
                flash('Form submission failed. Please try again.', 'danger')
                return redirect(request.referrer or '/')

        return func(*args, **kwargs)
    return wrapper

def csrf_protect_ajax(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if request.method in ['POST', 'PUT', 'DELETE']:
            token_in_session = session.get('_csrf_token', None)
            token_in_form = request.form.get('_csrf_token')

            if not token_in_session or token_in_form != token_in_session:
                # Return a JSON error instead of a redirect
                return jsonify({
                    'success': False,
                    'message': 'CSRF token is invalid or missing.'
                }), 400
        return func(*args, **kwargs)
    return wrapper

def token_required(f):
    @wraps(f)
    def wrapped_function(*args, **kwargs):
        # 1. Try to get token from 'Authorization' header
        #    e.g., "Authorization: Bearer <token>"
        auth_header = request.headers.get('Authorization')
        token = None
        if auth_header:
            # If you expect a "Bearer " prefix, remove it
            if auth_header.lower().startswith("bearer "):
                token = auth_header[7:].strip()
            else:
                token = auth_header.strip()

        # 2. If no token in Authorization header, check form data for a "key" field
        if not token:
            token = request.form.get('key', None)

        # 3. If still no token, deny
        if not token:
            return jsonify({"success": False, "message": "Access Denied"}),

        # 4. Query the database for a matching token record
        token_result = get_tokens(current_app.config['db'], token_value=token)

        if not token_result["success"]:
            return jsonify({"success": False, "message": token_result["message"]}), 500

        # If no matching tokens found, deny
        if not token_result["result"]:
            return jsonify({"success": False, "message": "Access Denied"}), 401

        # 5. (Optional) Check caller's IP address if the token has IP restrictions
        token_record = token_result["result"][0]
        ip_restrictions = token_record.get("token_ip_address")  # This should be a list
        caller_ip = request.remote_addr  # or use X-Forwarded-For logic if behind proxy

        # If the token has IP restrictions (i.e. non-empty list)
        if ip_restrictions:
            # If "*" is NOT present AND caller_ip is NOT in the list => deny
            # So "if '*' not in ip_restrictions AND caller_ip not in ip_restrictions => 403"
            if '*' not in ip_restrictions and caller_ip not in ip_restrictions:
                return jsonify({"success": False, "message": f"Access Denied: '{caller_ip}'"}), 403

        # If everything is fine, proceed
        return f(*args, **kwargs)

    return wrapped_function

def login_required(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        authenticated = session.get('authenticated')
        if not authenticated:
            current_app.config['logger'].debug(f"Redirecting User: Current session data: {dict(session)}")
            return redirect(url_for('base_site.base_site_index'))
        else:
            current_app.config['logger'].debug(f"User Authorized: Current session data: {dict(session)}")

        return func(*args, **kwargs)

    return wrapper

def login_or_token_required(f):
    """
    Decorator that allows access if EITHER:
      - The user is authenticated via the session (login_required logic), OR
      - A valid JWT token is present (token_required logic).
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        # 1) Check if the user is already authenticated via session
        authenticated = session.get("authenticated")
        if authenticated:
            # The user is logged in, so just proceed
            current_app.config["logger"].debug(f"User Authorized via session")
            return f(*args, **kwargs)

        # 1. Try to get token from 'Authorization' header
        #    e.g., "Authorization: Bearer <token>"
        auth_header = request.headers.get('Authorization')
        token = None
        if auth_header:
            # If you expect a "Bearer " prefix, remove it
            if auth_header.lower().startswith("bearer "):
                token = auth_header[7:].strip()
            else:
                token = auth_header.strip()

        # 2. If no token in Authorization header, check form data for a "key" field
        if not token:
            token = request.form.get('key', None)

        # 3. If still no token, deny
        if not token:
            return jsonify({"success": False, "message": "Access Denied"}),

        # 4. Query the database for a matching token record
        token_result = get_tokens(current_app.config['db'], token_value=token)

        if not token_result["success"]:
            return jsonify({"success": False, "message": token_result["message"]}), 500

        # If no matching tokens found, deny
        if not token_result["result"]:
            return jsonify({"success": False, "message": "Access Denied"}), 401

        # 5. (Optional) Check caller's IP address if the token has IP restrictions
        token_record = token_result["result"][0]
        ip_restrictions = token_record.get("token_ip_address")  # This should be a list
        caller_ip = request.remote_addr  # or use X-Forwarded-For logic if behind proxy

        # If the token has IP restrictions (i.e. non-empty list)
        if ip_restrictions:
            # If "*" is NOT present AND caller_ip is NOT in the list => deny
            # So "if '*' not in ip_restrictions AND caller_ip not in ip_restrictions => 403"
            if '*' not in ip_restrictions and caller_ip not in ip_restrictions:
                return jsonify({"success": False, "message": f"Access Denied: '{caller_ip}'"}), 403

        current_app.config["logger"].debug(f"User Authorized via token")
        return f(*args, **kwargs)

    return wrapper