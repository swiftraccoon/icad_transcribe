# routes/middleware.py

import os
import base64
from flask import request, session

def log_ip():
    """Logs the IP address of the incoming request."""
    from app import main_logger  # Import your logger instance here
    ip_address = request.remote_addr
    main_logger.debug(f"Request received from IP address: {ip_address}")

def generate_csrf_token():
    """Generates a CSRF token if not already in the session."""
    if '_csrf_token' not in session:
        session['_csrf_token'] = base64.urlsafe_b64encode(os.urandom(24)).decode('utf-8')
    return session['_csrf_token']

def inject_csrf_token():
    """Injects the CSRF token into the context processor."""
    return dict(csrf_token=generate_csrf_token())
