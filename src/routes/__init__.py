# routes/__init__.py

from flask import Flask
from routes.middleware import inject_csrf_token
from routes.base_site.base_site import base_site
from routes.auth.auth import auth
from routes.admin.admin import admin
from routes.api.system import system
from routes.api.talkgroup import talkgroup
from routes.api.transcribe import transcribe
from routes.api.whisper import whisper

def register_middlewares(app: Flask):
    """Registers global middlewares for the Flask app."""
    app.context_processor(inject_csrf_token)  # Inject CSRF token into templates