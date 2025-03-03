import os
import sqlite3
import sys
import time

from flask import Flask, request, render_template, jsonify
from flask.cli import load_dotenv

from flask_session import Session
from werkzeug.middleware.proxy_fix import ProxyFix

#Import Libraries
from lib.logging_module import CustomLogger
from lib.sqlite_module import SQLiteDatabase
from lib.utility_module import get_max_content_length
from lib.whisper_module import WhisperTranscribe

#Import Routes
from routes import base_site, auth, admin, system, talkgroup, transcribe, whisper, register_middlewares

app_name = "icad_transcribe"
__version__ = "2.1.4"

load_dotenv()

root_path = os.getcwd()
config_path = os.path.join(root_path, 'etc')
config_file_name = 'config.json'
config_file_path = os.path.join(config_path, config_file_name)
log_path = os.path.join(root_path, 'log')
log_file_name = f"{app_name}.log"
log_file_path = os.path.join(log_path, log_file_name)
var_path = os.path.join(root_path, 'var')
model_path = os.path.join(var_path, 'models')

if not os.path.exists(log_path):
    os.makedirs(log_path)

if not os.path.exists(config_path):
    os.makedirs(config_path)

if not os.path.exists(var_path):
    os.makedirs(var_path)

if not os.path.exists(model_path):
    os.makedirs(model_path)

# Start Logger
main_logger = CustomLogger(os.getenv('LOG_LEVEL', 1), f'{app_name}',
                           log_file_path, show_threads=True).logger

# Init Database
try:
    db = SQLiteDatabase()
    main_logger.info("SQLite Database connected successfully.")

except ValueError as e:
    # Typically raised if db_path is invalid or empty
    main_logger.error(f"Invalid SQLite database path: {e}")
    time.sleep(5)
    sys.exit(1)

except IsADirectoryError as e:
    # Raised if db_path is actually a directory
    main_logger.error(f"Database path is a directory, not a file: {e}")
    time.sleep(5)
    sys.exit(1)

except sqlite3.Error as e:
    # Raised if there's a lower-level SQLite error (I/O issues, etc.)
    main_logger.error(f"SQLite error during database initialization: {e}")
    time.sleep(5)
    sys.exit(1)

except Exception as e:
    # Catch-all for anything else unexpected
    main_logger.error(f"Unexpected error while connecting to the database: {e}")
    time.sleep(5)
    sys.exit(1)

# Init Whisper Model
try:
    # Attempt to create the WhisperTranscribe instance
    wt = WhisperTranscribe()
    main_logger.info("Whisper Transcribe initialized successfully.")

except KeyError as e:
    # Raised if required config keys are missing
    main_logger.error(f"Missing or invalid Whisper configuration key: {e}")
    time.sleep(5)
    sys.exit(1)

except ValueError as e:
    # For issues like unsupported devices ("Unsupported device type: x")
    main_logger.error(f"ValueError while starting Whisper Model: {e}")
    time.sleep(5)
    sys.exit(1)

except RuntimeError as e:
    # Potentially raised from GPU-related issues or nvidia-smi failures
    main_logger.error(f"RuntimeError while starting Whisper Model: {e}")
    time.sleep(5)
    sys.exit(1)

except Exception as e:
    # Catch-all for anything else
    main_logger.error(f"Unexpected error while starting Whisper Model: {e}")
    time.sleep(5)
    sys.exit(1)
app = Flask(__name__, template_folder='templates', static_folder='static')

app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)

# Load or create secret key
if not os.getenv('SECRET_KEY'):
    try:
        with open(os.path.join(root_path + '/etc', 'secret_key'), 'rb') as f:
            app.config['SECRET_KEY'] = f.read()
    except FileNotFoundError:
        secret_key = os.urandom(24)
        with open(os.path.join(root_path + '/etc', 'secret_key'), 'wb') as f:
            f.write(secret_key)
            app.config['SECRET_KEY'] = secret_key
else:
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

app.config['logger'] = main_logger
app.config['base_url'] = os.getenv('BASE_URL', 'localhost')
app.config['db'] = db
app.config['wt'] = wt

# set max content for file uploads
app.config['MAX_CONTENT_LENGTH'] = get_max_content_length(os.getenv('AUDIO_UPLOAD_MAX_FILE_SIZE_MB', 5))

# Session Configuration
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True

# Cookie Configuration
app.config['SESSION_COOKIE_SECURE'] = os.getenv('SESSION_COOKIE_SECURE', False)
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_DOMAIN'] = os.getenv('SESSION_COOKIE_DOMAIN', 'localhost')
app.config['SESSION_COOKIE_NAME'] = os.getenv('SESSION_COOKIE_NAME', 'localhost')
app.config['SESSION_COOKIE_PATH'] = os.getenv('SESSION_COOKIE_PATH', '/')
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Initializing the session
sess = Session()
sess.init_app(app)

# Register base site /
app.register_blueprint(base_site, url_prefix='/')

# Register auth routes /auth
app.register_blueprint(auth, url_prefix='/auth')

# Register Admin Routes /admin
app.register_blueprint(admin, url_prefix='/admin')

# Register API Routes /system
app.register_blueprint(system, url_prefix='/api/system')

# Register API Routes /talkgroup
app.register_blueprint(talkgroup, url_prefix='/api/talkgroup')

# Register API Routes /transcribe
app.register_blueprint(transcribe, url_prefix='/api/transcribe')

# Register API Routes /whisper
app.register_blueprint(whisper, url_prefix='/api/whisper')

# Register Middleware
register_middlewares(app)

# if __name__ == '__main__':
#     app.run(host="0.0.0.0", port=3001)
