# ICAD Transcribe

**ICAD Transcribe** is a Flask-based web application that processes and transcribes audio files using a [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) model. It provides:

1. **RESTful endpoints** for uploading and transcribing audio
2. **Configurable** Whisper model settings (decoding, prompts, VAD, tone-removal, amplification, etc.)
3. **Integration** with SQLite for application/user/config storage
4. **Robust pre-processing** (tone detection/removal, VAD filtering, amplification)
5. **API token management** for programmatic access
6. **Web-based** minimal admin panel for managing systems, talkgroups, transcription configs, etc.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Endpoints Overview](#endpoints-overview)
- [Transcription Flow](#transcription-flow)
- [Usage Examples](#usage-examples)

---

## Features

- **Upload & Transcribe**: Send audio via REST (`/api/transcribe/get`) to get back JSON transcripts (including word-level timestamps if enabled).
- **Multiple Configurations**: Create or customize multiple "transcribe configurations" with different Whisper settings—like beam size, temperature, VAD thresholds, prompt text, etc.
- **Tone Removal**: Detect and remove various alert tones (DTMF, MDC, two-tone, hi-low warble, etc.) from the audio or inject them as special segments in the transcript.
- **Voice Activity Detection (VAD)**: Filter out non-speech audio to reduce noise or only amplify speech segments.
- **Amplification**: Increase volume of speech segments while leaving non-speech or removed-tone sections silent.
- **GPU/CPU Support**: Automatically load the model on multiple GPUs (when `WHISPER_DEVICE=cuda`) or fallback to CPU.
- **SQLite Storage**: Store user accounts, tokens, and Whisper configurations in a local SQLite database.  
- **Session/Token Auth**: Endpoints for both a session-based admin UI and token-based programmatic usage.
- **File Validation**: Check MIME types, durations, etc., upon file upload.

---

## Project Structure

A quick overview of key directories and modules:

```
.
├── etc/
│   ├── secret_key           # Auto-generated secret key file if not provided
│   └── config.json          # (Optional) Additional config files
├── init_db/
│   └── transcribe_db.sql    # SQL schema for initial database creation
├── log/
│   └── icad_transcribe.log  # Default log output (auto-created)
├── static/
│   └── audio/               # Directory where uploaded/processed audio can be stored
├── templates/               # HTML templates for minimal admin pages
├── var/
│   └── models/              # Location of downloaded whisper models (faster-whisper)
├── src/
│   ├── lib/                 # Core library modules:
│   │   ├── address_handler.py
│   │   ├── audio_file_module.py
│   │   ├── audio_metadata_module.py
│   │   ├── exceptions_module.py
│   │   ├── gpu_handler.py
│   │   ├── logging_module.py
│   │   ├── replacement_handler.py
│   │   ├── sqlite_module.py
│   │   ├── system_module.py
│   │   ├── talkgroup_module.py
│   │   ├── token_module.py
│   │   ├── transcribe_text_module.py
│   │   ├── user_module.py
│   │   ├── utility_module.py
│   │   └── whisper_module.py
│   ├── routes/
│   │   ├── admin/       # Admin Blueprint (Dashboard pages)
│   │   ├── api/         # API Blueprints (system/talkgroup/transcribe/whisper)
│   │   ├── auth/        # Auth Blueprint (login/logout/token)
│   │   ├── decorators.py
│   │   └── middleware.py
│   └── app.py           # Main Flask Application Entry
└── requirements.txt      # Python dependencies
```

---

## Requirements

- **Python 3.9+** (Recommended; tested up to 3.12)
- [FFmpeg](https://ffmpeg.org/) (required by pydub to process audio)
- (Optional) **NVIDIA GPU** with `nvidia-smi` for CUDA-based transcription
- Python packages (typical):
  - `flask`, `flask_session`, `Werkzeug`
  - `faster-whisper`
  - `pydub`
  - `silero_vad`
  - `bcrypt`
  - `python-magic`
  - `numpy`
  - `sqlite3` (standard library)
  - etc.

See **requirements.txt** or the `pyproject.toml` (if present) for the full list.

---

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/TheGreatCodeholio/icad_transcribe
   cd icad_transcribe
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   Ensure that you also have FFmpeg installed and available on your system PATH.

4. **Initialize environment variables**:
   - You can create a `.env` file in the project root, or set them in your shell environment:
     ```bash
     # Example .env or environment export
     export SECRET_KEY="some-random-string"
     export LOG_LEVEL=1
     export WHISPER_MODEL="small.en"
     export WHISPER_DEVICE="cpu"            # or "cuda"
     export WHISPER_GPU_INDEXES="0,1"       # or "all", if using multiple GPUs
     export AUDIO_UPLOAD_MAX_FILE_SIZE_MB=5
     ```
   - If `SECRET_KEY` is not provided, a file `etc/secret_key` will be generated automatically.

5. **Database Initialization**:
   - On first run, the code automatically creates **SQLite** database file using `init_db/transcribe_db.sql`.
   - The default path is controlled by environment variable `SQLITE_DATABASE_PATH`. If not set, it defaults to `1` (which is replaced with an actual path).
     - Typically, you'd do:  
       `export SQLITE_DATABASE_PATH="./var/icad_transcribe.db"`
     - The app logs warnings if it doesn't find the DB. It will create tables and a default `admin` user with password `admin`.

---

## Configuration

In addition to any optional `config.json`, the application respects the following environment variables:

| Variable                               | Default                               | Description                                                                                                                                                                            |
|----------------------------------------|---------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **`SECRET_KEY`**                       | *Auto-generated*                      | Secret key for session signing. If undefined, the app creates/stores a random key in `etc/secret_key`.                                                                                 |
| **`LOG_LEVEL`**                        | `1` (DEBUG)                           | Logging verbosity, mapped to Python log levels: `1=DEBUG`, `2=INFO`, `3=WARNING`, `4=ERROR`, `5=CRITICAL`.                                                                             |
| **`WHISPER_MODEL_PATH`**               | `var/models`                          | Directory where Faster-Whisper models are downloaded/stored.                                                                                                                           |
| **`WHISPER_MODEL`**                    | `small.en`                            | Which Faster-Whisper model variant to load (e.g., `tiny`, `base`, `small`, `medium`, `large`, or an `.en` variant).                                                                    |
| **`WHISPER_DEVICE`**                   | `cpu`                                 | Device selection for inference: `cpu` or `cuda`.                                                                                                                                       |
| **`WHISPER_GPU_INDEXES`**              | `0`                                   | Comma-separated GPU indices to use (e.g. `0,1`) or `all` for every available GPU. Only relevant if `WHISPER_DEVICE=cuda`.                                                               |
| **`WHISPER_COMPUTE_TYPE`**             | `float16`                             | Data precision for Faster-Whisper (e.g., `float32`, `float16`, `int8_float16`).                                                                                                        |
| **`WHISPER_CPU_THREADS`**              | `4`                                   | Number of CPU threads to use if `WHISPER_DEVICE=cpu`.                                                                                                                                  |
| **`WHISPER_NUM_WORKER`**               | `1`                                   | Internal parallelism/workers used during model loading.                                                                                                                                |
| **`WHISPER_BATCHED`**                  | `False`                               | Whether to enable batched inference (`True` or `False`).                                                                                                                               |
| **`SQLITE_DATABASE_PATH`**             | `1`                                   | Path for the SQLite database file. If not set or invalid, the system attempts to create a default local DB file.                                                                       |
| **`BASE_URL`**                         | `localhost`                           | Used for references to the application’s base URL (e.g., in templates).                                                                                                                 |
| **`SESSION_COOKIE_SECURE`**            | `False`                               | If `True`, session cookies only sent over HTTPS.                                                                                                                                        |
| **`SESSION_COOKIE_DOMAIN`**            | `localhost`                           | The domain for session cookies.                                                                                                                                                         |
| **`SESSION_COOKIE_NAME`**              | `localhost`                           | Cookie name for the session.                                                                                                                                                            |
| **`SESSION_COOKIE_PATH`**              | `/`                                   | Path for which the session cookie is valid.                                                                                                                                             |
| **`SESSION_COOKIE_SAMESITE`**          | `Lax`                                 | Cross-site protection policy for session cookies (`Lax`, `Strict`, or `None`).                                                                                                          |
| **`AUDIO_UPLOAD_MAX_FILE_SIZE_MB`**    | `5`                                   | Maximum audio upload size in MB. If exceeded, the request is rejected.                                                                                                                  |
| **`AUDIO_UPLOAD_ALLOWED_MIMETYPES`**   | `audio/x-wav,audio/x-m4a,audio/mpeg`  | Comma-separated list of acceptable MIME types for uploaded audio.                                                                                                                       |
| **`AUDIO_UPLOAD_MIN_AUDIO_LENGTHS`**   | `0`                                   | Minimum audio file duration (in seconds). Set to `0` to allow any minimum.                                                                                                              |
| **`AUDIO_UPLOAD_MAX_AUDIO_LENGTH`**    | `300`                                 | Maximum audio file duration (in seconds). Default 300 allows up to 5 minutes.                                                                                                           |

> **Note**  
> - If you omit `SECRET_KEY`, a random one is generated and persisted at `etc/secret_key`.  
> - The `SQLITE_DATABASE_PATH` defaults to a special placeholder (`1`) that triggers automatic creation of a local DB under `var/`. You can override it with a real path (e.g., `export SQLITE_DATABASE_PATH=./var/icad_transcribe.db`).  
> - `WHISPER_BATCHED` should be set to a string `"True"` or `"False"` (the code interprets it as a boolean).  
> - `WHISPER_GPU_INDEXES` has no effect unless `WHISPER_DEVICE="cuda"`.  
> - `AUDIO_UPLOAD_*` variables control validations in `audio_file_module.py`.

Once your environment variables are configured, run the Flask app (or Gunicorn, etc.) as normal. The application reads and applies these values at startup.
---

## Running the Application

1. **Local development run**:
   ```bash
   (venv) python src/app.py
   ```
   The Flask server will start, typically at `http://127.0.0.1:5000`.

2. **Production**:  
   In production, run behind a WSGI container (e.g., gunicorn), then reverse-proxy from NGINX. For example:
   ```bash
   gunicorn --bind 0.0.0.0:3001 src.app:app
   ```

3. **Log in**:  
   Access the base site in your browser:  
   ```
   http://localhost:5000/
   ```
   The default admin username is `admin` with password `admin`.  
   Once logged in, you’ll see minimal admin pages:
   - `GET /admin/dashboard`
   - `GET /admin/configurations`
   - etc.

---

## Endpoints Overview

The app is organized into several **Flask Blueprints** under `src/routes/`:

### 1. Authentication

- **`POST /auth/login`**  
  - Form fields: `username`, `password`
  - Sets session upon success
- **`GET /auth/logout`**  
  - Clears session
- **`POST /auth/token/add`**  
  - Creates an API token for programmatic use
- **`GET /auth/token/get`**  
  - Lists tokens filtered by optional parameters
- **`POST /auth/token/update`** / **`delete`**  
  - Update or remove an API token

### 2. Transcription

- **`POST /api/transcribe/get`**  
  - Expects an `audio` file in form-data
  - Optional `transcribe_config_id` to pick custom settings
  - Returns JSON with transcript and segment details
  - Requires session login or token-based auth

### 3. System & Talkgroup Management

- **`POST /api/system/add`**, `update`, `delete`
- **`GET /api/system/get`**  
  - Supports query params for searching
- **`POST /api/talkgroup/add`**, `update`, `delete`
- **`GET /api/talkgroup/get`**

### 4. Whisper Configuration

- **`POST /api/whisper/config/add`**, `update`, `delete`
- **`GET /api/whisper/config/get`**

These allow you to create multiple transcription configurations, each referencing different advanced settings (like beam size, temperature arrays, VAD, tone removal, etc.).

### 5. Admin Pages

- **`GET /admin/dashboard`**  
  Basic admin home page.  
- **`GET /admin/configurations`**  
  Manage transcription configs.

---

## Transcription Flow

1. **Upload** an audio file (`POST /api/transcribe/get`) including any optional metadata:
   - The server **validates** the file (MIME type, length).
   - The server applies **pre-process** steps:  
     a) Tone detection/removal (silence inserted where tones were removed).  
     b) (Optionally) Voice Activity Detection to isolate speech.  
     c) (Optionally) Amplification of speech segments.  
   - The processed audio is then fed to **Faster-Whisper** with the chosen config.
2. **Segmentation**: The model returns segments (and word-level timestamps if enabled).
3. **Tone Injection**: If tones were detected, they can be re-injected as annotated text segments.
4. **Metadata**: Associates segments with any external metadata (e.g., talkgroup or source units).
5. **Response**: A JSON object with final transcript and an array of segments.

---

## Usage Examples

### 1. Simple `curl` example

```bash
curl -X POST \
     -F "audio=@audio_file.wav" \
     -F "transcribe_config_id=1" \
     http://localhost:5000/api/transcribe/get
```

- Returns JSON:
  ```json
  {
    "success": true,
    "message": "Transcribe Success!",
    "transcript": "Hello world, this is a test.",
    "segments": [
      {
        "segment_id": 1,
        "text": "Hello world, this is a test.",
        "words": [ ... word-level data ...],
        "start": 0.0,
        "end": 3.5
      }
    ],
    "process_time_seconds": 1.23
  }
  ```

### 2. Using an API token

If you generated a token `YOUR_TOKEN_HERE`, you can supply it in:

- A header: `Authorization: Bearer YOUR_TOKEN_HERE`
- Or form field: `key=YOUR_TOKEN_HERE`

For example:

```bash
curl -X POST \
     -H "Authorization: Bearer YOUR_TOKEN_HERE" \
     -F "audio=@audio_file.wav" \
     http://localhost:5000/api/transcribe/get
```