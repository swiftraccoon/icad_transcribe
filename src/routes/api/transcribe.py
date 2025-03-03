import logging
import os
import time
import uuid

from flask import Blueprint, request, jsonify, flash, redirect, url_for, render_template, current_app, session

from lib.audio_file_module import validate_audio_file, pre_process_audio
from lib.audio_metadata_module import get_audio_metadata
from lib.transcribe_text_module import process_transcribe_text
from lib.whisper_module import get_whisper_config
from routes.decorators import login_or_token_required

transcribe = Blueprint('transcribe', __name__)
module_logger = logging.getLogger('icad_dispatch.api.transcribe')

@transcribe.route('/get', defaults={'transcribe_config_id': 1}, methods=['POST'])
@transcribe.route('/get/<int:transcribe_config_id>', methods=['POST'])
@login_or_token_required
def api_transcribe(transcribe_config_id):
    start = time.time()

    audio_file = request.files.get('audio')

    # 1. Check if an audio file was actually uploaded
    if not audio_file:
        result = {"success": False, "message": "No audio file uploaded"}
        module_logger.error("No audio file uploaded")
        return jsonify(result), 400

    # 2. Validate the audio file (MIME type, duration, etc.)
    validation_result = validate_audio_file(
        audio_file=audio_file
    )

    if not validation_result["success"]:
        # If validation failed, return the reason to the client
        module_logger.error(f"Audio validation failed: {validation_result['message']}")
        return jsonify(validation_result), 400

    audio_segment = validation_result["result"]["audio"]
    duration_seconds = validation_result["result"]["duration"]
    sample_rate = validation_result["result"]["sample_rate"]

    module_logger.info(
        f"Audio File Validated: duration={duration_seconds:.2f}s, sample_rate={sample_rate}"
    )

    # Get transcribe_config_id if provided in form upload
    form_config_id = request.form.get('transcribe_config_id')
    if form_config_id is not None:
        try:
            transcribe_config_id = int(form_config_id)
        except ValueError:
            pass

    # Get the call data from the form
    call_data = get_audio_metadata(request.form)

    # get the transcribe configuration from transcribe_config_id
    transcribe_config = get_whisper_config(current_app.config['db'], transcribe_config_id=transcribe_config_id)

    if not transcribe_config.get('success'):
        return jsonify({
            "success": False,
            "message": f"Unable to get the transcribe configuration id {transcribe_config_id} {transcribe_config.get('message')}",
            "result": []
        }), 500

    # result is returned as a list, get the first item
    transcribe_config = transcribe_config.get('result')[0]

    tone_removal_configuration = transcribe_config.get('tone_removal_config')
    amplify_configuration = transcribe_config.get('amplify_config')
    vad_configuration = transcribe_config.get('vad_config')
    transcribe_config = transcribe_config.get('transcribe_config')


    # pre-process audio
    processed_audio, detected_tones = pre_process_audio(audio_segment, tone_removal_configuration, amplify_configuration, vad_configuration)

    # transcribe via Whisper
    segments, info = current_app.config['wt'].transcribe_audio(processed_audio, transcribe_config)

    # Process transcribe segments
    processed_segments, transcript_text = process_transcribe_text(segments, transcribe_config, tone_removal_configuration, call_data.get('sources', []), detected_tones)

    process_time = round((time.time() - start), 2)

    current_app.config['logger'].debug(f"Transcribe Result {transcript_text}\n{processed_segments}\n{process_time}")

    response = {"success": True, "message": "Transcribe Success!", "transcript": transcript_text, "segments": processed_segments, "process_time_seconds": process_time}

    # Save Audio if sent from Admin Panel
    if request.form.get('_csrf_token'):

        # Generate a new filename
        audio_filename = f"{uuid.uuid4()}.mp3"
        audio_save_path = os.path.join('static', 'audio', audio_filename)

        # Export/save
        audio_segment.export(audio_save_path, format='mp3')

        # Generate a usable URL (relative to your Flask static folder)
        audio_url = url_for('static', filename=f'audio/{audio_filename}')

        response["audio_src"] = audio_url

    return jsonify(response), 200