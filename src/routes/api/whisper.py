import logging
import time

from flask import Blueprint, request, jsonify, flash, redirect, url_for, render_template, current_app, session

from lib.utility_module import normalize_param
from lib.whisper_module import get_whisper_config, update_whisper_config_basic, update_whisper_config_decoding, \
    update_whisper_config_prompt, update_whisper_config_advanced, update_whisper_config_vad, add_whisper_config, \
    delete_whisper_config, update_whisper_config_amplify, update_whisper_config_tone_removal
from routes.decorators import login_or_token_required, login_required, csrf_protect_ajax

whisper = Blueprint('whisper', __name__)
module_logger = logging.getLogger('icad_dispatch.api.whisper')

@whisper.route('/config/add', methods=['POST'])
@login_required
@csrf_protect_ajax
def api_add_whisper_config():
    transcribe_config_name = request.form.get("transcribe_config_name")
    add_result = add_whisper_config(current_app.config['db'], transcribe_config_name)
    return jsonify(add_result)

@whisper.route('/config/update/basic', methods=['POST'])
@login_required
@csrf_protect_ajax
def api_update_whisper_basic():
    # Access form data
    transcribe_config_name = request.form.get("transcribe_config_name")

    result = update_whisper_config_basic(current_app.config['db'], request.form)

    if result['success']:
        return jsonify({'success': True, 'message': f"Whisper Basic Configuration for {transcribe_config_name} updated successfully"}), 201
    else:
        return jsonify({'success': False, 'message': f"Error Updating Whisper Configuration: {result['message']}"}), 500

@whisper.route('/config/update/decoding', methods=['POST'])
@login_required
@csrf_protect_ajax
def api_update_whisper_decoding():
    # Access form data
    transcribe_config_name = request.form.get("transcribe_config_name")

    result = update_whisper_config_decoding(current_app.config['db'], request.form)

    if result['success']:
        return jsonify({'success': True, 'message': f"Whisper Decoding Configuration for {transcribe_config_name} updated successfully"}), 201
    else:
        return jsonify({'success': False, 'message': f"Error Updating Whisper Decoding Configuration: {result['message']}"}), 500

@whisper.route('/config/update/prompt', methods=['POST'])
@login_required
@csrf_protect_ajax
def api_update_whisper_prompt():
    # Access form data
    transcribe_config_name = request.form.get("transcribe_config_name")

    current_app.config['logger'].debug(request.form)

    result = update_whisper_config_prompt(current_app.config['db'], request.form)

    if result.get('success', False):
        return jsonify({'success': True, 'message': f"Whisper Prompt Configuration for {transcribe_config_name} updated successfully"}), 201
    else:
        return jsonify({'success': False, 'message': f"Error Updating Whisper Prompt Configuration: {result.get('message', '')}"}), 500

@whisper.route('/config/update/advanced', methods=['POST'])
@login_required
@csrf_protect_ajax
def api_update_whisper_advanced():
    # Access form data
    transcribe_config_name = request.form.get("transcribe_config_name")

    result = update_whisper_config_advanced(current_app.config['db'], request.form)

    if result.get('success', True):
        return jsonify({'success': True, 'message': f"Whisper Advanced Settings for {transcribe_config_name} updated successfully"}), 201
    else:
        return jsonify({'success': False,'message': f"Error Updating Whisper Advanced Settings: {result.get('message', '')}"}), 500

@whisper.route('/config/update/vad', methods=['POST'])
@login_required
@csrf_protect_ajax
def api_update_whisper_vad():
    # Access form data
    transcribe_config_name = request.form.get("transcribe_config_name")

    print(request.form)

    result = update_whisper_config_vad(current_app.config['db'], request.form)

    print(result)

    if result.get('success'):
        return jsonify({'success': True,'message': f"Whisper VAD Settings for {transcribe_config_name} updated successfully"}), 201
    else:
        return jsonify({'success': False,'message': f"Error Updating Whisper VAD Settings: {result.get('message', '')}"}), 500

@whisper.route('/config/update/amplify', methods=['POST'])
@login_required
@csrf_protect_ajax
def api_update_whisper_amplify():
    # Access form data
    transcribe_config_name = request.form.get("transcribe_config_name")

    result = update_whisper_config_amplify(current_app.config['db'], request.form)

    if result.get('success'):
        return jsonify({'success': True,'message': f"Whisper Amplify Settings for {transcribe_config_name} updated successfully"}), 201
    else:
        return jsonify({'success': False,'message': f"Error Updating Whisper Amplify Settings: {result.get('message', '')}"}), 500

@whisper.route('/config/update/tone_removal', methods=['POST'])
@login_required
@csrf_protect_ajax
def api_update_whisper_tone_removal():
    # Access form data
    transcribe_config_name = request.form.get("transcribe_config_name")

    result = update_whisper_config_tone_removal(current_app.config['db'], request.form)

    if result.get('success'):
        return jsonify({'success': True,'message': f"Whisper Tone Removal Settings for {transcribe_config_name} updated successfully"}), 201
    else:
        return jsonify({'success': False,'message': f"Error Updating Whisper Tone Removal Settings: {result.get('message', '')}"}), 500

@whisper.route('/config/delete', methods=['POST'])
@login_required
@csrf_protect_ajax
def api_whisper_config_delete():
    transcribe_config_id = request.form.get("transcribe_config_id")

    try:
        transcribe_config_id = int(transcribe_config_id)
    except ValueError:
        return {"success": False, "message": f"Invalid transcribe config ID: {transcribe_config_id}"}

    result = delete_whisper_config(current_app.config['db'], transcribe_config_id)
    return jsonify(result)

@whisper.route('/config/get', methods=['GET'])
@login_or_token_required
def api_whisper_config_get():
    search_params = request.args.to_dict(flat=False)

    # Process parameters using normalize_param
    corrected_params = {
        'transcribe_config_id': normalize_param(search_params, 'transcribe_config_id', int),
        'transcribe_config_name': normalize_param(search_params, 'transcribe_config_name', str)
    }

    # Call get_systems with corrected parameters
    result = get_whisper_config(
        current_app.config['db'],
        transcribe_config_id=corrected_params['transcribe_config_id'],
        transcribe_config_name=corrected_params['transcribe_config_name']
    )

    return jsonify(result)