import logging
import re

from flask import Blueprint, request, jsonify, flash, redirect, url_for, render_template, current_app, session

from lib.system_module import add_system, get_systems, update_system_general, delete_system, \
    update_system_whisper_basic, update_system_whisper_decoding, update_system_whisper_prompt, \
    update_system_whisper_advanced, update_system_whisper_vad
from lib.utility_module import normalize_param
from routes.decorators import login_required, csrf_protect_ajax

system = Blueprint('system', __name__)
module_logger = logging.getLogger('icad_dispatch.api.system')

@system.route('/add', methods=['POST'])
@login_required
@csrf_protect_ajax
def api_system_add():
    # Access form data
    system_decimal = request.form.get('system_decimal')
    system_name = request.form.get('system_name')
    stream_url = request.form.get('stream_url')  # Optional field

    # Validate required fields
    if not system_decimal or not system_name:
        return jsonify({'success': False, 'message': 'System ID and System Name are required!'}), 400

    # Process the form data (e.g., add to the database)
    result = add_system(current_app.config['db'], request.form)

    if result['success']:
        return jsonify({'success': True, 'message': f'System {system_name} added successfully!'}), 201
    else:
        return jsonify({'success': False, 'message': result['message']}), 500

@system.route('/update/general', methods=['POST'])
@login_required
@csrf_protect_ajax
def api_system_update_general():
    # Access form data
    radio_system_id = request.form.get('radio_system_id')
    system_decimal = request.form.get('system_decimal')
    system_name = request.form.get('system_name')

    # Validate required fields
    if not system_decimal or not system_name or not radio_system_id:
        return jsonify({'success': False, 'message': 'System ID and System Name are required!'}), 400

    # Process the form data (e.g., add to the database)
    result = update_system_general(current_app.config['db'], request.form)

    if result['success']:
        return jsonify({'success': True, 'message': f'System {system_name} added successfully!'}), 201
    else:
        return jsonify({'success': False, 'message': result['message']}), 500

@system.route('/update/whisper/basic', methods=['POST'])
@login_required
@csrf_protect_ajax
def api_system_update_whisper_basic():
    # Access form data
    radio_system_name = request.form.get('system_name')

    current_app.config['logger'].debug(request.form)
    result = update_system_whisper_basic(current_app.config['db'], request.form)

    if result['success']:
        return jsonify({'success': True, 'message': f"Whisper Basic Configuration for {radio_system_name} updated successfully"}), 201
    else:
       return jsonify({'success': False, 'message': f"Error Updating Whisper Configuration: {result['message']}"}), 500

@system.route('/update/whisper/decoding', methods=['POST'])
@login_required
@csrf_protect_ajax
def api_system_update_whisper_decoding():
    # Access form data
    radio_system_name = request.form.get('system_name')

    current_app.config['logger'].debug(request.form)
    result = update_system_whisper_decoding(current_app.config['db'], request.form)

    if result['success']:
        return jsonify({'success': True, 'message': f"Whisper Decoding Configuration for {radio_system_name} updated successfully"}), 201
    else:
        return jsonify({'success': False, 'message': f"Error Updating Whisper Decoding Configuration: {result['message']}"}), 500

@system.route('/update/whisper/prompt', methods=['POST'])
@login_required
@csrf_protect_ajax
def api_system_update_whisper_prompt():
    # Access form data
    radio_system_name = request.form.get('system_name')

    current_app.config['logger'].debug(request.form)

    result = update_system_whisper_prompt(current_app.config['db'], request.form)

    if result.get('success', False):
        return jsonify({'success': True, 'message': f"Whisper Prompt Configuration for {radio_system_name} updated successfully"}), 201
    else:
        return jsonify({'success': False, 'message': f"Error Updating Whisper Prompt Configuration: {result.get('message', '')}"}), 500

@system.route('/update/whisper/advanced', methods=['POST'])
@login_required
@csrf_protect_ajax
def api_system_update_whisper_advanced():
    # Access form data
    radio_system_name = request.form.get('system_name')
    current_app.config['logger'].debug(request.form)

    result = update_system_whisper_advanced(current_app.config['db'], request.form)

    if result.get('success', True):
        return jsonify({'success': True, 'message': f"Whisper Advanced Settings for {radio_system_name} updated successfully"}), 201
    else:
        return jsonify({'success': False,'message': f"Error Updating Whisper Advanced Settings: {result.get('message', '')}"}), 500

@system.route('/update/whisper/vad', methods=['POST'])
@login_required
@csrf_protect_ajax
def api_system_update_whisper_vad():
    # Access form data
    radio_system_name = request.form.get('system_name')
    current_app.config['logger'].debug(request.form)

    result = update_system_whisper_vad(current_app.config['db'], request.form)


    if not result.get('success', True):
        return jsonify({'success': True,'message': f"Whisper VAD Settings for {radio_system_name} updated successfully"}), 201
    else:
        return jsonify({'success': False,'message': f"Error Updating Whisper VAD Settings: {result.get('message', '')}"}), 500


@system.route('/delete', methods=['POST'])
@login_required
@csrf_protect_ajax
def api_system_delete():
    # Access form data
    radio_system_id = request.form.get('radio_system_id')
    radio_system_name = request.form.get('system_name')

    # Validate required fields
    if not radio_system_id:
        return jsonify({'success': False, 'message': 'Radio System ID is required!'}), 400

    try:
        radio_system_id = int(radio_system_id)
    except ValueError:
        return jsonify({'success': False, 'message': 'Radio System ID is invalid!'}), 400

    # Process the form data (e.g., add to the database)
    result = delete_system(current_app.config['db'], radio_system_id)

    if result['success']:
        return jsonify({'success': True, 'message': f'System {radio_system_name} deleted successfully!'}), 201
    else:
        return jsonify({'success': False, 'message': result['message']}), 500

@system.route('/get', methods=['GET'])
@login_required
def api_system_get():
    search_params = request.args.to_dict(flat=False)

    # Process parameters using normalize_param
    corrected_params = {
        'radio_system_id': normalize_param(search_params, 'radio_system_id', int),
        'system_decimal': normalize_param(search_params, 'system_decimal', int),
        'system_name': normalize_param(search_params, 'system_name', str),
        'include_talkgroups': search_params.get('include_talkgroups', ['False'])[0].lower() == 'true',
        'include_config': search_params.get('include_config', ['False'])[0].lower() == 'true'
    }

    # Call get_systems with corrected parameters
    result = get_systems(
        current_app.config['db'],
        radio_system_id=corrected_params['radio_system_id'],
        system_decimal=corrected_params['system_decimal'],
        system_name=corrected_params['system_name'],
        include_talkgroups=corrected_params['include_talkgroups'],
        include_config=corrected_params['include_config']
    )

    return jsonify(result)
