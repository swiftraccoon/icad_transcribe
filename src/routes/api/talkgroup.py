import logging

from flask import Blueprint, request, jsonify, flash, redirect, url_for, render_template, current_app, session

from lib.talkgroup_module import add_or_update_system_talkgroup, delete_talkgroup, get_talkgroups
from lib.utility_module import normalize_param
from routes.decorators import login_required, csrf_protect_ajax

talkgroup = Blueprint('talkgroup', __name__)
module_logger = logging.getLogger('icad_transcribe.api.talkgroup')

@talkgroup.route('/add', methods=['POST'])
@login_required
@csrf_protect_ajax
def api_talkgroup_add():
    # Access form data
    radio_system_id = request.form.get('radio_system_id')
    system_name = request.form.get('system_name')
    talkgroup_decimal = request.form.get('talkgroup_decimal')
    talkgroup_description = request.form.get('talkgroup_description')

    # Validate required fields
    if not radio_system_id or not talkgroup_decimal or not talkgroup_description:
        return jsonify({'success': False, 'message': 'System ID and Talkgroup Description and Talkgroup Decimal are required!'}), 400

    # Process the form data (e.g., add to the database)
    result = add_or_update_system_talkgroup(current_app.config['db'], request.form)

    if result['success']:
        return jsonify({'success': True, 'message': f'Talkgroup {talkgroup_description} added to {system_name} successfully!'}), 201
    else:
        return jsonify({'success': False, 'message': result['message']}), 500

@talkgroup.route('/update', methods=['POST'])
@login_required
@csrf_protect_ajax
def api_talkgroup_update_general():
    # Access form data
    talkgroup_id = request.form.get('talkgroup_id')
    talkgroup_decimal = request.form.get('talkgroup_decimal')
    talkgroup_description = request.form.get('talkgroup_description')

    # Validate required fields
    if not talkgroup_id or not talkgroup_decimal or not talkgroup_description:
        return jsonify({'success': False, 'message': 'System ID, Talkgroup Description and Talkgroup Decimal are required!'}), 400

    try:
        talkgroup_id = int(talkgroup_id)
    except ValueError:
        return jsonify({'success': False, 'message': 'Talkgroup ID is invalid!'}), 400

    # Process the form data (e.g., add to the database)
    result = add_or_update_system_talkgroup(current_app.config['db'], request.form)

    if result['success']:
        return jsonify({'success': True, 'message': f'Talkgroup {talkgroup_description} updated successfully!'}), 201
    else:
        return jsonify({'success': False, 'message': result['message']}), 500

@talkgroup.route('/delete', methods=['POST'])
@login_required
@csrf_protect_ajax
def api_talkgroup_delete():
    # Access form data
    talkgroup_id = request.form.get('talkgroup_id')
    talkgroup_description = request.form.get('talkgroup_description')

    # Validate required fields
    if not talkgroup_id:
        return jsonify({'success': False, 'message': 'Talkgroup ID is required!'}), 400

    try:
        talkgroup_id = int(talkgroup_id)
    except ValueError:
        return jsonify({'success': False, 'message': 'Radio System ID is invalid!'}), 400

    # Process the form data (e.g., add to the database)
    result = delete_talkgroup(current_app.config['db'], talkgroup_id)

    if result['success']:
        return jsonify({'success': True, 'message': f'Talkgroup {talkgroup_description} deleted successfully!'}), 201
    else:
        return jsonify({'success': False, 'message': result['message']}), 500

@talkgroup.route('/get', methods=['GET'])
@login_required
def api_talkgroup_get():
    search_params = request.args.to_dict(flat=False)

    # Process parameters using normalize_param
    corrected_params = {
        'radio_system_id': normalize_param(search_params, 'radio_system_id', int),
        'talkgroup_id': normalize_param(search_params, 'talkgroup_id', int),
        'talkgroup_decimal': normalize_param(search_params, 'talkgroup_decimal', int),
        'talkgroup_description': normalize_param(search_params, 'talkgroup_description', str),
        'talkgroup_alpha_tag': normalize_param(search_params, 'talkgroup_alpha_tag', str),
        'talkgroup_service_tag': normalize_param(search_params, 'talkgroup_service_tag', str),
        'include_config': search_params.get('include_config', ['False'])[0].lower() == 'true'
    }

    # Call get_systems with corrected parameters
    result = get_talkgroups(
        current_app.config['db'],
        radio_system_id=corrected_params['radio_system_id'],
        talkgroup_id=corrected_params['talkgroup_id'],
        talkgroup_decimal=corrected_params['talkgroup_decimal'],
        talkgroup_description=corrected_params['talkgroup_description'],
        talkgroup_alpha_tag=corrected_params['talkgroup_alpha_tag'],
        talkgroup_service_tag=corrected_params['talkgroup_service_tag'],
        include_config=corrected_params['include_config']
    )

    return jsonify(result)