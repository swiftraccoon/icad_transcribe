import logging

from flask import Blueprint, request, jsonify, flash, redirect, url_for, render_template, current_app, session


base_site = Blueprint('base_site', __name__)
module_logger = logging.getLogger('icad_transcribe.base_site')

@base_site.route('/')
def base_site_index():
    return render_template('index.html')