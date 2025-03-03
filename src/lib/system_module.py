import json
import logging

from lib.talkgroup_module import get_talkgroups
from lib.utility_module import parse_nullable_float, parse_nullable_string, parse_float, parse_bool_as_int, \
    parse_csv_of_floats, parse_nullable_int, parse_int, parse_csv_of_integers, \
    parse_nullable_csv_of_floats

module_logger = logging.getLogger('icad_dispatch.system_module')

transcribe_config_fields = [
    "transcribe_config_id",
    "language",
    "task",
    "log_progress",
    "condition_on_previous_text",
    "suppress_blank",
    "word_timestamps",
    "without_timestamps",
    "multilingual",
    "vad_filter",
    "batch_size",
    "beam_size",
    "best_of",
    "no_repeat_ngram_size",
    "language_detection_segments",
    "max_new_tokens",
    "chunk_length",
    "patience",
    "length_penalty",
    "repetition_penalty",
    "prompt_reset_on_temperature",
    "max_initial_timestamp",
    "hallucination_silence_threshold",
    "compression_ratio_threshold",
    "log_prob_threshold",
    "no_speech_threshold",
    "language_detection_threshold",
    "temperature",
    "suppress_tokens",
    "clip_timestamps",
    "prefix",
    "initial_prompt",
    "prepend_punctuations",
    "append_punctuations",
    "hotwords",
]

vad_config_fields = [
    "vad_config_id",
    "threshold",
    "neg_threshold",
    "min_speech_duration_ms",
    "min_silence_duration_ms",
    "speech_pad_ms",
]

def check_for_system(db, system_name=None, system_decimal=None, radio_system_id=None):
    if not radio_system_id:
        query = f"SELECT * FROM radio_systems rs WHERE rs.system_name = %s"
        params = (system_name,)
    elif system_decimal:
        query = f"SELECT * FROM radio_systems rs WHERE rs.system_decimal = %s"
        params = (system_decimal,)
    else:
        query = f"SELECT * FROM radio_systems rs WHERE rs.radio_system_id = %s"
        params = (radio_system_id,)

    result = db.execute_query(query, params, fetch_mode='one')
    return result.get("result", {})

def add_system(db, system_data):
    """
    Add a new system to the database.

    Args:
        db (MySQLDatabase): The database wrapper instance.
        system_data (dict): A dictionary containing the new systems information. system_decimal, system_name, stream_url

    Returns:
        dict: A dictionary with success status and the inserted system ID.
    """
    if not system_data.get('system_decimal'):
        return {'success': False, 'message': 'System decimal field is required'}

    if not system_data.get('system_name'):
        return {'success': False, 'message': 'System name field is required'}

    decimal_result = check_for_system(db, system_decimal=system_data['system_decimal'])
    if decimal_result:
        return {'success': False, 'message': 'A system with that system decimal id already exists'}

    name_result = check_for_system(db, system_name=system_data['system_name'])
    if name_result:
        return {'success': False, 'message': 'A system with that system name already exists'}

    query = """
        INSERT INTO radio_systems (system_decimal, system_name, stream_url)
        VALUES (%s, %s, %s)
    """
    params = (system_data.get('system_decimal'), system_data.get('system_name'), system_data.get('stream_url'))
    result = db.execute_commit(query, params, return_row_id=True)

    if result.get('success') and result.get('result'):
        add_default_result = add_system_default_settings(db, result.get('result'))
        if add_default_result.get("success"):
            result["message"] = f"New system {system_data.get('system_name')} added."
        else:
            # Roll back if default config was not inserted
            delete_system(db, system_decimal=system_data.get('system_decimal'))
            result["message"] = add_default_result.get("message")

    return result

def add_system_default_settings(db, radio_system_id):
    try:

        # Insert default Tone Settings
        transcribe_config_result = db.execute_commit(
            "INSERT INTO transcribe_config (radio_system_id) VALUES (%s)",
            (radio_system_id,), return_row_id=True
        )

        if not transcribe_config_result.get('success'):
            module_logger.error(f'Error adding default transcribe configuration: {transcribe_config_result.get("message")}')
            return {"success": False, "message": f'Error adding default transcribe configuration: {transcribe_config_result.get("message")}', 'result':None}

        # Insert default email settings
        vad_result = db.execute_commit(
            "INSERT INTO vad_config (transcribe_config_id) VALUES (%s)",
            (transcribe_config_result.get("result"),)
        )


        module_logger.info(f"Default settings inserted successfully for radio_system_id: {radio_system_id}")
        return {"success": True, "message": f"Default settings inserted successfully"}
    except Exception as e:
        module_logger.error(f"Failed to insert default settings for system_id: {radio_system_id}. Error: {e}")
        return {"success": False, "messsage": f"Unexpected error inserting default settings: {e}", "result":None}

def update_system_general(db, system_data):
    """
    Update an existing system's details, including its tokens if the decimal changes.

    Args:
        db (MySQLDatabase): The database wrapper instance.
        system_data (dict): A dictionary containing the system information. radio_system_id, system_decimal, system_name, stream_url

    Returns:
        dict: A dictionary with success status, affected row count, and any error messages.
    """
    try:
        radio_system_id = system_data.get('radio_system_id')
        if radio_system_id:
            radio_system_id = int(radio_system_id)
        system_decimal = system_data.get('system_decimal')
        if system_decimal:
            system_decimal = int(system_decimal)
        system_name = system_data.get('system_name')
        stream_url = system_data.get('stream_url')
    except ValueError:
        return {'success': False, 'message': 'One or more fields as not of the right data type.'}

    if not radio_system_id:
        return {'success': False, 'message': 'radio_system_id field is required'}

    # Fetch the current system to check the old decimal
    current_system_query = """
        SELECT system_decimal
        FROM radio_systems
        WHERE radio_system_id = %s
    """
    current_system_result = db.execute_query(current_system_query, (radio_system_id,), fetch_mode="one")

    if not current_system_result['success'] or not current_system_result['result']:
        module_logger.error(f"Failed to update system {system_name} with ID {radio_system_id}: {current_system_result['message']}")
        return {'success': False, 'message': 'System not found or query failed'}

    old_decimal = current_system_result['result']['system_decimal']

    # Prepare fields to update
    fields = []
    params = []

    if system_decimal is not None and system_decimal != old_decimal:
        fields.append('system_decimal = %s')
        params.append(system_decimal)
    if system_name:
        fields.append("system_name = %s")
        params.append(system_name)
    if stream_url:
        fields.append("stream_url = %s")
        params.append(stream_url)

    if not fields:
        module_logger.error(f"Failed to update system {system_name} with ID {radio_system_id}: No fields provided")
        return {'success': False, 'message': 'No fields to update'}

    # Update the system
    update_system_query = f"""
        UPDATE radio_systems
        SET {', '.join(fields)}
        WHERE radio_system_id = %s
    """
    params.append(radio_system_id)
    update_result = db.execute_commit(update_system_query, params)

    if not update_result['success']:
        module_logger.error(f"Failed to update system {system_name}: {update_result['message']}")
        return {'success': False, 'message': 'Failed to update system'}

    return {
        'success': True,
        'message': 'System updated successfully',
        'result': []
    }

def update_system_whisper_basic(db, whisper_data):
    """
    Update a system's Basic Whisper Configuration.

    Args:
        db (SQLiteDatabase): The database wrapper instance.
        whisper_data (dict): A dictionary containing the system information. radio_system_id, language, task, log_progress

    Returns:
        dict: A dictionary with success status, affected row count, and any error messages.
    """
    try:
        # Validate Inputs
        radio_system_id = whisper_data.get('radio_system_id')
        if radio_system_id:
            radio_system_id = int(radio_system_id)
        else:
            raise ValueError

        language = whisper_data.get('language')
        if language is not None:
            if len(language) > 2:
                raise ValueError
        task = whisper_data.get('task')
        if task is None:
            raise ValueError
        if task not in ['transcribe', "translate"]:
            raise ValueError

        log_progress = whisper_data.get('log_progress')
        if log_progress is None:
            raise ValueError
        log_progress = int(log_progress)
    except ValueError as ve:
        return {'success': False, 'message': 'One or more fields as not of the right data type.'}

    update_query = f"""
    UPDATE 
      transcribe_config
    SET 
      language = %s, 
      task = %s, 
      log_progress= %s
    WHERE
      radio_system_id = %s
    AND
      talkgroup_id IS NULL
    """
    params = (language, task, log_progress, radio_system_id)

    update_result = db.execute_commit(update_query, params)

    return update_result

def update_system_whisper_decoding(db, whisper_data):
    """
    Update a system's Decoding Whisper Configuration.

    Args:
        db (SQLiteDatabase): The database wrapper instance.
        whisper_data (dict): A dictionary containing the system information:
          - radio_system_id (int)
          - beam_size (int)
          - best_of (int)
          - patience (float)
          - length_penalty (float)
          - repetition_penalty (float)
          - no_repeat_ngram_size (int)
          - temperature (list of floats, provided as comma‐separated string)
          - compression_ratio_threshold (float)
          - log_prob_threshold (float)  # user’s form key: systemWhisperDecodingLogProbThreshold
          - no_speech_threshold (float)
          - max_new_tokens (int) - nullable
          - chunk_length (int) - nullable

    Returns:
        dict: { 'success': bool, 'message': str, ... }
    """
    try:
        # 1) radio_system_id must be present & valid
        radio_system_id = parse_int(
            whisper_data.get('radio_system_id'),
            'radio_system_id'
        )

        # 2) Non‐nullable int fields
        beam_size = parse_int(
            whisper_data.get('beam_size'),
            'beam_size'
        )
        best_of = parse_int(
            whisper_data.get('best_of'),
            'best_of'
        )
        no_repeat_ngram_size = parse_int(
            whisper_data.get('no_repeat_ngram_size'),
            'no_repeat_ngram_size'
        )

        # 3) Non‐nullable float fields
        patience = parse_float(
            whisper_data.get('patience'),
            'patience'
        )
        length_penalty = parse_float(
            whisper_data.get('length_penalty'),
            'length_penalty'
        )
        repetition_penalty = parse_float(
            whisper_data.get('repetition_penalty'),
            'repetition_penalty'
        )

        # The user’s snippet calls this "log_prob_threshold"
        # instead of "log_probability_threshold":
        log_prob_threshold = parse_float(
            whisper_data.get('log_prob_threshold'),
            'log_prob_threshold'
        )

        compression_ratio_threshold = parse_float(
            whisper_data.get('compression_ratio_threshold'),
            'compression_ratio_threshold'
        )
        no_speech_threshold = parse_float(
            whisper_data.get('no_speech_threshold'),
            'no_speech_threshold'
        )

        # 4) Temperature is a required comma‐separated list of floats
        temperature = parse_csv_of_floats(
            whisper_data.get('temperature'),
            'temperature'
        )

        # 5) Nullable int fields
        max_new_tokens = parse_nullable_int(
            whisper_data.get('max_new_tokens'),
            'max_new_tokens'
        )
        chunk_length = parse_nullable_int(
            whisper_data.get('chunk_length'),
            'chunk_length'
        )

    except ValueError as ve:
        # If any parse or validation fails, we catch it here
        return {
            'success': False,
            'message': f'Error getting Whisper Decoding data: {ve}'
        }

    # 6) Assuming these columns exist in your DB schema:
    update_query = """
        UPDATE transcribe_config
           SET beam_size = %s,
               best_of = %s,
               patience = %s,
               length_penalty = %s,
               repetition_penalty = %s,
               no_repeat_ngram_size = %s,
               temperature = %s,
               compression_ratio_threshold = %s,
               log_probability_threshold = %s,
               no_speech_threshold = %s,
               max_new_tokens = %s,
               chunk_length = %s
         WHERE radio_system_id = %s
           AND talkgroup_id IS NULL
    """

    temperature_str = json.dumps(temperature)

    params = (
        beam_size,
        best_of,
        patience,
        length_penalty,
        repetition_penalty,
        no_repeat_ngram_size,
        temperature_str,  # store as JSON
        compression_ratio_threshold,
        log_prob_threshold,
        no_speech_threshold,
        max_new_tokens,
        chunk_length,
        radio_system_id
    )

    update_result = db.execute_commit(update_query, params)
    return update_result

def update_system_whisper_prompt(db, whisper_data):
    """
    Update a system's Prompt Whisper Configuration.

    Args:
        db (SQLiteDatabase): The database wrapper instance.
        whisper_data (dict): A dictionary containing the form data, e.g.:
            - radio_system_id (required, int)
            - condition_on_previous_text (required, bool -> stored as int(1 or 0))
            - prompt_reset_on_temperature (required float)
            - initial_prompt (nullable string)
            - prefix (nullable string)
            - prepend_punctuations (nullable string)
            - append_punctuations (nullable string)
            - suppress_blank (required bool -> stored as int(1 or 0))
            - suppress_tokens (required, comma‐separated ints -> stored as JSON in DB)
            - without_timestamps (required bool -> stored as int(1 or 0))
            - word_timestamps (required bool -> stored as int(1 or 0))
            - max_initial_timestamp (required float)
            - hallucination_silence_threshold (nullable float)
            - hotwords (nullable string)

    Returns:
        dict: A dictionary with { 'success': bool, 'message': str, ... }
              or the DB result on success.
    """
    try:
        # radio_system_id must be present & valid integer
        radio_system_id = parse_int(
            whisper_data.get('radio_system_id'),
            'radio_system_id'
        )

        condition_on_previous_text = parse_bool_as_int(
            whisper_data.get('condition_on_previous_text'),
            'condition_on_previous_text'
        )

        prompt_reset_on_temperature = parse_float(
            whisper_data.get('prompt_reset_on_temperature'),
            'prompt_reset_on_temperature'
        )

        initial_prompt = parse_nullable_string(
            whisper_data.get('initial_prompt'),
            'initial_prompt'
        )

        prefix = parse_nullable_string(
            whisper_data.get('prefix'),
            'prefix'
        )

        prepend_punctuations = parse_nullable_string(
            whisper_data.get('prepend_punctuations'),
            'prepend_punctuations'
        )

        append_punctuations = parse_nullable_string(
            whisper_data.get('append_punctuations'),
            'append_punctuations'
        )

        suppress_blank = parse_bool_as_int(
            whisper_data.get('suppress_blank'),
            'suppress_blank'
        )

        # 1) Parse comma‐separated tokens -> list of ints
        suppress_tokens_list = parse_csv_of_integers(
            whisper_data.get('suppress_tokens'),
            'suppress_tokens'
        )

        # 2) Convert that list to a JSON string for storage
        suppress_tokens_json = json.dumps(suppress_tokens_list)

        without_timestamps = parse_bool_as_int(
            whisper_data.get('without_timestamps'),
            'without_timestamps'
        )

        word_timestamps = parse_bool_as_int(
            whisper_data.get('word_timestamps'),
            'word_timestamps'
        )

        max_initial_timestamp = parse_float(
            whisper_data.get('max_initial_timestamp'),
            'max_initial_timestamp'
        )

        hallucination_silence_threshold = parse_nullable_float(
            whisper_data.get('hallucination_silence_threshold'),
            'hallucination_silence_threshold'
        )

        hotwords = parse_nullable_string(
            whisper_data.get('hotwords'),
            'hotwords'
        )

    except ValueError as ve:
        # If any parse or validation fails, we get here
        return {
            'success': False,
            'message': f'Error getting Whisper Prompt data: {ve}'
        }

    # Build your SQL update query
    update_query = """
        UPDATE transcribe_config
           SET condition_on_previous_text = %s,
               prompt_reset_on_temperature = %s,
               initial_prompt = %s,
               prefix = %s,
               prepend_punctuations = %s,
               append_punctuations = %s,
               suppress_blank = %s,
               suppress_tokens = %s,
               without_timestamps = %s,
               word_timestamps = %s,
               max_initial_timestamp = %s,
               hallucination_silence_threshold = %s,
               hotwords = %s
         WHERE radio_system_id = %s
           AND talkgroup_id IS NULL
    """

    # The parameters must match your placeholders in the UPDATE.
    params = (
        condition_on_previous_text,
        prompt_reset_on_temperature,
        initial_prompt,
        prefix,
        prepend_punctuations,
        append_punctuations,
        suppress_blank,
        suppress_tokens_json,
        without_timestamps,
        word_timestamps,
        max_initial_timestamp,
        hallucination_silence_threshold,
        hotwords,
        radio_system_id
    )

    update_result = db.execute_commit(update_query, params)

    return update_result

def update_system_whisper_advanced(db, whisper_data):
    """
    Update a system's 'Advanced' Whisper Configuration.

    Args:
        db (SQLiteDatabase): Database wrapper instance.
        whisper_data (dict): Form data, which may include:
            - radio_system_id (int, required)
            - clip_timestamps (nullable CSV of floats)
            - multilingual (bool -> stored as int(1 or 0))
            - language_detection_threshold (float, required)
            - language_detection_segments (int, required)
    Returns:
        dict: { 'success': bool, 'message': str } or DB result on success.
    """
    try:
        # 1) radio_system_id must be present & valid integer
        radio_system_id = parse_int(
            whisper_data.get('radio_system_id'),
            'radio_system_id'
        )

        # 2) Parse clip_timestamps as a *nullable* CSV of floats
        #    e.g. "0.0,0.1,0.2" => [0.0, 0.1, 0.2]
        clip_timestamps_list = parse_nullable_csv_of_floats(
            whisper_data.get('clip_timestamps'),
            'clip_timestamps'
        )
        # If not None, store it as JSON
        clip_timestamps_json = None
        if clip_timestamps_list is not None:
            clip_timestamps_json = json.dumps(clip_timestamps_list)

        # 3) multilingual as a bool -> stored as 1/0 in DB
        multilingual = parse_bool_as_int(
            whisper_data.get('multilingual'),
            'multilingual'
        )

        # 4) language_detection_threshold (required float)
        language_detection_threshold = parse_float(
            whisper_data.get('language_detection_threshold'),
            'language_detection_threshold'
        )

        # 5) language_detection_segments (required int)
        language_detection_segments = parse_int(
            whisper_data.get('language_detection_segments'),
            'language_detection_segments'
        )

    except ValueError as ve:
        return {
            'success': False,
            'message': f'Error getting Whisper Advanced data: {ve}'
        }

    # Build your SQL update query
    update_query = """
        UPDATE transcribe_config
           SET clip_timestamps = %s,
               multilingual = %s,
               language_detection_threshold = %s,
               language_detection_segments = %s
         WHERE radio_system_id = %s
           AND talkgroup_id IS NULL
    """

    # Prepare parameters; note the JSON or None for clip_timestamps
    params = (
        clip_timestamps_json,
        multilingual,
        language_detection_threshold,
        language_detection_segments,
        radio_system_id
    )

    update_result = db.execute_commit(update_query, params)

    return update_result

def update_system_whisper_vad(db, whisper_data):
    """
     Update a system's VAD (Voice Activity Detection) Whisper Configuration.

     Args:
         db (SQLiteDatabase): The database wrapper instance.
         whisper_data (dict): Form data that may include:
             - radio_system_id (required int)
             - talkgroup_config_id (required int)
             - vad_enabled (bool -> stored as 1 or 0)
             - threshold (float, required)
             - neg_threshold (float, nullable)
             - min_speech_duration_ms (int, required)
             - min_silence_duration_ms (int, required)
             - speech_pad_ms (int, required)

     Returns:
         dict or similar: A result object indicating success/failure and possibly rows affected.
     """
    try:
        # --- Parse/Validate Inputs ---
        radio_system_id = parse_int(
            whisper_data.get("radio_system_id"),
            "radio_system_id"
        )

        talkgroup_config_id = parse_int(
            whisper_data.get("talkgroup_config_id"),
            "talkgroup_config_id"
        )

        vad_enabled = parse_bool_as_int(
            whisper_data.get("vad_enabled"),
            "vad_enabled"
        )

        threshold = parse_float(
            whisper_data.get("threshold"),
            "threshold"
        )

        neg_threshold = parse_nullable_float(
            whisper_data.get("neg_threshold"),
            "neg_threshold"
        )

        min_speech_duration_ms = parse_int(
            whisper_data.get("min_speech_duration_ms"),
            "min_speech_duration_ms"
        )

        min_silence_duration_ms = parse_int(
            whisper_data.get("min_silence_duration_ms"),
            "min_silence_duration_ms"
        )

        speech_pad_ms = parse_int(
            whisper_data.get("speech_pad_ms"),
            "speech_pad_ms"
        )

    except ValueError as ve:
        return {
            "success": False,
            "message": f"Error getting VAD data: {ve}"
        }

    # ----------------------------------------------------
    # 1) Update the vad_config table
    # ----------------------------------------------------
    update_query_vad = """
        UPDATE vad_config
           SET threshold              = ?,
               neg_threshold          = ?,
               min_speech_duration_ms = ?,
               min_silence_duration_ms= ?,
               speech_pad_ms         = ?
         WHERE talkgroup_config_id    = ?
    """
    params_vad = (
        threshold,
        neg_threshold,
        min_speech_duration_ms,
        min_silence_duration_ms,
        speech_pad_ms,
        talkgroup_config_id
    )

    vad_result = db.execute_commit(
        update_query_vad,
        params_vad
    )

    # If the first update fails, return the error
    if not vad_result["success"]:
        return {
            "success": False,
            "message": f"Failed to update vad_config: {vad_result['message']}",
            "result": vad_result["result"]
        }

    # ----------------------------------------------------
    # 2) Update the transcribe_config table (vad_enabled)
    # ----------------------------------------------------
    update_query_transcribe = """
        UPDATE transcribe_config
           SET vad_enabled = ?
         WHERE radio_system_id = ?
           AND talkgroup_id    IS NULL
    """
    params_transcribe = (vad_enabled, radio_system_id)

    transcribe_result = db.execute_commit(
        update_query_transcribe,
        params_transcribe
    )

    if not transcribe_result["success"]:
        return {
            "success": False,
            "message": f"Failed to update transcribe_config (vad_enabled): {transcribe_result['message']}",
            "result": transcribe_result["result"]
        }

    # If both updates succeed, return a combined success
    return {
        "success": True,
        "message": "VAD configuration updated successfully.",
        "result": {
            "vad_config_update": vad_result["result"],
            "transcribe_config_update": transcribe_result["result"]
        }
    }

def delete_system(db, radio_system_id=None, system_decimal=None):
    """
    Delete a system by its ID.

    Args:
        db (MySQLDatabase): The database wrapper instance.
        radio_system_id (int): The ID of the system to delete.

    Returns:
        dict: A dictionary with success status and affected row count.
    """

    if not radio_system_id and not system_decimal:
        return {"success": False, "message": "No radio system ID or system decimal ID provided", "result": None}


    if radio_system_id:
        query = "DELETE FROM radio_systems WHERE radio_system_id = %s"
        params = (radio_system_id,)
    else:
        query = "DELETE FROM radio_systems WHERE system_decimal = %s"
        params = (system_decimal,)

    result = db.execute_commit(query, params)
    return result

def get_systems(db, radio_system_id=None, system_decimal=None, system_name=None, include_talkgroups=False, include_config=False):
    """
    Retrieve systems by different filters.

    Args:
        db (MySQLDatabase): The database wrapper instance.
        radio_system_id (int or list): The ID(s) of the system(s) to retrieve.
        system_decimal (int or list): Optional system decimal ID(s) to filter by.
        system_name (str): Optional system name to filter by.
        include_talkgroups (bool): Whether to include talkgroups under each system.

    Returns:
        dict: A dictionary with success status and system(s) data, optionally including talkgroups.
    """

    if not include_config:
        query = """
            SELECT 
              rs.radio_system_id, 
              rs.system_decimal, 
              rs.system_name, 
              rs.stream_url
            FROM radio_systems AS rs
        """
    else:

        # Get config if requested
        query = """
        SELECT
           rs.radio_system_id,
           rs.system_decimal,
           rs.system_name,
           rs.stream_url,
           tsc.transcribe_config_id,
           tsc.language,
           tsc.task,
           tsc.log_progress,
           tsc.condition_on_previous_text,
           tsc.suppress_blank,
           tsc.word_timestamps,
           tsc.without_timestamps,
           tsc.multilingual,
           tsc.vad_filter,
           tsc.batch_size,
           tsc.beam_size,
           tsc.best_of,
           tsc.no_repeat_ngram_size,
           tsc.language_detection_segments,
           tsc.max_new_tokens,
           tsc.chunk_length,
           tsc.patience,
           tsc.length_penalty,
           tsc.repetition_penalty,
           tsc.prompt_reset_on_temperature,
           tsc.max_initial_timestamp,
           tsc.hallucination_silence_threshold,
           tsc.compression_ratio_threshold,
           tsc.log_prob_threshold,
           tsc.no_speech_threshold,
           tsc.language_detection_threshold,
           tsc.temperature,
           tsc.suppress_tokens,
           tsc.clip_timestamps,
           tsc.prefix,
           tsc.initial_prompt,
           tsc.prepend_punctuations,
           tsc.append_punctuations,
           tsc.hotwords,
       
           -- VAD config columns (only if present)
           vc.vad_config_id,
           vc.threshold,
           vc.neg_threshold,
           vc.min_speech_duration_ms,
           vc.min_silence_duration_ms,
           vc.speech_pad_ms
       
       FROM radio_systems AS rs
       LEFT JOIN transcribe_config AS tsc
              ON rs.radio_system_id = tsc.radio_system_id
             AND tsc.talkgroup_id IS NULL
       LEFT JOIN vad_config AS vc
              ON tsc.transcribe_config_id = vc.transcribe_config_id
        """

    conditions = []
    params = []

    # Handle radio_system_id as a single value or a list
    if radio_system_id is not None:
        if isinstance(radio_system_id, list):
            placeholders = ', '.join(['%s'] * len(radio_system_id))
            conditions.append(f"rs.radio_system_id IN ({placeholders})")
            params.extend(radio_system_id)
        else:
            conditions.append("rs.radio_system_id = %s")
            params.append(radio_system_id)

    # Handle system_decimal as a single value or a list
    if system_decimal is not None:
        if isinstance(system_decimal, list):
            placeholders = ', '.join(['%s'] * len(system_decimal))
            conditions.append(f"rs.system_decimal IN ({placeholders})")
            params.extend(system_decimal)
        else:
            conditions.append("rs.system_decimal = %s")
            params.append(system_decimal)

    # Handle system_name as a single value or a list
    if system_name is not None:
        if isinstance(system_name, list):
            placeholders = ', '.join(['%s'] * len(system_name))
            conditions.append(f"rs.system_name IN ({placeholders})")
            params.extend(system_name)
        else:
            conditions.append("rs.system_name = %s")
            params.append(system_name)


    # Construct WHERE clause
    where_clause = f" WHERE {' AND '.join(conditions)} GROUP BY rs.radio_system_id ORDER BY rs.system_name;" if conditions else " WHERE 1=1 GROUP BY rs.radio_system_id ORDER BY rs.system_name;"
    query = query + where_clause

    module_logger.info(f'Requested Query: {query}')
    module_logger.info(f'Requested Params: {params}')

    # Execute the query
    systems_result = db.execute_query(query, tuple(params))

    if not systems_result.get("success"):
        module_logger.error(f"Failed to retrieve systems: {systems_result['message']}")
        return {
            'success': False,
            'message': f"Failed to retrieve systems: {systems_result['message']}",
            'result': []
        }

    systems = systems_result["result"]

    module_logger.debug(f'Systems Result: {systems}')

    for system in systems:
        if include_talkgroups:
            talkgroups_result = get_talkgroups(db, radio_system_id=system["radio_system_id"], include_config=include_config)
            if talkgroups_result.get("success"):
                system["talkgroups"] = talkgroups_result["result"]
            else:
                system["talkgroups"] = []

        # restructure system dict
        if include_config:
            # Build the transcribe_config sub-dict
            transcribe_config_data = {}
            for field in transcribe_config_fields:

                raw_value = system.pop(field, None)

                if field in ["temperature", "suppress_tokens", "clip_timestamps"]:
                    if raw_value is not None:
                        try:
                            value = json.loads(raw_value)
                        except (ValueError, TypeError):
                            value = raw_value
                    else:
                        value = raw_value
                else:
                    value = raw_value

                transcribe_config_data[field] = value

            # Build the vad_parameters sub-dict
            vad_config_data = {}
            for field in vad_config_fields:
                vad_config_data[field] = system.pop(field, None)

            # Attach vad_parameters to transcribe_config
            transcribe_config_data["vad_parameters"] = vad_config_data

            # Attach transcribe_config to system
            system["transcribe_config"] = transcribe_config_data

    return {
        'success': True,
        'message': 'Systems retrieved successfully.',
        'result': systems
    }
