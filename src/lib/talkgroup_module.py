import logging

module_logger = logging.getLogger('icad_dispatch.talkgroup_module')

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

def delete_talkgroup(db, talkgroup_id):
    """
    Delete a talkgroup by its ID.

    Args:
        db (MySQLDatabase): The database wrapper instance.
        talkgroup_id (int): The ID of the talkgroup to delete.

    Returns:
        dict: A dictionary with success status and affected row count.
    """
    query = "DELETE FROM radio_system_talkgroups WHERE talkgroup_id = %s"
    params = (talkgroup_id,)

    result = db.execute_commit(query, params)
    return result

def add_or_update_system_talkgroup(db, talkgroup_data):
    """
    Adds or updates a record in the radio_system_talkgroups table.

    If talkgroup_data contains an 'talkgroup_id', this function will attempt an UPDATE.
    If 'talkgroup_id' is missing or None, it will INSERT a new record.

    Args:
        db: The database wrapper instance.
        talkgroup_data (dict): A dictionary that may include:
            - talkgroup_id (int, optional)
            - radio_system_id (int, required)
            - talkgroup_decimal (int, optional)
            - talkgroup_description (str, required)
            - talkgroup_alpha_tag (str, required)
            - talkgroup_service_tag (str, required)

    Returns:
        dict: The result from db.execute_commit(), containing:
              {
                'success': bool,
                'message': str,
                'result': last_insert_id or rowcount/[]
              }
    """
    # Extract fields from email_data
    talkgroup_id = talkgroup_data.get('talkgroup_id')
    radio_system_id = talkgroup_data.get('radio_system_id')
    talkgroup_decimal = talkgroup_data.get('talkgroup_decimal')
    talkgroup_description = talkgroup_data.get('talkgroup_description')
    talkgroup_alpha_tag = talkgroup_data.get('talkgroup_alpha_tag')
    talkgroup_service_tag = talkgroup_data.get('talkgroup_service_tag')

    # If talkgroup_id is present, we do an UPDATE; otherwise, an INSERT
    if talkgroup_id is not None:
        # -- UPDATE --
        update_query = """
            UPDATE radio_system_talkgroups
               SET talkgroup_decimal = %s,
                   talkgroup_description = %s,
                   talkgroup_alpha_tag = %s,
                   talkgroup_service_tag = %s
             WHERE talkgroup_id = %s
        """
        params = (talkgroup_decimal, talkgroup_description, talkgroup_alpha_tag, talkgroup_service_tag, talkgroup_id)

        # Use return_count=True to get the affected row count
        result = db.execute_commit(
            query=update_query,
            params=params,
            return_count=True
        )
        return result

    else:

        # Validate required fields
        if radio_system_id is None:
            return {
                'success': False,
                'message': (
                    "Missing required fields. "
                    "Ensure 'radio_system_id' is provided."
                ),
                'result': []
            }

        # -- INSERT --
        insert_query = """
            INSERT INTO radio_system_talkgroups (radio_system_id, talkgroup_decimal, talkgroup_description, talkgroup_alpha_tag, talkgroup_service_tag)
            VALUES (%s, %s, %s, %s, %s)
        """
        params = (radio_system_id, talkgroup_decimal, talkgroup_description, talkgroup_alpha_tag, talkgroup_service_tag)

        # Use return_row=True to get the last inserted ID
        result = db.execute_commit(
            query=insert_query,
            params=params,
            return_row_id=True
        )
        return result

def get_talkgroups(db, radio_system_id, talkgroup_id=None, talkgroup_decimal=None, talkgroup_description=None, talkgroup_alpha_tag=None, talkgroup_service_tag=None, include_config=None):
    """
    Retrieve talkgroups by different filters, requiring system_id.

    Args:
        db (MySQLDatabase): The database wrapper instance.
        radio_system_id (int): The ID of the system to filter by.
        talkgroup_id (int): Optional talkgroup ID to filter by.
        talkgroup_description (str): Optional talkgroup description to filter by.
        talkgroup_alpha_tag (str): Optional talkgroup alpha tag to filter by.
        talkgroup_service_tag (str): Optional talkgroup service tag to filter by.

    Returns:
        dict: A dictionary with success status and talkgroup(s) data.
    """

    if not include_config:
        query = """
            SELECT 
                tg.talkgroup_id, 
                tg.talkgroup_decimal, 
                tg.talkgroup_description, 
                tg.talkgroup_alpha_tag, 
                tg.talkgroup_service_tag
            FROM 
                radio_system_talkgroups tg
        """
    else:
        query = """
           SELECT
               tg.talkgroup_id,
               tg.talkgroup_decimal,
               tg.talkgroup_description,
               tg.talkgroup_alpha_tag,
               tg.talkgroup_service_tag,
            
               -- Transcribe config columns
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
            
               -- VAD config columns
               vc.vad_config_id,
               vc.threshold,
               vc.neg_threshold,
               vc.min_speech_duration_ms,
               vc.min_silence_duration_ms,
               vc.speech_pad_ms
           FROM radio_system_talkgroups AS tg
           LEFT JOIN transcribe_config AS tsc 
                  ON tg.talkgroup_id = tsc.talkgroup_id AND tg.radio_system_id = tsc.radio_system_id
           LEFT JOIN vad_config AS vc
                  ON tsc.transcribe_config_id = vc.transcribe_config_id
        """
    params = []
    if radio_system_id:
        query += ' WHERE tg.radio_system_id = %s'
        params.append(radio_system_id)
    elif talkgroup_id:
        query += " WHERE tg.talkgroup_id = %s"
        params.append(talkgroup_id)
    else:
        return {'success': False, 'message': 'Radio system ID or Talkgroup ID is required.'}

    if talkgroup_decimal:
        query += " AND tg.talkgroup_decimal = %s"
        params.append(talkgroup_decimal)
    if talkgroup_description:
        query += " AND tg.talkgroup_description = %s"
        params.append(talkgroup_description)
    if talkgroup_alpha_tag:
        query += " AND tg.talkgroup_alpha_tag = %s"
        params.append(talkgroup_alpha_tag)
    if talkgroup_service_tag:
        query += " AND tg.talkgroup_service_tag = %s"
        params.append(talkgroup_service_tag)

    talkgroups_result = db.execute_query(query, tuple(params))

    talkgroups = talkgroups_result["result"]
    # If we DO have config in the query, post-process the rows

    for talkgroup in talkgroups:
        # restructure talkgroup dict
        if include_config:

            # Build the transcribe_config sub-dict
            transcribe_config_data = {}
            for field in transcribe_config_fields:
                transcribe_config_data[field] = talkgroup.pop(field, None)

            # Build the vad_parameters sub-dict
            vad_config_data = {}
            for field in vad_config_fields:
                vad_config_data[field] = talkgroup.pop(field, None)

            # Attach vad_parameters to transcribe_config
            transcribe_config_data["vad_parameters"] = vad_config_data

            # Attach transcribe_config to talkgroup
            talkgroup["transcribe_config"] = transcribe_config_data

    return talkgroups_result