import json
import logging

module_logger = logging.getLogger('icad_transcribe.audio_metadata')


def get_audio_metadata(uploaded_data):
    call_data = {}
    call_data["systemLabel"]     = uploaded_data.get("systemLabel")
    call_data["talkgroup"]       = uploaded_data.get("talkgroup")
    call_data["talkgroupGroup"]  = uploaded_data.get("talkgroupGroup")
    call_data["talkgroupLabel"]  = uploaded_data.get("talkgroupLabel")
    call_data["talkgroupTag"]    = uploaded_data.get("talkgroupTag")

    for list_field in ("frequencies", "patches", "sources"):
        raw_value = uploaded_data.get(list_field)
        if raw_value:
            try:
                call_data[list_field] = json.loads(raw_value)
            except json.JSONDecodeError:
                call_data[list_field] = []
        else:
            call_data[list_field] = []

    return call_data