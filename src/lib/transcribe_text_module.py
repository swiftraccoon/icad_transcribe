import logging

module_logger = logging.getLogger("icad_transcribe.transcribe_text")



def process_transcribe_text(segments, transcribe_configuration, tone_removal_configuration, sources_list, tone_list):
    segment_texts = []
    segments_data = []
    segment_count = 0
    for segment in segments:
        segment_count += 1
        segment_texts.append(segment.text.strip())
        text = []
        word_id = 0

        if transcribe_configuration.get("word_timestamps", False):
            for word in segment.words:
                word_id += 1
                text.append({'word_id': word_id, 'word': word.word, 'start': word.start, 'end': word.end})
        else:
            text = []

        segments_data.append(
            {"segment_id": segment_count, "text": segment.text.strip(), "words": text, "unit_tag": "",
             "start": segment.start, "end": segment.end}
        )

    module_logger.debug(segments_data)

    if tone_removal_configuration.get('tone_removal'):
        segments_data = _inject_alert_tone_segments(segments_data, transcribe_configuration, tone_list)

    if sources_list:
        segments_data = _associate_segments_with_unit(segments_data, sources_list)

    if segments_data:
        transcribe_text = " ".join(segment['text'] for segment in segments_data if segment['text'])
    else:
        transcribe_text = "No Transcript"

    # Ensure transcribe_text is always a string
    if not transcribe_text.strip():
        transcribe_text = "No Transcript"

    return segments_data, transcribe_text

def _associate_segments_with_unit(segments, src_list):
    if not segments or not src_list:
        return segments

    # Sort src_list by pos ascending
    src_list_sorted = sorted(src_list, key=lambda x: x['pos'])

    for segment in segments:
        start_time = segment['start']
        # Filter only those with pos <= segment start
        valid_srcs = [s for s in src_list_sorted if s['pos'] <= start_time]

        if valid_srcs:
            # If we have one or more valid sources, pick the one with the largest pos
            chosen_src = valid_srcs[-1]
        else:
            # If nothing is <= segment start, pick the earliest source
            chosen_src = src_list_sorted[0]

        # If chosen_src['src'] != -1, we set unit_tag
        if chosen_src['src'] != -1:
            segment['unit_tag'] = chosen_src['tag'] if chosen_src['tag'] else chosen_src['src']
        else:
            segment['unit_tag'] = 0

    return segments


def _inject_alert_tone_segments(whisper_segments, transcribe_configuration, detected_tones):
    """
    Insert tone segments in chronological order, splitting or trimming
    whisper_segments to avoid overlap. Returns a single combined,
    sorted list of segments with no overlapping times.
    """
    # Convert to a list we can manipulate
    all_segments = list(whisper_segments)

    if not detected_tones:
        return whisper_segments

    # 1) Build "alert_segments" from detected_tones
    alert_segments = []
    for tone_type, tone_list in detected_tones.items():
        for tone in tone_list:
            alert_segment = _build_alert_segment(tone_type, tone, transcribe_configuration)
            if alert_segment:
                alert_segments.append(alert_segment)

    # 2) Insert each alert segment into the timeline,
    #    splitting speech segments if they overlap
    for alert_seg in alert_segments:
        all_segments = _insert_alert_segment(all_segments, alert_seg)

    # 3) Sort everything by start time
    all_segments.sort(key=lambda x: x['start'])

    # 4) Re-assign segment_id
    for i, seg in enumerate(all_segments, start=1):
        seg['segment_id'] = i

    return all_segments


def _build_alert_segment(tone_type, tone, transcribe_configuration):
    """
    Create a single alert segment dict from a tone dictionary
    like: {'start': 5.772, 'end': 9.821, 'detected': [...], etc.}
    """
    start = tone.get("start")
    end = tone.get("end")
    if start is None or end is None:
        return None  # invalid data

    # Build base text
    if tone_type == "two_tone":
        tone_text = f"[Two Tone ({str(tone.get('detected', ''))})]"
    elif tone_type == "long_tone":
        tone_text = f"[Long Tone ({str(tone.get('detected', ''))})]"
    elif tone_type == "hi_low_tone":
        tone_text = f"[High Low ({str(tone.get('detected', ''))})]"
    elif tone_type == "mdc":
        tone_text = f"[MDC ({str(tone.get('UnitID', ''))})]"
    elif tone_type == "dtmf":
        tone_text = f"[DTMF ({str(tone.get('digit', ''))})]"
    else:
        tone_text = "[Tones]"

    alert_seg = {
        "start": start,
        "end": end,
        "text": tone_text,
        "unit_tag": 0,
        "segment_id": -1,  # placeholder, will be reassigned
        "words": []
    }

    # Optionally build words if needed
    if transcribe_configuration.get('word_timestamps', False):
        alert_seg['words'] = [{
            'word_id': 1,
            'word': tone_text,
            'start': start,
            'end': end
        }]

    return alert_seg


def _insert_alert_segment(all_segments, alert_seg):
    """
    Insert `alert_seg` into `all_segments`.
    If there's overlap with a speech segment, we split or trim
    the speech so that times do not overlap.
    """
    new_segments = []
    inserted = False

    for seg in all_segments:
        # If seg is completely before alert_seg, keep as-is
        if seg['end'] <= alert_seg['start']:
            new_segments.append(seg)

        # If seg is completely after alert_seg, and we haven't inserted yet,
        # insert alert_seg first, then seg
        elif seg['start'] >= alert_seg['end']:
            if not inserted:
                new_segments.append(alert_seg)
                inserted = True
            new_segments.append(seg)

        else:
            # Overlap case:
            # seg.start < alert_seg.end AND seg.end > alert_seg.start
            # We can split seg into up to two pieces:
            #  -- left piece if seg.start < alert_seg.start
            #  -- right piece if seg.end > alert_seg.end

            # 1) Left piece
            if seg['start'] < alert_seg['start']:
                left_seg = dict(seg)  # shallow copy
                left_seg['end'] = alert_seg['start']
                # If you are storing word-level timestamps, also trim words
                if left_seg['words']:
                    left_seg['words'] = _trim_words(left_seg['words'], left_seg['start'], left_seg['end'])
                new_segments.append(left_seg)

            # 2) Insert the alert segment if not already done
            if not inserted:
                new_segments.append(alert_seg)
                inserted = True

            # 3) Right piece
            if seg['end'] > alert_seg['end']:
                right_seg = dict(seg)  # shallow copy
                right_seg['start'] = alert_seg['end']
                # trim words for the right piece
                if right_seg['words']:
                    right_seg['words'] = _trim_words(right_seg['words'], right_seg['start'], right_seg['end'])
                new_segments.append(right_seg)

    # If alert_seg never got inserted (it might be after all), append it
    if not inserted:
        new_segments.append(alert_seg)

    return new_segments


def _trim_words(words, seg_start, seg_end):
    """
    If you have word-level timestamps, trim or drop words that fall
    outside [seg_start, seg_end].
    """
    trimmed = []
    word_id = 0
    for w in words:
        w_start = w['start']
        w_end = w['end']
        # Check if it overlaps [seg_start, seg_end]
        if w_end > seg_start and w_start < seg_end:
            # Clip if partially outside
            w_start = max(w_start, seg_start)
            w_end = min(w_end, seg_end)
            word_id += 1
            trimmed.append({
                'word_id': word_id,
                'word': w['word'],
                'start': w_start,
                'end': w_end
            })
    return trimmed