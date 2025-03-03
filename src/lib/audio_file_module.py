import io
import logging
import os.path
import uuid
from typing import Any

import magic
import numpy as np
from icad_tone_detection import tone_detect
from pydub import AudioSegment
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

module_logger = logging.getLogger('icad_transcribe.audio_file')



def validate_audio_file(audio_file):
    """
    Validates an uploaded audio file based on its MIME type and duration.

    This function:
      1. Checks the file object's existence.
      2. Detects the file's MIME type using `python-magic`.
      3. Verifies that the MIME type is in the allowed list.
      4. Loads the audio using pydub to determine its duration and sample rate.
      5. Enforces min/max duration constraints (if provided).

    Args:
        audio_file (file-like object):
            A file-like object (e.g., `werkzeug.FileStorage`, `io.BytesIO`) containing audio data.

    Returns:
        dict:
            A dictionary containing:
            - "success" (bool): Whether validation passed or failed.
            - "message" (str): A message explaining the result.
            - "result" (dict or None): If "success" is True, contains:
                {
                    "duration": float,       # Duration in seconds
                    "sample_rate": int,      # Original sample rate
                    "audio": AudioSegment    # pydub AudioSegment
                }
            Otherwise, None if validation fails.
    """

    allowed_mimetypes_str = os.getenv('AUDIO_UPLOAD_ALLOWED_MIMETYPES', 'audio/x-wav,audio/x-m4a,audio/mpeg')
    allowed_mimetypes = [m.strip() for m in allowed_mimetypes_str.split(',') if m.strip()]
    try:
        min_audio_length = float(os.getenv('AUDIO_UPLOAD_MIN_AUDIO_LENGTHS', 0))
        max_audio_length = float(os.getenv('AUDIO_UPLOAD_MAX_AUDIO_LENGTH', 300))
    except ValueError:
        min_audio_length = 0.0
        max_audio_length = 300.0

    # Check basic parameter validity
    if not audio_file:
        return {
            "success": False,
            "message": "No file provided or invalid file object.",
            "result": None
        }

    if not allowed_mimetypes or not isinstance(allowed_mimetypes, (list, tuple, set)):
        return {
            "success": False,
            "message": "Allowed mime types not provided or invalid.",
            "result": None
        }

    # Ensure min and max are numeric
    if not isinstance(min_audio_length, (int, float)) or not isinstance(max_audio_length, (int, float)):
        return {
            "success": False,
            "message": "Minimum and maximum audio length must be numbers.",
            "result": None
        }

    try:
        # Detect MIME type using python-magic
        # Read a small buffer from the file to guess the MIME type
        head_bytes = audio_file.read(1024)
        mimetype = magic.from_buffer(head_bytes, mime=True)
        audio_file.seek(0)  # reset the pointer
    except Exception as e:
        module_logger.debug(f"Error detecting MIME type: {e}")
        return {
            "success": False,
            "message": f"Error reading file for MIME detection: {str(e)}",
            "result": None
        }

    # Check if mimetype is allowed
    if mimetype not in allowed_mimetypes:
        return {
            "success": False,
            "message": f"Disallowed MIME type '{mimetype}'. Must be one of: {allowed_mimetypes}",
            "result": None
        }

    try:
        # Parse the entire file content for AudioSegment
        audio_bytes = audio_file.read()
        audio_file.seek(0)
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
        duration_seconds = audio_segment.duration_seconds
        original_sample_rate = audio_segment.frame_rate
    except Exception as e:
        # This can catch pydub's CouldntDecodeError or other I/O issues
        module_logger.debug(f"Error loading audio file with pydub: {e}")
        return {
            "success": False,
            "message": f"Error loading audio file: {str(e)}",
            "result": None
        }

    # Check maximum length (if > 0)
    if max_audio_length > 0 and duration_seconds > max_audio_length:
        return {
            "success": False,
            "message": f"File duration {duration_seconds:.2f}s exceeds max of {max_audio_length}s.",
            "result": None
        }

    # Check minimum length (if > 0)
    if min_audio_length > 0 and duration_seconds < min_audio_length:
        return {
            "success": False,
            "message": f"File duration {duration_seconds:.2f}s is below min of {min_audio_length}s.",
            "result": None
        }

    # If all checks pass, return a success response
    return {
        "success": True,
        "message": "Valid audio file.",
        "result": {
            "duration": duration_seconds,
            "sample_rate": original_sample_rate,
            "audio": audio_segment
        }
    }

def pre_process_audio(audio: AudioSegment, tone_removal_configuration: dict, amplify_configuration: dict, vad_configuration: dict) -> tuple[AudioSegment | Any, dict | None]:
    """
    Pre-processes an audio file by detecting and removing tones, optionally applying voice amplification,
    and running VAD for speech detection.

    Steps:
        1. Detect tones in the audio.
        2. Remove tone regions using tone metadata.
        3. (Optional) Get voice segments via Silero VAD.
        4. (Optional) Maximize volume of voice segments.
        5. (Optional) Insert silences back where tones were removed to preserve original length.

    Args:
        audio (AudioSegment):
            The input audio file as a pydub AudioSegment object.
        tone_removal_configuration (dict):
            Configuration for tone detection and removal:
              - tone_removal (bool): Whether or not to remove detected tones.
              - matching_threshold (float): % threshold for matching frequencies. Default: 2.5.
              - time_resolution_ms (int): Time resolution for STFT in ms. Default: 25.
              - tone_a_min_length (float): Minimum length in seconds for A tone. Default: 0.8.
              - tone_b_min_length (float): Minimum length in seconds for B tone. Default: 2.8.
              - long_tone_min_length (float): Minimum length for a long tone. Default: 3.8.
              - hi_low_interval (float): Max allowed interval between hi-low tones. Default: 0.2s.
              - hi_low_min_alternations (int): Min number of alternations for a valid hi-low warble. Default: 6.
        amplify_configuration (dict):
            Configuration for speech amplification:
              - amplify (bool): Whether to apply speech amplification. Default: True.
              - margin_factor (float): How close to full-scale (0..1). Default 0.99.
              - safety_db (float): Additional dB headroom below 0 dBFS. Default 0.4.
              - chunk_ms (int): Size of each chunk (ms) for local peak normalization. Default 100.
        vad_configuration (dict):
            Configuration for Voice Activity Detection:
              - vad_filter (bool): Whether to apply VAD or not.
              - threshold (float): VAD sensitivity threshold [0..1]. Default: 0.3.
              - min_speech_duration_ms (int): Minimum speech length in ms. Default: 50.
              - min_silence_duration_ms (int): Minimum silence length in ms. Default: 100.

    Returns:
        AudioSegment:
            Processed audio after optional tone removal, optional VAD-based amplification,
            and preservation of original length.
        detected_tones:
            Dict with the detected tones or None if tone removal is false
    """

    processed_audio = None
    tone_list = None
    removed_timestamps = []

    # 2. Detect and remove tones
    if tone_removal_configuration.get('tone_removal'):
        tone_list = detect_tones(audio, tone_removal_configuration)
        processed_audio, removed_timestamps = _remove_tones(audio, tone_list)
    else:
        processed_audio = audio

    # 2. Get Speech Segments From Silero VAD and Replace Non-Speech sections with silence. Then amplify if its enabled.
    if vad_configuration.get('enabled'):
        speech_timestamps = _get_speech_timestamps_silero_vad(processed_audio, vad_configuration)
        processed_audio = _replace_non_speech_with_silence(processed_audio, speech_timestamps)

        # 3. Maximize voice volume, but only if VAD and Amplify are enabled.
        if amplify_configuration.get("amplify"):
            processed_audio = _amplify_segments(processed_audio, speech_timestamps, amplify_configuration)

    # 4. Restore exact-length silences
    if tone_removal_configuration.get('tone_removal'):
        processed_audio = _restore_tone_silence(processed_audio, removed_timestamps, len(audio), audio.frame_rate)

    # 5. Export Final Audio Segment for Debug.
    audio_debug_path = f"static/audio/debug"
    if not os.path.exists(audio_debug_path):
        os.makedirs(audio_debug_path)
    audio_debug_output_file = f"{uuid.uuid4()}_pre_process.wav"
    processed_audio.export(os.path.join(audio_debug_path, audio_debug_output_file), format='wav')
    module_logger.debug(f"Exported processed audio to {os.path.join(audio_debug_path, audio_debug_output_file)}")

    # 6. Return Final Audio Segment after processing.
    return processed_audio, tone_list

def detect_tones(audio: AudioSegment, tone_removal_configuration: dict):
    """
   Detects various tone types (two-tone, long tone, hi-low tone, MDC, DTMF) in an audio clip.

   Uses an internal STFT-based approach to identify tonal regions. Returns a dict
   of detected tones, each containing a list of detected tone segments with start/end times.

   Args:
       audio (AudioSegment):
           The audio segment to analyze.
       tone_removal_configuration (dict):
           Configuration dictionary containing:
             - matching_threshold (float): % threshold for matching frequencies (Default: 2.5).
             - time_resolution_ms (int): STFT window size in ms (Default: 25).
             - tone_a_min_length (float): Min length (s) of A tone for two-tone detection (Default: 0.8).
             - tone_b_min_length (float): Min length (s) of B tone for two-tone detection (Default: 2.8).
             - hi_low_interval (float): Max interval (s) between hi-low tones (Default: 0.2).
             - hi_low_min_alternations (int): Min alternations for hi-low warble (Default: 6).
             - long_tone_min_length (float): Min length (s) for a long tone (Default: 3.8).

   Returns:
       dict:
           A dictionary of detected tones, keyed by tone type:
           {
               "two_tone": [...],
               "long_tone": [...],
               "hi_low_tone": [...],
               "mdc": [...],
               "dtmf": [...]
           }
           Each list contains tone segments with "start" and "end" in seconds.
   """
    tone_result = tone_detect(audio,
                              matching_threshold=tone_removal_configuration.get('matching_threshold', 2.5),
                              time_resolution_ms=tone_removal_configuration.get('time_resolution_ms', 25),
                              tone_a_min_length=tone_removal_configuration.get('tone_a_min_length', 0.7),
                              tone_b_min_length=tone_removal_configuration.get('tone_b_min_length', 2.7),
                              hi_low_interval=tone_removal_configuration.get('hi_low_interval', 0.2),
                              hi_low_min_alternations=tone_removal_configuration.get('hi_low_min_alternations', 6),
                              long_tone_min_length=tone_removal_configuration.get('long_tone_min_length', 3.8))

    detect_result = {"two_tone": tone_result.two_tone_result,
                     "long_tone": tone_result.long_result,
                     "hi_low_tone": tone_result.hi_low_result,
                     "mdc": tone_result.mdc_result,
                     "dtmf": tone_result.dtmf_result}

    module_logger.debug(f"Tones Detected: {detect_result}")

    return detect_result

def _remove_tones(audio: AudioSegment, tone_json: dict):
    """
    Removes tone segments from the audio and returns the processed audio plus a list of removed intervals.

    This function merges overlapping tone intervals, then cuts them out of the audio.
    The resulting audio is shorter, but can be restored later by re-inserting silence.

    Args:
        audio (AudioSegment):
            The original audio from which tones will be removed.
        tone_json (dict):
            A dictionary where each key is a tone type, and its value is a list of
            dictionaries containing 'start' and 'end' times (in seconds).

    Returns:
        tuple:
            (processed_audio, removed_timestamps)
            processed_audio (AudioSegment): Audio with tone segments removed.
            removed_timestamps (list): A list of (start_ms, end_ms) intervals for each removed tone.
    """
    tone_ranges = []

    # 1) Gather and convert tone start/end times to milliseconds
    for tone_type, tone_list in tone_json.items():
        for tone_info in tone_list:
            start_ms = int(round(tone_info["start"] * 1000))
            end_ms = int(round(tone_info["end"] * 1000))
            tone_ranges.append((start_ms, end_ms))

    # 2) Sort and merge overlapping/adjacent ranges
    merged_tone_ranges = _merge_ranges(tone_ranges)

    # 3) Construct new audio without tones
    processed_audio = AudioSegment.silent(duration=0, frame_rate=audio.frame_rate)
    removed_timestamps = []  # Track exact places where tones were removed
    last_end = 0

    for (start_ms, end_ms) in merged_tone_ranges:
        start_ms = max(0, start_ms)
        end_ms = min(len(audio), end_ms)

        # Append the segment **exactly** between tones
        if start_ms > last_end:
            processed_audio += audio[last_end:start_ms]

        # Track removed tone region
        removed_timestamps.append((start_ms, end_ms))

        # Skip the tone segment itself
        last_end = end_ms

    # 4) Append the remaining part of the original audio
    if last_end < len(audio):
        processed_audio += audio[last_end:]

    module_logger.debug(f"Removed tones. New length: {len(processed_audio)}ms (Original: {len(audio)}ms)")

    return processed_audio, removed_timestamps

def _merge_ranges(ranges, gap=0):
    """
    Merges overlapping or adjacent ranges into a single continuous interval if they're within `gap` ms.

    Args:
        ranges (list of tuples):
            A list of (start_ms, end_ms) intervals in ascending order.
        gap (int):
            A threshold in milliseconds. If the next range is within `gap` ms of the current one, they merge.

    Returns:
        list of tuples:
            A merged list of (start_ms, end_ms) intervals, ensuring no overlaps or tiny gaps exist.
    """
    if not ranges:
        return []

    merged = []
    ranges.sort(key=lambda x: x[0])

    current_start, current_end = ranges[0]

    for i in range(1, len(ranges)):
        start, end = ranges[i]
        # If the next range overlaps or is within 'gap' ms of the current range
        if start <= current_end + gap:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    # Add the final merged range
    merged.append((current_start, current_end))

    return merged

def _get_speech_timestamps_silero_vad(audio, vad_configuration):
    """
    Detects speech segments in the given audio using Silero VAD and returns precise timestamps.

    This function converts a pydub AudioSegment into a 16 kHz, mono float32 waveform in the
    range [-1, 1], then runs the Silero Voice Activity Detector with user-defined thresholds
    and parameters. It does not alter the audio itself, only returns the speech intervals.

    Args:
        audio (AudioSegment):
            The input audio to analyze. Must be compatible with pydub (e.g., WAV, MP3).
            Internally, it is downsampled to 16 kHz, mono, 16-bit.
        vad_configuration (dict):
            A dictionary containing VAD parameters:
              - threshold (float): Main sensitivity threshold [0..1].
              - neg_threshold (float): Background noise threshold (optional).
              - min_speech_duration_ms (int): Minimum speech length for a valid segment (ms).
              - min_silence_duration_ms (int): Minimum silence length before a speech segment ends (ms).
              - speech_pad_ms (int): Extra padding applied to each speech segment (ms).

    Returns:
        list of dict:
            A list of detected speech intervals, each element has:
              {
                "start": <start_time_seconds>,
                "end":   <end_time_seconds>
              }
            Example:
              [
                {"start": 1.5, "end": 3.6},
                {"start": 10.0, "end": 13.6},
                ...
              ]
    """
    module_logger.debug("Processing audio with Silero VAD...")

    # Convert pydub AudioSegment to NumPy int16 array
    samples = np.array(audio.get_array_of_samples(), dtype=np.int16)

    # Normalize to float32 range [-1, 1]
    samples = samples.astype(np.float32) / 32768.0

    # Load Silero VAD model
    model = load_silero_vad()
    speech_timestamps = get_speech_timestamps(
        samples,
        model,
        threshold = vad_configuration.get('threshold', 0.5),
        neg_threshold = vad_configuration.get('neg_threshold'),
        min_speech_duration_ms = vad_configuration.get('min_speech_duration_ms', 250),
        min_silence_duration_ms = vad_configuration.get('min_silence_duration_ms', 250),
        speech_pad_ms = vad_configuration.get('speech_pad_ms', 250),
        sampling_rate = 16000,
        return_seconds = True
    )

    module_logger.debug(f"Silero VAD detected speech timestamps: {speech_timestamps}")

    return speech_timestamps

def _replace_non_speech_with_silence(audio: AudioSegment, speech_timestamps: list) -> AudioSegment:
    """
    Mutes all audio outside the specified speech timestamps, preserving only speech segments.

    This function iterates over the speech intervals and copies the original audio in those intervals.
    Everything else is replaced with silence, so the final audio has the same length but zero amplitude
    where speech does not exist.

    Args:
        audio (AudioSegment):
            The original audio to modify. Typically 16 kHz, mono for VAD.
        speech_timestamps (list of dict):
            Each dict in the list has "start" and "end" in seconds, e.g.:
              [
                {"start": 1.5, "end": 3.6},
                {"start": 10.0, "end": 13.6},
                ...
              ]

    Returns:
        AudioSegment:
            A new AudioSegment of equal duration to `audio`. Speech intervals remain intact,
            and everything else is zero amplitude.
    """
    module_logger.debug("Muting non-speech portions...")

    # 1) Convert to float32 [-1, 1]
    samples_16k = np.array(audio.get_array_of_samples(), dtype=np.float32)
    max_abs = 2 ** (8 * audio.sample_width - 1)
    samples_16k /= max_abs

    # 2) Create an output buffer of zeros
    out_samples_16k = np.zeros_like(samples_16k, dtype=np.float32)

    # 3) Copy speech segments from the original array
    for seg in speech_timestamps:
        start_idx = int(seg["start"] * 16000)
        end_idx   = int(seg["end"]   * 16000)

        # Ensure valid bounds
        start_idx = max(start_idx, 0)
        end_idx   = min(end_idx, len(samples_16k))

        if end_idx > start_idx:
            out_samples_16k[start_idx:end_idx] = samples_16k[start_idx:end_idx]
        else:
            module_logger.debug(f"Skipping empty or invalid segment: {seg}")

    # 4) Convert back to int16
    out_int16 = (out_samples_16k * max_abs).clip(-max_abs, max_abs - 1).astype(np.int16)

    # 5) Build new AudioSegment
    out_audio = AudioSegment(
        data=out_int16.tobytes(),
        sample_width=2,
        frame_rate=16000,
        channels=1
    )
    module_logger.debug("Finished muting non-speech.")
    return out_audio

def _amplify_segments(audio: AudioSegment, speech_timestamps: list, amplify_configuration: dict) -> AudioSegment:
    """
    Amplifies detected speech segments independently, leaving non-speech regions as silence.

    This function processes each speech interval in small chunks (e.g., 100ms), calculates
    the maximum amplitude per chunk, and applies a proportional gain to bring it near
    a target peak. Non-speech remains at zero amplitude.

    Args:
        audio (AudioSegment):
            The input audio (16 kHz, mono) with non-speech possibly already set to silence.
        speech_timestamps (list of dict):
            A list of speech intervals, each with "start" and "end" in seconds.
        amplify_configuration (dict):
            A dict containing:
              - margin_factor (float): How close to full-scale (0..1). Default 0.99.
              - safety_db (float): Additional dB headroom below 0 dBFS. Default 0.4.
              - chunk_ms (int): Size of each chunk (ms) for local peak normalization. Default 100.

    Returns:
        AudioSegment:
            A new AudioSegment where the speech intervals have been amplified to a near-0 dBFS
            target, and non-speech remains silent.
    """
    module_logger.debug("Running Speech Amplifier...")

    samples_16k = np.array(audio.get_array_of_samples(), dtype=np.float32)
    max_abs = 2 ** (8 * audio.sample_width - 1)
    samples_16k /= max_abs

    # Compute target amplitude
    safety_linear = 10.0 ** (-amplify_configuration.get('safety_db', 0.5) / 20.0)
    target_peak   = amplify_configuration.get('margin_factor', 0.99) * safety_linear
    chunk_size    = int((amplify_configuration.get('chunk_ms', 100) / 1000) * 16000)  # Convert ms to samples

    # Create output buffer of zeros
    out_samples_16k = np.zeros_like(samples_16k, dtype=np.float32)

    # Process each speech segment
    for seg in speech_timestamps:
        start_idx = int(seg["start"] * 16000)
        end_idx   = int(seg["end"]   * 16000)

        start_idx = max(start_idx, 0)
        end_idx   = min(end_idx, len(samples_16k))

        module_logger.debug(f"Processing segment: {seg}")

        # Split the segment into smaller chunks
        for chunk_start in range(start_idx, end_idx, chunk_size):
            chunk_end = min(chunk_start + chunk_size, end_idx)
            chunk_samples = samples_16k[chunk_start:chunk_end]

            # Find max amplitude in this chunk
            chunk_max_amp = np.max(np.abs(chunk_samples)) if len(chunk_samples) > 0 else 0
            if chunk_max_amp > 0:
                gain_factor = target_peak / chunk_max_amp
                module_logger.debug(f"Chunk {chunk_start}-{chunk_end}: Max Amp={chunk_max_amp:.4f}, Gain Factor={gain_factor:.4f}")

                out_samples_16k[chunk_start:chunk_end] = (chunk_samples * gain_factor).clip(-1, 1)
            else:
                module_logger.debug(f"Chunk {chunk_start}-{chunk_end}: No valid amplitude, skipping.")

    # Convert float32 back to int16
    out_int16 = (out_samples_16k * max_abs).clip(-max_abs, max_abs - 1).astype(np.int16)

    out_audio = AudioSegment(
        data=out_int16.tobytes(),
        sample_width=2,
        frame_rate=16000,
        channels=1
    )

    module_logger.debug("Speech amplification complete.")
    return out_audio

def _restore_tone_silence(shortened_audio: AudioSegment, removed_timestamps: list, original_length_ms: int, frame_rate: int = None):
    """
    Restores exact-length silence at the places where tones were removed.

    :param shortened_audio: AudioSegment with tones removed.
    :param removed_timestamps: List of (start_ms, end_ms) where silence was removed.
    :param original_length_ms: Length of the original audio.
    :param frame_rate: Sample rate of the output.
    :return: Restored audio with silence exactly where tones were removed.
    """
    if frame_rate is None:
        frame_rate = shortened_audio.frame_rate

    # 1) Create a silent base audio of exact original length
    output = AudioSegment.silent(duration=original_length_ms, frame_rate=frame_rate)

    # 2) Overlay non-tone segments from shortened_audio
    short_idx = 0  # Pointer into shortened_audio (ms)
    last_end = 0   # Last tone end position in original timeline (ms)

    for (tone_start, tone_end) in removed_timestamps:
        tone_start = max(0, min(tone_start, original_length_ms))
        tone_end = max(0, min(tone_end, original_length_ms))

        # Overlay preserved speech segments
        if tone_start > last_end:
            gap_duration = tone_start - last_end
            gap_chunk = shortened_audio[short_idx: short_idx + gap_duration]
            output = output.overlay(gap_chunk, position=last_end)
            short_idx += len(gap_chunk)  # Ensure we track the exact amount used

        # Move past the tone segment
        last_end = tone_end

    # 3) Overlay any remaining audio after the last tone
    if last_end < original_length_ms and short_idx < len(shortened_audio):
        gap_chunk = shortened_audio[short_idx:]
        output = output.overlay(gap_chunk, position=last_end)

    print(f"[DEBUG] Restored silences. Final length: {len(output)}ms (Original: {original_length_ms}ms)")

    return output