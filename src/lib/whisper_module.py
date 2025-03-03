import concurrent.futures
import json
import logging
import os.path
import shutil
import traceback
from datetime import datetime, timedelta
import random

import numpy as np
from faster_whisper import WhisperModel, BatchedInferencePipeline, download_model

from lib.exceptions_module import WhisperConfigError, WhisperModelAccessError, TranscriptionError
from lib.utility_module import merge_dicts, parse_int, parse_float, parse_csv_of_floats, parse_nullable_int, \
    parse_bool_as_int, parse_nullable_string, parse_csv_of_integers, parse_nullable_float, parse_nullable_csv_of_floats

from lib.gpu_handler import get_available_gpus, get_gpu_memory

module_logger = logging.getLogger("icad_transcribe.whisper")

class DummyTranscriptionInfo:
    def __init__(self):
        self.segments = []

class WhisperTranscribe:
    """
    A class to manage initialization of Whisper models based on a configuration dictionary.

    :raises KeyError: If required keys are missing from config.
    :raises ValueError: If the device type is unsupported.
    :raises Exception: For unexpected errors during model initialization.
    """

    def __init__(self):
        self.whisper_models = {}
        self.gpu_count = 0
        self.batched = os.getenv('WHISPER_BATCHED', False)
        self.model_path = os.getenv('WHISPER_MODEL_PATH', 'var/models')
        self.model = os.getenv('WHISPER_MODEL', 'small.en')
        self.device = os.getenv('WHISPER_DEVICE', 'cpu')
        self.gpu_indexes = os.getenv('WHISPER_GPU_INDEXES', 0)
        self.compute_type = os.getenv('WHISPER_COMPUTE_TYPE', 'float16')
        self.cpu_threads = os.getenv('WHISPER_CPU_THREADS', 4)
        self.num_workers = os.getenv('WHISPER_NUM_WORKER', 1)
        # Construct model path for reference (not strictly necessary if you do it in initialize_models)
        self.model_path = os.path.join(
            os.getenv('WHISPER_MODEL_PATH', "var/models"),
            os.getenv('WHISPER_MODEL', "small.en"),
        )

        # Initialize the models immediately (you can also delay this if needed)
        try:
            self.initialize_models()
        except Exception as e:
            module_logger.error(f"Failed to initialize WhisperTranscribe: {e}")
            # Re-raise so the caller can handle it (or you can choose to handle fully here)
            raise

    def initialize_models(self):
        """
        Initialize Whisper models based on the configuration. Supports CPU or CUDA (GPU).
        :raises ValueError: If the device is unsupported (not 'cpu' or 'cuda').
        :raises Exception: For unexpected errors during model loading.
        """

        # Resolve absolute model path
        if not self.model_path.startswith("/"):
            root_path = os.getcwd()
            self.model_path = os.path.join(root_path, self.model_path)

        module_logger.debug(f"Using model path: {self.model_path}")

        if not os.path.isdir(self.model_path):
            module_logger.warning(f"Creating Model path {self.model_path}")
            os.makedirs(self.model_path)


        try:
            if self.device == "cpu":
                self._initialize_cpu_model()

            elif self.device == "cuda":
                self._initialize_gpu_models()

            else:
                raise ValueError(f"Unsupported device type: {self.device}")

            module_logger.info("All Whisper models initialized successfully.")

        except Exception as e:
            module_logger.error(f"Error while initializing Whisper models: {e}")
            raise

    def _initialize_cpu_model(self):
        """Helper method for CPU-only initialization."""
        module_logger.info(f"Initializing Whisper model {self.model} on <<CPU>>")
        model = WhisperModel(
            self.model,
            device="cpu",
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
            num_workers=self.num_workers,
            download_root=self.model_path
        )
        self.whisper_models[0] = BatchedInferencePipeline(model=model) if self.batched else model

    def _initialize_gpu_models(self):
        """Helper method for GPU (CUDA) initialization, with support for 'gpu_indexes'."""
        module_logger.info(f"Initializing Whisper model {self.model} on <<GPU>>")

        try:
            self.gpu_count = get_available_gpus()
            module_logger.info(f"Found {self.gpu_count} GPU(s).")
        except Exception as e:
            module_logger.error(f"Could not detect <<GPUs>> properly: {e}")
            raise

        if self.gpu_count == 0:
            # If device='cuda' but no GPUs, fallback to CPU
            module_logger.warning("No GPUs detected. Falling back to <<CPU>>.")
            self._initialize_cpu_model()
            return

        gpu_indexes = self.gpu_indexes

        if gpu_indexes.lower() == "all":
            # Use every GPU index in [0, self.gpu_count)
            indexes_to_use = range(self.gpu_count)
        else:
            valid_indexes = []
            # Split the CSV string (e.g. "0,1,5") into separate items
            for idx_str in gpu_indexes.split(","):
                idx_str = idx_str.strip()
                # Attempt to parse each chunk as an integer
                try:
                    idx = int(idx_str)
                except ValueError:
                    module_logger.warning(f"Ignoring invalid GPU index '{idx_str}' (not an integer).")
                    continue

                # Check if the parsed index is within valid range
                if 0 <= idx < self.gpu_count:
                    valid_indexes.append(idx)
                else:
                    module_logger.warning(
                        f"Ignoring invalid GPU index {idx}; valid range is [0..{self.gpu_count - 1}]."
                    )

            indexes_to_use = valid_indexes

        # Now indexes_to_use is either range(self.gpu_count) or a list of valid GPU indices.


        # If no device index is valid
        if not indexes_to_use:
            module_logger.warning("No valid <<GPU>> indexes found or provided. Falling back to <<CPU.>>")
            self._initialize_cpu_model()
            return

        # Helper to load the model on a GPU using concurrent.futures
        def _load_gpu_model(gpu_index):
            """
            Helper function that loads a single GPU model.
            Raises exceptions on failure so we can handle them in the main thread.
            """
            try:
                available_memory = get_gpu_memory(gpu_index)
                module_logger.info(
                    f"GPU {gpu_index} has {available_memory}MB free; loading model..."
                )

                model = WhisperModel(
                    self.model,
                    device="cuda",
                    device_index=gpu_index,
                    compute_type=self.compute_type,
                    num_workers=self.num_workers,
                    download_root=self.model_path
                )
                if self.batched:
                    pipeline = BatchedInferencePipeline(model=model)
                    module_logger.info(f"Initialized batched model {self.model} on GPU {gpu_index}.")
                    return gpu_index, pipeline
                else:
                    module_logger.info(f"Initialized single model {self.model} on GPU {gpu_index}.")
                    return gpu_index, model
            except Exception as e:
                # This exception will propagate back to the main thread in future.result()
                module_logger.error(f"Failed to initialize model {self.model} on GPU {gpu_index}: {e}")
                raise

        # Dictionary to track each future to its GPU index
        futures_to_gpu = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(indexes_to_use)) as executor:
            for gpu_index in indexes_to_use:
                future = executor.submit(_load_gpu_model, gpu_index)
                futures_to_gpu[future] = gpu_index

            # Collect results as futures complete
            for future in concurrent.futures.as_completed(futures_to_gpu):
                gpu_index = futures_to_gpu[future]
                try:
                    index_returned, model_obj = future.result()
                    self.whisper_models[index_returned] = model_obj
                except Exception as e:
                    # GPU failed to load model
                    module_logger.warning(f"Skipping GPU {gpu_index}")
                    continue

        if not self.whisper_models:
            module_logger.error("No GPU models were successfully initialized. Fallback to CPU or raise error.")
            # fallback to CPU:
            self._initialize_cpu_model()

    def transcribe_audio(self, audio_segment, transcribe_config: dict):
        """Transcribe the given audio file using the initialized models."""
        module_logger.info(f"Starting transcription of audio file.")

        module_logger.debug(f"Using Whisper Configuration: {transcribe_config}")

        # Get raw samples as a NumPy array (typically they're int16)
        samples = np.array(audio_segment.get_array_of_samples())

        # Convert to float32 in range -1.0 to +1.0
        samples = samples.astype(np.float32) / 32768.0

        transcribe_config["audio"] = samples

        if not self.batched:
            transcribe_config.pop('batch_size')

        if self.gpu_count <= 1:
            model = self.whisper_models[0]
        else:
            model = random.choice(list(self.whisper_models.values()))

        # Call the transcribe method
        try:
            segments, info = model.transcribe(**transcribe_config)
        except Exception as e:
            traceback.print_exc()
            module_logger.error(f"Error transcribing audio: {e}")
            info = DummyTranscriptionInfo()
            segments = info.segments

        module_logger.info(f"Finished transcription of audio file.")
        return segments, info

def get_whisper_config(db, transcribe_config_id=None, transcribe_config_name=None):

    base_query = f"""
    SELECT
      -- Transcribe config columns
      tsc.transcribe_config_id,
      tsc.language,
      tsc.transcribe_config_name,
      tsc.log_progress,
      tsc.condition_on_previous_text,
      tsc.suppress_blank,
      tsc.word_timestamps,
      tsc.without_timestamps,
      tsc.multilingual,
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
      vc.vad_filter,
      vc.threshold,
      vc.neg_threshold,
      vc.min_speech_duration_ms,
      vc.min_silence_duration_ms,
      vc.speech_pad_ms,
      
      -- Amplify config columns
      ac.amplify,
      ac.margin_factor,
      ac.safety_db,
      ac.chunk_ms,
      
      -- Tone Removal config columns
      tr.tone_removal,
      tr.matching_threshold,
      tr.time_resolution_ms,
      tr.tone_a_min_length,
      tr.tone_b_min_length,
      tr.hi_low_interval,
      tr.hi_low_min_alternations,
      tr.long_tone_min_length,
      tr.enable_mdc,
      tr.enable_dtmf
      
    FROM 
      transcribe_config tsc
    LEFT JOIN 
      vad_config AS vc
    ON 
      tsc.transcribe_config_id = vc.transcribe_config_id
    LEFT JOIN 
      amplify_config AS ac
    ON 
      tsc.transcribe_config_id = ac.transcribe_config_id
    LEFT JOIN 
      tone_removal_config AS tr
    ON 
      tsc.transcribe_config_id = tr.transcribe_config_id       
    """


    if transcribe_config_id:
        base_query += " WHERE tsc.transcribe_config_id = %s ORDER BY tsc.transcribe_config_id"
        params = (transcribe_config_id,)
    elif transcribe_config_name:
        base_query += " WHERE tsc.transcribe_config_name = %s ORDER BY tsc.transcribe_config_id"
        params = (transcribe_config_name,)
    else:
        base_query += " ORDER BY tsc.transcribe_config_id"
        params = None

    config_result = db.execute_query(base_query, params)

    if not config_result.get("success"):
        return config_result

    config_result = config_result["result"]

    final_result = []

    for whisper_config in config_result:
        transcribe_config_data = {"transcribe_config": {}}
        for field in transcribe_config_fields:
            raw_value = whisper_config.pop(field, None)

            if field in ["temperature", "suppress_tokens"]:
                if raw_value is not None:
                    try:
                        value = json.loads(raw_value)
                    except (ValueError, TypeError):
                        value = raw_value
                else:
                    value = raw_value
            else:
                value = raw_value

            if field == "transcribe_config_id":
                transcribe_config_data['transcribe_config_id'] = value
            elif field == "transcribe_config_name":
                transcribe_config_data['transcribe_config_name'] = value
            else:
                transcribe_config_data["transcribe_config"][field] = value

        # Build the vad_config sub-dict
        vad_config_data = {}
        for field in vad_config_fields:
            vad_config_data[field] = whisper_config.pop(field, None)

        # Attach vad_config to transcribe_config
        transcribe_config_data["vad_config"] = vad_config_data

        # Build the amplify_config sub-dict
        amplify_config_data = {}
        for field in amplify_config_fields:
            amplify_config_data[field] = whisper_config.pop(field, None)

        # Attach amplify_config to transcribe_config
        transcribe_config_data["amplify_config"] = amplify_config_data

        # Build the tone_removal_config sub-dict
        tone_removal_config_data = {}
        for field in tone_removal_config_fields:
            tone_removal_config_data[field] = whisper_config.pop(field, None)

        # Attach tone_removal_config to transcribe_config
        transcribe_config_data["tone_removal_config"] = tone_removal_config_data

        final_result.append(transcribe_config_data)

    return {"success": True, "message": "Retrieved transcribe configurations successfully.", "result": final_result}

def add_whisper_config(db, transcribe_config_name):

    # 1) Check if this config name already exists
    check_result = get_whisper_config(db, transcribe_config_name=transcribe_config_name)
    if not check_result["success"]:
        return {
            "success": False,
            "message": f'Unexpected error during check: {check_result["message"]}',
            "result": None
        }

    # If we found at least one config with the given name, return an error
    if check_result["result"]:
        return {
            "success": False,
            "message": f"Transcribe config name '{transcribe_config_name}' already exists.",
            "result": None
        }

    # 2) Insert a new row into transcribe_config
    transcribe_config_result = db.execute_commit(
        "INSERT INTO transcribe_config (transcribe_config_name) VALUES (%s)",
        (transcribe_config_name,), return_row_id=True)

    if not transcribe_config_result.get('success'):
        module_logger.error(f'Error adding default transcribe configuration: {transcribe_config_result.get("message")}')
        return {"success": False, "message": f'Error adding default transcribe configuration: {transcribe_config_result.get("message")}', 'result':None}

    # 3 Insert a new row in to vad_config settings
    vad_result = db.execute_commit("INSERT INTO vad_config (transcribe_config_id) VALUES (%s)", (transcribe_config_result.get("result"),))
    if not vad_result.get('success'):
        delete_whisper_config(db, transcribe_config_result.get("result"))
        return {"success": False, "message": "Error inserting new transcribe VAD configuration", "result": None}

    # 4 Insert a new row in to amplify_config
    amplify_result = db.execute_commit("INSERT INTO amplify_config (transcribe_config_id) VALUES (%s)", (transcribe_config_result.get("result"),))
    if not amplify_result.get('success'):
        delete_whisper_config(db, transcribe_config_result.get("result"))
        return {"success": False, "message": "Error inserting new transcribe Amplify configuration", "result": None}

    #5 Insert a new row in the tone_removal_config
    tone_removal_result = db.execute_commit("INSERT INTO tone_removal (transcribe_config_id) VALUES (%s)", (transcribe_config_result.get("result"),))
    if not tone_removal_result.get('success'):
        delete_whisper_config(db, transcribe_config_result.get("result"))
        return {"success": False, "message": "Error inserting new transcribe Toen Removal configuration", "result": None}

    module_logger.info(f"New transcribe configuration added successfully {transcribe_config_name}")
    return {"success": True, "message": f"New transcribe configuration added successfully {transcribe_config_name}", "result": transcribe_config_result.get("result")}

def update_whisper_config_basic(db, whisper_data):
    """
    Update a Basic Whisper Configuration.

    Args:
        db (SQLiteDatabase): The database wrapper instance.
        whisper_data (dict): A dictionary containing the system information. transcribe_config_id, language, transcribe_config_name, log_progress

    Returns:
        dict: A dictionary with success status, affected row count, and any error messages.
    """
    try:
        # Validate Inputs
        transcribe_config_id = whisper_data.get('transcribe_config_id')
        if transcribe_config_id:
            transcribe_config_id = int(transcribe_config_id)
        else:
            raise ValueError

        language = whisper_data.get('language')
        if language is not None:
            if len(language) > 2:
                raise ValueError
        transcribe_config_name = whisper_data.get('transcribe_config_name')
        if transcribe_config_name is None:
            raise ValueError

        log_progress = whisper_data.get('log_progress')
        if log_progress is None:
            raise ValueError
        log_progress = int(log_progress)
    except ValueError as ve:
        return {'success': False, 'message': 'One or more fields as not of the right data type.'}

    if transcribe_config_id == 1:
        # Don't allow name change if default
        update_query = f"""
                            UPDATE 
                              transcribe_config
                            SET 
                              language = %s,  
                              log_progress= %s
                            WHERE
                              transcribe_config_id = %s
                        """
        params = (language, log_progress, transcribe_config_id)
    else:
        update_query = f"""
                            UPDATE 
                              transcribe_config
                            SET 
                              language = %s, 
                              transcribe_config_name = %s, 
                              log_progress= %s
                            WHERE
                              transcribe_config_id = %s
                        """
        params = (language, transcribe_config_name, log_progress, transcribe_config_id)


    update_result = db.execute_commit(update_query, params)

    return update_result

def update_whisper_config_decoding(db, whisper_data):
    """
    Update a Decoding Whisper Configuration.

    Args:
        db (SQLiteDatabase): The database wrapper instance.
        whisper_data (dict): A dictionary containing the system information:
          - transcribe_config_id (int)
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
        # 1) transcribe_config_id must be present & valid
        transcribe_config_id = parse_int(
            whisper_data.get('transcribe_config_id'),
            'transcribe_config_id'
        )

        # 2) Non‐nullable int fields
        batch_size = parse_int(
            whisper_data.get('batch_size'),
            'batch_size'
        )

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
           SET batch_size = %s,
               beam_size = %s,
               best_of = %s,
               patience = %s,
               length_penalty = %s,
               repetition_penalty = %s,
               no_repeat_ngram_size = %s,
               temperature = %s,
               compression_ratio_threshold = %s,
               log_prob_threshold = %s,
               no_speech_threshold = %s,
               max_new_tokens = %s,
               chunk_length = %s
         WHERE transcribe_config_id = %s
    """

    temperature_str = json.dumps(temperature)

    params = (
        batch_size,
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
        transcribe_config_id
    )

    update_result = db.execute_commit(update_query, params)
    return update_result

def update_whisper_config_prompt(db, whisper_data):
    """
    Update a system's Prompt Whisper Configuration.

    Args:
        db (SQLiteDatabase): The database wrapper instance.
        whisper_data (dict): A dictionary containing the form data, e.g.:
            - transcribe_config_id (required, int)
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

        transcribe_config_id = parse_int(
            whisper_data.get('transcribe_config_id'),
            'transcribe_config_id'
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
         WHERE transcribe_config_id = %s
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
        transcribe_config_id
    )

    update_result = db.execute_commit(update_query, params)

    return update_result

def update_whisper_config_advanced(db, whisper_data):
    """
    Update a system's 'Advanced' Whisper Configuration.

    Args:
        db (SQLiteDatabase): Database wrapper instance.
        whisper_data (dict): Form data, which may include:
            - transcribe_config_id (int, required)
            - clip_timestamps (nullable CSV of floats)
            - multilingual (bool -> stored as int(1 or 0))
            - language_detection_threshold (float, required)
            - language_detection_segments (int, required)
    Returns:
        dict: { 'success': bool, 'message': str } or DB result on success.
    """
    try:
        # 1) transcribe_config_id must be present & valid integer
        transcribe_config_id = parse_int(
            whisper_data.get('transcribe_config_id'),
            'transcribe_config_id'
        )

        # 2) Parse clip_timestamps as a *nullable* CSV of floats
        #    e.g. "0.0,0.1,0.2" => [0.0, 0.1, 0.2]
        clip_timestamps_str = parse_nullable_string(
            whisper_data.get('clip_timestamps'),
            'clip_timestamps'
        )

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
         WHERE transcribe_config_id = %s
    """

    # Prepare parameters; note the JSON or None for clip_timestamps
    params = (
        clip_timestamps_str,
        multilingual,
        language_detection_threshold,
        language_detection_segments,
        transcribe_config_id
    )

    update_result = db.execute_commit(update_query, params)

    return update_result

def update_whisper_config_vad(db, whisper_data):
    """
     Update a system's VAD (Voice Activity Detection) Whisper Configuration.

     Args:
         db (SQLiteDatabase): The database wrapper instance.
         whisper_data (dict): Form data that may include:
             - radio_system_id (required int)
             - transcribe_config_id (required int)
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
        transcribe_config_id = parse_int(
            whisper_data.get('transcribe_config_id'),
            'transcribe_config_id'
        )

        #the form sends both 1 an 0 when enabled and just 0 when disabled
        enabled_values = whisper_data.getlist('vad_filter')
        vad_filter_val = int('1' in enabled_values)

        vad_enabled = vad_filter_val

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
           SET vad_filter             = %s,
               threshold              = %s,
               neg_threshold          = %s,
               min_speech_duration_ms = %s,
               min_silence_duration_ms= %s,
               speech_pad_ms         = %s
         WHERE transcribe_config_id    = %s
    """
    params_vad = (
        vad_enabled,
        threshold,
        neg_threshold,
        min_speech_duration_ms,
        min_silence_duration_ms,
        speech_pad_ms,
        transcribe_config_id
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

    # If both updates succeed, return a combined success
    return {
        "success": True,
        "message": "VAD configuration updated successfully.",
        "result": []
    }

def update_whisper_config_tone_removal(db, whisper_data):
    """
     Update a tone removal Whisper Configuration.

     Args:
         db (SQLiteDatabase): The database wrapper instance.
         whisper_data (dict): Form data that may include:
             - transcribe_config_id (required int)
             - tone_removal (bool -> stored as 1 or 0)
             - matching_threshold (float, required)
             - tone_a_min_length (float, required)
             - tone_b_min_length (float, required)
             - hit_low_interval (float, required)
             - hi_low_min_alternations (int, required)
             - long_tone_min_length (float, required)
             - enable_mdc (bool -> stores as 1 or 0)
             - enable_dtmf (bool -> stores as 1 or 0)

     Returns:
         dict or similar: A result object indicating success/failure and possibly rows affected.
     """
    try:
        # --- Parse/Validate Inputs ---
        transcribe_config_id = parse_int(
            whisper_data.get('transcribe_config_id'),
            'transcribe_config_id'
        )

        tone_removal = parse_int(
            whisper_data.get("tone_removal"),
            "tone_removal"
        )

        matching_threshold = parse_float(
            whisper_data.get("matching_threshold"),
            "matching_threshold"
        )

        tone_a_min_length = parse_float(
            whisper_data.get("tone_a_min_length"),
            "tone_a_min_length"
        )

        tone_b_min_length = parse_float(
            whisper_data.get("tone_b_min_length"),
            "tone_b_min_length"
        )

        hi_low_interval = parse_float(
            whisper_data.get("hi_low_interval"),
            "hi_low_interval"
        )

        hi_low_min_alternations = parse_int(
            whisper_data.get("hi_low_min_alternations"),
            "hi_low_min_alternations"
        )

        long_tone_min_length = parse_float(
            whisper_data.get("long_tone_min_length"),
            "long_tone_min_length"
        )

        enable_mdc = parse_int(
            whisper_data.get("enable_mdc"),
            "enable_mdc"
        )

        enable_dtmf = parse_int(
            whisper_data.get("enable_dtmf"),
            "enable_dtmf"
        )

    except ValueError as ve:
        return {
            "success": False,
            "message": f"Error getting VAD data: {ve}"
        }

    # ----------------------------------------------------
    # 1) Update the vad_config table
    # ----------------------------------------------------
    update_query_tone_removal = """
        UPDATE tone_removal_config
           SET tone_removal             = %s,
               matching_threshold       = %s,
               tone_a_min_length        = %s,
               tone_b_min_length        = %s,
               hi_low_interval          = %s,
               hi_low_min_alternations  = %s,
               long_tone_min_length     = %s,
               enable_mdc               = %s,
               enable_dtmf              = %s
         WHERE transcribe_config_id     = %s
    """
    params_tone_removal = (
        tone_removal,
        matching_threshold,
        tone_a_min_length,
        tone_b_min_length,
        hi_low_interval,
        hi_low_min_alternations,
        long_tone_min_length,
        enable_mdc,
        enable_dtmf,
        transcribe_config_id
    )

    update_result = db.execute_commit(
        update_query_tone_removal,
        params_tone_removal
    )

    if not update_result["success"]:
        return {
            "success": False,
            "message": f"Failed to update tone_removal_config: {update_result['message']}",
            "result": update_result["result"]
        }

    return {
        "success": True,
        "message": "Tone Removal configuration updated successfully.",
        "result": []
    }

def update_whisper_config_amplify(db, whisper_data):
    """
     Update a configurations AMPLIFY Whisper Configuration.

     Args:
         db (SQLiteDatabase): The database wrapper instance.
         whisper_data (dict): Form data that may include:
             - transcribe_config_id (required int)
             - amplify (bool -> stored as 1 or 0)
             - margin_factor (float, required)
             - safety_db (float, required)

     Returns:
         dict or similar: A result object indicating success/failure and possibly rows affected.
     """
    try:
        # --- Parse/Validate Inputs ---
        transcribe_config_id = parse_int(
            whisper_data.get('transcribe_config_id'),
            'transcribe_config_id'
        )

        amplify_enabled = parse_int(
            whisper_data.get("amplify"),
            "amplify"
        )

        margin_factor = parse_float(
            whisper_data.get("margin_factor"),
            "margin_factor"
        )

        safety_db = parse_float(
            whisper_data.get("safety_db"),
            "safety_db"
        )

    except ValueError as ve:
        return {
            "success": False,
            "message": f"Error getting Amplify data: {ve}"
        }

    # ----------------------------------------------------
    # Update the amplify_config table
    # ----------------------------------------------------
    update_query_amplify = """
        UPDATE amplify_config
           SET amplify                 = %s,
               margin_factor           = %s,
               safety_db               = %s
         WHERE transcribe_config_id    = %s
    """
    params_vad = (
        amplify_enabled,
        margin_factor,
        safety_db,
        transcribe_config_id
    )

    amplify_result = db.execute_commit(
        update_query_amplify,
        params_vad
    )

    if not amplify_result["success"]:
        return {
            "success": False,
            "message": f"Failed to update amplify_config: {amplify_result['message']}",
            "result": amplify_result["result"]
        }

    return {
        "success": True,
        "message": "Amplify configuration updated successfully.",
        "result": []
    }

def delete_whisper_config(db, transcribe_config_id):
    """
    Delete a configuration by its ID.

    Args:
        db (MySQLDatabase): The database wrapper instance.
        transcribe_config_id (int): The ID of the system to delete.

    Returns:
        dict: A dictionary with success status and affected row count.
    """

    if not transcribe_config_id:
        return {"success": False, "message": "No transcribe config ID provided", "result": None}

    if transcribe_config_id == 1:
        return {"success": False, "message": "Default config can't not be deleted.", "result": None}

    query = "DELETE FROM transcribe_config WHERE transcribe_config_id = %s"
    params = (transcribe_config_id,)

    result = db.execute_commit(query, params)
    return result

transcribe_config_fields = [
    "transcribe_config_id",
    "transcribe_config_name",
    "language",
    "log_progress",
    "condition_on_previous_text",
    "suppress_blank",
    "word_timestamps",
    "without_timestamps",
    "multilingual",
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
    "vad_filter",
    "threshold",
    "neg_threshold",
    "min_speech_duration_ms",
    "min_silence_duration_ms",
    "speech_pad_ms",
]

amplify_config_fields = [
    "amplify",
    "margin_factor",
    "safety_db",
    "chunk_ms"
]

tone_removal_config_fields = [
    'tone_removal',
    'matching_threshold',
    'time_resolution_ms',
    'tone_a_min_length',
    'tone_b_min_length',
    'hi_low_interval',
    'hi_low_min_alternations',
    'long_tone_min_length',
    'enable_mdc',
    'enable_dtmf'
]