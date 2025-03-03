CREATE TABLE IF NOT EXISTS users
(
    user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_username TEXT NOT NULL UNIQUE,
    user_password TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS api_tokens
(
    token_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    token            TEXT   NOT NULL UNIQUE,
    token_name       TEXT   NOT NULL UNIQUE,
    token_expiration BIGINT NOT NULL DEFAULT 0,
    token_ip_address TEXT   DEFAULT NULL,
    user_id          INTEGER,
    FOREIGN KEY (user_id)
        REFERENCES users (user_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS transcribe_config
(
    transcribe_config_id            INTEGER PRIMARY KEY AUTOINCREMENT,

    -- String fields
    language                        TEXT    DEFAULT NULL, -- nullable
    transcribe_config_name          TEXT    NOT NULL UNIQUE,
    prefix                          TEXT    DEFAULT NULL, -- nullable
    initial_prompt                  TEXT    DEFAULT NULL, -- nullable
    prepend_punctuations            TEXT    DEFAULT '"''“¿([{-', -- nullable
    append_punctuations             TEXT    DEFAULT '"''.。,，!！?？:：”)]}、', -- nullable
    hotwords                        TEXT    DEFAULT NULL, -- nullable

    -- Boolean fields
    log_progress                    BOOLEAN NOT NULL DEFAULT 0,
    condition_on_previous_text      BOOLEAN NOT NULL DEFAULT 1,
    suppress_blank                  BOOLEAN NOT NULL DEFAULT 1,
    word_timestamps                 BOOLEAN NOT NULL DEFAULT 0,
    without_timestamps              BOOLEAN NOT NULL DEFAULT 0,
    multilingual                    BOOLEAN NOT NULL DEFAULT 0,

    -- Integer fields
    batch_size                      INTEGER NOT NULL DEFAULT 8,
    beam_size                       INTEGER NOT NULL DEFAULT 5,
    best_of                         INTEGER NOT NULL DEFAULT 5,
    no_repeat_ngram_size            INTEGER NOT NULL DEFAULT 0,
    language_detection_segments     INTEGER NOT NULL DEFAULT 1,
    max_new_tokens                  INTEGER DEFAULT NULL, -- nullable
    chunk_length                    INTEGER DEFAULT NULL, -- nullable

    -- Float fields
    patience                        REAL    NOT NULL DEFAULT 1.0,
    length_penalty                  REAL    NOT NULL DEFAULT 1.0,
    repetition_penalty              REAL    NOT NULL DEFAULT 1.0,
    prompt_reset_on_temperature     REAL    NOT NULL DEFAULT 0.5,
    max_initial_timestamp           REAL    NOT NULL DEFAULT 1.0,
    hallucination_silence_threshold REAL    DEFAULT NULL, -- nullable
    compression_ratio_threshold     REAL    NOT NULL DEFAULT 2.4,
    log_prob_threshold              REAL    NOT NULL DEFAULT -1.0,
    no_speech_threshold             REAL    NOT NULL DEFAULT 0.6,
    language_detection_threshold    REAL    NOT NULL DEFAULT 0.5,

    -- Fields that may be arrays/lists (store as TEXT or JSON)
    temperature                     TEXT    NOT NULL DEFAULT '[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]',
    suppress_tokens                 TEXT    NOT NULL DEFAULT '[-1]',
    clip_timestamps                 TEXT    DEFAULT NULL
);


CREATE TABLE IF NOT EXISTS vad_config
(
    vad_config_id           INTEGER PRIMARY KEY AUTOINCREMENT,

    -- The foreign key references transcribe_config.id
    transcribe_config_id    INTEGER NOT NULL,
    vad_filter                      BOOLEAN NOT NULL DEFAULT 0,
    threshold               REAL    NOT NULL DEFAULT 0.5,
    neg_threshold           REAL    DEFAULT NULL,
    min_speech_duration_ms  INTEGER NOT NULL DEFAULT 250,
    min_silence_duration_ms INTEGER NOT NULL DEFAULT 100,
    speech_pad_ms           INTEGER NOT NULL DEFAULT 30,

    -- For referential integrity
    FOREIGN KEY (transcribe_config_id)
        REFERENCES transcribe_config (transcribe_config_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS amplify_config
(
    amplify_config_id           INTEGER PRIMARY KEY AUTOINCREMENT,

    -- The foreign key references transcribe_config.id
    transcribe_config_id    INTEGER NOT NULL,
    amplify                 BOOLEAN NOT NULL DEFAULT 0,
    safety_db               REAL    NOT NULL DEFAULT 0.99,
    margin_factor           REAL    NOT NULL DEFAULT 0.5,
    chunk_ms                INTEGER NOT NULL DEFAULT 100,

    -- For referential integrity
    FOREIGN KEY (transcribe_config_id)
        REFERENCES transcribe_config (transcribe_config_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS tone_removal_config
(
    tone_removal_config_id           INTEGER PRIMARY KEY AUTOINCREMENT,

    -- The foreign key references transcribe_config.id
    transcribe_config_id            INTEGER NOT NULL,
    tone_removal                    BOOLEAN NOT NULL DEFAULT 0,
    matching_threshold              REAL    NOT NULL DEFAULT 2.5,
    time_resolution_ms              INTEGER NOT NULL DEFAULT 25,
    tone_a_min_length               REAL    NOT NULL DEFAULT 0.7,
    tone_b_min_length               REAL    NOT NULL DEFAULT 2.7,
    hi_low_interval                 REAL    NOT NULL DEFAULT 0.2,
    hi_low_min_alternations         INTEGER NOT NULL DEFAULT 6,
    long_tone_min_length            REAL    NOT NULL DEFAULT 3.8,
    enable_mdc                      BOOLEAN NOT NULL DEFAULT 0,
    enable_dtmf                     BOOLEAN NOT NULL DEFAULT 0,

    -- For referential integrity
    FOREIGN KEY (transcribe_config_id)
        REFERENCES transcribe_config (transcribe_config_id)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);