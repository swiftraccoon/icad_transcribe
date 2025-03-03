class WhisperConfigError(Exception):
    """Exception raised when whisper configuration data cannot be retrieved."""
    pass

class WhisperModelAccessError(Exception):
    """Exception raised when the model.bin file cannot be accessed due to permissions."""
    pass

class TranscriptionError(Exception):
    """Exception raised when audio transcription fails."""
    pass
