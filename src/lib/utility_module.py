import json
import logging
import time

module_logger = logging.getLogger('icad_transcribe.utility')

def get_max_content_length(default_size_mb=5):
    try:
        # Attempt to retrieve and convert the max file size to an integer
        max_file_size = int(default_size_mb)
    except (ValueError, TypeError) as e:
        # Log the error and exit if the value is not an integer or not convertible to one
        module_logger.error(f'Max File Size Must be an Integer: {e}')
        time.sleep(5)
        exit(1)
    else:
        # Return the size in bytes
        return max_file_size * 1024 * 1024

def merge_dicts(original, updates):
    """
    Recursively merge two dictionaries.

    Args:
        original (dict): The original dictionary to be updated.
        updates (dict): The dictionary with updates.
    """
    for key, value in updates.items():
        if isinstance(value, dict) and key in original and isinstance(original[key], dict):
            merge_dicts(original[key], value)
        else:
            original[key] = value

def deep_update(original, updates):
    """
    Recursively update 'original' dict with values from 'updates'.
    If a key exists in both and both values are dicts, recurse.
    Otherwise, overwrite the value in 'original' with 'updates'.
    """
    for key, value in updates.items():
        if (key in original
                and isinstance(original[key], dict)
                and isinstance(value, dict)):
            deep_update(original[key], value)
        else:
            original[key] = value
    return original


def load_json(input_data):
    """
    Loads JSON data from a file-like object or a string.

    Parameters:
    -----------
    input_data : str or file-like object
        The input containing JSON data. If input_data has a `read` method, it will be treated
        as a file-like object, and the function will attempt to read from it. If input_data is
        a string, the function will attempt to decode it as JSON directly.

    Returns:
    --------
    tuple
        A tuple where the first element is the loaded JSON data or None if an error occurs,
        and the second element is an error message or None if no error occurs.
    """
    try:
        # Check if input_data is file-like; it must have a 'read' method
        if hasattr(input_data, 'read'):
            # Assuming input_data is a file-like object with 'read' method
            data = json.loads(input_data.read())
        else:
            # Assuming input_data is a string
            data = json.loads(input_data)
        return {"success": True, "message": "JSON loaded successfully", "result": data}
    except json.JSONDecodeError as e:
        return {"success": False, "message": f"JSON Decode Error: {e}", "result": []}
    except Exception as e:
        return {"success": False, "message": f"Unexpected Error loading JSON data: {e}", "result": []}

def normalize_param(search_params, param_name, expected_type=int):
    """
    Normalize query parameters to handle integers, strings, comma-separated lists, or Python-style lists.

    Args:
        search_params (dict): Query parameters as a dictionary.
        param_name (str): The parameter name to normalize.
        expected_type (type): The expected type of the elements (e.g., int, str).

    Returns:
        int, str, list, or None: A single value or list of values in the expected type, or None if the parameter is missing.

    Raises:
        ValueError: If elements cannot be converted to the expected type.
    """
    if param_name in search_params and search_params[param_name]:
        values = search_params[param_name]

        # Convert single-item list to a single value
        if isinstance(values, list) and len(values) == 1:
            values = values[0]

        # Handle Python-style lists (e.g., [fire dispatch,law-tac])
        if isinstance(values, str) and values.startswith("[") and values.endswith("]"):
            try:
                values = values[1:-1].split(",")
                return [expected_type(x.strip()) for x in values]
            except ValueError:
                raise ValueError(f"All elements in '{param_name}' must be convertible to {expected_type.__name__}.")

        # Handle comma-separated values (e.g., fire dispatch,law-tac)
        if isinstance(values, str) and "," in values:
            try:
                return [expected_type(x.strip()) for x in values.split(",")]
            except ValueError:
                raise ValueError(f"All elements in '{param_name}' must be convertible to {expected_type.__name__}.")

        # Handle single value (e.g., fire dispatch or an integer like 1234)
        try:
            return expected_type(values)
        except ValueError:
            raise ValueError(f"'{param_name}' must be convertible to {expected_type.__name__}.")

    return None

def parse_bool_as_int(value: str, field_name: str) -> int:
    """
    Parses a string "1" -> int(1), "0" -> int(0).
    Raises ValueError if invalid.
    """
    if value == "1":
        return 1
    elif value == "0":
        return 0
    raise ValueError(f"{field_name} must be '1' or '0'. Got: {value!r}")

def parse_float(value: str, field_name: str) -> float:
    """
    Parse a float from a string. Raises ValueError if invalid or missing.
    """
    if value is None or value.strip() == "":
        raise ValueError(f"{field_name} is required and must be a float.")
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{field_name} must be a valid float. Got: {value!r}")

def parse_int(value: str, field_name: str) -> int:
    """
    Parse a non‐nullable integer field. Raises ValueError if invalid or missing.
    """
    if value is None or value.strip() == "":
        raise ValueError(f"{field_name} is required and must be an integer.")
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{field_name} must be a valid integer. Got: {value!r}")

def parse_nullable_int(value: str, field_name: str) -> int | None:
    """
    Parse an integer or return None if empty.
    """
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{field_name} must be an integer or empty. Got: {value!r}")

def parse_nullable_float(value: str, field_name: str) -> float | None:
    """
    Parse a float from a string, or return None if empty/None.
    """
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{field_name} must be a valid float or empty. Got: {value!r}")

def parse_nullable_string(value: str, field_name: str) -> str | None:
    """
    Return the string if non-empty, else None. Allows a truly empty field to become None.
    """
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None

def parse_csv_of_floats(value: str, field_name: str) -> list[float]:
    """
    Parse a comma‐separated string of floats, e.g. "0.0,0.2,0.4".
    Raises ValueError if any item is not a valid float or if the value is empty.
    """
    if value is None or value.strip() == "":
        raise ValueError(f"{field_name} is required and must be a comma‐separated list of floats.")
    try:
        return [float(item.strip()) for item in value.split(",")]
    except ValueError:
        raise ValueError(f"{field_name} must be a comma‐separated list of floats. Got: {value!r}")

def parse_csv_of_integers(value: str, field_name: str) -> list[int]:
    """
    Parse a comma‐separated string of integers, e.g. "1,2,3".
    Raises ValueError if any item is not a valid float or if the value is empty.
    """
    if value is None or value.strip() == "":
        raise ValueError(f"{field_name} is required and must be a comma‐separated list of integers.")
    try:
        return [int(item.strip()) for item in value.split(",")]
    except ValueError:
        raise ValueError(f"{field_name} must be a comma‐separated list of floats. Got: {value!r}")

def parse_nullable_csv_of_floats(value: str | None, field_name: str) -> list[float] | None:
    """
    Parse a comma‐separated string of floats, e.g. "0.0,0.2,0.4".
    If empty or None, return None.
    Raises ValueError if any item is not a valid float.
    """
    if value is None or not value.strip():
        return None
    try:
        return [float(item.strip()) for item in value.split(",")]
    except ValueError:
        raise ValueError(f"{field_name} must be a comma‐separated list of floats. Got: {value!r}")
