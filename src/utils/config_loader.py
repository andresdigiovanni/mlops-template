import os
from typing import Any, Dict

import yaml


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Loads the YAML configuration file and returns a dictionary.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the file doesn't exist or parsing fails.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if config is None:
                raise FileNotFoundError("Configuration file is empty or malformed.")
            return config

    except yaml.YAMLError as e:
        raise FileNotFoundError(f"Error parsing the config file: {e}")
