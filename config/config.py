# config/config.py
import yaml
from pathlib import Path

def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load configuration from a YAML file.

    Parameters:
    - config_path (str): Path to the YAML configuration file.

    Returns:
    - config (dict): Configuration as a dictionary.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    return config
