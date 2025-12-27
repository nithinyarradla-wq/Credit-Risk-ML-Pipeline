"""Configuration loader utility for the credit risk pipeline."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


class ConfigLoader:
    """Loads and manages configuration from YAML files."""

    _instance: Optional["ConfigLoader"] = None
    _config: Optional[Dict[str, Any]] = None

    def __new__(cls) -> "ConfigLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        if self._config is None:
            self._config_path = config_path or self._find_config_path()
            self._config = self._load_config()

    def _find_config_path(self) -> str:
        """Find the config file path relative to the project root."""
        current = Path(__file__).resolve()
        for parent in current.parents:
            config_file = parent / "config" / "config.yaml"
            if config_file.exists():
                return str(config_file)
        raise FileNotFoundError("Could not find config.yaml in project hierarchy")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(self._config_path, "r") as f:
            return yaml.safe_load(f)

    @property
    def config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary."""
        return self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot-notation key."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value

    def get_project_root(self) -> Path:
        """Get the project root directory."""
        return Path(self._config_path).parent.parent

    def get_data_path(self, data_type: str) -> Path:
        """Get path for data directories."""
        root = self.get_project_root()
        path_key = f"paths.{data_type}"
        relative_path = self.get(path_key, f"data/{data_type}")
        return root / relative_path


def get_config() -> ConfigLoader:
    """Get the singleton configuration loader instance."""
    return ConfigLoader()
