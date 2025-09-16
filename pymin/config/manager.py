"""
Configuration management for PyMin package.
Supports loading, merging, and validating configuration files in YAML format.
"""

import dataclasses
import logging
from pathlib import Path

import yaml

from .defaults import UNSET, CONFIG_PATH, DEFAULT_CONFIG_PATH
from .models import PyMinConfig


class ConfigManager:
    """Configuration manager for CMS ECAL scales and smearings analysis."""

    def __init__(self, logger=UNSET):
        self.config_dir = Path(__file__).parent.parent / CONFIG_PATH
        if logger is not UNSET:
            self.logger = logger
        else:

            self.logger = logging.getLogger(__name__)

    def load_config(
        self, config_file: str = UNSET, workflow: str = UNSET
    ) -> PyMinConfig:
        """Load configuration from YAML files with inheritance."""
        # Start with default config
        base_config = self._load_yaml(self.config_dir / DEFAULT_CONFIG_PATH)

        # Override with workflow-specific config if specified
        if workflow != UNSET and workflow:
            workflow_file = self.config_dir / "workflows" / f"{workflow}.yaml"
            if workflow_file.exists():
                workflow_config = self._load_yaml(workflow_file)
                base_config = self._merge_configs(base_config, workflow_config)

        # Override with user-specific config if specified
        if config_file != UNSET and config_file:
            user_config = self._load_yaml(Path(config_file))
            base_config = self._merge_configs(base_config, user_config)

        return self._dict_to_dataclass(base_config)

    def _load_yaml(self, file_path: Path) -> dict:
        """Load a YAML file and return its contents as a dictionary."""
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    def _merge_configs(self, base: dict, override: dict) -> dict:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    def _dict_to_dataclass(self, config_dict: dict) -> PyMinConfig:
        """Convert nested dictionary to dataclass structure."""
        return PyMinConfig(**config_dict)

    def create_config_template(self, output_path: str):
        """Create a template configuration file."""
        template = PyMinConfig()
        config_dict = self._dataclass_to_dict(template)

        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def _dataclass_to_dict(self, obj) -> dict:
        """Convert dataclass to dictionary for YAML serialization."""
        if dataclasses.is_dataclass(obj):
            result = {}
            for attr_field in dataclasses.fields(obj):
                value = getattr(obj, attr_field.name)
                if value is not UNSET:
                    if dataclasses.is_dataclass(value):
                        result[attr_field.name] = self._dataclass_to_dict(value)
                    elif isinstance(value, dict):
                        result[attr_field.name] = value
                    else:
                        result[attr_field.name] = value
            return result
        else:
            return obj
